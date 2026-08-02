//! The attention projections and the two dense layers' feed-forward networks,
//! which are 66% of a decode step — 52% for the five projections of every layer
//! and 14% for the two feed-forward networks.
//!
//! **The term is the multiply, not the decode.** The experts came here because
//! decoding them was 32 GB a step; these come here for a different reason. Of a
//! step, `dequantize_blocks_into` is 12% and `ops::linear` is 66%, and it is
//! these weights that most of both is spent on — so what a dispatch replaces
//! here is mostly the arithmetic. A serial f32 dot product is what no compiler
//! may vectorise, because f32 addition is not associative and reassociating it
//! is not a transformation LLVM is allowed to make; the 9 GB of dequantisation
//! that fed it comes off as well, but it was never the larger half.
//!
//! **Every one of these is read in full by every token.** That is what separates
//! them from the experts: a routed bank is 256 experts of which a step reads six,
//! so wrapping one is mostly a promise not to read it, and the gather is what
//! keeps that promise. A projection has no such axis — there is nothing here to
//! be selective about — so these are [`PackedProjection`]s, which is a bank of
//! one expert every row goes through, and the only thing that stays packed is
//! the weight itself.
//!
//! **Twenty-six dispatches a layer that routes and no submission of its own** —
//! eighteen where the two dense layers hold a feed-forward network in place of
//! two banks, and one command buffer for as many layers as a run merges. `q`, `k`, `v` and `r` consume the same normed hidden state and
//! nothing of each other; the two short convolutions consume two of those and
//! are consumed by the two head norms and the attention step; `o_proj` consumes
//! what the step produced; the layer's first convolution consumes that and adds
//! the layer's input as it writes; the second norm consumes the sum; the MLP's
//! dispatches consume what the norm left; and the layer's second convolution
//! consumes what the MLP answered with and adds the sum the norm was taken of.
//! Every arrow in that is a device buffer, so the whole of a layer is one
//! command buffer — see [`LayerDevice`], which is where an activation is formed
//! and consumed without the CPU seeing it twenty-five times over, and
//! [`ModelLayers`], which is where the buffer stays open across the layer
//! boundary too. That is what a [`Batch`] is for: the last forty-one submissions
//! a decode step made were worth 12.7 ms of it.
//!
//! **What took longest to reach was not the arithmetic but the state.** Four of
//! those operations write something that outlives the call — three convolutions'
//! windows, and the span the step attends over — so a backend could not run them
//! without also holding that state, and a backend that did not run them had to
//! close its command buffer in the middle of the layer to let the CPU.
//! [`Projections::layer`] is the seam that moved first and [`DecoderDevice`] is
//! where it stopped, with [`AttentionCache::seen`] the whole of what a sequence
//! carries once the rest is here.

use std::cell::RefCell;

use inkling_core::attention::{AttentionCache, AttentionStep, LayerStep, Projections, Qkvr, Sdpa};
use inkling_core::layer::{DecoderCache, DecoderDevice, DecoderStep, Experts, LayerMlp, Passed};
use inkling_core::mask::BandedMask;
use inkling_core::ops::{MlpProjections, Projection};
use inkling_core::profile::{self, Op};
use inkling_core::weights::{LayerBackend, LayerBanks, LayerPacked, Packed, PackedMlp};

use crate::attention::{AttentionError, FusedAttention, KeyValues, LayerAttention, Step};
use crate::buffer::{Buffer, Landing};
use crate::device::{Device, MetalError};
use crate::experts::{ExpertKernels, LayerExperts};
use crate::kernel::{Batch, Submitted};
use crate::matmul::{MatmulError, PackedMatmul, PackedProjection, Pending, together};
use crate::norm::{LayerNorm, RmsNorm};
use crate::sconv::{LayerConv, ShortConvolution};
use crate::swiglu::SwiGlu;

/// One attention layer's five projections on the device.
///
/// The mirror of [`DecodedProjections`](inkling_core::DecodedProjections), and
/// it holds the same relation to it that
/// [`ExpertBanks`](crate::ExpertBanks) holds to
/// [`PackedExperts`](inkling_core::PackedExperts): the arithmetic is the
/// checkpoint's, and what changes is that no weight is ever decoded to memory.
#[derive(Debug)]
pub struct LayerProjections<'a> {
    /// The norm whose output the first four consume, resident beside them.
    ///
    /// Here rather than left to the CPU because of what it lets the four do:
    /// its answer stays in a device buffer they read directly, so the normed
    /// hidden state is never a `Vec<f32>` anywhere — see
    /// [`LayerProjections::normed_qkvr`].
    input_layernorm: LayerNorm<'a>,
    q_proj: PackedProjection<'a>,
    k_proj: PackedProjection<'a>,
    v_proj: PackedProjection<'a>,
    r_proj: PackedProjection<'a>,
    o_proj: PackedProjection<'a>,
    /// The attention step between the four and the fifth, resident for the same
    /// reason the norm in front of them is: what it holds is the layer's band,
    /// and what that buys is that `o_proj` reads a buffer rather than a value
    /// this process formed — see [`LayerProjections::attend`].
    attention: LayerAttention<'a>,
    /// The two short convolutions between `k`, `v` and the step, resident for
    /// the reason the norm and the step are and for one more: they carry a
    /// window from one call to the next, so where they run decides where that
    /// window lives — see [`LayerProjections::layer`].
    k_sconv: LayerConv<'a>,
    v_sconv: LayerConv<'a>,
    /// The two RMSNorms over each head's channels, resident for the same reason
    /// again. The query's writes the `[heads, queries, head_dim]` the step reads
    /// and the key's writes into the span, so between them the last two values
    /// a layer's attention formed here stop being formed here at all.
    q_norm: LayerNorm<'a>,
    k_norm: LayerNorm<'a>,
}

/// The four kernels a layer's own operations dispatch through, compiled once for
/// the whole model.
///
/// Held together because that is what they are: none of the four names a shape,
/// so one pipeline each serves all forty-two layers, and a layer standing itself
/// up needs all four. Compiling them per layer would be forty-two trips through
/// the Metal compiler for four source strings.
#[derive(Debug)]
pub struct LayerKernels {
    matmul: PackedMatmul,
    norm: RmsNorm,
    conv: ShortConvolution,
    attention: FusedAttention,
}

impl LayerKernels {
    pub fn compile(device: &Device) -> Result<Self, MetalError> {
        Ok(Self {
            matmul: PackedMatmul::new(device)?,
            norm: RmsNorm::new(device)?,
            conv: ShortConvolution::new(device)?,
            attention: FusedAttention::new(device)?,
        })
    }

    /// The packed matmul, which the head and the expert banks dispatch through
    /// too — they are the same kernel over the same format, and a second
    /// compilation of it would be a second pipeline for one source string.
    pub fn matmul(&self) -> &PackedMatmul {
        &self.matmul
    }

    /// The attention step, which the MTP heads dispatch through too: a head's
    /// block is a decoder layer, so the operation is the same one and the
    /// source names no shape.
    pub fn attention(&self) -> &FusedAttention {
        &self.attention
    }
}

impl<'a> LayerProjections<'a> {
    /// Wrap a layer's five projections where the checkpoint mapped them, with
    /// the norm, the two convolutions and the attention step that sit among
    /// them.
    ///
    /// Nothing checks here that they are one layer's, and nothing here could:
    /// what the five widths have to be is
    /// [`AttentionConfig`](inkling_core::AttentionConfig)'s to say, and
    /// [`Attention::new`](inkling_core::Attention::new) is where they are asked
    /// — of whichever backend answered, so that the two cannot differ.
    ///
    /// Which is why the mapping from name to tensor is what this module's tests
    /// are about. Two of the five pairs the checkpoint gives are the same shape
    /// both ways round — `q_proj` against `o_proj`, `k_proj` against `v_proj` —
    /// so a slot filled from the wrong name is a layer that stands up, checks
    /// out and attends to the wrong thing.
    pub fn wrap(
        device: &'a Device,
        kernels: &'a LayerKernels,
        layer: &LayerPacked<'a>,
        slack: usize,
    ) -> Result<Self, ProjectionError> {
        let (config, packed) = (layer.config, &layer.attention);
        let matmul = &kernels.matmul;
        let sconv = |weight: &[f32]| {
            LayerConv::with_slack(device, &kernels.conv, config.kv_channels(), weight, slack)
        };
        let head_norm =
            |weight: &[f32]| LayerNorm::new(device, &kernels.norm, weight, config.rms_norm_eps);
        Ok(Self {
            input_layernorm: LayerNorm::new(
                device,
                &kernels.norm,
                &layer.input_layernorm,
                config.rms_norm_eps,
            )?,
            attention: LayerAttention::new(device, &kernels.attention, config, &layer.rel_proj)?,
            k_sconv: sconv(&layer.k_sconv)?,
            v_sconv: sconv(&layer.v_sconv)?,
            q_norm: head_norm(&layer.q_norm)?,
            k_norm: head_norm(&layer.k_norm)?,
            q_proj: whole(device, matmul, &packed.q_proj)?,
            k_proj: whole(device, matmul, &packed.k_proj)?,
            v_proj: whole(device, matmul, &packed.v_proj)?,
            r_proj: whole(device, matmul, &packed.r_proj)?,
            o_proj: whole(device, matmul, &packed.o_proj)?,
        })
    }

    /// Take back the last `rows` timesteps of everything this layer's attention
    /// carries for a sequence: the keys, the values, and the two convolution
    /// windows behind the key and the value.
    ///
    /// All three or none — a sequence whose keys stop one timestep before its
    /// windows do is one that still attends, over a position it half took back.
    pub fn rewind(&self, rows: usize) {
        self.attention.rewind(rows);
        self.k_sconv.rewind(rows);
        self.v_sconv.rewind(rows);
    }

    /// That the step being asked for is over the shape this layer was wrapped
    /// for.
    ///
    /// The shape is checked rather than taken, for the reason
    /// [`LayerProjections::normed_qkvr`] checks its norm: the step carries the
    /// widths its caller derived and this holds the widths it was wrapped for,
    /// and the two are separate copies of one fact. A step paired with another
    /// layer's band is buffers of a plausible size read under the wrong shape,
    /// which is a wrong answer rather than a panic.
    fn shaped_for(&self, sdpa: Sdpa, mask: BandedMask<'_>) {
        let wrapped = self.attention.config();
        assert_eq!(
            [
                sdpa.heads(),
                sdpa.kv_heads(),
                sdpa.head_dim(),
                mask.d_rel(),
                mask.rel_extent(),
                mask.sliding(),
            ],
            [
                wrapped.heads,
                wrapped.kv_heads,
                wrapped.head_dim,
                wrapped.d_rel,
                self.attention.rel_extent(),
                wrapped.sliding,
            ],
            "the layer's step against the shape wrapped for it"
        );
    }

    /// Take the layer's state for a sequence that has seen `keys` keys, with
    /// room for `queries` more.
    ///
    /// The span and the two convolution windows are one sequence's state and are
    /// started over together — a sequence that has seen no keys has seen no
    /// timesteps either. Which sequence's they are is
    /// [`LayerAttention::hold`]'s to refuse, and it refuses for all three: the
    /// windows advance exactly when the span does.
    fn starting(&self, keys: usize, queries: usize) {
        if keys == 0 {
            self.k_sconv.restart();
            self.v_sconv.restart();
        }
        self.attention
            .hold(keys, queries)
            .unwrap_or_else(|err| panic!("the layer's span did not grow: {err}"));
    }

    /// The whole of a layer's attention encoded into `batch`: eleven dispatches
    /// from the hidden state to what `o_proj` returns, with nothing in between
    /// leaving the device.
    ///
    /// The order is the layer's own and every arrow in it is a buffer:
    ///
    /// ```text
    /// input_layernorm ─┬─ q_proj ────────────── q_norm ──┐
    ///                  ├─ k_proj ── k_sconv ─── k_norm ──┤ (into the span)
    ///                  ├─ v_proj ── v_sconv ─────────────┤ (into the span)
    ///                  └─ r_proj ────────────────────────┴─ attend ─ o_proj
    /// ```
    ///
    /// **The value has no head norm and the key does**, which is not a symmetry
    /// this lost: the reference norms the key after its convolution and leaves
    /// the value alone. So the value's convolution is the last thing to touch it
    /// and lands its rows in the span directly, where the key's convolution
    /// lands in a buffer its norm reads.
    /// Everything a call has to settle before a dispatch is encoded: that the
    /// step is over the shape this was wrapped for, that its convolutions and
    /// head norms are the ones this holds, and that the span and windows belong
    /// to this sequence. Answers how many queries the call is.
    ///
    /// Apart from [`LayerProjections::encoding`] because a caller running the
    /// whole decoder layer has one more window to start over — the layer's own
    /// residual convolution — and has to do it before the same command buffer is
    /// opened.
    fn beginning(&self, cache: &mut AttentionCache, step: LayerStep<'_>, queries: usize) -> usize {
        self.shaped_for(step.sdpa, step.mask);
        self.starting(cache.seen(), queries);
        assert_eq!(
            [
                step.k_sconv.kernel_size(),
                step.v_sconv.kernel_size(),
                step.q_norm.len(),
                step.k_norm.len(),
            ],
            [
                self.k_sconv.taps(),
                self.v_sconv.taps(),
                self.q_norm.width(),
                self.k_norm.width(),
            ],
            "the layer's convolutions and head norms against the ones wrapped for it"
        );
        queries
    }

    /// The layer's input layernorm over rows already on the device, or `None`
    /// where the step says they arrive normalised — in which case the four
    /// projections read those rows themselves.
    ///
    /// Apart from [`LayerProjections::encoding`] because of what else reads the
    /// rows it was given. A caller running the whole decoder layer adds the
    /// layer's first residual to `x`, so `x` has to outlive the norm over it;
    /// one that runs the attention alone has nothing else to do with it.
    ///
    /// `weight` and `eps` arrive and are checked rather than used: this holds
    /// the same norm already, uploaded once at wrap time where the CPU path
    /// widens its copy out of the mapping on every step.
    pub(crate) fn input_norm(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
        step: LayerStep<'_>,
    ) -> Result<Option<Buffer<f32>>, MetalError> {
        let Some(weight) = step.input_layernorm else {
            return Ok(None);
        };
        assert_eq!(
            weight.len(),
            self.input_layernorm.width(),
            "the layer's norm against the one wrapped for it"
        );
        assert_eq!(step.eps, self.input_layernorm.eps(), "rms_norm_eps");
        self.input_layernorm.encode(batch, x).map(Some)
    }

    fn encoding(
        &self,
        batch: &mut Batch<'_>,
        span: &mut KeyValues,
        normed: &mut Buffer<f32>,
        step: LayerStep<'_>,
    ) -> Result<Pending, MatmulError> {
        let device = self.q_proj.device();
        let mut q = self.q_proj.encode_over(batch, normed)?.buffer();
        let mut k = self.k_proj.encode_over(batch, normed)?.buffer();
        let mut v = self.v_proj.encode_over(batch, normed)?.buffer();
        let mut rel = self.r_proj.encode_over(batch, normed)?.buffer();

        let queries = q.len() / (step.sdpa.heads() * step.sdpa.head_dim());
        let (keys, values) = span.landings();
        let mut k = self.k_sconv.encode(batch, &mut k, None, 1.0)?;
        self.v_sconv.encode_over(batch, &mut v, None, 1.0, values)?;

        let mut headed = device.zeroed::<f32>(q.len())?;
        self.q_norm.encode_over(
            batch,
            &mut q,
            step.q_taus,
            Landing {
                out: &mut headed,
                groups: step.sdpa.heads(),
                stride: queries,
                base: 0,
            },
        )?;
        self.k_norm.encode_over(batch, &mut k, None, keys)?;

        // **The span grows here rather than when the batch completes**, because
        // the step below is what has to see this call's keys and it is in the
        // same command buffer as the two dispatches that wrote them. Metal's
        // default dispatch type is serial, so those writes are done before the
        // step reads them — and a span that grew after the wait would attend
        // over the previous step's keys and leave this token out of its own row.
        span.appended(queries);
        let mut attended = self.attention.encode_over(
            batch,
            span,
            &mut headed,
            &mut rel,
            step.bias_taus,
            step.q_offset,
        )?;
        self.o_proj.encode_over(batch, &mut attended)
    }
}

/// What standing one layer's own weights up on the device can fail with, which
/// is a multiply's failures and the attention step's.
#[derive(Debug, thiserror::Error)]
pub enum ProjectionError {
    #[error(transparent)]
    Metal(#[from] MetalError),

    #[error(transparent)]
    Matmul(#[from] MatmulError),

    #[error(transparent)]
    Attention(#[from] AttentionError),
}

impl Projections for LayerProjections<'_> {
    /// The four that consume the normed hidden state, in one command buffer.
    ///
    /// Four submissions become one, which at 225 microseconds a submission is
    /// 675 µs a layer and 28 ms of a decode step — against the 105 µs the
    /// arithmetic of one of these projections takes. They are independent of
    /// each other, so the only thing the batch orders is what they cost.
    fn qkvr(&self, x: &[f32]) -> Qkvr {
        let [q, k, v, r] = together(self.q_proj.device(), |batch| {
            Ok([
                self.q_proj.encode(batch, x)?,
                self.k_proj.encode(batch, x)?,
                self.v_proj.encode(batch, x)?,
                self.r_proj.encode(batch, x)?,
            ])
        })
        .unwrap_or_else(|err| panic!("the layer's projections did not run: {err}"));
        Qkvr { q, k, v, r }
    }

    /// The layer's input layernorm and those same four, in one command buffer,
    /// with the normed state never leaving the device.
    ///
    /// **This is the pattern the rest of the residency is made of.** The four
    /// dispatches read the buffer the norm's dispatch wrote — Metal's default
    /// dispatch type is serial, so the ordering is the command buffer's — and
    /// what used to be a `Vec<f32>` formed on the CPU, copied over four times
    /// and dropped is now a value that exists only in device memory. The
    /// submission count does not move: five dispatches where there were four,
    /// in the one command buffer that already held them.
    ///
    /// `weight` and `eps` arrive and are checked rather than used: this holds
    /// the same norm already, uploaded once at wrap time where the CPU path
    /// widens its copy out of the mapping on every step.
    ///
    /// Two copies of one tensor, then, and what says they agree is that both are
    /// `{layer}.input_layernorm.weight` widened out of the same read-only
    /// mapping — [`CheckpointWeights`](inkling_core::CheckpointWeights) names it
    /// once for the backend and once for the layer, and a checkpoint does not
    /// change under a running process. The assertions cover the two things that
    /// are cheap to check and would be a wrong answer rather than a panic: a
    /// norm of another layer's width, and an `eps` from another config.
    /// Comparing the values themselves would cost more per call than the
    /// normalisation does.
    fn normed_qkvr(&self, x: &[f32], weight: &[f32], eps: f32) -> Qkvr {
        assert_eq!(
            weight.len(),
            self.input_layernorm.width(),
            "the layer's norm against the one wrapped for it"
        );
        assert_eq!(eps, self.input_layernorm.eps(), "rms_norm_eps");

        let device = self.q_proj.device();
        let [q, k, v, r] = together(device, |batch| {
            let mut input = device.buffer(x)?;
            let mut normed = self.input_layernorm.encode(batch, &mut input)?;
            Ok([
                self.q_proj.encode_over(batch, &mut normed)?,
                self.k_proj.encode_over(batch, &mut normed)?,
                self.v_proj.encode_over(batch, &mut normed)?,
                self.r_proj.encode_over(batch, &mut normed)?,
            ])
        })
        .unwrap_or_else(|err| panic!("the layer's norm and projections did not run: {err}"));
        Qkvr { q, k, v, r }
    }

    /// The attention step and `o_proj`, in one command buffer, with the mask the
    /// step needs never built and what the step produced never leaving the
    /// device.
    ///
    /// **Two tensors this process would otherwise hold do not exist.** The
    /// additive `[heads, queries, keys]` mask is not formed at all — the kernel
    /// derives each entry where it scores the key it belongs to — and the step's
    /// own `[queries, heads * head_dim]` answer stays in the buffer `o_proj`
    /// reads, which is the pattern [`LayerProjections::normed_qkvr`] establishes
    /// at the other end of the layer.
    ///
    /// **The step alone, over a span the caller holds.** A layer's own step is
    /// [`LayerProjections::layer`]'s, which reads the span this layer keeps and
    /// puts the whole of a layer's attention in one command buffer; this is what
    /// is left for a caller that has the keys and values here — the oracle the
    /// cases in this module measure that one against.
    fn attend(&self, step: AttentionStep<'_>) -> Vec<f32> {
        self.shaped_for(step.sdpa, step.mask);

        let [out] = together(self.q_proj.device(), |batch| {
            let mut attended = self.attention.encode(
                batch,
                Step {
                    q: step.q,
                    k: step.k,
                    v: step.v,
                    rel: step.rel,
                    taus: step.taus,
                    q_offset: step.q_offset,
                },
            )?;
            Ok([self.o_proj.encode_over(batch, &mut attended)?])
        })
        .unwrap_or_else(|err| panic!("the layer's attention step did not run: {err}"));
        out
    }

    /// The whole layer, over a span of keys and values this layer keeps.
    ///
    /// **What the residency buys here is the copy that grows with the
    /// context.** [`LayerProjections::attend`] above is handed the whole cached
    /// span as a slice and allocates and copies all of it onto the device on
    /// every layer of every step, where every key but the newest was already
    /// there. Held by [`LayerAttention`], a step copies
    /// the key it made — and the `[keys, kv_heads * head_dim]` to `[kv_heads,
    /// keys, head_dim]` transpose the CPU path runs over the whole span
    /// alongside it becomes the indexing of that one write.
    ///
    /// **One submission a layer.** Nothing between the hidden state this is
    /// handed and the `[queries, hidden]` it returns is a value this process
    /// forms or reads: the normed state, the four projections' outputs, the two
    /// convolutions', the two head norms' and the attention step's are each a
    /// buffer the next dispatch reads.
    ///
    /// **The state is what made that possible.** Four of the eleven write
    /// something that outlives the call — the two convolutions' windows, and the
    /// keys and values the step attends over — so running them here is only
    /// coherent because the layer holds all four. What crosses back is the
    /// answer and nothing else.
    fn layer(&self, cache: &mut AttentionCache, step: LayerStep<'_>) -> Option<Vec<f32>> {
        let queries = self.beginning(cache, step, step.x.len() / self.input_layernorm.width());
        let device = self.q_proj.device();
        let mut span = self.attention.span();
        let [out] = together(device, |batch| {
            let mut x = device.buffer(step.x)?;
            let mut normed = self.input_norm(batch, &mut x, step)?;
            let normed = normed.as_mut().unwrap_or(&mut x);
            Ok([self.encoding(batch, &mut span, normed, step)?])
        })
        .unwrap_or_else(|err| panic!("the layer's attention did not run: {err}"));
        cache.appended(queries);
        cache.convolved(queries);
        Some(out)
    }

    fn q_proj(&self) -> &dyn Projection {
        &self.q_proj
    }

    fn k_proj(&self) -> &dyn Projection {
        &self.k_proj
    }

    fn v_proj(&self) -> &dyn Projection {
        &self.v_proj
    }

    fn r_proj(&self) -> &dyn Projection {
        &self.r_proj
    }

    fn o_proj(&self) -> &dyn Projection {
        &self.o_proj
    }
}

/// One dense layer's feed-forward network on the device.
///
/// `3 x [16384, 4096]`, which is the widest weight in the model below the head
/// and four and a half times a layer's five attention projections together. Two
/// layers of forty-two have one.
#[derive(Debug)]
pub struct DenseFfn<'a> {
    gate_proj: PackedProjection<'a>,
    up_proj: PackedProjection<'a>,
    down_proj: PackedProjection<'a>,
    /// The activation between the pair and the third, which is not a weight and
    /// belongs to no layer — one pipeline serves the whole model, and it is here
    /// for the reason [`ExpertBanks`](crate::ExpertBanks) holds one: encoded
    /// between them, the network is four dispatches in one command buffer and
    /// nothing it computes is ever a `Vec<f32>`.
    swiglu: &'a SwiGlu,
}

impl<'a> DenseFfn<'a> {
    /// Wrap a dense layer's three where the checkpoint mapped them. Whether they
    /// pair is [`DenseMlp`](inkling_core::DenseMlp)'s to say, and `gate_proj`
    /// against `up_proj` is the pair that pairs either way round.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        swiglu: &'a SwiGlu,
        packed: &PackedMlp<'a>,
    ) -> Result<Self, MatmulError> {
        Ok(Self {
            gate_proj: whole(device, matmul, &packed.gate_proj)?,
            up_proj: whole(device, matmul, &packed.up_proj)?,
            down_proj: whole(device, matmul, &packed.down_proj)?,
            swiglu,
        })
    }

    /// The whole network encoded into `batch`, over rows a dispatch already left
    /// on the device: the pair, the activation between them, and `down` over
    /// what it produced.
    ///
    /// The same four dispatches an expert bank is — see
    /// [`ExpertBanks::encode`](crate::ExpertBanks::encode) — over a bank of one
    /// expert every row goes through. What is left for the caller is the trailing
    /// `global_scale`, which is `InklingDenseMLP`'s rather than `SwiGLUMLP`'s and
    /// is not 1.
    pub(crate) fn encode_into(
        &self,
        batch: &mut Batch<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Pending, MatmulError> {
        let mut gate = self.gate_proj.encode_over(batch, x)?.buffer();
        let mut up = self.up_proj.encode_over(batch, x)?.buffer();
        self.swiglu.encode(batch, &mut gate, &mut up)?;
        self.down_proj.encode_over(batch, &mut gate)
    }
}

impl MlpProjections for DenseFfn<'_> {
    /// The two that consume the same input, in one command buffer — the same
    /// bargain [`LayerProjections::qkvr`] strikes, over a pair.
    fn gate_up(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let [gate, up] = together(self.gate_proj.device(), |batch| {
            Ok([
                self.gate_proj.encode(batch, x)?,
                self.up_proj.encode(batch, x)?,
            ])
        })
        .unwrap_or_else(|err| panic!("the feed-forward network did not run: {err}"));
        (gate, up)
    }

    fn gate_proj(&self) -> &dyn Projection {
        &self.gate_proj
    }

    fn up_proj(&self) -> &dyn Projection {
        &self.up_proj
    }

    fn down_proj(&self) -> &dyn Projection {
        &self.down_proj
    }
}

/// A whole packed tensor as the projection it is: every row, and one expert
/// every row of a call goes through.
///
/// The head is the only weight that wants fewer rows than it holds — its 966
/// padding rows are the truncation [`inkling_core::head`] describes — and a
/// layer's projections have no padding to stop short of.
fn whole<'a>(
    device: &'a Device,
    matmul: &'a PackedMatmul,
    packed: &Packed<'a>,
) -> Result<PackedProjection<'a>, MatmulError> {
    PackedProjection::wrap_packed(device, matmul, packed, packed.slices())
}

/// Every layer on the device, which for Inkling-Small is 42 attentions, two
/// dense feed-forward networks and forty pairs of expert banks.
///
/// 1.19 GB of packed projections beside 137 GB of packed banks — 9.6 GB and 1.1
/// TB were either decoded — wrapped where the checkpoint mapped them and holding
/// no resident set of their own. That is the residency policy stated: all of it,
/// at load, in about six milliseconds, holding nothing. The alternatives — copy
/// every layer at construction, or copy a layer the first time a token routes
/// into it — are both answers to a question about how much of 137 GB to move,
/// and the answer here is none of it.
#[derive(Debug)]
pub struct ModelLayers<'a> {
    /// Indexed by layer, `None` where nothing here answers for one — which is a
    /// layer the CPU keeps, and is how a partial handover stays expressible.
    layers: Vec<Option<LayerDevice<'a>>>,
    device: &'a Device,
    /// What a run may retain before it has to end — see
    /// [`ModelLayers::carries`] and [`RETAINED_BUDGET`], which is what
    /// [`ModelLayers::wrap`] puts here.
    budget: u64,
    /// The command buffer a run of layers is being encoded into, and what the
    /// last of them left in it — `None` between runs.
    ///
    /// Behind a cell for the reason a layer's own resident tensors are: a run
    /// belongs to the call that started it and not to this, which is borrowed
    /// immutably by every layer it holds.
    carried: RefCell<Option<Carried<'a>>>,
    /// The run's command buffers that have been committed and not yet waited
    /// for, oldest first — which is the order one queue runs them in and the
    /// order they are waited for in.
    ///
    /// Empty between runs, and that is what
    /// [`LayerBackend::rewind`](inkling_core::LayerBackend::rewind) asserts on
    /// beside the open buffer: a window this side shifts is a window a dispatch
    /// still in flight may be reading.
    flight: RefCell<Vec<Submitted<'a>>>,
}

/// A run of layers part way through: the command buffer they are being encoded
/// into, the last layer encoded, the `[tokens, hidden]` it wrote, and what the
/// device had allocated when the run opened.
///
/// **Nothing here has run yet.** The dispatches are in the buffer and the buffer
/// is not committed, so what `rows` names is memory the next layer's first
/// dispatch will read and nobody will look at in between — which is the whole of
/// what a merged run is.
struct Carried<'a> {
    batch: Batch<'a>,
    at: usize,
    rows: Buffer<f32>,
    /// [`Device::allocated_bytes`] as this run opened, which is what the same
    /// reading is measured against to say what the run is holding.
    opened: u64,
}

/// By hand because a [`Batch`] is not printable, and because what a reader wants
/// of a run part way through is where it has got to rather than what is in it.
impl std::fmt::Debug for Carried<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Carried")
            .field("at", &self.at)
            .field("rows", &self.rows.len())
            .finish_non_exhaustive()
    }
}

/// What a merged run may hold before it ends and another begins.
///
/// **A run retains every intermediate of every layer in it until the command
/// buffer completes**, and that is what merging is paid for in: the round trips
/// of every layer boundary it crosses, against the memory those layers' values
/// are held in a moment longer. What the cost is *not* is a row count. A layer
/// allocates the same buffers for three rows as for one — a normed state, four
/// projections, two convolutions, two head norms, what the step and `o_proj`
/// produced, and the eight expert rows a token routes through — and what changes
/// with the call is how long each of them is. So the thing to bound is the
/// bytes, and a row count bounds them only for one width of model.
///
/// **What this bounds is what merging adds to a peak, and that is exact.** A run
/// ends once it has reached this budget, so what it holds is this plus the layer
/// that crossed it — and a layer holds what it would have held on its own,
/// unmerged. So a call whose own layer already reaches this budget merges
/// nothing and costs exactly what it costs today, which is what keeps a long
/// prefill one submission a layer; and no call anywhere can peak more than this
/// budget above the call it replaced.
///
/// **The figure is the deepest block this engine can ask for.** The checkpoint
/// ships eight multi-token prediction heads, so the widest verify block a round
/// can propose is nine rows, and this is set above what nine rows of the whole
/// stack come to by the arithmetic of the checkpoint's own shapes — which is
/// what makes every block this engine can speculate one command buffer.
///
/// **And the arithmetic agrees with what a step allocates.** A decode step
/// retains 17.6 MiB, which
/// `the_generated_tokens_match_the_oracle_with_the_model_on_the_device` prints,
/// so this admits about nine rows of these shapes — the deepest block the eight
/// heads can propose, which `what_a_speculative_round_costs_and_what_it_buys`
/// reports at every width through nine as the two submissions a single row
/// takes, one for the run of layers and one for the head. The same figure is
/// what keeps a prefill out: ten rows already pass this, so any prompt reaches
/// it at its first layer and stays a submission a layer.
const RETAINED_BUDGET: u64 = 160 << 20;

/// How many dispatches a run encodes into one command buffer before committing
/// it and carrying on into the next, without waiting for either.
///
/// **A command buffer executes nothing until it is committed**, so a run that
/// commits once at the end has the device idle for every microsecond it spent
/// encoding — 4.4 ms of a 26.4 ms decode step, ahead of an 18.6 ms wait rather
/// than inside it. Committing part way through is what puts the two beside each
/// other, and the arithmetic says the CPU wins the race easily: a decode step's
/// dispatches are 4.5 µs each to encode and 16 µs each to run, so once the first
/// buffer is in flight nothing this process does can fall behind the device.
///
/// **So what this number decides is the ramp and not the overlap**: it is how
/// much encoding happens before the GPU has anything at all, against a fixed
/// cost per command buffer the driver charges whatever is in it. Both ends are
/// visible in a sweep of it over a decode step — 43 submissions read 20.93 ms
/// and 3 read 22.28, where 15 read 19.88 — and the middle of that range is flat
/// enough that no two runs order it the same way, so this is chosen as about
/// three layers rather than fitted to the best figure.
///
/// It is dispatches rather than layers because both costs it trades are per
/// dispatch, and because a run may be handed a block of nine rows as easily as
/// one and the dispatches are the same either way.
pub const DISPATCHES_A_SUBMISSION: usize = 64;

/// One whole decoder layer on the device: its attention, the convolution and
/// residual add behind that, the second norm, its MLP, and the convolution and
/// residual add behind *that*.
///
/// **What this is that [`LayerProjections`] is not is the MLP**, and that is the
/// whole of why it exists. Everything else here was already reachable from the
/// attention — the convolution on the residual path reads what `o_proj` wrote,
/// the add reads the layer's input, the second norm reads the add — but a
/// backend that stopped at the norm would have closed its command buffer there
/// for the MLP's first dispatch to open another. Holding both, it does not; and
/// holding the MLP is what let the second residual path follow, since what that
/// convolution reads is the one value the MLP produces.
#[derive(Debug)]
pub struct LayerDevice<'a> {
    attention: LayerProjections<'a>,
    /// The convolution on the layer's first residual path, which carries the
    /// layer's input as a second addend — see [`LayerConv::encode_over`].
    attn_sconv: LayerConv<'a>,
    /// The layer's second norm, between that add and the MLP.
    post_attention_layernorm: LayerNorm<'a>,
    /// `None` where this backend holds the layer's attention and not its MLP,
    /// which is the partial handover [`LayerBackend::decoder`] answers `None`
    /// for.
    mlp: Option<LayerMlpDevice<'a>>,
    /// The convolution on the layer's second residual path, which carries `h`
    /// as a second addend the way the first carries the layer's input — and
    /// which is the last dispatch of the layer, so what it writes is what the
    /// next layer is handed.
    mlp_sconv: LayerConv<'a>,
}

/// Whichever MLP a layer index called for, on the device: `InklingDenseMLP`
/// below `dense_mlp_idx` and `InklingSparseMoE` above it.
///
/// The mirror of [`LayerMlp`](inkling_core::LayerMlp), and boxed on both sides
/// because a layer holds one of them and the two are hundreds of bytes apart.
#[derive(Debug)]
enum LayerMlpDevice<'a> {
    Dense(Box<DenseFfn<'a>>),
    Sparse(Box<LayerExperts<'a>>),
}

/// What a wrapped stack is, beside the weights: how many layers it has, the
/// width they map through, and how many timesteps each has to be able to give
/// back.
///
/// Three numbers rather than three arguments because they are one answer — the
/// model's shape, as the thing wrapping it has to be told — and because the
/// last of them is the only one a caller decides: `slack` is how far ahead the
/// run will speculate, and a run that speculates nothing passes zero and wraps
/// what this always wrapped. See [`crate::LayerConv::with_slack`].
#[derive(Debug, Clone, Copy)]
pub struct StackShape {
    pub layers: usize,
    pub dim: usize,
    pub slack: usize,
}

impl<'a> ModelLayers<'a> {
    /// Wrap every projection `packed` names and every bank `banks` names, over
    /// the stack `stack` describes.
    ///
    /// The stack's length is stated rather than read off the last entry: a
    /// backend answering for none of the last layers would otherwise report a
    /// shorter stack than the model has, and "past the stack" and "left to the
    /// CPU" would stop being answerable apart.
    pub fn wrap(
        device: &'a Device,
        kernels: &'a LayerKernels,
        experts: ExpertKernels<'a>,
        packed: &[LayerPacked<'a>],
        banks: &[LayerBanks<'a>],
        stack: StackShape,
    ) -> Result<Self, ProjectionError> {
        let StackShape { layers, dim, slack } = stack;
        let mut wrapped: Vec<Option<LayerDevice<'a>>> = (0..layers).map(|_| None).collect();
        for layer in packed {
            let residual =
                |weight: &[f32]| LayerConv::with_slack(device, &kernels.conv, dim, weight, slack);
            wrapped[layer.layer] = Some(LayerDevice {
                attention: LayerProjections::wrap(device, kernels, layer, slack)?,
                attn_sconv: residual(&layer.attn_sconv)?,
                post_attention_layernorm: LayerNorm::new(
                    device,
                    &kernels.norm,
                    &layer.post_attention_layernorm,
                    layer.config.rms_norm_eps,
                )?,
                mlp: layer
                    .dense_mlp
                    .map(|mlp| {
                        DenseFfn::wrap(device, &kernels.matmul, experts.swiglu, &mlp)
                            .map(|ffn| LayerMlpDevice::Dense(Box::new(ffn)))
                    })
                    .transpose()?,
                mlp_sconv: residual(&layer.mlp_sconv)?,
            });
        }
        for bank in banks {
            let Some(held) = wrapped[bank.layer].as_mut() else {
                continue;
            };
            let sparse = LayerExperts::wrap(device, experts, bank, dim)?;
            held.mlp = Some(LayerMlpDevice::Sparse(Box::new(sparse)));
        }
        Ok(Self {
            layers: wrapped,
            device,
            budget: RETAINED_BUDGET,
            carried: RefCell::new(None),
            flight: RefCell::new(Vec::new()),
        })
    }

    /// How many of the stack's layers are here at all, which is how many have
    /// their attention projections here.
    pub fn layers(&self) -> usize {
        self.layers.iter().flatten().count()
    }

    /// How many of those have a dense feed-forward network here, which is how
    /// many are dense.
    pub fn dense_layers(&self) -> usize {
        self.held(|mlp| matches!(mlp, LayerMlpDevice::Dense(_)))
    }

    /// How many have expert banks here, which is how many are MoE.
    pub fn expert_layers(&self) -> usize {
        self.held(|mlp| matches!(mlp, LayerMlpDevice::Sparse(_)))
    }

    fn held(&self, of: impl Fn(&LayerMlpDevice<'a>) -> bool) -> usize {
        self.layers
            .iter()
            .flatten()
            .filter(|layer| layer.mlp.as_ref().is_some_and(&of))
            .count()
    }

    fn layer(&self, layer: usize) -> Option<&LayerDevice<'a>> {
        self.layers.get(layer)?.as_ref()
    }
}

/// The seam [`inkling_core::weights`] names, so that a layer standing itself up
/// does not know whether any of its weights was ever decoded.
impl LayerBackend for ModelLayers<'_> {
    fn attention(&self, layer: usize) -> Option<&dyn Projections> {
        Some(&self.layer(layer)?.attention as &dyn Projections)
    }

    fn dense_mlp(&self, layer: usize) -> Option<&dyn MlpProjections> {
        match self.layer(layer)?.mlp.as_ref()? {
            LayerMlpDevice::Dense(ffn) => Some(&**ffn as &dyn MlpProjections),
            LayerMlpDevice::Sparse(_) => None,
        }
    }

    fn experts(&self, layer: usize) -> Option<&dyn Experts> {
        match self.layer(layer)?.mlp.as_ref()? {
            LayerMlpDevice::Sparse(experts) => Some(&**experts as &dyn Experts),
            LayerMlpDevice::Dense(_) => None,
        }
    }

    /// A layer whose attention *and* MLP are both here, which is the condition
    /// for one command buffer — and `None` for one that is only half here, which
    /// still runs, one submission either side of the norm the CPU would apply.
    ///
    /// The stack rather than the layer answers, because whether a layer's output
    /// crosses back is a question about the layer *after* it — see
    /// [`ModelLayers::run`].
    fn decoder(&self, layer: usize) -> Option<&dyn DecoderDevice> {
        self.whole(layer)?;
        Some(self as &dyn DecoderDevice)
    }

    /// Every layer this holds, because a sequence is in every one of them.
    ///
    /// A layer the CPU kept has nothing here to take back and is skipped rather
    /// than refused: its state is the cache, which the caller has already
    /// rewound.
    ///
    /// **Between runs, and that is what the first line says.** A rewind reads
    /// and writes windows the device holds, so the dispatches that wrote them
    /// have to have run — and a run part way through is a command buffer with
    /// dispatches in it that have not. There is no such run here when a caller
    /// asks: what closes one is the last layer of the stack, whose rows every
    /// caller reads back before it can know there is anything to take back.
    fn rewind(&self, rows: usize) {
        assert!(
            self.carried.borrow().is_none() && self.flight.borrow().is_empty(),
            "a rewind while a run of layers is still being encoded or still in flight"
        );
        for layer in self.layers.iter().flatten() {
            layer.rewind(rows);
        }
    }
}

impl LayerDevice<'_> {
    /// Take back the last `rows` timesteps of everything this layer holds for
    /// the sequence in flight: its attention's keys and two windows, and the
    /// two convolutions on its residual paths.
    ///
    /// Four windows and a span, which is the whole of what
    /// [`DecoderCache`](inkling_core::DecoderCache) holds for a layer that runs
    /// here — see [`crate::LayerConv::rewind`] for what they need of the
    /// caller.
    fn rewind(&self, rows: usize) {
        self.attention.rewind(rows);
        self.attn_sconv.rewind(rows);
        self.mlp_sconv.rewind(rows);
    }
}

impl<'a> ModelLayers<'a> {
    /// Layer `layer` where this holds the whole of it.
    fn whole(&self, layer: usize) -> Option<&LayerDevice<'a>> {
        let held = self.layer(layer)?;
        held.mlp.as_ref()?;
        Some(held)
    }

    /// Whether what layer `layer` produces can stay where it is, which is the
    /// whole of what decides a round trip.
    ///
    /// **Two conditions, and the second is the memory one.** The layer after
    /// this one has to be here whole, or there is nobody to read the buffer. And
    /// the run has to have room: what a merged run holds is every intermediate
    /// of every layer in it until the buffer completes, which at a decode step
    /// is a few megabytes and over a long enough prefill would be gigabytes —
    /// while what merging buys shrinks as the work inside a submission grows, a
    /// round trip being the same 290 microseconds whatever is in it. So a run
    /// ends where the bytes it is holding reach [`RETAINED_BUDGET`], and a call
    /// wide enough that one layer reaches it on its own is one command buffer a
    /// layer.
    ///
    /// **`retained` is measured rather than derived**, as the bytes this device
    /// has allocated since the run opened. Nothing allocated inside a run can be
    /// freed before it completes, because the command buffer holds a reference
    /// to everything bound into it — so the reading is what the run is still
    /// carrying, and it needs no model of what a layer's dispatches ask for. A
    /// span that doubled while the run was open is counted too, which overstates
    /// a run by a buffer that outlives it and is the conservative direction.
    fn carries(&self, layer: usize, retained: u64) -> bool {
        retained < self.budget && self.whole(layer + 1).is_some()
    }

    /// The layer encoded into the run's command buffer, opening one where there
    /// is none and waiting for it where there is nothing left to carry to.
    fn encode(
        &self,
        layer: usize,
        cache: &mut DecoderCache,
        step: DecoderStep<'_>,
        held: &LayerDevice<'a>,
    ) -> Result<Passed, ProjectionError> {
        let mut carried = self.carried.borrow_mut();
        let (mut batch, mut x, opened) = match carried.take() {
            Some(run) => {
                assert_eq!(
                    run.at + 1,
                    layer,
                    "layer {layer} was handed what layer {} left",
                    run.at
                );
                (run.batch, run.rows, run.opened)
            }
            None => {
                // A run that opens while another's buffers are still in flight
                // would allocate against a budget that has already been spent
                // and would attend over a span a running dispatch is still
                // appending to. Every way a run can end drains them, so this
                // holds by construction and is asserted where it would break.
                debug_assert!(
                    self.flight.borrow().is_empty(),
                    "a run of layers opened while another was still in flight"
                );
                // Read before the row this call is handed is allocated, so that
                // the first thing a run holds is charged to it.
                let opened = self.device.allocated_bytes();
                (
                    self.device.batch()?,
                    self.device.buffer(step.attention.x)?,
                    opened,
                )
            }
        };

        let rows = held.encode_into(&mut batch, cache, step, &mut x)?;
        if self.carries(layer, self.device.allocated_bytes() - opened) {
            *carried = Some(Carried {
                batch: self.relayed(batch)?,
                at: layer,
                rows,
                opened,
            });
            return Ok(Passed::Carried(step.queries));
        }

        self.flight.borrow_mut().push(batch.submit());
        self.landed()?;
        Ok(Passed::Rows(profile::timed(Op::Readback, || rows.to_vec())))
    }

    /// The buffer the rest of the run encodes into: the one it is holding, or a
    /// fresh one where that has enough in it to be worth the device starting on
    /// — see [`DISPATCHES_A_SUBMISSION`].
    ///
    /// The committed one is not waited for. Ordering is the queue's, so what the
    /// next buffer's first dispatch reads is what this one's last dispatch
    /// wrote, and what the run is holding is unchanged: a command buffer retains
    /// what is bound into it until it completes, whether or not anybody is
    /// waiting, which is what makes [`ModelLayers::carries`]'s budget still the
    /// bound on all of them together.
    /// The buffer the next one is opened before this one is committed, so that
    /// a device that will not open one leaves nothing in flight to be waited
    /// for by a run that has already given up.
    fn relayed(&self, batch: Batch<'a>) -> Result<Batch<'a>, ProjectionError> {
        if batch.dispatches() < DISPATCHES_A_SUBMISSION {
            return Ok(batch);
        }
        let next = self.device.batch()?;
        self.flight.borrow_mut().push(batch.submit());
        Ok(next)
    }

    /// Every command buffer the run committed, waited for oldest first.
    ///
    /// One queue runs them in that order, so waiting for the last would be
    /// enough to know they had all finished — but not enough to know none of
    /// them failed, and not enough to charge each what it cost. So each is
    /// waited for, and all but the last of them are already done.
    ///
    /// **Every one of them, and the first failure reported afterwards.** A `?`
    /// inside the loop would leave the buffers behind a failing one committed
    /// and never waited for, and would empty `flight` as it unwound — so the
    /// next rewind would find nothing in flight and shift a window a running
    /// dispatch is still reading.
    fn landed(&self) -> Result<(), ProjectionError> {
        let mut failed = None;
        for submitted in self.flight.borrow_mut().drain(..) {
            if let Err(err) = submitted.wait() {
                failed.get_or_insert(err);
            }
        }
        failed.map_or(Ok(()), |err| Err(err.into()))
    }
}

/// A run of decoder layers in one command buffer.
///
/// **Where a run ends is where somebody has to read what it produced**, and for
/// a decode step over a stack this holds whole that is the head. Layer `i`'s
/// output is layer `i+1`'s input and nothing else reads it, so the buffer stays
/// open across the layer boundary and what crosses back is the last layer's
/// answer alone — 42 round trips a step become one, and 41 uploads and 41
/// readbacks stop happening at all.
///
/// **What forces a run to end early is stated rather than discovered**: a layer
/// this does not hold whole, the last layer of the stack, or a run that has
/// reached the bytes it may hold — see [`ModelLayers::carries`], which is where
/// the memory a run holds is traded against the round trips it saves.
impl DecoderDevice for ModelLayers<'_> {
    fn run(&self, layer: usize, cache: &mut DecoderCache, step: DecoderStep<'_>) -> Option<Passed> {
        let held = self.whole(layer)?;
        Some(
            self.encode(layer, cache, step, held)
                .unwrap_or_else(|err| panic!("the layer did not run: {err}")),
        )
    }
}

/// The whole of one decoder layer, encoded.
///
/// **Twenty-six dispatches and no submission at all**, where the same operations
/// asked for a piece at a time are two submissions and three CPU rows between
/// them. Eleven are the attention's — see [`LayerProjections::layer`], which is
/// this one step in — and every value from the hidden state this is handed to
/// the one it answers with is a buffer the next dispatch reads: what `o_proj`
/// wrote, the first convolution's rows with the layer's input already added,
/// what the second norm made of that, the gate's logits, the experts the top-k
/// took out of them, each bank's two halves with the activation between them,
/// the softmax over the eight logits that selection named, both banks' rows
/// weighted by it, and the second convolution's rows with `h` already added.
///
/// **What a sequence carries is what decided all of it.** Four operations here
/// write state that outlives the call — the span the step attends over and three
/// convolutions' windows — and the last of those three is `mlp_sconv`, whose
/// rows are what the *next* layer reads. So neither end of a layer is a value
/// this process forms, and whether either crosses back is
/// [`ModelLayers::run`]'s question rather than this one's.
impl LayerDevice<'_> {
    fn encode_into(
        &self,
        batch: &mut Batch<'_>,
        cache: &mut DecoderCache,
        step: DecoderStep<'_>,
        x: &mut Buffer<f32>,
    ) -> Result<Buffer<f32>, ProjectionError> {
        let mlp = self
            .mlp
            .as_ref()
            .expect("a layer run whole holds its own MLP");
        let attention = &self.attention;
        let queries = attention.beginning(cache.attention(), step.attention, step.queries);
        assert_eq!(
            [step.attn_sconv.kernel_size(), step.mlp_sconv.kernel_size()],
            [self.attn_sconv.taps(), self.mlp_sconv.taps()],
            "the layer's residual convolutions against the ones wrapped for it"
        );
        assert_eq!(
            step.post_attention_layernorm.len(),
            self.post_attention_layernorm.width(),
            "the layer's second norm against the one wrapped for it"
        );
        assert_eq!(
            step.eps,
            self.post_attention_layernorm.eps(),
            "rms_norm_eps"
        );

        // Both residual convolutions' windows are this sequence's, and they
        // advance exactly when the span and the two windows inside attention do
        // — which `beginning` has already started over if this sequence has seen
        // nothing.
        if cache.attention().seen() == 0 {
            self.attn_sconv.restart();
            self.mlp_sconv.restart();
        }

        let mut span = attention.attention.span();
        let mut normed = attention
            .input_norm(batch, x, step.attention)?
            .expect("a decoder layer normalises the state it is handed");
        let mut attended = attention
            .encoding(batch, &mut span, &mut normed, step.attention)?
            .buffer();
        let mut h = self.attn_sconv.encode(batch, &mut attended, Some(x), 1.0)?;
        let mut normed = self.post_attention_layernorm.encode(batch, &mut h)?;
        let (projected, scale) = self.projected(batch, &mut normed, queries, mlp, step.mlp)?;
        let out = self
            .mlp_sconv
            .encode(batch, &mut projected.buffer(), Some(&mut h), scale)?;

        // **The count advances here rather than when the buffer completes**, for
        // the reason the span's own does — see `LayerProjections::encoding`. A
        // run of layers is encoded before any of it runs, so a sequence that
        // counted its keys at the wait would count them all after the last
        // layer of the run had already attended.
        cache.attention().appended(queries);
        cache.attention().convolved(queries);
        cache.convolved(queries);
        Ok(out)
    }

    /// What the layer's MLP produced, and the scale its rows still carry.
    ///
    /// **A dense layer's rows are not finished and a routed layer's are.**
    /// `InklingDenseMLP` multiplies what its three projections produced by a
    /// learned `global_scale`, outside the `SwiGLUMLP` body it shares with other
    /// models; a routed layer's two scales are already in the weights its own
    /// router applied — see
    /// [`LayerRouter::encode_weights`](crate::LayerRouter::encode_weights),
    /// where three of the four ways of misreading that gate live. So the scale
    /// goes to the convolution that reads these rows rather than into a dispatch
    /// of its own.
    ///
    /// The pairing is checked rather than assumed: what this holds and the
    /// [`LayerMlp`] the step describes are two copies of one fact, and a layer
    /// whose MLP is not the one wrapped for it would otherwise scale a routed
    /// layer's output by a dense layer's constant.
    fn projected(
        &self,
        batch: &mut Batch<'_>,
        normed: &mut Buffer<f32>,
        queries: usize,
        held: &LayerMlpDevice<'_>,
        mlp: LayerMlp<'_>,
    ) -> Result<(Pending, f32), ProjectionError> {
        Ok(match (held, mlp) {
            (LayerMlpDevice::Dense(ffn), LayerMlp::Dense(dense)) => {
                (ffn.encode_into(batch, normed)?, dense.scale())
            }
            (LayerMlpDevice::Sparse(experts), LayerMlp::Sparse(_)) => {
                (experts.encode_into(batch, normed, queries)?, 1.0)
            }
            _ => panic!("the layer's MLP is not the one wrapped for it"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkling_core::Checkpoint;
    use inkling_core::attention::AttentionConfig;
    use inkling_core::fixture::{self, deviation};
    use inkling_core::mask::BandedMask;
    use inkling_core::ops::{DenseProjection, rms_norm};
    use inkling_core::weights::PackedAttention;

    use inkling_core::attention::{AttentionProjections, AttentionWeights};
    use inkling_core::layer::{DecoderLayer, DecoderWeights, Hidden, NoExperts};
    use inkling_core::ops::DenseMlp;

    use crate::combine::MoeCombine;
    use crate::dense::DenseMatmul;
    use crate::matmul::testing::{Case, pack};
    use crate::router::{Router, RouterWeights};
    use crate::testing::device;

    /// The packed tensors `just dump-quant-fixture` cut out of the checkpoint
    /// that are weights, each with the values MLX decoded it to stored beside
    /// it.
    ///
    /// They are what a hermetic test has to hand out, and they are what makes
    /// the assertions below possible at all: a slot's answer is checked against
    /// *the tensor that slot was named*, so a slot filled from the wrong name is
    /// a different weight rather than a different width.
    ///
    /// The fixture's fourth, `code_grid`, is left out. It is every code under
    /// every scale byte, `0xff` included, so it decodes to infinities and is a
    /// decoder's case rather than a weight anything can be measured against.
    const TENSORS: [&str; 3] = ["dense_ffn", "vocab_padding", "routed_expert"];

    /// The same account as `matmul::tests::TOLERANCE`: decoding is exact on both
    /// sides, so what separates a dispatch from the CPU is summation order, and
    /// what a wrong tensor produces is decades away rather than ulps.
    const TOLERANCE: f32 = 6e-6;

    fn packed<'a>(ckpt: &'a Checkpoint, name: &str) -> Packed<'a> {
        Packed::open(ckpt, name).expect("the fixture holds the slice packed")
    }

    /// The width every tensor in the fixture maps from, which is the model's.
    const IN_DIM: usize = 4096;

    /// A stand-in for a layer's `rms_norm_eps`, which is not 1e-6 so that a path
    /// defaulting to a round number would be a different answer.
    const EPS: f32 = 1.5625e-5;

    /// The checkpoint's own `sconv_kernel_size`, which is what a window of the
    /// wrong depth would be measured against.
    const KERNEL_SIZE: usize = 4;

    /// A row spread over both signs, so that a reduction cancels the way a
    /// trained one does.
    fn row(in_dim: usize) -> Vec<f32> {
        (0..in_dim).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect()
    }

    /// A norm weight that is not all ones, so that a path which dropped it would
    /// be a different answer rather than the same one.
    fn layernorm(width: usize) -> Vec<f32> {
        (0..width).map(|i| 0.5 + (i % 13) as f32 / 16.0).collect()
    }

    /// The shape the layer's attention step is over.
    ///
    /// The checkpoint's own numbers rather than anything the fixture's three
    /// tensors imply: those are whichever slices `just dump-quant-fixture` cut
    /// out, and what this module's cases are about is which *name* fills which
    /// slot. Whether a step reproduces the reference is the step's own tests'
    /// question.
    fn shape() -> AttentionConfig {
        AttentionConfig {
            hidden: IN_DIM,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            d_rel: 16,
            sliding: 512,
            rms_norm_eps: EPS,
            log_scaling: None,
        }
    }

    /// A band that is not all one value, for the reason the norm weight above is
    /// not all ones.
    fn rel_proj() -> Vec<f32> {
        let extent = shape().sliding;
        (0..shape().d_rel * extent)
            .map(|i| ((i % 29) as f32 - 14.0) / 32.0)
            .collect()
    }

    /// That every projection of `named` answers with the tensor it was named,
    /// against MLX's own decoding of that tensor.
    fn each_answers(ckpt: &Checkpoint, named: &[(&str, &dyn Projection)]) {
        for (name, projection) in named {
            let x = row(projection.in_dim());
            let weight = fixture::f32s(&fixture::tensor(ckpt, &format!("{name}.dequantized")));
            let want = DenseProjection::new(x.len(), &weight).forward(&x);

            assert_eq!(projection.out_dim(), want.len(), "{name} maps to");
            let deviation = deviation(&projection.forward(&x), &want);
            assert!(deviation <= TOLERANCE, "{name}: deviation {deviation:e}");
        }
    }

    /// The two tensors the last two slots take, which is what the rounds below
    /// exchange: `(r_proj, o_proj)`.
    const LAST_TWO: [(&str, &str); 2] = [(TENSORS[0], TENSORS[1]), (TENSORS[1], TENSORS[0])];

    /// One attention layer's five tensors: the first three are the fixture's
    /// three, and the last two are whichever round this is.
    fn attention<'a>(ckpt: &'a Checkpoint, (r_proj, o_proj): (&str, &str)) -> PackedAttention<'a> {
        PackedAttention {
            q_proj: packed(ckpt, TENSORS[0]),
            k_proj: packed(ckpt, TENSORS[1]),
            v_proj: packed(ckpt, TENSORS[2]),
            r_proj: packed(ckpt, r_proj),
            o_proj: packed(ckpt, o_proj),
        }
    }

    /// One layer as `CheckpointWeights` hands it over, with the fixture's tensors
    /// in the five slots and this module's stand-ins for everything else.
    fn layer_packed<'a>(ckpt: &'a Checkpoint, names: (&str, &str)) -> LayerPacked<'a> {
        LayerPacked {
            layer: 0,
            attention: attention(ckpt, names),
            dense_mlp: None,
            input_layernorm: layernorm(IN_DIM),
            post_attention_layernorm: layernorm(IN_DIM),
            attn_sconv: residual_sconv(),
            mlp_sconv: residual_sconv(),
            rel_proj: rel_proj(),
            k_sconv: sconv(),
            v_sconv: sconv(),
            q_norm: head_weight(),
            k_norm: head_weight(),
            config: shape(),
        }
    }

    /// A head-norm weight over the channels of one head, not all ones for the
    /// reason the layer norm's is not.
    fn head_weight() -> Vec<f32> {
        layernorm(shape().head_dim)
    }

    /// A convolution kernel that is not all one value, for the reason the norm
    /// weight above is not all ones — and not palindromic per channel, so a
    /// kernel read backwards would be a different answer.
    fn sconv() -> Vec<f32> {
        (0..shape().kv_channels() * KERNEL_SIZE)
            .map(|i| ((i % 7) as f32 - 3.0) / 8.0)
            .collect()
    }

    /// The same, over the hidden state's channels rather than the key's — which
    /// is what the two convolutions on a layer's residual path are over.
    fn residual_sconv() -> Vec<f32> {
        (0..IN_DIM * KERNEL_SIZE)
            .map(|i| ((i % 11) as f32 - 5.0) / 8.0)
            .collect()
    }

    /// Each of the five names wraps the tensor it was given.
    ///
    /// This is the mistake the widths cannot catch. `q_proj` and `o_proj` are
    /// both `[4096, 4096]` in the checkpoint and `k_proj` and `v_proj` are both
    /// `[1024, 4096]`, so either pair exchanged here produces a layer that
    /// stands up, passes every shape check there is, and attends to the wrong
    /// thing.
    ///
    /// Three distinct tensors over five slots means two slots have to repeat two
    /// others, and which two is what the rounds exchange: the first round
    /// repeats at `(q, r)` and `(k, o)`, the second at `(k, r)` and `(q, o)`. No
    /// pair of slots holds the same weight in both, so every one of the ten
    /// exchanges of two names is a wrong answer in at least one round.
    #[test]
    fn each_of_an_attention_layers_five_names_wraps_the_tensor_it_was_given() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let ckpt = fixture::open(fixture::MXFP4);

        for (r_proj, o_proj) in LAST_TWO {
            let five = LayerProjections::wrap(
                &device,
                &kernels,
                &layer_packed(&ckpt, (r_proj, o_proj)),
                0,
            )
            .expect("the five wrap");

            each_answers(
                &ckpt,
                &[
                    (TENSORS[0], five.q_proj()),
                    (TENSORS[1], five.k_proj()),
                    (TENSORS[2], five.v_proj()),
                    (r_proj, five.r_proj()),
                    (o_proj, five.o_proj()),
                ],
            );
        }
    }

    /// And each of a feed-forward network's three does. `silu` goes on the gate
    /// and not on the up, and the two are the same shape, so a network wrapped
    /// with them exchanged is one of exactly the right widths and the wrong
    /// activation.
    #[test]
    fn each_of_a_feed_forward_networks_three_names_wraps_the_tensor_it_was_given() {
        let Some(device) = device() else { return };
        let matmul = PackedMatmul::new(&device).expect("the packed matmul compiles");
        let ckpt = fixture::open(fixture::MXFP4);

        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let three = DenseFfn::wrap(
            &device,
            &matmul,
            &swiglu,
            &PackedMlp {
                gate_proj: packed(&ckpt, TENSORS[0]),
                up_proj: packed(&ckpt, TENSORS[1]),
                down_proj: packed(&ckpt, TENSORS[2]),
            },
        )
        .expect("the three wrap");

        each_answers(
            &ckpt,
            &[
                (TENSORS[0], three.gate_proj()),
                (TENSORS[1], three.up_proj()),
                (TENSORS[2], three.down_proj()),
            ],
        );
    }

    /// The layer's norm and the four projections that consume it, in one
    /// command buffer, against normalising here and asking for the four.
    ///
    /// **The claim the whole seam rests on.** What the device path produces is
    /// never seen by this process — the normed hidden state exists only in a
    /// buffer between two dispatches — so the only way to say it is the right
    /// value is to run the same four projections over a normed state formed
    /// here and compare what came out the far end.
    ///
    /// Two rows rather than one, because a norm reduces over the last axis and a
    /// kernel that reduced over the buffer would agree on a single row.
    ///
    /// The third answer is what says the norm happened at all: the same four
    /// projections over the raw input, which is a layer that stands up and
    /// attends to an unnormalised state.
    #[test]
    fn a_layers_norm_and_its_four_projections_answer_what_the_cpu_answers() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let ckpt = fixture::open(fixture::MXFP4);
        let weight = layernorm(IN_DIM);

        // The fixture's third tensor is a two-expert bank, which as a whole
        // projection maps from 131072 — so the five here are the two that map
        // from the model's own width, which is what asking the four for one
        // input needs them to share.
        let of = |name| packed(&ckpt, name);
        let five = LayerProjections::wrap(
            &device,
            &kernels,
            &LayerPacked {
                attention: PackedAttention {
                    q_proj: of(TENSORS[0]),
                    k_proj: of(TENSORS[1]),
                    v_proj: of(TENSORS[0]),
                    r_proj: of(TENSORS[1]),
                    o_proj: of(TENSORS[0]),
                },
                ..layer_packed(&ckpt, LAST_TWO[0])
            },
            0,
        )
        .expect("the five wrap");

        let x = [row(IN_DIM), row(IN_DIM).iter().map(|v| v * 3.0).collect()].concat();
        let fused = five.normed_qkvr(&x, &weight, EPS);
        let apart = five.qkvr(&rms_norm(&x, &weight, EPS));
        let unnormed = five.qkvr(&x);

        for (name, got, want, raw) in [
            ("q", &fused.q, &apart.q, &unnormed.q),
            ("k", &fused.k, &apart.k, &unnormed.k),
            ("v", &fused.v, &apart.v, &unnormed.v),
            ("r", &fused.r, &apart.r, &unnormed.r),
        ] {
            let agreed = deviation(got, want);
            assert!(agreed <= TOLERANCE, "{name}: deviation {agreed:e}");
            assert!(
                deviation(got, raw) > TOLERANCE,
                "{name}: the norm did not reach the answer"
            );
        }
    }

    /// The attention step and `o_proj` in one command buffer, against building
    /// the mask here and asking for `o_proj`.
    ///
    /// **The claim the second half of the seam rests on**, and the same shape of
    /// claim the norm above makes: what the device path produces in the middle
    /// is never seen by this process — the step's answer exists only in a buffer
    /// between two dispatches — so the only way to say it is the right value is
    /// to run the same `o_proj` over a step taken here and compare what came out
    /// the far end.
    ///
    /// Two queries at a nonzero offset over a span that outruns neither, because
    /// the offset is what a single query at position zero cannot expose.
    ///
    /// The third answer is what says the band reached the step at all: the same
    /// call with the offset dropped, which is a layer that attends over almost
    /// nothing because every key but the first sits at a distance the band reads
    /// as a position that has not happened yet.
    #[test]
    fn a_layers_attention_step_answers_what_the_cpu_answers() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let ckpt = fixture::open(fixture::MXFP4);

        let five = LayerProjections::wrap(&device, &kernels, &layer_packed(&ckpt, LAST_TWO[0]), 0)
            .expect("the five wrap");

        let config = shape();
        let (queries, keys, offset) = (2, 40, 7);
        let (heads, kv_heads, head_dim) = (config.heads, config.kv_heads, config.head_dim);
        let band = rel_proj();
        let (q, k, v) = (
            row(heads * queries * head_dim),
            row(kv_heads * keys * head_dim),
            row(kv_heads * keys * head_dim),
        );
        let rel = row(queries * heads * config.d_rel);
        let attending = |q_offset| AttentionStep {
            sdpa: Sdpa::new(heads, kv_heads, head_dim),
            mask: BandedMask::new(config.d_rel, &band, config.sliding),
            q: &q,
            k: &k,
            v: &v,
            rel: &rel,
            taus: None,
            q_offset,
        };

        let fused = five.attend(attending(offset));
        let apart = five.o_proj().forward(&attending(offset).on_the_cpu());
        let unplaced = five.o_proj().forward(&attending(0).on_the_cpu());

        let agreed = deviation(&fused, &apart);
        assert!(agreed <= TOLERANCE, "deviation {agreed:e}");
        assert!(
            deviation(&fused, &unplaced) > TOLERANCE,
            "the cache's offset did not reach the answer"
        );
    }

    /// Which layer's projections answer for which layer, and which layers this
    /// answers for at all.
    ///
    /// A layer nothing here holds is `None` rather than absent, because that is
    /// what the CPU path reads as "decode them yourself" — and a layer past the
    /// stack is `None` too rather than an index off the end. The stack here is
    /// two longer than the last layer wrapped, which is the case that says the
    /// two answers are not the same answer.
    #[test]
    fn a_layer_this_does_not_hold_and_a_layer_past_the_stack_are_left_to_the_cpu() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let ckpt = fixture::open(fixture::MXFP4);

        // Inkling's shape with a hole punched in it: a dense layer, a gap the
        // CPU keeps, a MoE layer, and two more layers nothing was handed.
        const LAYERS: usize = 5;
        let packed_layers = [
            LayerPacked {
                layer: 0,
                input_layernorm: layernorm(IN_DIM),
                post_attention_layernorm: layernorm(IN_DIM),
                attn_sconv: residual_sconv(),
                mlp_sconv: residual_sconv(),
                rel_proj: rel_proj(),
                k_sconv: sconv(),
                v_sconv: sconv(),
                q_norm: head_weight(),
                k_norm: head_weight(),
                config: shape(),
                attention: attention(&ckpt, LAST_TWO[0]),
                dense_mlp: Some(PackedMlp {
                    gate_proj: packed(&ckpt, TENSORS[0]),
                    up_proj: packed(&ckpt, TENSORS[1]),
                    down_proj: packed(&ckpt, TENSORS[2]),
                }),
            },
            LayerPacked {
                layer: 2,
                input_layernorm: layernorm(IN_DIM),
                post_attention_layernorm: layernorm(IN_DIM),
                attn_sconv: residual_sconv(),
                mlp_sconv: residual_sconv(),
                rel_proj: rel_proj(),
                k_sconv: sconv(),
                v_sconv: sconv(),
                q_norm: head_weight(),
                k_norm: head_weight(),
                config: shape(),
                attention: attention(&ckpt, LAST_TWO[1]),
                dense_mlp: None,
            },
        ];
        let dense = DenseMatmul::new(&device).expect("the dense matmul compiles");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let router = Router::new(&device).expect("the router compiles");
        let weights = RouterWeights::new(&device).expect("the weighting compiles");
        let combine = MoeCombine::new(&device).expect("the combine compiles");
        let experts = ExpertKernels {
            matmul: kernels.matmul(),
            dense: &dense,
            swiglu: &swiglu,
            router: &router,
            weights: &weights,
            combine: &combine,
        };
        let projections = ModelLayers::wrap(
            &device,
            &kernels,
            experts,
            &packed_layers,
            &[],
            StackShape {
                layers: LAYERS,
                dim: IN_DIM,
                slack: 0,
            },
        )
        .expect("the layers wrap");

        assert_eq!(projections.layers(), 2, "two of the five");
        assert_eq!(projections.dense_layers(), 1);
        assert_eq!(projections.expert_layers(), 0, "no banks were handed over");
        assert!(projections.dense_mlp(0).is_some());
        assert!(
            projections.dense_mlp(2).is_none(),
            "a layer that routes to experts has no feed-forward network"
        );
        assert!(
            projections.attention(1).is_none(),
            "a layer left to the CPU"
        );
        assert!(projections.attention(4).is_none(), "the last of the stack");
        assert!(projections.attention(LAYERS).is_none(), "past the stack");

        // **A whole layer is only reachable where the whole layer is here.** The
        // dense one has its feed-forward network and answers for the layer; the
        // MoE one was handed no banks, so it still runs — one submission either
        // side of the norm the CPU applies — and says so by declining.
        assert!(projections.decoder(0).is_some(), "a layer held whole");
        assert!(
            projections.decoder(2).is_none(),
            "a layer whose MLP is not here"
        );
        assert!(projections.decoder(1).is_none(), "a layer left to the CPU");
        assert!(projections.decoder(LAYERS).is_none(), "past the stack");

        // And the two layers wrapped are not each other's, which is what an
        // index off by one would produce. They differ in `o_proj` alone.
        for (layer, (_, o_proj)) in [(0, LAST_TWO[0]), (2, LAST_TWO[1])] {
            let five = projections.attention(layer).expect("a wrapped layer");
            each_answers(&ckpt, &[(o_proj, five.o_proj())]);
        }
    }

    /// A layer narrow enough to stand up out of synthetic weights, which is what
    /// running a *whole* one needs.
    ///
    /// The fixture's three tensors are `[64, 4096]`, `[64, 4096]` and a
    /// two-expert `[2, 32, 4096]` bank — every one of them mapping *from* the
    /// model's width — so no three of them are a feed-forward network, and a
    /// layer without an MLP is exactly the layer this case is about. Uploaded
    /// codes are, and the widths below are Inkling's own ratios shrunk: two
    /// heads over one key head, a query as wide as the hidden state, and a
    /// feed-forward network half of it.
    struct Narrow<'a> {
        device: &'a Device,
        kernels: &'a LayerKernels,
        swiglu: &'a SwiGlu,
        /// Timesteps every one of the layer's four windows can give back, which
        /// is zero for every case but the one that rewinds.
        slack: usize,
    }

    /// The shapes a `Narrow` layer is built to, which have to be whole groups of
    /// 32 codes on every width a weight maps *from*.
    const NARROW: AttentionConfig = AttentionConfig {
        hidden: 128,
        heads: 2,
        kv_heads: 1,
        head_dim: 64,
        d_rel: 8,
        sliding: 32,
        rms_norm_eps: EPS,
        log_scaling: None,
    };

    /// The width the narrow layer's feed-forward network maps through.
    const NARROW_FFN: usize = 64;

    /// `InklingDenseMLP`'s trailing scale, which is not 1 in the checkpoint and
    /// must not be here — a path that dropped it would agree with a path that
    /// dropped it.
    const GLOBAL_SCALE: f32 = 1.75;

    impl<'a> Narrow<'a> {
        /// One packed weight of the given shape, uploaded — the seed is what
        /// makes two of them differ, and against weights that agreed an
        /// exchanged pair would change nothing.
        fn weight(&self, seed: u32, in_dim: usize, out_dim: usize) -> PackedProjection<'a> {
            let case = Case::seeded(seed, in_dim, out_dim, 1);
            PackedProjection::upload(
                self.device,
                self.kernels.matmul(),
                in_dim,
                out_dim,
                &pack(&case.codes),
                &case.scales,
            )
            .expect("the weight's shapes pair")
        }

        fn norm(&self, weight: &[f32]) -> LayerNorm<'a> {
            LayerNorm::new(self.device, &self.kernels.norm, weight, NARROW.rms_norm_eps)
                .expect("the norm uploads")
        }

        fn conv(&self, channels: usize, weight: &[f32]) -> LayerConv<'a> {
            LayerConv::with_slack(
                self.device,
                &self.kernels.conv,
                channels,
                weight,
                self.slack,
            )
            .expect("the kernel uploads")
        }

        /// Three of those layers as a stack a run can be driven across, holding
        /// `budget` bytes before it has to end.
        ///
        /// Three because two cannot tell a run that carried once from a run that
        /// carries everything — the middle layer is the only one that both
        /// consumes a carried buffer and leaves one.
        fn stack(&self, weights: &NarrowWeights, budget: u64) -> ModelLayers<'a> {
            self.stack_of(weights, budget, &[0, 0x99, 0xdd])
        }

        /// A stack a salt to the layer, for a case that needs more of them than
        /// the three a carried run is settled by: a run also splits at a
        /// dispatch count, and one of these layers is eighteen dispatches, so a
        /// stack that reaches [`DISPATCHES_A_SUBMISSION`] has to be longer than
        /// the stack that says what carrying means.
        ///
        /// Every salt different, for [`Narrow::layer`]'s reason: a stack of one
        /// layer repeated cannot tell a run that ran them in order from one that
        /// did not.
        fn stack_of(&self, weights: &NarrowWeights, budget: u64, salts: &[u32]) -> ModelLayers<'a> {
            stack(
                self.device,
                salts
                    .iter()
                    .map(|salt| self.layer(weights, *salt))
                    .collect(),
                budget,
            )
        }

        /// The whole layer on the device, with a dense feed-forward network in
        /// the MLP slot.
        ///
        /// `salt` is what makes two of these different layers rather than one
        /// layer twice, which a case that runs a *pair* of them needs: against
        /// two layers built from the same codes, running them in the wrong order
        /// would change nothing.
        fn layer(&self, weights: &NarrowWeights, salt: u32) -> LayerDevice<'a> {
            let (heads, head_dim) = (NARROW.heads, NARROW.head_dim);
            let kv = NARROW.kv_channels();
            let seed = |base: u32| base + salt;
            LayerDevice {
                attention: LayerProjections {
                    input_layernorm: self.norm(&weights.input_layernorm),
                    attention: LayerAttention::new(
                        self.device,
                        &self.kernels.attention,
                        NARROW,
                        &weights.rel_proj,
                    )
                    .expect("the step stands up"),
                    k_sconv: self.conv(kv, &weights.k_sconv),
                    v_sconv: self.conv(kv, &weights.v_sconv),
                    q_norm: self.norm(&weights.q_norm),
                    k_norm: self.norm(&weights.k_norm),
                    q_proj: self.weight(seed(0x11), NARROW.hidden, heads * head_dim),
                    k_proj: self.weight(seed(0x22), NARROW.hidden, kv),
                    v_proj: self.weight(seed(0x33), NARROW.hidden, kv),
                    r_proj: self.weight(seed(0x44), NARROW.hidden, heads * NARROW.d_rel),
                    o_proj: self.weight(seed(0x55), heads * head_dim, NARROW.hidden),
                },
                attn_sconv: self.conv(NARROW.hidden, &weights.attn_sconv),
                post_attention_layernorm: self.norm(&weights.post_attention_layernorm),
                mlp: Some(LayerMlpDevice::Dense(Box::new(DenseFfn {
                    gate_proj: self.weight(seed(0x66), NARROW.hidden, NARROW_FFN),
                    up_proj: self.weight(seed(0x77), NARROW.hidden, NARROW_FFN),
                    down_proj: self.weight(seed(0x88), NARROW_FFN, NARROW.hidden),
                    swiglu: self.swiglu,
                }))),
                mlp_sconv: self.conv(NARROW.hidden, &weights.mlp_sconv),
            }
        }
    }

    /// The narrow layer's own small tensors, held so that both sides can borrow
    /// the same ones.
    struct NarrowWeights {
        input_layernorm: Vec<f32>,
        post_attention_layernorm: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        k_sconv: Vec<f32>,
        v_sconv: Vec<f32>,
        attn_sconv: Vec<f32>,
        mlp_sconv: Vec<f32>,
        rel_proj: Vec<f32>,
    }

    impl NarrowWeights {
        /// None of them all one value, and no two of them equal — so a slot
        /// filled from the wrong name is a different answer rather than the same
        /// one.
        fn new() -> Self {
            let of = |len: usize, salt: usize| -> Vec<f32> {
                (0..len)
                    .map(|i| 0.5 + ((i * 7 + salt) % 13) as f32 / 16.0)
                    .collect()
            };
            let kv = NARROW.kv_channels();
            Self {
                input_layernorm: of(NARROW.hidden, 1),
                post_attention_layernorm: of(NARROW.hidden, 2),
                q_norm: of(NARROW.head_dim, 3),
                k_norm: of(NARROW.head_dim, 4),
                k_sconv: of(kv * KERNEL_SIZE, 5),
                v_sconv: of(kv * KERNEL_SIZE, 6),
                attn_sconv: of(NARROW.hidden * KERNEL_SIZE, 7),
                mlp_sconv: of(NARROW.hidden * KERNEL_SIZE, 8),
                rel_proj: of(NARROW.d_rel * NARROW.sliding, 9),
            }
        }

        /// The same tensors as `CheckpointWeights` hands a layer, around
        /// whichever projections answer for it.
        fn decoder<'w>(&'w self, five: &'w dyn Projections) -> DecoderWeights<'w> {
            DecoderWeights {
                attention: AttentionWeights {
                    projections: AttentionProjections::backend(five),
                    q_norm: &self.q_norm,
                    k_norm: &self.k_norm,
                    k_sconv: &self.k_sconv,
                    v_sconv: &self.v_sconv,
                    rel_proj: &self.rel_proj,
                },
                input_layernorm: &self.input_layernorm,
                post_attention_layernorm: &self.post_attention_layernorm,
                attn_sconv: &self.attn_sconv,
                mlp_sconv: &self.mlp_sconv,
            }
        }
    }

    /// `[rows, hidden]` spread over both signs, so that a reduction cancels the
    /// way a trained one does.
    fn hidden_rows(rows: usize) -> Vec<f32> {
        (0..rows * NARROW.hidden)
            .map(|i| ((i * 31 % 67) as f32 - 33.0) / 33.0)
            .collect()
    }

    /// **The whole layer in one command buffer is the same layer.**
    ///
    /// This is the claim the seam rests on and the only way to make it is to run
    /// both, because what the fused path forms in the middle is never a value
    /// this process sees: the convolved residual, the second norm's output and
    /// the rows the feed-forward network answered exist only in buffers between
    /// dispatches. So one `DecoderLayer` is driven twice over the same weights —
    /// once with the device holding the whole layer, once with it holding only
    /// the attention and the three operations behind it left here — and what is
    /// compared is the far end.
    ///
    /// **Two calls against one cache**, because three of the layer's windows and
    /// its span are the *device's* rather than the cache's on the fused path, and
    /// a prefill that agreed alone would say nothing about the call that reads
    /// what it left. And a third call from a fresh cache after them, which has to
    /// reproduce the prefill: the windows are the layer's, so what starts them
    /// over is a cache that has seen nothing, and a fused layer that never
    /// restarted its residual convolution would answer the second sequence with
    /// the first one's tail.
    #[test]
    fn a_whole_layer_in_one_command_buffer_answers_what_its_pieces_answer() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let narrow = Narrow {
            device: &device,
            kernels: &kernels,
            swiglu: &swiglu,
            slack: 0,
        };
        let weights = NarrowWeights::new();
        let stack = stack(&device, vec![narrow.layer(&weights, 0)], RETAINED_BUDGET);
        let held = stack.layer(0).expect("the layer is here");

        let mlp = DenseMlp::backend(
            NARROW.hidden,
            NARROW_FFN,
            held.mlp.as_ref().and_then(dense).expect("a dense layer"),
            GLOBAL_SCALE,
        );
        let layer = DecoderLayer::new(
            NARROW,
            weights.decoder(&held.attention),
            LayerMlp::Dense(mlp),
        );

        let (x, more) = (hidden_rows(3), hidden_rows(2));
        let sequence = |device: Option<&dyn DecoderDevice>| {
            let cache = &mut layer.cache();
            let prefill = layer
                .forward(0, cache, Hidden::Rows(&x), &NoExperts, device)
                .rows();
            let rest = layer
                .forward(0, cache, Hidden::Rows(&more), &NoExperts, device)
                .rows();
            (prefill, rest)
        };

        let fused = sequence(Some(&stack as &dyn DecoderDevice));
        let apart = sequence(None);
        for (what, got, want) in [
            ("the prefill", &fused.0, &apart.0),
            ("the continuation", &fused.1, &apart.1),
        ] {
            let agreed = deviation(got, want);
            assert!(agreed <= TOLERANCE, "{what}: deviation {agreed:e}");
        }
        assert!(
            deviation(&fused.1, &apart.0[..more.len()]) > TOLERANCE,
            "two calls of one sequence that agreed would say nothing about the cache"
        );

        // The same first call again, from a cache that has seen nothing — which
        // is what says the fused layer starts its own windows and span over
        // rather than continuing the sequence just run.
        let again = sequence(Some(&stack as &dyn DecoderDevice));
        assert_eq!(again.0, fused.0, "a second sequence's prefill");
        assert_eq!(again.1, fused.1, "a second sequence's continuation");
    }

    /// **Two layers in one command buffer are the same two layers**, and one
    /// submission rather than two.
    ///
    /// Both halves are the commit. That the answers agree says the schedule
    /// changed no arithmetic; that the submission count halves while the
    /// dispatch count does not is the whole reason for it, and it is the half a
    /// test of the values alone would let slip.
    ///
    /// **Three layers rather than two**, because the middle one is the only one
    /// that both consumes a carried buffer and leaves one — the first opens the
    /// run and the last closes it, and a backend that got the middle case wrong
    /// would still pass a pair.
    ///
    /// **And more than one row, because a block of guesses is one.** What a
    /// merged run holds is every intermediate of every layer in it until the
    /// buffer completes, and that is bytes rather than rows — so a call of three
    /// merges the way a call of one does, and what says otherwise is
    /// [`ModelLayers::carries`] over a budget, which the case below drives. The
    /// second call against the same caches is what says the state a carried run
    /// leaves behind is the state the next call reads.
    ///
    /// The three layers are not each other's: they differ in every packed
    /// weight, so a run that ran one of them twice, or ran them in another
    /// order, would be a different answer rather than the same one.
    #[test]
    fn a_run_of_layers_in_one_command_buffer_answers_what_they_answer_apart() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let narrow = Narrow {
            device: &device,
            kernels: &kernels,
            swiglu: &swiglu,
            slack: 0,
        };
        let weights = NarrowWeights::new();
        let stack = narrow.stack(&weights, RETAINED_BUDGET);
        let run = Merged::over(&stack, &weights);

        // A decode step and a block of guesses, which merge alike.
        for rows in [1, 3] {
            let before = device.submissions();
            let merged = run.sequence(Some(&stack as &dyn DecoderDevice), rows);
            let submissions = device.submissions() - before;

            let apart = run.sequence(None, rows);
            for (what, got, want) in [
                ("the first call", &merged.0, &apart.0),
                ("the second", &merged.1, &apart.1),
            ] {
                let agreed = deviation(got, want);
                assert!(agreed <= TOLERANCE, "{rows} rows, {what}: {agreed:e}");
            }
            assert_ne!(merged.0, merged.1, "a second call that read no state");
            assert_eq!(
                submissions, 2,
                "{rows} rows: one submission a call over three layers"
            );
        }
    }

    /// **A run long enough commits part way through and keeps encoding into the
    /// next command buffer**, and what comes out is what the same layers answer
    /// one submission at a time.
    ///
    /// Five layers because that is what it takes: one of these is eighteen
    /// dispatches, so a run splits after the fourth and the fifth is what makes
    /// the second buffer a buffer somebody carries into rather than the tail of
    /// the first. Three layers, which every case above is settled by, never
    /// reach [`DISPATCHES_A_SUBMISSION`] at all.
    ///
    /// The answers are compared because the split is a scheduling decision and
    /// has to be nothing else. The submissions are compared because that is the
    /// decision: a run that stopped splitting would still answer this.
    #[test]
    fn a_run_that_reaches_its_dispatch_count_commits_and_carries_on_encoding() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let narrow = Narrow {
            device: &device,
            kernels: &kernels,
            swiglu: &swiglu,
            slack: 0,
        };
        let weights = NarrowWeights::new();
        let stack = narrow.stack_of(&weights, RETAINED_BUDGET, &[0, 0x99, 0xdd, 0x33, 0x77]);
        let run = Merged::over(&stack, &weights);
        const ROWS: usize = 2;

        let (_, split) = run.retaining(&device, Some(&stack as &dyn DecoderDevice), ROWS);
        assert_eq!(
            split, 2,
            "five layers of eighteen dispatches against a split at {DISPATCHES_A_SUBMISSION}"
        );

        let merged = run.sequence(Some(&stack as &dyn DecoderDevice), ROWS);
        let apart = run.sequence(None, ROWS);
        for (what, got, want) in [
            ("the first call", &merged.0, &apart.0),
            ("the second", &merged.1, &apart.1),
        ] {
            let agreed = deviation(got, want);
            assert!(agreed <= TOLERANCE, "{what}: {agreed:e}");
        }
        assert_ne!(merged.0, merged.1, "a second call that read no state");
    }

    /// **What ends a run is the bytes it is holding**, and the same three layers
    /// over the same rows are one command buffer, two or three depending only on
    /// what they are allowed to hold.
    ///
    /// The budgets are derived from a measurement rather than written down,
    /// because what a layer of these retains is a shape this case does not own —
    /// every buffer between two dispatches of a narrow layer, which the layer
    /// decides and would have to be restated here to be asserted. So one run
    /// with a budget of nothing says what a layer costs, and the two budgets
    /// after it are a fraction of it.
    ///
    /// Three layers because two cannot tell a run of two from a run of
    /// everything, and the answers are compared at every budget: a schedule that
    /// changed an answer would be the finding, whichever way the bytes went.
    #[test]
    fn a_run_ends_where_the_bytes_it_holds_reach_its_budget() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let narrow = Narrow {
            device: &device,
            kernels: &kernels,
            swiglu: &swiglu,
            slack: 0,
        };
        let weights = NarrowWeights::new();
        const ROWS: usize = 2;

        // A budget of nothing is a submission a layer, which is what a run of
        // one layer allocates — and three of them is what the whole call does.
        let alone = narrow.stack(&weights, 0);
        let run = Merged::over(&alone, &weights);
        let (bytes, submissions) = run.retaining(&device, Some(&alone as &dyn DecoderDevice), ROWS);
        assert_eq!(submissions, 3, "a submission a layer");
        let layer = bytes / 3;
        assert!(layer > 0, "a layer that allocated nothing");

        let want = run.sequence(None, ROWS);
        for (budget, submissions, what) in [
            (layer - 1, 3, "a budget one layer cannot fit in"),
            (layer + 1, 2, "a budget of two layers"),
            (3 * layer, 1, "a budget of the whole stack"),
        ] {
            let held = narrow.stack(&weights, budget);
            let run = Merged::over(&held, &weights);
            let (_, got) = run.retaining(&device, Some(&held as &dyn DecoderDevice), ROWS);
            assert_eq!(got, submissions, "{what}");

            let answered = run.sequence(Some(&held as &dyn DecoderDevice), ROWS);
            for (which, got, want) in [
                ("the first call", &answered.0, &want.0),
                ("the second", &answered.1, &want.1),
            ] {
                let agreed = deviation(got, want);
                assert!(agreed <= TOLERANCE, "{what}, {which}: {agreed:e}");
            }
        }
    }

    /// Three layers of the same stack driven as one, which is what a run is
    /// about: the first opens it, the last closes it, and the middle one both
    /// consumes a carried buffer and leaves one.
    struct Merged<'a> {
        layers: Vec<DecoderLayer<'a>>,
    }

    impl<'a> Merged<'a> {
        fn over(stack: &'a ModelLayers<'_>, weights: &'a NarrowWeights) -> Self {
            let layers = (0..stack.layers())
                .map(|at| {
                    let held = stack.layer(at).expect("the layer is here");
                    DecoderLayer::new(
                        NARROW,
                        weights.decoder(&held.attention),
                        LayerMlp::Dense(DenseMlp::backend(
                            NARROW.hidden,
                            NARROW_FFN,
                            held.mlp.as_ref().and_then(dense).expect("a dense layer"),
                            GLOBAL_SCALE,
                        )),
                    )
                })
                .collect();
            Self { layers }
        }

        /// Two calls of `rows` against one set of caches, so that what the
        /// second answers is what the first left behind.
        fn sequence(
            &self,
            device: Option<&dyn DecoderDevice>,
            rows: usize,
        ) -> (Vec<f32>, Vec<f32>) {
            let caches = &mut self
                .layers
                .iter()
                .map(DecoderLayer::cache)
                .collect::<Vec<DecoderCache>>();
            let mut through = |x: &[f32]| {
                let mut h = Passed::Rows(x.to_vec());
                for (at, layer) in self.layers.iter().enumerate() {
                    h = layer.forward(at, &mut caches[at], h.handed(), &NoExperts, device);
                }
                h.rows()
            };
            let first = through(&hidden_rows(rows));
            (first, through(&hidden_rows(rows)))
        }

        /// The bytes one call of `rows` allocated and the command buffers it
        /// went in.
        ///
        /// A stack that has already run rather than a fresh one, so that what is
        /// counted is what a call of these rows costs every time it is made: a
        /// span grows by doubling and a window is started over, and either would
        /// charge whichever call happened to be first for memory the calls after
        /// it do not allocate at all.
        fn retaining(
            &self,
            device: &Device,
            held: Option<&dyn DecoderDevice>,
            rows: usize,
        ) -> (u64, u64) {
            self.sequence(held, rows);
            let (bytes, submissions) = (device.allocated_bytes(), device.submissions());
            self.sequence(held, rows);
            (
                (device.allocated_bytes() - bytes) / 2,
                (device.submissions() - submissions) / 2,
            )
        }
    }

    /// A stack of layers this backend holds whole, which is what a merged run is
    /// asked of — see [`ModelLayers::run`].
    /// **A rejected speculative token leaves nothing behind in a stack that ran
    /// on the device.**
    ///
    /// The CPU path states this over its own five layers — see
    /// [`inkling_core::model`] — and it has to hold here for a different
    /// reason: what a layer left behind is not the cache the caller holds but
    /// the span and the four windows the device holds, so a rewind that reached
    /// only the first would leave a sequence that still runs and attends over a
    /// position it half took back.
    ///
    /// Driven a row at a time over two layers, which is the shape that makes it
    /// worth driving: a decode step is one command buffer spanning both layers,
    /// so the dispatches that write the windows a rewind moves are the ones a
    /// merged run defers — and the wait that has to have happened before a
    /// caller can ask is the one at the end of the stack.
    ///
    /// Exact equality, because both runs are the same dispatches over the same
    /// floats and the only thing a rewind changes is which call wrote a window.
    #[test]
    fn rewinding_a_run_of_layers_that_ran_on_the_device_leaves_them_where_it_found_them() {
        let Some(device) = device() else { return };
        let kernels = LayerKernels::compile(&device).expect("the kernels compile");
        let swiglu = SwiGlu::new(&device).expect("the swiglu compiles");
        let weights = NarrowWeights::new();
        let (x, more) = (hidden_rows(1), hidden_rows(1));
        let wrong: Vec<f32> = more.iter().map(|value| -3.0 * value).collect();

        let run = |slack: usize, rejected: Option<&[f32]>| {
            let narrow = Narrow {
                device: &device,
                kernels: &kernels,
                swiglu: &swiglu,
                slack,
            };
            let stack = stack(
                &device,
                vec![narrow.layer(&weights, 0), narrow.layer(&weights, 0x99)],
                RETAINED_BUDGET,
            );
            let layers: Vec<DecoderLayer<'_>> = (0..2)
                .map(|at| {
                    let held = stack.layer(at).expect("the layer is here");
                    DecoderLayer::new(
                        NARROW,
                        weights.decoder(&held.attention),
                        LayerMlp::Dense(DenseMlp::backend(
                            NARROW.hidden,
                            NARROW_FFN,
                            held.mlp.as_ref().and_then(dense).expect("a dense layer"),
                            GLOBAL_SCALE,
                        )),
                    )
                })
                .collect();

            let held = Some(&stack as &dyn DecoderDevice);
            let caches = &mut [
                DecoderCache::speculating(NARROW, NARROW.hidden, KERNEL_SIZE, slack),
                DecoderCache::speculating(NARROW, NARROW.hidden, KERNEL_SIZE, slack),
            ];
            let through = |x: &[f32], caches: &mut [DecoderCache; 2]| {
                let mut h = Passed::Rows(x.to_vec());
                for (at, layer) in layers.iter().enumerate() {
                    h = layer.forward(at, &mut caches[at], h.handed(), &NoExperts, held);
                }
                h.rows()
            };

            through(&x, caches);
            if let Some(rejected) = rejected {
                through(rejected, caches);
                let rows = rejected.len() / NARROW.hidden;
                for cache in caches.iter_mut() {
                    cache.rewind(rows);
                }
                LayerBackend::rewind(&stack, rows);
            }
            through(&more, caches)
        };

        assert_eq!(run(1, Some(&wrong)), run(1, None), "a row taken back");
        assert_eq!(run(0, None), run(1, None), "windows that kept slack");
    }

    /// A stack of layers this backend holds whole, which is what a merged run is
    /// asked of — see [`ModelLayers::run`].
    ///
    /// `budget` rather than [`RETAINED_BUDGET`] because these layers are 32
    /// wide: what a real layer holds at the widest call this engine is measured
    /// at is what the constant is derived from, and reaching it here would take
    /// a call of a hundred thousand rows. A budget in the same relation to
    /// *these* layers is what makes the same two cases drivable.
    fn stack<'a>(device: &'a Device, held: Vec<LayerDevice<'a>>, budget: u64) -> ModelLayers<'a> {
        ModelLayers {
            layers: held.into_iter().map(Some).collect(),
            device,
            budget,
            carried: RefCell::new(None),
            flight: RefCell::new(Vec::new()),
        }
    }

    /// The feed-forward network of a layer that has one, for a caller that holds
    /// the layer rather than the network.
    fn dense<'m>(mlp: &'m LayerMlpDevice<'_>) -> Option<&'m dyn MlpProjections> {
        match mlp {
            LayerMlpDevice::Dense(ffn) => Some(&**ffn as &dyn MlpProjections),
            LayerMlpDevice::Sparse(_) => None,
        }
    }
}
