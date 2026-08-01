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
//! **One submission a layer, and eleven dispatches in it.** `q`, `k`, `v` and
//! `r` consume the same normed hidden state and nothing of each other; the two
//! short convolutions consume two of those and are consumed by the two head
//! norms and the attention step; and `o_proj` consumes what the step produced.
//! Every arrow in that is a device buffer, so the whole of a layer's attention
//! is one command buffer — see [`LayerProjections::layer`], which is where an
//! activation is formed and consumed without the CPU seeing it, ten times over.
//! That is what a [`Batch`] is for, at 157 µs a
//! marginal submission it is worth 6.6 ms of a decode step.
//!
//! **What took longest to reach was not the arithmetic but the state.** Four of
//! those operations write something that outlives the call — the two
//! convolutions' windows, and the keys and values the step attends over — so a
//! backend could not run them without also holding that state, and a backend
//! that did not run them had to close its command buffer in the middle of the
//! layer to let the CPU. [`Projections::layer`] is the seam that moved, and
//! [`AttentionCache::seen`] is what a sequence carries once the rest is here.

use inkling_core::attention::{AttentionCache, AttentionStep, LayerStep, Projections, Qkvr, Sdpa};
use inkling_core::layer::{
    DecoderCache, DecoderDevice, DecoderHalves, DecoderStep, Experts, LayerMlp,
};
use inkling_core::mask::BandedMask;
use inkling_core::ops::{MlpProjections, Projection};
use inkling_core::profile::{self, Op};
use inkling_core::weights::{LayerBackend, LayerBanks, LayerPacked, Packed, PackedMlp};

use crate::attention::{AttentionError, FusedAttention, KeyValues, LayerAttention, Step};
use crate::buffer::{Buffer, Landing};
use crate::device::{Device, MetalError};
use crate::experts::{Dispatched, ExpertKernels, LayerExperts};
use crate::kernel::Batch;
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
    ) -> Result<Self, ProjectionError> {
        let (config, packed) = (layer.config, &layer.attention);
        let matmul = &kernels.matmul;
        let sconv =
            |weight: &[f32]| LayerConv::new(device, &kernels.conv, config.kv_channels(), weight);
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
    fn beginning(&self, cache: &mut AttentionCache, step: LayerStep<'_>) -> usize {
        self.shaped_for(step.sdpa, step.mask);
        let queries = step.x.len() / self.input_layernorm.width();
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

    fn device(&self) -> &Device {
        self.q_proj.device()
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
        let mut k = self.k_sconv.encode(batch, &mut k, None)?;
        self.v_sconv.encode_over(batch, &mut v, None, values)?;

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
        let queries = self.beginning(cache, step);
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
}

/// One decoder layer on the device: its attention, the convolution and residual
/// add behind that, the second norm, and its MLP.
///
/// **What this is that [`LayerProjections`] is not is the MLP**, and that is the
/// whole of why it exists. Everything else here was already reachable from the
/// attention — the convolution on the residual path reads what `o_proj` wrote,
/// the add reads the layer's input, the second norm reads the add — but a
/// backend that stopped at the norm would have closed its command buffer there
/// for the MLP's first dispatch to open another. Holding both, it does not.
#[derive(Debug)]
pub struct LayerDevice<'a> {
    attention: LayerProjections<'a>,
    /// The convolution on the layer's residual path, which carries the layer's
    /// input as a second addend — see [`LayerConv::encode_over`].
    attn_sconv: LayerConv<'a>,
    /// The layer's second norm, between that add and the MLP.
    post_attention_layernorm: LayerNorm<'a>,
    /// `None` where this backend holds the layer's attention and not its MLP,
    /// which is the partial handover [`LayerBackend::decoder`] answers `None`
    /// for.
    mlp: Option<LayerMlpDevice<'a>>,
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

impl<'a> ModelLayers<'a> {
    /// Wrap every projection `packed` names and every bank `banks` names, over a
    /// stack of `layers` mapping through `dim`.
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
        layers: usize,
        dim: usize,
    ) -> Result<Self, ProjectionError> {
        let mut wrapped: Vec<Option<LayerDevice<'a>>> = (0..layers).map(|_| None).collect();
        for layer in packed {
            wrapped[layer.layer] = Some(LayerDevice {
                attention: LayerProjections::wrap(device, kernels, layer)?,
                attn_sconv: LayerConv::new(device, &kernels.conv, dim, &layer.attn_sconv)?,
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
            });
        }
        for bank in banks {
            let Some(held) = wrapped[bank.layer].as_mut() else {
                continue;
            };
            let sparse = LayerExperts::wrap(device, experts, bank, dim)?;
            held.mlp = Some(LayerMlpDevice::Sparse(Box::new(sparse)));
        }
        Ok(Self { layers: wrapped })
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
    fn decoder(&self, layer: usize) -> Option<&dyn DecoderDevice> {
        let held = self.layer(layer)?;
        held.mlp.as_ref()?;
        Some(held as &dyn DecoderDevice)
    }
}

/// The whole of one decoder layer in one command buffer.
///
/// **Twenty-three dispatches and one submission**, where the same operations
/// asked for a piece at a time are two submissions and three CPU rows between
/// them. Eleven are the attention's — see [`LayerProjections::layer`], which is
/// this one step in — and every value between them and the rows the MLP answered
/// is a buffer the next dispatch reads: what `o_proj` wrote, the convolution's
/// rows with the layer's input already added, what the second norm made of that,
/// the gate's logits, the experts the top-k took out of them, and each bank's
/// two halves with the activation between them.
///
/// **Where it stops is where the routing's weights are.** They are a softmax
/// over eight numbers from logits a dispatch in this same buffer produced, and
/// three of the four ways of misreading this gate live in that softmax — see
/// [`SparseMoe::weighted`](inkling_core::moe::SparseMoe::weighted) — so the rows
/// both banks answered come back to be weighted here. `h` comes back beside
/// them, because the layer's second residual is added to it on that side too.
impl DecoderDevice for LayerDevice<'_> {
    fn run(&self, cache: &mut DecoderCache, step: DecoderStep<'_>) -> Option<DecoderHalves> {
        let mlp = self.mlp.as_ref()?;
        Some(
            self.encode(cache, step, mlp)
                .unwrap_or_else(|err| panic!("the layer did not run: {err}")),
        )
    }
}

impl LayerDevice<'_> {
    fn encode(
        &self,
        cache: &mut DecoderCache,
        step: DecoderStep<'_>,
        mlp: &LayerMlpDevice<'_>,
    ) -> Result<DecoderHalves, ProjectionError> {
        let attention = &self.attention;
        let queries = attention.beginning(cache.attention(), step.attention);
        assert_eq!(
            step.attn_sconv.kernel_size(),
            self.attn_sconv.taps(),
            "the layer's residual convolution against the one wrapped for it"
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

        // The residual convolution's window is this sequence's, and it advances
        // exactly when the span and the two windows inside attention do — which
        // `beginning` has already started over if this sequence has seen nothing.
        if cache.attention().seen() == 0 {
            self.attn_sconv.restart();
        }

        let device = attention.device();
        let mut span = attention.attention.span();
        let mut batch = device.batch()?;

        let mut x = device.buffer(step.attention.x)?;
        let mut normed = attention
            .input_norm(&mut batch, &mut x, step.attention)?
            .expect("a decoder layer normalises the state it is handed");
        let mut attended = attention
            .encoding(&mut batch, &mut span, &mut normed, step.attention)?
            .buffer();
        let mut h = self
            .attn_sconv
            .encode(&mut batch, &mut attended, Some(&mut x))?;
        let mut normed = self.post_attention_layernorm.encode(&mut batch, &mut h)?;
        let dispatched = match mlp {
            LayerMlpDevice::Dense(ffn) => {
                Dispatch::Dense(ffn.encode_into(&mut batch, &mut normed)?)
            }
            LayerMlpDevice::Sparse(experts) => {
                Dispatch::Sparse(experts.encode_into(&mut batch, &mut normed, queries)?)
            }
        };
        batch.wait()?;
        cache.attention().appended(queries);

        let h = profile::timed(Op::Readback, || h.to_vec());
        Ok(DecoderHalves {
            projected: dispatched.weighted(h.len(), step.mlp),
            h,
        })
    }
}

/// What a layer's MLP left on the device, which is one value for a dense layer
/// and four for one that routes.
enum Dispatch {
    Dense(Pending),
    Sparse(Dispatched),
}

impl Dispatch {
    /// The `[tokens, hidden]` the MLP produced, read back and finished on this
    /// side.
    ///
    /// **Both arms leave something here, and neither is arithmetic a kernel
    /// declined.** A dense layer's trailing `global_scale` is
    /// `InklingDenseMLP`'s rather than `SwiGLUMLP`'s, and a routed layer's
    /// weights are a softmax over the logits its own gate produced — the half of
    /// the router where three of the four ways of misreading it live.
    fn weighted(self, values: usize, mlp: LayerMlp<'_>) -> Vec<f32> {
        match (self, mlp) {
            (Self::Dense(rows), LayerMlp::Dense(dense)) => dense.scaled(rows.take()),
            (Self::Sparse(dispatched), LayerMlp::Sparse(moe)) => {
                let answered = dispatched.answered();
                moe.weighted(values, &answered.logits, &answered.picked, &answered.banks)
                    .total()
            }
            _ => panic!("the layer's MLP is not the one wrapped for it"),
        }
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
    use inkling_core::layer::{DecoderLayer, DecoderWeights, NoExperts};
    use inkling_core::ops::DenseMlp;

    use crate::dense::DenseMatmul;
    use crate::matmul::testing::{Case, pack};
    use crate::router::Router;
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
            let five =
                LayerProjections::wrap(&device, &kernels, &layer_packed(&ckpt, (r_proj, o_proj)))
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

        let five = LayerProjections::wrap(&device, &kernels, &layer_packed(&ckpt, LAST_TWO[0]))
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
        let experts = ExpertKernels {
            matmul: kernels.matmul(),
            dense: &dense,
            swiglu: &swiglu,
            router: &router,
        };
        let projections = ModelLayers::wrap(
            &device,
            &kernels,
            experts,
            &packed_layers,
            &[],
            LAYERS,
            IN_DIM,
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
            LayerConv::new(self.device, &self.kernels.conv, channels, weight)
                .expect("the kernel uploads")
        }

        /// The whole layer on the device, with a dense feed-forward network in
        /// the MLP slot.
        fn layer(&self, weights: &NarrowWeights) -> LayerDevice<'a> {
            let (heads, head_dim) = (NARROW.heads, NARROW.head_dim);
            let kv = NARROW.kv_channels();
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
                    q_proj: self.weight(0x11, NARROW.hidden, heads * head_dim),
                    k_proj: self.weight(0x22, NARROW.hidden, kv),
                    v_proj: self.weight(0x33, NARROW.hidden, kv),
                    r_proj: self.weight(0x44, NARROW.hidden, heads * NARROW.d_rel),
                    o_proj: self.weight(0x55, heads * head_dim, NARROW.hidden),
                },
                attn_sconv: self.conv(NARROW.hidden, &weights.attn_sconv),
                post_attention_layernorm: self.norm(&weights.post_attention_layernorm),
                mlp: Some(LayerMlpDevice::Dense(Box::new(DenseFfn {
                    gate_proj: self.weight(0x66, NARROW.hidden, NARROW_FFN),
                    up_proj: self.weight(0x77, NARROW.hidden, NARROW_FFN),
                    down_proj: self.weight(0x88, NARROW_FFN, NARROW.hidden),
                    swiglu: self.swiglu,
                }))),
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
        };
        let weights = NarrowWeights::new();
        let held = narrow.layer(&weights);

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
            let prefill = layer.forward(cache, &x, &NoExperts, device);
            let rest = layer.forward(cache, &more, &NoExperts, device);
            (prefill, rest)
        };

        let fused = sequence(Some(&held as &dyn DecoderDevice));
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
        let again = sequence(Some(&held as &dyn DecoderDevice));
        assert_eq!(again.0, fused.0, "a second sequence's prefill");
        assert_eq!(again.1, fused.1, "a second sequence's continuation");
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
