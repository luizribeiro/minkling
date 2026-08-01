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
use inkling_core::mask::BandedMask;
use inkling_core::ops::{MlpProjections, Projection};
use inkling_core::weights::{LayerPacked, Packed, PackedMlp, ProjectionBackend};

use crate::attention::{AttentionError, FusedAttention, KeyValues, LayerAttention, Step};
use crate::buffer::Landing;
use crate::device::{Device, MetalError};
use crate::kernel::Batch;
use crate::matmul::{MatmulError, PackedMatmul, PackedProjection, Pending, together};
use crate::norm::{LayerNorm, RmsNorm};
use crate::sconv::{LayerConv, ShortConvolution};

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
    fn encoding(
        &self,
        batch: &mut Batch<'_>,
        span: &mut KeyValues,
        step: LayerStep<'_>,
    ) -> Result<Pending, MatmulError> {
        let device = self.q_proj.device();
        let mut normed = match step.input_layernorm {
            Some(weight) => {
                assert_eq!(
                    weight.len(),
                    self.input_layernorm.width(),
                    "the layer's norm against the one wrapped for it"
                );
                assert_eq!(step.eps, self.input_layernorm.eps(), "rms_norm_eps");
                self.input_layernorm.encode(batch, step.x)?
            }
            None => device.buffer(step.x)?,
        };
        let mut q = self.q_proj.encode_over(batch, &mut normed)?.buffer();
        let mut k = self.k_proj.encode_over(batch, &mut normed)?.buffer();
        let mut v = self.v_proj.encode_over(batch, &mut normed)?.buffer();
        let mut rel = self.r_proj.encode_over(batch, &mut normed)?.buffer();

        let queries = q.len() / (step.sdpa.heads() * step.sdpa.head_dim());
        let (keys, values) = span.landings();
        let mut k = self.k_sconv.encode(batch, &mut k)?;
        self.v_sconv.encode_over(batch, &mut v, values)?;

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
    /// Four submissions become one, which at 206 microseconds a submission is
    /// 618 µs a layer and 26 ms of a decode step — against the 105 µs the
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

        let [q, k, v, r] = together(self.q_proj.device(), |batch| {
            let mut normed = self.input_layernorm.encode(batch, x)?;
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

        let mut span = self.attention.span();
        let [out] = together(self.q_proj.device(), |batch| {
            Ok([self.encoding(batch, &mut span, step)?])
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
}

impl<'a> DenseFfn<'a> {
    /// Wrap a dense layer's three where the checkpoint mapped them. Whether they
    /// pair is [`DenseMlp`](inkling_core::DenseMlp)'s to say, and `gate_proj`
    /// against `up_proj` is the pair that pairs either way round.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        packed: &PackedMlp<'a>,
    ) -> Result<Self, MatmulError> {
        Ok(Self {
            gate_proj: whole(device, matmul, &packed.gate_proj)?,
            up_proj: whole(device, matmul, &packed.up_proj)?,
            down_proj: whole(device, matmul, &packed.down_proj)?,
        })
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

/// Every layer's own projections on the device, which for Inkling-Small is 42
/// layers of five and two of three more.
///
/// 1.19 GB of packed bytes — 9.6 GB were it decoded — wrapped where the
/// checkpoint mapped them and holding no resident set of their own. The same
/// bargain [`ModelExperts`](crate::ModelExperts) strikes over its 138 GB, and
/// cheap enough for the same reason that there is no residency question to
/// answer.
#[derive(Debug)]
pub struct ModelProjections<'a> {
    /// Indexed by layer, `None` where nothing here answers for one — which is a
    /// layer the CPU keeps, and is how a partial handover stays expressible.
    layers: Vec<Option<Layer<'a>>>,
}

/// One layer's own projections: attention's five, and the feed-forward network
/// of a layer that has one.
#[derive(Debug)]
struct Layer<'a> {
    attention: LayerProjections<'a>,
    dense_mlp: Option<DenseFfn<'a>>,
}

impl<'a> ModelProjections<'a> {
    /// Wrap every projection `packed` names, over a stack of `layers`.
    ///
    /// The stack's length is stated rather than read off the last entry, for the
    /// reason [`ModelExperts::wrap`](crate::ModelExperts::wrap) states it: a
    /// backend answering for none of the last layers would otherwise report a
    /// shorter stack than the model has, and "past the stack" and "left to the
    /// CPU" would stop being answerable apart.
    pub fn wrap(
        device: &'a Device,
        kernels: &'a LayerKernels,
        packed: &[LayerPacked<'a>],
        layers: usize,
    ) -> Result<Self, ProjectionError> {
        let mut wrapped: Vec<Option<Layer<'a>>> = (0..layers).map(|_| None).collect();
        for layer in packed {
            wrapped[layer.layer] = Some(Layer {
                attention: LayerProjections::wrap(device, kernels, layer)?,
                dense_mlp: layer
                    .dense_mlp
                    .map(|mlp| DenseFfn::wrap(device, &kernels.matmul, &mlp))
                    .transpose()?,
            });
        }
        Ok(Self { layers: wrapped })
    }

    /// How many of the stack's layers have their attention projections here.
    pub fn layers(&self) -> usize {
        self.layers.iter().flatten().count()
    }

    /// How many of those also have a feed-forward network here, which is how
    /// many are dense.
    pub fn dense_layers(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .filter(|layer| layer.dense_mlp.is_some())
            .count()
    }

    fn layer(&self, layer: usize) -> Option<&Layer<'a>> {
        self.layers.get(layer)?.as_ref()
    }
}

/// The seam [`inkling_core::weights`] names, so that a layer standing itself up
/// does not know whether its projections were ever decoded.
impl ProjectionBackend for ModelProjections<'_> {
    fn attention(&self, layer: usize) -> Option<&dyn Projections> {
        Some(&self.layer(layer)?.attention as &dyn Projections)
    }

    fn dense_mlp(&self, layer: usize) -> Option<&dyn MlpProjections> {
        Some(self.layer(layer)?.dense_mlp.as_ref()? as &dyn MlpProjections)
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

        let three = DenseFfn::wrap(
            &device,
            &matmul,
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
        let projections = ModelProjections::wrap(&device, &kernels, &packed_layers, LAYERS)
            .expect("the layers wrap");

        assert_eq!(projections.layers(), 2, "two of the five");
        assert_eq!(projections.dense_layers(), 1);
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

        // And the two layers wrapped are not each other's, which is what an
        // index off by one would produce. They differ in `o_proj` alone.
        for (layer, (_, o_proj)) in [(0, LAST_TWO[0]), (2, LAST_TWO[1])] {
            let five = projections.attention(layer).expect("a wrapped layer");
            each_answers(&ckpt, &[(o_proj, five.o_proj())]);
        }
    }
}
