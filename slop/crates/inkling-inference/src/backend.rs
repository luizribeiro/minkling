//! Where the model's multiplies run, and what a run holds to get them there.
//!
//! The choice is made once, before a token is decoded, and it reaches the engine
//! as two handovers: the [`Projection`](inkling_core::Projection) `lm_head`
//! multiplies through, and every layer — its five attention projections, and
//! either the experts it routes into or the feed-forward network the two dense
//! ones hold in their place. Between them
//! they are every weight this engine has a kernel for, and a layer's own small
//! tensors go over with its projections because nothing else reads them: its
//! input layernorm, the kernels of the two convolutions inside attention, the
//! two head norms behind them, the band the attention step contracts against —
//! which is the one handover that is an operation rather than a weight — and the
//! second norm and both residual convolutions behind all of it. What is left on
//! the CPU is nothing of a layer: it hands one over `[tokens, hidden]` and takes
//! back `[tokens, hidden]`.
//!
//! Nothing downstream branches on the choice — not the generation loop, not the
//! server, not the head's own arithmetic — which is what makes "the CPU path
//! still works" a claim a caller can check by rerunning the same command with
//! one word changed.
//!
//! # The weight is not moved at all
//!
//! `lm_head` is 411 MB of codes under 26 MB of scales, and copying that onto the
//! device took 49 ms against the 1.4 ms a dispatch against it takes — so it was
//! copied once, at load time, rather than per call. Wrapped where the checkpoint
//! mapped it, the same 0.41 GiB is handed over in 52 microseconds and costs no
//! resident set of its own, which is what makes "once at load time" stop being a
//! decision worth defending.
//!
//! What is still owned by the [`CheckpointWeights`] the command holds until it
//! exits is the *binding*: a `MTLBuffer` over pages the mapping owns, which
//! borrows the checkpoint and cannot outlive it.

use std::time::Instant;

use anyhow::{Context, Result};
use inkling_core::mtp::{CheckpointHeads, FRONTIER};
use inkling_core::{Checkpoint, CheckpointWeights, Config, TextConfig};
use inkling_metal::{
    DenseMatmul, Device, ExpertGrouping, ExpertKernels, LayerKernels, LayerNorm, ModelHeads,
    ModelLayers, ModelTail, MoeCombine, Numerics, PackedProjection, Router, RouterWeights,
    StackShape, SwiGlu, TailWeights,
};

use crate::{Backend, LABEL};

/// What a Metal-backed run holds for its whole life: the device, and the
/// compiled kernels everything on it shares.
///
/// None of them is about a weight. No kernel's source names a shape, so one of
/// each serves the whole model — and all of them are opened before the
/// checkpoint, so that a machine this cannot run on says so in a millisecond
/// rather than after mapping 130 GiB.
///
/// The five a layer does not reach are the experts': the dense matmul is the
/// router's gate, which is the single weight the quantiser left in bfloat16; the
/// router is the top-k over what that gate produced and the softmax over the
/// eight logits it picked out; the SwiGLU is the activation between a bank's two
/// halves; and the combine is both banks' rows weighted by that softmax. All but
/// the first are no weight at all.
#[derive(Debug)]
pub struct Gpu {
    device: Device,
    kernels: LayerKernels,
    dense: DenseMatmul,
    swiglu: SwiGlu,
    router: Router,
    grouping: ExpertGrouping,
    weights: RouterWeights,
    combine: MoeCombine,
}

impl Gpu {
    fn open(numerics: Numerics) -> Result<Self> {
        let device = Device::open().context("opening a Metal device")?;
        let kernels =
            LayerKernels::compiling(&device, numerics).context("compiling a layer's kernels")?;
        let dense = DenseMatmul::new(&device).context("compiling the dense matmul")?;
        let swiglu = SwiGlu::new(&device).context("compiling the swiglu")?;
        let router = Router::new(&device).context("compiling the router")?;
        let grouping = ExpertGrouping::new(&device).context("compiling the expert grouping")?;
        let weights = RouterWeights::new(&device).context("compiling the router's weighting")?;
        let combine = MoeCombine::new(&device).context("compiling the combine")?;
        Ok(Self {
            device,
            kernels,
            dense,
            swiglu,
            router,
            grouping,
            weights,
            combine,
        })
    }

    /// The device itself, for a caller measuring what a submission through it
    /// costs rather than running one.
    ///
    /// Everything else here hands out something wrapped, because a caller that
    /// only wants to run the model never has to name the device. A caller that
    /// wants the driver's own clock on a command buffer does — see
    /// [`inkling_metal::Device::record_round_trips`], which is what a wait that
    /// is neither encode nor execution has to be divided by.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Which arithmetic this device's innermost accumulation uses, read back off
    /// the kernel that was compiled rather than off the flag that asked for it.
    pub fn numerics(&self) -> Numerics {
        self.kernels.matmul().numerics()
    }

    /// The model's final norm, muP divide and `lm_head`, wrapped for whoever
    /// will run the rows in front of them.
    ///
    /// Asked for twice on a speculating run — once by the stack and once by the
    /// heads — because what each of them holds has to be reachable from the
    /// command buffer it is already encoding into. What that costs is a second
    /// binding over the same mapped pages and a second 16 KB copy of the norm's
    /// weight, and no bytes of `lm_head` either time.
    fn tail<'a>(&'a self, weights: &TailWeights<'a>) -> Result<Option<ModelTail<'a>>> {
        Ok(ModelTail::wrap(
            &self.device,
            self.kernels.norm(),
            self.kernels.matmul(),
            self.kernels.argmax(),
            weights,
        )?)
    }

    /// One dispatch of one row, to submit into a gap where nothing else is
    /// running.
    ///
    /// **What it is for is a question about the machine rather than about this
    /// engine**: an Apple GPU that has been idle for a hundred milliseconds
    /// charges the next dispatch after it about three and a half milliseconds
    /// more than it charges the same dispatch back to back, and what nothing
    /// here says is whether what the part is holding on to is *a dispatch* or an
    /// occupancy. A caller that submits this into the gap and re-measures has
    /// asked it.
    ///
    /// The smallest thing this crate has a kernel for, so that what it costs is
    /// close to what a submission costs and not to what a step does: one row
    /// through a norm is 206 microseconds of submission around a dispatch of
    /// microseconds. The answer is thrown away — nothing reads a row of ones
    /// normalised — and the cost is reported by whoever asked for it.
    pub fn keep_warm(&self, width: usize) -> Result<KeepWarm<'_>> {
        let ones = vec![1.0f32; width];
        Ok(KeepWarm {
            norm: LayerNorm::new(&self.device, self.kernels.norm(), &ones, 1e-6)
                .context("wrapping a keep-warm dispatch")?,
            row: ones,
        })
    }

    /// The six a MoE layer dispatches through, which is five of this and the
    /// packed matmul every projection in the model shares.
    fn expert_kernels(&self) -> ExpertKernels<'_> {
        ExpertKernels {
            matmul: self.kernels.matmul(),
            dense: &self.dense,
            swiglu: &self.swiglu,
            router: &self.router,
            grouping: &self.grouping,
            weights: &self.weights,
            combine: &self.combine,
        }
    }
}

/// A dispatch cheap enough to be worth submitting for its own sake — see
/// [`Gpu::keep_warm`], which is where what it is for is stated.
pub struct KeepWarm<'a> {
    norm: LayerNorm<'a>,
    /// The row it norms, which is the weight it norms against: what is being
    /// submitted is a dispatch, and the numbers in it are not read by anything.
    row: Vec<f32>,
}

impl KeepWarm<'_> {
    /// One dispatch, submitted and waited for, as a step's own are.
    pub fn dispatch(&self) -> Result<()> {
        self.norm
            .forward(&self.row)
            .context("submitting a keep-warm dispatch")?;
        Ok(())
    }
}

/// The device a backend needs, or nothing at all for the one that needs none.
///
/// Separate from [`weights`] because of what has to outlive what: the uploaded
/// projection borrows the device and the kernel, so they are opened by the
/// caller and live in its scope, above the weights that point at them.
pub fn open(backend: Backend, numerics: Numerics) -> Result<Option<Gpu>> {
    match backend {
        Backend::Cpu => Ok(None),
        Backend::Metal => Gpu::open(numerics).map(Some),
    }
}

/// The checkpoint's weights, with the head wherever the backend put it.
///
/// The head is cut to the vocabulary on the way over, so the 966 padding rows
/// the checkpoint carries are never indexed — the truncation
/// [`inkling_core::head`] describes, honoured by not reaching the bytes.
///
/// `speculation` is how many timesteps a layer has to be able to give back,
/// which is how far ahead this run will guess: the state a rejected token
/// reached lives where the layer ran, so a device holding the layers has to be
/// told before it wraps them. A run that speculates nothing asks for none.
///
/// `slots` is how many sequences the run will advance together — see
/// [`StackShape::slots`], where what a slot costs is stated. A run serving one
/// request asks for one, which is the batch of one the whole engine was before
/// there was a batch.
pub fn weights<'a>(
    gpu: Option<&'a Gpu>,
    ckpt: &'a Checkpoint,
    config: &'a TextConfig,
    speculation: usize,
    slots: usize,
) -> Result<CheckpointWeights<'a>> {
    let weights = CheckpointWeights::open(ckpt, config)?;
    let Some(gpu) = gpu else {
        eprintln!(
            "{:<LABEL$}cpu, every weight decoded on the way through",
            "backend"
        );
        return Ok(weights);
    };

    let rows = weights.head().vocab();
    let started = Instant::now();
    let head = PackedProjection::wrap_packed(
        &gpu.device,
        gpu.kernels.matmul(),
        &weights.head_packed(),
        rows,
    )
    .context("giving lm_head to the Metal device")?;

    let tail = gpu
        .tail(&tail_weights(&weights, config))
        .context("giving the model's tail to the Metal device")?;
    let banks = weights.expert_banks();
    let packed = weights.layer_projections();
    let layers = ModelLayers::wrap(
        &gpu.device,
        &gpu.kernels,
        gpu.expert_kernels(),
        &packed,
        &banks,
        tail,
        StackShape {
            layers: config.num_hidden_layers,
            dim: config.hidden_size,
            slack: speculation,
            slots,
        },
    )
    .context("giving the model's layers to the Metal device")?;
    // The numerics are said here rather than only obeyed, for the reason the
    // backend itself is: the first thing a report about a wrong token has to say
    // is which path produced it, and a line a run prints is a line that can be
    // pasted into one.
    eprintln!(
        "{:<LABEL$}metal, {} numerics, {rows} rows of lm_head and all {} layers — {} MoE banks, \
         {} dense feed-forward networks — wrapped in {:.2?}",
        "backend",
        gpu.numerics().named(),
        layers.layers(),
        layers.expert_layers(),
        layers.dense_layers(),
        started.elapsed()
    );

    Ok(weights
        .with_head(Box::new(head))
        .with_backend(Box::new(layers)))
}

/// What the back of the model is, out of the checkpoint the front of it came
/// from.
///
/// Assembled here rather than by each backend because the four pieces are one
/// answer, for the reason [`CheckpointWeights::generator`] assembles the same
/// three: a tail built from one checkpoint's norm and another's head would run.
pub fn tail_weights<'a>(weights: &CheckpointWeights<'a>, config: &TextConfig) -> TailWeights<'a> {
    TailWeights {
        norm: weights.final_norm().to_vec(),
        eps: config.rms_norm_eps,
        mup: weights.head().mup(),
        head: weights.head_packed(),
        vocab: weights.head().vocab(),
    }
}

/// The multi-token prediction heads, with their multiplies wherever the backend
/// puts them.
///
/// **Opened only when a run means to speculate**, because what they cost to
/// have is what a caller who does not want them should not pay: 160 tensors of
/// bfloat16, 4.2 GiB of mapping, and a scratch on the CPU path that is larger
/// than everything else the process holds.
///
/// A checkpoint whose config declares no `mtp_config` has no heads to open, and
/// a checkpoint that declares them and does not ship them is the error this
/// returns — both of which a caller answers the same way, by decoding one token
/// at a time.
pub fn heads<'a>(
    gpu: Option<&'a Gpu>,
    ckpt: &'a Checkpoint,
    config: &Config,
    depth: usize,
    tail: &TailWeights<'a>,
    slots: usize,
) -> Result<Option<CheckpointHeads<'a>>> {
    if depth == 0 {
        return Ok(None);
    }
    let mtp = config
        .mtp_config
        .as_ref()
        .context("this checkpoint's config declares no MTP heads to speculate with")?;

    let started = Instant::now();
    let heads = CheckpointHeads::open(ckpt, &config.text_config, mtp)
        .context("mapping the MTP heads, which this checkpoint's shards do not hold")?;
    let Some(gpu) = gpu else {
        eprintln!(
            "{:<LABEL$}{} heads, {depth} deep, every weight widened on the way through",
            "mtp",
            heads.heads()
        );
        return Ok(Some(heads));
    };

    let packed = heads.head_projections();
    let wrapped = ModelHeads::wrap(
        &gpu.device,
        &gpu.kernels,
        &gpu.dense,
        &gpu.swiglu,
        &packed,
        gpu.tail(tail)
            .context("giving the model's tail to the Metal device")?,
        FRONTIER,
        slots,
    )
    .context("giving the MTP heads to the Metal device")?;
    eprintln!(
        "{:<LABEL$}{} heads wrapped in {:.2?}, {depth} of them a round",
        "mtp",
        wrapped.heads(),
        started.elapsed()
    );
    Ok(Some(heads.with_backend(Box::new(wrapped))))
}
