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
//! second norm and residual convolution behind all of it. What is left on the
//! CPU is the half of each router that weights what it chose, and the second
//! convolution and residual add that read what it weighted.
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
use inkling_core::{Checkpoint, CheckpointWeights, TextConfig};
use inkling_metal::{
    DenseMatmul, Device, ExpertKernels, LayerKernels, ModelLayers, PackedProjection, Router,
    RouterWeights, SwiGlu,
};

use crate::LABEL;
use crate::args::Backend;

/// What a Metal-backed run holds for its whole life: the device, and the
/// compiled kernels everything on it shares.
///
/// None of them is about a weight. No kernel's source names a shape, so one of
/// each serves the whole model — and all of them are opened before the
/// checkpoint, so that a machine this cannot run on says so in a millisecond
/// rather than after mapping 130 GiB.
///
/// The four a layer does not reach are the experts': the dense matmul is the
/// router's gate, which is the single weight the quantiser left in bfloat16; the
/// router is the top-k over what that gate produced and the softmax over the
/// eight logits it picked out; and the SwiGLU is the activation between a bank's
/// two halves. All but the first are no weight at all.
#[derive(Debug)]
pub struct Gpu {
    device: Device,
    kernels: LayerKernels,
    dense: DenseMatmul,
    swiglu: SwiGlu,
    router: Router,
    weights: RouterWeights,
}

impl Gpu {
    fn open() -> Result<Self> {
        let device = Device::open().context("opening a Metal device")?;
        let kernels = LayerKernels::compile(&device).context("compiling a layer's kernels")?;
        let dense = DenseMatmul::new(&device).context("compiling the dense matmul")?;
        let swiglu = SwiGlu::new(&device).context("compiling the swiglu")?;
        let router = Router::new(&device).context("compiling the router")?;
        let weights = RouterWeights::new(&device).context("compiling the router's weighting")?;
        Ok(Self {
            device,
            kernels,
            dense,
            swiglu,
            router,
            weights,
        })
    }

    /// The five a MoE layer dispatches through, which is four of this and the
    /// packed matmul every projection in the model shares.
    fn expert_kernels(&self) -> ExpertKernels<'_> {
        ExpertKernels {
            matmul: self.kernels.matmul(),
            dense: &self.dense,
            swiglu: &self.swiglu,
            router: &self.router,
            weights: &self.weights,
        }
    }
}

/// The device a backend needs, or nothing at all for the one that needs none.
///
/// Separate from [`weights`] because of what has to outlive what: the uploaded
/// projection borrows the device and the kernel, so they are opened by the
/// caller and live in its scope, above the weights that point at them.
pub fn open(backend: Backend) -> Result<Option<Gpu>> {
    match backend {
        Backend::Cpu => Ok(None),
        Backend::Metal => Gpu::open().map(Some),
    }
}

/// The checkpoint's weights, with the head wherever the backend put it.
///
/// The head is cut to the vocabulary on the way over, so the 966 padding rows
/// the checkpoint carries are never indexed — the truncation
/// [`inkling_core::head`] describes, honoured by not reaching the bytes.
pub fn weights<'a>(
    gpu: Option<&'a Gpu>,
    ckpt: &'a Checkpoint,
    config: &'a TextConfig,
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

    let banks = weights.expert_banks();
    let packed = weights.layer_projections();
    let layers = ModelLayers::wrap(
        &gpu.device,
        &gpu.kernels,
        gpu.expert_kernels(),
        &packed,
        &banks,
        config.num_hidden_layers,
        config.hidden_size,
    )
    .context("giving the model's layers to the Metal device")?;
    eprintln!(
        "{:<LABEL$}metal, {rows} rows of lm_head and all {} layers — {} MoE banks, {} dense \
         feed-forward networks — wrapped in {:.2?}",
        "backend",
        layers.layers(),
        layers.expert_layers(),
        layers.dense_layers(),
        started.elapsed()
    );

    Ok(weights
        .with_head(Box::new(head))
        .with_backend(Box::new(layers)))
}
