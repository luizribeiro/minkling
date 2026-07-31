//! Where the model's largest projection runs, and what a run holds to get it
//! there.
//!
//! The choice is made once, before a token is decoded, and it reaches the engine
//! as one thing: the [`Projection`](inkling_core::Projection) `lm_head`
//! multiplies through. Nothing downstream branches on it — not the generation
//! loop, not the server, not the head's own arithmetic — which is what makes
//! "the CPU path still works" a claim a caller can check by rerunning the same
//! command with one word changed.
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
use inkling_metal::{Device, PackedMatmul, PackedProjection};

use crate::LABEL;
use crate::args::Backend;

/// What a Metal-backed run holds for its whole life: the device, and the
/// compiled kernel every packed projection on it shares.
///
/// Neither is about a weight. The kernel's source names no shape, so one of
/// these serves the whole model — and both are opened before the checkpoint, so
/// that a machine this cannot run on says so in a millisecond rather than after
/// mapping 130 GiB.
#[derive(Debug)]
pub struct Gpu {
    device: Device,
    matmul: PackedMatmul,
}

impl Gpu {
    fn open() -> Result<Self> {
        let device = Device::open().context("opening a Metal device")?;
        let matmul = PackedMatmul::new(&device).context("compiling the packed matmul")?;
        Ok(Self { device, matmul })
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
    let head =
        PackedProjection::wrap_packed(&gpu.device, &gpu.matmul, &weights.head_packed(), rows)
            .context("giving lm_head to the Metal device")?;
    eprintln!(
        "{:<LABEL$}metal, {rows} rows of lm_head wrapped in {:.2?}",
        "backend",
        started.elapsed()
    );
    Ok(weights.with_head(Box::new(head)))
}
