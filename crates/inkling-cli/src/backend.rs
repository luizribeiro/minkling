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
//! # The weight is uploaded once and lives as long as the process
//!
//! `lm_head` is 411 MB of codes under 26 MB of scales, and the upload of it
//! takes 49 ms against the 1.4 ms a dispatch against it takes. A projection that
//! uploaded per call would therefore spend thirty times the multiply it enables,
//! so it is uploaded once, at load time, into shared storage the CPU and the GPU
//! both address — and owned by the [`CheckpointWeights`] the command holds until
//! it exits.
//!
//! What that costs is measured rather than assumed: over the same four-token
//! generation the resident set peaks at 20.79 GiB with the head on the device
//! against 20.36 GiB without it. The 0.44 GiB is the buffers; the pages they
//! were copied from stay mapped, and no weight is decoded to make them, so the
//! model does not become two models.

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
/// The head is cut to the vocabulary on the way up, so the 966 padding rows the
/// checkpoint carries are not uploaded — the truncation
/// [`inkling_core::head`] describes, honoured by not moving the bytes.
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
        PackedProjection::upload_packed(&gpu.device, &gpu.matmul, &weights.head_packed(), rows)
            .context("uploading lm_head to the Metal device")?;
    eprintln!(
        "{:<LABEL$}metal, {rows} rows of lm_head uploaded in {:.2?}",
        "backend",
        started.elapsed()
    );
    Ok(weights.with_head(Box::new(head)))
}
