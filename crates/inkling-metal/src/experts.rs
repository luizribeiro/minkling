//! The routed expert matmuls, which are 73% of a decode step.
//!
//! A decode step is dequantisation-bandwidth bound and decodes about 44 GB: 32
//! of them are the experts, 9 the layer projections and 3.3 the head. This is
//! the 32.
//!
//! **Nothing is uploaded.** The forty MoE layers' banks are 137 GB of packed
//! bytes — the whole checkpoint but for its two ends — and a policy that copied
//! them onto the device would take two minutes at load and hold a second copy of
//! the model. [`Device::wrap`](crate::Device::wrap) is the other side of that:
//! every bank is handed over where the checkpoint mapped it, in about 40
//! microseconds a gibibyte and with no resident set of its own, and what the GPU
//! then reads are the file's own pages. Wrapping a bank nobody routes to costs
//! nothing, so *every* bank is wrapped at load and the residency question the
//! three-way choice was about stops being one.
//!
//! **One dispatch a projection, indexed by the gathered expert list.** The other
//! way to spend a layer is a dispatch per selected expert, and the arithmetic is
//! the same either way — the same six banks read, the same 250 untouched. What
//! differs is what surrounds it. Six experts by three projections by forty
//! layers is 720 dispatches for the routed banks alone, and a dispatch costs
//! about 170 microseconds to encode, commit and wait for against the 13
//! microseconds one expert's 4 MB takes to read at 267 GB/s: the GPU would be
//! idle for thirteen parts in fourteen and a decode step would carry 0.2 s of
//! floor. Gathered, a layer is six dispatches — gate, up and down of each bank —
//! and a step is 240 of them, or 41 ms of encoding under 16 ms of reading.
//!
//! It is still not free, and it is worth saying which way the remaining cost
//! runs: the shared bank is two experts and gets three dispatches of its own,
//! the same as the routed bank's six. Merging the two banks' gate and up into
//! single dispatches is the next thing to try if the encoding ever matters.
//!
//! **The SwiGLU stays on the CPU.** Between `gate_proj` and `down_proj` sits
//! `silu(gate) * up` over `[rows, 2048]`, which for a decode step is eight rows
//! — 16384 multiplies against the 4.3 GB the dispatches around it read. A kernel
//! for it would be a fourth dispatch a bank to save nothing measurable, and the
//! buffers it would avoid touching are shared storage the CPU addresses anyway.

use inkling_core::layer::Experts;
use inkling_core::moe::Gathered;
use inkling_core::ops::swiglu;
use inkling_core::weights::PackedExperts;

use crate::device::Device;
use crate::matmul::{MatmulError, PackedBank, PackedMatmul};

/// One `SwitchGLU`'s three banks on the device: `[experts, hidden_dim, dim]`
/// gate and up projections beside `[experts, dim, hidden_dim]` down projections.
///
/// The mirror of [`PackedExperts`], which is the same three banks left in the
/// mapping — and holds the same relation to it that
/// [`PackedProjection`](crate::PackedProjection) holds to
/// [`PackedRows`](inkling_core::PackedRows): the arithmetic is the checkpoint's,
/// and what changes is that no weight is ever decoded to memory.
#[derive(Debug)]
pub struct ExpertBanks<'a> {
    gate_proj: PackedBank<'a>,
    up_proj: PackedBank<'a>,
    down_proj: PackedBank<'a>,
}

impl<'a> ExpertBanks<'a> {
    /// Wrap a checkpoint's three banks, `dim` wide in and out.
    ///
    /// The width between is read off the tensors rather than taken, because it
    /// is the one thing the three shapes have to agree about: `gate_proj` and
    /// `up_proj` map `dim` to it and `down_proj` maps it back, and a bank paired
    /// with another layer's would still be three tensors of plausible shape.
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        banks: &PackedExperts<'a>,
        dim: usize,
    ) -> Result<Self, MatmulError> {
        let gate_proj = PackedBank::wrap(device, matmul, &banks.gate_proj(), dim)?;
        let hidden_dim = gate_proj.out_dim();
        Ok(Self {
            up_proj: PackedBank::wrap(device, matmul, &banks.up_proj(), dim)?,
            down_proj: PackedBank::wrap(device, matmul, &banks.down_proj(), hidden_dim)?,
            gate_proj,
        })
    }

    pub fn experts(&self) -> usize {
        self.gate_proj.experts()
    }

    /// Every gathered row through the expert it named, as the SwiGLU MLP an
    /// expert is.
    ///
    /// Three dispatches over the same expert list: `x @ gate^T` and `x @ up^T`
    /// against the same rows, and `silu(gate) * up` through `down`.
    pub fn forward(&self, gathered: Gathered<'_>) -> Result<Vec<f32>, MatmulError> {
        let chosen: Vec<u32> = gathered
            .experts()
            .iter()
            .map(|expert| {
                u32::try_from(*expert).unwrap_or_else(|_| panic!("expert {expert} is a wide index"))
            })
            .collect();

        let mut gate = self.gate_proj.multiply(&chosen, gathered.rows())?;
        swiglu(&mut gate, &self.up_proj.multiply(&chosen, gathered.rows())?);
        self.down_proj.multiply(&chosen, &gate)
    }
}

/// One MoE layer's two banks, which is what a layer reaches its experts through.
///
/// The routed bank is 256 experts of which a token reads six and the shared bank
/// is two every token reads, and nothing else separates them — the same three
/// dispatches over the same gathered list, differing in how much of the bank the
/// list names.
#[derive(Debug)]
pub struct LayerExperts<'a> {
    routed: ExpertBanks<'a>,
    shared: ExpertBanks<'a>,
}

impl<'a> LayerExperts<'a> {
    pub fn wrap(
        device: &'a Device,
        matmul: &'a PackedMatmul,
        routed: &PackedExperts<'a>,
        shared: &PackedExperts<'a>,
        dim: usize,
    ) -> Result<Self, MatmulError> {
        Ok(Self {
            routed: ExpertBanks::wrap(device, matmul, routed, dim)?,
            shared: ExpertBanks::wrap(device, matmul, shared, dim)?,
        })
    }
}

/// The seam [`inkling_core::layer`] names, so that a layer running its MoE does
/// not know whether an expert was ever decoded.
///
/// Infallible where [`ExpertBanks::forward`] is not, for the reason
/// [`PackedProjection`](crate::PackedProjection)'s side of the same bargain is:
/// a dispatch that does not complete is not a condition a decode step can do
/// anything about.
impl Experts for LayerExperts<'_> {
    fn routed(&self, gathered: Gathered<'_>) -> Vec<f32> {
        through(&self.routed, gathered)
    }

    fn shared(&self, gathered: Gathered<'_>) -> Vec<f32> {
        through(&self.shared, gathered)
    }
}

fn through(banks: &ExpertBanks<'_>, gathered: Gathered<'_>) -> Vec<f32> {
    banks
        .forward(gathered)
        .unwrap_or_else(|err| panic!("the expert matmul did not run: {err}"))
}
