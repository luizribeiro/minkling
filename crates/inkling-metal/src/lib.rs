//! Metal backend.
//!
//! Kernels are compiled at runtime from source via `newLibraryWithSource:`
//! rather than precompiled into a `.metallib`. Ahead-of-time compilation needs
//! `xcrun metal`, which ships with Xcode and is not available inside the Nix
//! devshell, so runtime compilation is what keeps `nix develop` self-contained.
//! MLX does the same at the layer this replaces — `mx.fast.metal_kernel` hands
//! the driver a source string — so this is a cost the reference already pays.
//!
//! Four things stand between a source string and a result, and this crate is
//! each of them: a [`Device`] to compile and run against, a [`Kernel`] compiled
//! from source, [`Buffer`]s the CPU and GPU both address, and a [`Grid`] saying
//! how many threads to run.
//!
//! On top of those sits the operation the whole engine is made of: [`matmul`],
//! which multiplies against weights that stay MXFP4-packed, and [`dense`],
//! which is the same operation over the one weight the quantiser left in
//! bfloat16. Beside them sits [`norm`], which is the first here that consumes
//! activations rather than a weight — and so the first whose output is worth
//! leaving on the device — and [`attention`], which consumes nothing but
//! activations and is the one operation in the model with no weight to multiply
//! against at all. [`sconv`] is where a layer's position comes from, and the one
//! kernel here that carries state from one call to the next. [`swiglu`] is the
//! smallest of them and is here for what it lets the two matmuls either side of
//! it do rather than for its own arithmetic, [`router`] is the one that decides
//! which weights another dispatch will read and what the rows they produce are
//! worth, and [`combine`] is where those rows and those weights meet.

pub mod argmax;
pub mod attention;
pub mod buffer;
pub mod combine;
pub mod dense;
pub mod device;
pub mod experts;
pub mod grouping;
pub mod heads;
pub mod kernel;
pub mod matmul;
pub mod norm;
pub mod numerics;
pub mod projections;
pub mod router;
pub mod sampling;
pub mod sconv;
pub mod swiglu;
pub mod tail;

#[cfg(test)]
mod testing;

pub use argmax::{GreedyArgmax, Vocabulary};
pub use attention::{AttentionError, FusedAttention, LayerAttention, Step};
pub use buffer::{Arg, Buffer, Bytes, Element, Inline, Landing, Mapped};
pub use combine::MoeCombine;
pub use dense::{DenseMatmul, DenseWeight};
pub use device::{Device, MetalError, RoundTrip};
pub use experts::{ExpertBanks, ExpertKernels, LayerExperts};
pub use grouping::{ExpertGrouping, Grouped};
pub use heads::ModelHeads;
pub use kernel::{Batch, Grid, Kernel, Submitted};
pub use matmul::{MatmulError, Multiply, PackedBank, PackedMatmul, PackedProjection};
pub use norm::{LayerNorm, RmsNorm};
pub use numerics::Numerics;
pub use projections::{
    DISPATCHES_A_SUBMISSION, DenseFfn, LayerDevice, LayerKernels, LayerProjections, ModelLayers,
    StackShape,
};
pub use router::{LayerRouter, Router, RouterWeights, RoutingWeights};
pub use sconv::{LayerConv, ShortConvolution};
pub use swiglu::SwiGlu;
pub use tail::{ModelTail, TailWeights};
