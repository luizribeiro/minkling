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

pub mod buffer;
pub mod device;
pub mod kernel;

#[cfg(test)]
mod testing;

pub use buffer::{Arg, Buffer, Element};
pub use device::{Device, MetalError};
pub use kernel::{Grid, Kernel};
