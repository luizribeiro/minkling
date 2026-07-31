//! Metal backend.
//!
//! Kernels are compiled at runtime from source via `newLibraryWithSource:`
//! rather than precompiled into a `.metallib`. Ahead-of-time compilation needs
//! `xcrun metal`, which ships with Xcode and is not available inside the Nix
//! devshell, so runtime compilation is what keeps `nix develop` self-contained.
//! MLX does the same at the layer this replaces — `mx.fast.metal_kernel` hands
//! the driver a source string — so this is a cost the reference already pays.

pub mod device;

#[cfg(test)]
mod testing;

pub use device::{Device, MetalError};
