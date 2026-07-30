//! Metal backend.
//!
//! Kernels are compiled at runtime from source via `newLibraryWithSource:`
//! rather than precompiled into a `.metallib`. Ahead-of-time compilation needs
//! `xcrun metal`, which ships with Xcode and is not available inside the Nix
//! devshell, so runtime compilation is what keeps `nix develop` self-contained.
