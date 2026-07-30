{
  description = "inklingrs - Inkling inference engine for Apple Silicon";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    # Metal is the only backend, so there is no meaningful non-Darwin build.
    flake-utils.lib.eachSystem [ "aarch64-darwin" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            rust
            pkgs.cargo-nextest
            pkgs.cargo-flamegraph

            python
            pkgs.uv

            pkgs.just
            pkgs.hyperfine
          ];

          # uv builds the reference venv from PyPI wheels; nixpkgs has no
          # working Metal-enabled mlx, and a nix-built one cannot compile
          # shaders inside the sandbox.
          env = {
            UV_PYTHON = "${python}/bin/python3.12";
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            export INKLINGRS_ROOT="$PWD"
            export HF_XET_HIGH_PERFORMANCE=1
            export HF_HOME="''${HF_HOME:-$HOME/.cache/huggingface}"
          '';
        };
      });
}
