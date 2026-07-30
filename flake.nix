{
  description = "inklingrs - Inkling inference engine for Apple Silicon";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      git-hooks,
    }:
    # Metal is the only backend, so there is no meaningful non-Darwin build.
    flake-utils.lib.eachSystem [ "aarch64-darwin" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };

        rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        python = pkgs.python312;

        pre-commit = git-hooks.lib.${system}.run {
          src = ./.;

          # Context lines in a diff carry significant trailing whitespace, so
          # the whitespace hooks would corrupt these into unappliable patches.
          excludes = [ "^reference/patches/" ];

          hooks = {
            rustfmt = {
              enable = true;
              packageOverrides = {
                cargo = rust;
                rustfmt = rust;
              };
            };
            clippy = {
              enable = true;
              packageOverrides = {
                cargo = rust;
                clippy = rust;
              };
              settings.denyWarnings = true;
            };

            nixfmt.enable = true;

            ruff.enable = true;
            ruff-format.enable = true;

            shellcheck.enable = true;

            trim-trailing-whitespace.enable = true;
            end-of-file-fixer.enable = true;
            check-merge-conflicts.enable = true;
          };
        };
      in
      {
        checks.pre-commit = pre-commit;

        devShells.default = pkgs.mkShell {
          packages = [
            rust
            pkgs.cargo-nextest
            pkgs.cargo-flamegraph

            python
            pkgs.uv

            pkgs.just
            pkgs.hyperfine
          ]
          ++ pre-commit.enabledPackages;

          # uv builds the reference venv from PyPI wheels; nixpkgs has no
          # working Metal-enabled mlx, and a nix-built one cannot compile
          # shaders inside the sandbox.
          env = {
            UV_PYTHON = "${python}/bin/python3.12";
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            ${pre-commit.shellHook}
            export INKLINGRS_ROOT="$PWD"
            export HF_XET_HIGH_PERFORMANCE=1
            export HF_HOME="''${HF_HOME:-$HOME/.cache/huggingface}"
          '';
        };
      }
    );
}
