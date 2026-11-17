{
  description = "GPU-owned simulator: Vulkan/wgpu-hal + NVIDIA CUDA Rust";

  inputs = {
    cuda-rust.url = "github:NVIDIA/cuda-rust/348895e421a92d13a2c3a843ccb20ed1fb8b7f8b?dir=cuda-oxide";
    nixpkgs.follows = "cuda-rust/nixpkgs";
    rust-overlay.follows = "cuda-rust/rust-overlay";
  };

  outputs = { self, cuda-rust, nixpkgs, rust-overlay }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ (import rust-overlay) ];
        config.allowUnfree = true;
      };
      # Cranelift is ONLY for host iteration. cargo oxide keeps its stable,
      # LLVM-backed compiler and its own codegen backend for device kernels.
      fastRust = pkgs.rust-bin.nightly."2026-10-21".minimal.override {
        extensions = [ "rustc-codegen-cranelift-preview" ];
      };
      fast = pkgs.writeShellScriptBin "sim-fast" ''
        set -eu
        if [ "$#" -eq 0 ]; then set -- build; fi
        export RUSTC=${fastRust}/bin/rustc
        export RUSTDOC=${fastRust}/bin/rustdoc
        export CARGO_PROFILE_DEV_CODEGEN_BACKEND=cranelift
        export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-../.sim-target}/cranelift"
        exec ${fastRust}/bin/cargo "$@" -Zcodegen-backend
      '';
      shell = extra: pkgs.mkShell {
        inputsFrom = [ cuda-rust.devShells.${system}.default ];
        packages = with pkgs; [ mold pkg-config vulkan-loader vulkan-tools vulkan-validation-layers ffmpeg-headless git ] ++ extra;
        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ]}:''${LD_LIBRARY_PATH:-}
          export VK_LAYER_PATH=${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d
          export CUDA_OXIDE_DEVICE_ARCH=sm_120
        '';
      };
    in {
      devShells.${system} = {
        default = shell [ ];
        fast = shell [ fast ];
      };
    };
}
