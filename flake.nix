{
  description = "Visual encoder (DINOv3 distillation) dev shell: uv-managed PyTorch CUDA";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python313;
      # PyPI wheels (torch + bundled CUDA, raylib) dlopen these; NixOS has no FHS /usr/lib.
      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        libGL
        libx11
        libxcursor
        libxrandr
        libxinerama
        libxi
        libxext
        libxkbcommon
        wayland
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.typst
          python
          pkgs.uv
        ];
        env = {
          UV_PYTHON = "${python}/bin/python3";
          UV_PYTHON_DOWNLOADS = "never";
        };
        shellHook = ''
          # /run/opengl-driver/lib provides the host libcuda.so matching the kernel driver.
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          uv sync --quiet
          # Plain exports (not `source activate`) so direnv can carry them into non-bash shells.
          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$VIRTUAL_ENV/bin:$PATH"
        '';
      };
    };
}
