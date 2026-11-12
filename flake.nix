{
  description = "PyTorch CUDA notebook environment for visual encoder training";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true; # CUDA is unfree
          cudaSupport = true;
          cudaCapabilities = [ "12.0" ]; # RTX 5070 Ti (Blackwell, sm_120)
        };
      };
      python = pkgs.python3.withPackages (
        ps: with ps; [
          torch
          torchvision
          numpy
          pandas
          matplotlib
          tqdm
          jupyterlab
          ipykernel
          ipywidgets
        ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ python ];
        shellHook = ''
          echo "torch $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"
          echo "run: jupyter lab"
        '';
      };
    };
}
