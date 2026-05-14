{
  description = "Python RL scheduling environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-stable.url = "github:NixOS/nixpkgs/nixos-25.05";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, nixpkgs-stable, utils }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs-stable.legacyPackages.${system};
      python = pkgs.python312;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          (python.withPackages (ps: with ps; [
            # Core numerics / data
            numpy
            pandas
            matplotlib
            pillow
            sympy
            networkx

            # RL / ML
            gymnasium
            stable-baselines3
            (python.pkgs.callPackage ./sb3-contrib.nix {} )
            torch
            torchvision

            # TensorBoard
            tensorboard
            tensorboard-data-server
            protobuf
            grpcio
            markdown
            werkzeug
            absl-py

            # Utilities
            cloudpickle
            filelock
            fsspec
            packaging
            six
            typing-extensions
            tzdata
            pytz
            scikit-posthocs
            scipy
            python-dateutil
            snakemake
          ]))
        ] ++ [pkgs.snakemake pkgs.graphviz pkgs.just];

        shellHook = ''
          echo "HeraSched Environment Loaded"
          echo "Python: $(python --version)"
          echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
          echo "Stable-Baselines3: $(python -c 'import stable_baselines3; print(stable_baselines3.__version__)')"
          echo "Gymnasium: $(python -c 'import gymnasium; print(gymnasium.__version__)')"

          export PYTHONDONTWRITEBYTECODE=1
          export PYTHONOPTIMIZE=1
        '';
      };
    };
}
