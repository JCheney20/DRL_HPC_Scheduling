# Environment Setup

The project ships two ways to get a working environment. **Nix is the
reproducible source of truth** (pinned by `flake.lock`); pip/Conda via
`requirements.txt` is the portable best-effort fallback.

## Nix (recommended)

Fully reproducible — exact package versions are pinned in `flake.lock`, so you
get the same environment the results were produced with.

1. **Install Nix** (if not already installed):
   ```bash
   sh <(curl -L https://nixos.org/nix/install) --daemon
   ```

2. **Enable flakes** (add to `~/.config/nix/nix.conf`):
   ```
   experimental-features = nix-command flakes
   ```

3. **Enter the development shell** from the repo root:
   ```bash
   nix develop
   ```

The dev shell provides Python 3.12 with the full stack (torch, stable-baselines3,
sb3-contrib, gymnasium, scipy, scikit-posthocs, paretoset, …) plus `snakemake`,
`graphviz`, `just`, `apptainer`, and the SLURM executor plugins. The exact
package list lives in `flake.nix`; exact versions in `flake.lock`.

## pip / Conda (fallback)

Not guaranteed to be bit-for-bit reproducible — treat as best-effort. The
package list is derived from `flake.nix`; versions are unpinned so the resolver
can find a consistent set.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Conda users can create an env and `pip install -r requirements.txt` inside it.
Two system tools are **not** installable this way — install them separately if
you need them: `graphviz` (DAG export) and `apptainer`/`singularity` (container
runtime on HPC).

## Verification

```bash
python -c "import torch, stable_baselines3, sb3_contrib, gymnasium, scipy, paretoset; print('all imports OK')"
```

Because scripts are a namespace package under `src/`, always invoke them as
modules from the repo root, e.g. `python -m src.train_agents …` (not
`python src/train_agents.py`). A quick end-to-end check:

```bash
just dry_run_smoke      # validates the Snakemake DAG without executing
```

## GPU support

`torch` is used as-is from the environment. Check CUDA visibility:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

On HPC the Apptainer container is run with `--nv` (configured in
`profiles/slurm/config.yaml`) so the host GPU driver is mounted into the
container. Under Nix, `torch` from nixpkgs bundles its CUDA runtime; the host
driver is exposed via `/run/opengl-driver/lib` (already set in the dev shell).

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`** — you invoked a script by path.
Run it as a module from the repo root: `python -m src.<name>`.

**`ModuleNotFoundError: No module named 'sb3_contrib'`** — under pip, `pip install
sb3-contrib`; under Nix, make sure you're inside `nix develop`.

**Nix build fails with "unfree package"** — allow unfree packages:
`~/.config/nixpkgs/config.nix` → `{ allowUnfree = true; }` (the flake already
sets `config.allowUnfree = true` for its own package set).
