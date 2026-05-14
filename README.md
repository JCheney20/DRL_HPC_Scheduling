# HPC-DRL-Scheduler (Consolidation Repository)

<<<<<<< HEAD
This repository is the target clean structure for final project packaging.

Current active implementation still occurs in `github_repos/herasched/` in the main workspace.
=======
This repository is the consolidated structure for final project packaging and public release.

Active implementation now lives in this repository under `src/`.
>>>>>>> e7ed95b (update:ver1.1)

## Purpose

- host a clean, public-facing project structure;
- collect finalized training/evaluation/statistics workflow components;
<<<<<<< HEAD
=======
- provide a stable entry point for documentation and artefacts;
>>>>>>> e7ed95b (update:ver1.1)
- support migration to a future GitHub Wiki-based documentation flow.

## Current Status

<<<<<<< HEAD
- consolidation repo structure exists;
- docs scaffolding is in progress under `docs/`;
- operational source of truth remains root `AGENTS.md` in the parent workspace;
- implementation source of truth remains `github_repos/herasched/` until migration checkpoint.
=======
- consolidated pipeline code lives under `src/`;
- docs scaffolding is in progress under `docs/`;
- operational source of truth remains root `AGENTS.md` in the parent workspace.
>>>>>>> e7ed95b (update:ver1.1)

## Planned Consolidation Scope

This repo will eventually contain:

- training entry points and configs;
- evaluation and aggregation scripts;
- statistical analysis scripts and outputs;
- reproducibility documentation and runbooks;
<<<<<<< HEAD
- milestone-aligned presentation summaries and public references.

## Interim Documentation Pack

See `docs/` for templates used to track:
=======
- migration-ready Snakemake workflow and profiles;
- milestone-aligned presentation summaries and public references.

## Documentation Pack

See `docs/` for living documents that track:
>>>>>>> e7ed95b (update:ver1.1)

- methodology protocol;
- split and leakage policy;
- local workflow;
- HPC workflow;
- Snakemake pipeline;
- reproducibility checklist;
- Submission 2 evidence mapping.

## Structure

```
Project_Github/
<<<<<<< HEAD
├── docs/
├── training/
├── evaluation/
├── statistical_analysis/
├── data/
├── presentations/
└── tests/
=======
├── src/
├── docs/
├── data/
└── presentations/
>>>>>>> e7ed95b (update:ver1.1)
```

## Migration Rule

Do not migrate partially defined workflow code.

Promote components into this repo only when they satisfy:

- stable CLI and output contracts;
- smoke-tested locally;
- aligned with data governance and reproducibility rules;
- documented in `docs/`.
<<<<<<< HEAD
=======

## Next Migration Milestone

- document Apptainer/Nix container strategy in `docs/workflow_hpc.md`;
- add Snakemake Slurm profile and cluster submission notes;
- export the DAG figure and reference it in `docs/methodology_protocol.md`;
- keep root limited to `README.md`, `Snakefile`, `justfile`, `flake.nix` (plus future Apptainer/Slurm assets);
- move any custom Nix code into a `nix/` directory (referenced from `flake.nix`).
>>>>>>> e7ed95b (update:ver1.1)
