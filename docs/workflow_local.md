# Local Workflow Runbook

Validate the full pipeline locally before HPC deployment: catch schema and
integration failures early with a fast, tiny end-to-end run. All scripts run as
modules (`python -m src.<name>`); the normal path is the `just` targets, which
wrap Snakemake.

## 1. Prerequisites

- Nix dev shell active (`nix develop`) or `requirements.txt` installed.
- Traces present: `data/physical_job.csv`, `data/deeplearn_job.csv` (committed).
- Topologies present: `data/topology/{physical_topology.txt,deeplearn_topology.txt,nodes.csv}`.

Splits are **not** committed — `make_split` regenerates them on first run.

## 2. Fast path (recommended)

```bash
nix develop
just dry_run_smoke     # validate the smoke DAG (no execution)
just run_smoke         # tiny end-to-end run on physical_job
```

`config.smoke.yaml` drives this: 2 seeds, 2 algorithms (`maskable_a2c`,
`maskable_dqn`), 2 baselines (`fcfs`, `lcfs`), `save_interval=100 ×
total_saving=2 = 200` steps, `window_size=16`, `n_envs=1`, `eval_max_steps=5`.
The goal is to exercise every rule and file handoff, not to produce meaningful
results. Outputs land in `result/physical_job/`.

Switch trace: `just run_smoke TRACE=deeplearn_job`.

## 3. Manual stage-by-stage (for debugging a single rule)

Run from the repo root. `make_split` first; everything else reads the manifest
(`logs/run_log.csv`) written by training.

```bash
# 0. Time-aware split (idempotent)
python -m src.make_split --src physical_job --ratio 0.7 --out-dir data/splits/

# 1. Train one agent (tiny)
python -m src.train_agents \
  --algorithm maskable_a2c --name smoke_a2c --use-masking \
  --trace data/splits/physical_job_dev70.tsv \
  --save_interval 100 --total_saving 2 --seed 123456

# 2. Evaluate (reads logs/run_log.csv)
python -m src.evaluate_agents --manifest logs/run_log.csv \
  --output-dir result/physical_job/eval_runs \
  --filter-seed 123456 --filter-algo maskable_a2c --deterministic

# 3. Aggregate → 4. Stats → 5. Select best
python -m src.aggregate_results  ...
python -m src.statistical_test   ...
python -m src.select_best        ...
```

(Prefer `just run_smoke` — it wires the exact arguments for you.)

## 4. Holdout guard

Training refuses any trace path containing `holdout` (case-insensitive) and
fails fast before environment setup, so the final holdout can never be used for
tuning. To confirm the guard trips:

```bash
python -m src.train_agents --algorithm maskable_dqn --name guard_check \
  --trace data/splits/physical_job_holdout30.tsv \
  --save_interval 100 --total_saving 1 --seed 123456
# expected: immediate holdout-guard error, no training
```

## 5. Smoke gate checklist

- [ ] split artefacts generated (`*_dev70.tsv`, `*_holdout30.tsv`) and metadata JSON logged
- [ ] holdout guard fails fast on the holdout trace
- [ ] smoke matrix completes without traceback
- [ ] rewards finite; action masks valid for maskable algorithms
- [ ] eval, aggregate, and stats stages produce their outputs
- [ ] run metadata has a non-null `git_commit`

## 6. Output layout

```
data/splits/          # regenerated splits (gitignored)
trained_model/<trace>/<seed>/<algo>/selector/<step>.zip
result/<trace>/eval_runs/ aggregate/ stats/ best/ baseline/
logs/                 # run_log.csv, per-rule snakemake logs
plots/                # figures (gitignored)
```

## 7. Common failure modes

- `No module named 'src'` → invoked by path; use `python -m src.<name>`.
- Holdout trace used for training → guard blocks it (by design).
- DQN replay OOM on dict observations → reduce `--buffer-size` for smoke.
- Missing split → run `make_split` first (or just use `just run_smoke`).
