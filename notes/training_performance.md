# Training Performance Notes

Non-invasive performance / resource changes made alongside the float32 wrapper (see
[obs_wrapper.md](obs_wrapper.md)). None modify HPCsim; each either preserves behaviour exactly or is
applied uniformly across all algorithms.

## 1. Per-algorithm SLURM resources (`Snakefile` `train_agent`)

Previously every algorithm requested `mem_mb=98304` (96 GB) and `cpus_per_task=21`. That
over-provisions both ways:

- **DQN / maskable_DQN** run a *single* env (no SubprocVecEnv) → need ~1–4 cores, but carry the 150k
  replay buffer → need the 96 GB.
- **PPO / A2C (+ maskable)** run 20 SubprocVecEnv workers → need the 21 cores, but have no replay
  buffer (float32 rollout ~9 GB) → 48 GB is ample.

Resources are now wildcard callables:

| algorithm | mem_mb | cpus_per_task |
|---|---|---|
| dqn / maskable_dqn | 98304 (96 GB) | 4 |
| others | 49152 (48 GB) | 21 (N_ENVS + 1) |

Behaviour-preserving (resources don't change results). Tighter requests pack more jobs per node →
shorter queue across the 60-job sweep. Values are conservative — after the first sweep, run
`just slurm_report` and tighten from observed peak RSS / CPU efficiency.

## 2. Fewer checkpoints (`config.yaml`)

`save_interval 100k→300k`, `total_saving 30→10` — unchanged 3M training steps, but 10 snapshots
instead of 30. The pipeline evaluates only the final model (`3000000.zip`), so the intermediate
snapshots were largely unused. On-policy checkpoints are ~1.9 GB each (the `[4096,2048,1024]` pi+vf
nets), so this cuts scratch I/O ~3×. Behaviour-identical (training loop unchanged).

## 3. A2C wall-clock + the timeout diagnosis

A2C / maskable_A2C hit the wall-clock limit (empty logs — SIGKILL discards buffered stdout;
`PYTHONUNBUFFERED=1` is now exported in the train rule so future crashes flush their traceback).
`runtime` raised 480→720 min (partition max 14 h).

**Why A2C is slow (not a bug):** on-policy algos use *separate* pi and vf nets
(`utils.resolve_algorithm_config`), each with a `56090×4096 ≈ 230M`-param first layer → ~480M params.
A2C updates every `n_steps=5 × 20 = 100` steps → ~30,000 backprops through that net over 3M steps (vs
PPO's ~7,300), each at a tiny 100-sample batch that underutilises the GPU.

**Decision — keep Wang et al. (2025)'s `[4096,2048,1024]` architecture (fidelity), and diagnose
before changing anything:**

1. Run one A2C job (now unbuffered) and read `time/fps` from the first SB3 log dump (~10k steps). If
   `3,000,000 / fps > ~43,200 s (12 h)`, A2C won't finish and we apply a contingency.
2. **Contingency A (preferred):** raise A2C `n_steps` 5→32 — ~6× fewer updates and a 640-sample batch
   (better GPU use), ~4–6× faster on the update portion, *without* touching net_arch. `n_steps` is a
   per-algorithm hyperparameter (PPO independently uses `n_steps=2048`), justified on
   compute-feasibility grounds and applied to all A2C seeds. *If Wang pinned A2C's `n_steps`, match
   them instead.*
3. **Contingency B (last resort):** reduce net_arch uniformly across all algorithms (documented
   resource concession). Estimated wall-clock benefit: A2C ~2–3×, DQN ~1.5–2×, PPO ~1.2–1.5× (PPO is
   partly simulator-bound); models ~4× smaller. This deviates from Wang's architecture and is taken
   only if A / wall-clock can't make A2C feasible.

## Considered but not changed (would alter results / methodology)

- **The ~230M-param first layer** is the dominant cost for *every* algorithm, but it is Wang's
  specified architecture — reducing it is Contingency B above, not a free change.
- **`torch.compile` on the policy**: behaviour-preserving in principle and helps all algos, but
  fragile with SB3 + the maskable variants' dynamic mask shapes, and uncertain (~1.2–1.5×). Not
  attempted.
