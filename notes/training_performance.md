# Training Performance Notes

Non-invasive performance / resource changes made alongside the float32 wrapper (see
[obs_wrapper.md](obs_wrapper.md)). None modify HPCsim; each either preserves behaviour exactly or is
applied uniformly across all algorithms.

## 1. Uniform SLURM resources + DQN vectorization (`Snakefile` `train_agent`)

**First attempt (per-algorithm sizing):** DQN ran a *single* env so it was given few cores (4) but
most of the RAM (for its 150k replay buffer); PPO/A2C got the cores (21 = 20 workers + main) but far
less RAM. Behaviour-preserving, but it assumed single-env DQN was viable.

**It was not, on the L4s.** Single-env DQN could not complete 3M steps in 12 h and produced no log
table at all — the wall is HPCsim's per-step observation build (56,090 values in Python), not the
GPU. On-policy reaches ~184 fps only because 20 `SubprocVecEnv` workers step in parallel (~9 fps
each); single-env DQN gets just one env's worth. Extra cores and TF32 speed the *network*, not
env-stepping, so neither helps.

**Fix — vectorize DQN too** (`use_vec = n_envs > 1`; `MaskableDQN` is `support_multi_env=True` and
already fully vectorized). DQN now collects from 20 workers like the on-policy algorithms and tracks
the same ~4–5 h env-bound ceiling. Resources are therefore **uniform** across all six treatments:

| resource | value |
|---|---|
| mem_mb | 120000 (120 GB of 128 GB nodes; 125 GB tripped a node RAM limit) |
| cpus_per_task | N_ENVS + 1 (21) |
| runtime | 720 min (12 h ceiling, kept as safety; all algos expected < 5 h) |

DQN's 150k float32 replay (~66 GB) + 20 workers (~40 GB) ≈ 106 GB fits the 120 GB request; one job
per node (GPU-bound). **Methodological note:** with a VecEnv, DQN's gradient steps =
`total_steps / (train_freq × n_envs)` — fewer updates per experience than single-env, a
compute-feasibility concession held constant across all seeds. After the first full sweep, run
`just slurm_report` and tighten from observed peak RSS / CPU efficiency.

## 2. Fewer checkpoints (`config.yaml`)

`save_interval 100k→300k`, `total_saving 30→10` — unchanged 3M training steps, but 10 snapshots
instead of 30. The pipeline evaluates only the final model (`3000000.zip`), so the intermediate
snapshots were largely unused. On-policy checkpoints are ~1.9 GB each (the `[4096,2048,1024]` pi+vf
nets), so this cuts scratch I/O ~3×. Behaviour-identical (training loop unchanged).

## 3. A2C wall-clock + the timeout diagnosis

**Resolved empirically:** at 20 envs A2C and maskable_A2C finish in ~4.4 h (fps ~184–188), well
inside the 12 h ceiling, so neither contingency below was needed. Kept for the record.

A2C / maskable_A2C initially hit the wall-clock limit (empty logs — SIGKILL discards buffered stdout;
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

## 4. TF32 (already enabled, all algorithms)

`configure_compute()` (called at the start of `main()`) sets `torch.backends.cuda.matmul.allow_tf32
= True`, so **every run — including the earlier PPO/A2C timings — already used TF32.** On the L4
(Ada, cc 8.9) fp32 matmuls run on the tensor cores. The benefit is concentrated in large-batch
training (DQN's 2048-sample updates; PPO's rollout updates) and is largely hidden behind the
env-bound on-policy collection, so it does not materially move on-policy wall-clock — but it is a
uniform, behaviour-preserving speedup applied identically to all algorithms.

## Considered but not changed (would alter results / methodology)

- **The ~230M-param first layer** is the dominant cost for *every* algorithm, but it is Wang's
  specified architecture — reducing it is Contingency B above, not a free change.
- **`torch.compile` on the policy**: behaviour-preserving in principle and helps all algos, but
  fragile with SB3 + the maskable variants' dynamic mask shapes, and uncertain (~1.2–1.5×). Not
  attempted.
