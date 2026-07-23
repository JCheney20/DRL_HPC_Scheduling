# Process Notes for Paper

Practical notes from running the full sweep on the SLURM cluster. Intended for
the methodology / experimental-setup and limitations sections.

## Compute environment

- Nodes: 128 GB RAM, GPU (`gres=gpu:1`), 24 cores. Partition `main`.
- Each `(seed × algorithm)` treatment is one SLURM job (Snakemake fans them out).
- Training: 3M timesteps, `n_envs=20` (SubprocVecEnv) for all six treatments (DQN vectorized too).

## DQN needed substantially more RAM than the on-policy algorithms

DQN (and MaskableDQN) are off-policy and keep an experience **replay buffer**;
PPO/A2C are on-policy and discard each rollout after the update. The replay
buffer is what drives DQN's memory footprint.

The observation is a `gym.spaces.Dict` totalling **56,090 values**, cast to
`float32` at the environment boundary (see [obs_wrapper.md](obs_wrapper.md)) =
**≈ 219 KB per observation**. It is dominated by the cluster vector
(`5 × max_jobs = 5 × 10,260 = 51,300` values); the queue vector is only 3,585.
Stable-Baselines3 stores both `obs` and `next_obs` for every transition, so each
stored transition costs ≈ **0.44 MB**. Because the observation is a `Dict`, SB3
forbids `optimize_memory_usage`, so the 2× obs/next_obs cost cannot be avoided.

Consequences:

- `buffer_size` maps almost linearly to RAM: `≈ buffer_size × 0.44 MB`.
- A conventional `buffer_size=1,000,000` would need ~440 GB — far beyond any
  node. Even 300k needs ~132 GB, still over the 128 GB nodes.
- Buffer sized to fit: **`buffer_size=150,000` ≈ 66 GB**, requested with
  `mem_mb=120000` (120 GB) to leave headroom for the 20 env workers (~40 GB),
  the model, and the CUDA context. (125 GB tripped a node RAM limit.)
- On-policy runs (PPO/A2C) use only a fraction of the replay allowance; the
  120 GB request is provisioned for the DQN worst case and shared across the rule.

Paper-ready framing: the large per-observation footprint (a direct consequence
of the per-job cluster encoding) makes off-policy replay memory-bound on this
problem, capping the practical replay-buffer size well below values common in
the DQN literature (e.g. 1M for Atari, where each frame is ~7 KB).

## A2C wall-clock

A2C is update-bound: it updates on a very short rollout (SB3 default `n_steps=5`),
a gradient step roughly every `5 × n_envs = 100` environment steps — far more
update passes than PPO for the same 3M steps. It is the slowest on-policy
algorithm on the L4 nodes but finishes comfortably at **~4.6 h** (maskable_A2C
similar) under `runtime=720` (12 h). An earlier `runtime=480` (8 h) attempt was
too tight and produced empty logs (SIGKILL discards buffered stdout), which is
why the ceiling was raised to 720. A per-algorithm `n_steps` increase was
diagnosed as a contingency but not needed (see `training_performance.md §3`).

## A2C numerical stability: entropy floor + reverting non-standard advantage normalization

A `maskable_a2c` seed crashed with a `MaskableCategorical` `Simplex()` violation —
the distribution's probabilities no longer summed to one because the policy logits
had gone non-finite. Two coupled causes, fixed together:

**1. Entropy collapse (`ent_coef=0.0`).** On the ~230M-parameter `[4096,2048,1024]`
first layer the policy saturated to ~zero entropy within 20k steps (`entropy_loss
≈ 0`, all losses ~1e-7). Adding **`ent_coef=0.01`** (the value SB3's own A2C example
uses) keeps entropy up. This was necessary but *not sufficient*: with the bonus the
run survived to ~110k steps and then crashed the same way, so entropy was not the
whole story.

**2. Non-standard advantage normalization (the real driver).** `a2c_mask.py`
defaults `normalize_advantage=True` — normalization that stock SB3 A2C does **not**
perform (the source even comments it as "not present in the original
implementation"). Over A2C's tiny 100-sample rollout (`n_steps=5 × 20 envs`), once
the value function fits well (`explained_variance ≈ 0.83`) the true advantages are
~0, so `(adv − mean) / (std + 1e-8)` divides near-zero residuals by a near-zero std
and **rescales pure noise to unit scale**. The policy then takes a full-strength
gradient step on noise every update; the value fit collapses (`explained_variance`
fell `0.83 → 0.0001` in 100 iterations) and the logits diverge to `NaN`. This is
precisely why stock A2C keeps `normalize_advantage=False`. Fix: correct the
`MaskableA2C` default at its source — `a2c_mask.py`'s `normalize_advantage` default
is changed from `True` to `False`, matching canonical A2C. This is a return to the
standard algorithm, not a new hyperparameter, and it is `maskable_a2c`-specific —
stock `a2c` (SB3's own `A2C`) already defaults to `False`, which is why only the
maskable variant diverged.

The two fixes live at the layer each belongs to. `ent_coef=0.01` is a tuning choice
for this config, so it is set on the A2C construction path (`a2c` and `maskable_a2c`)
alongside the other hyperparameters; the advantage-normalization fix is a defect in
the custom class, so it is corrected as that class's default (no per-run override).
PPO normalizes over large minibatches (stable) and DQN has no analogue. The masking
itself was not implicated: an explicit all-false-mask guard (`a2c_mask.py`) never
triggered. Because training is seeded the failure is deterministic per seed, so it
needed a fix rather than a retry.

## Checkpoint save_freq must account for n_envs (silent no-save on on-policy)

On-policy jobs (PPO/A2C) reported `[DONE]` yet left empty model folders. Root
cause: SB3's `CheckpointCallback` counts *callback calls*, not environment steps,
and with a 20-env `SubprocVecEnv` one call advances `n_envs` steps. Passing
`save_freq=save_interval` (in env-steps) therefore meant the callback's call
counter only reached `total_timesteps / n_envs` and never hit `save_freq`, so
**zero** checkpoints were written — and there was no explicit final save. DQN
(single-env) was unaffected, which is why only the on-policy folders were empty.
Fix: (1) `save_freq = max(save_interval // n_envs, 1)` so cadence is in env-steps
for every algorithm, and (2) an explicit `model.save()` of the final model at the
manifest path after `learn()` — robust to PPO overshooting `total_timesteps` to a
full-rollout boundary. This is the file the evaluator loads.

## Evaluation is single-env and full-trace (60 min was not enough)

`evaluate_agents.py` rolls a trained policy deterministically over the *entire*
evaluation trace in a **single** environment — the same per-step Python
observation rebuild that bounds training, but without the 20-env parallelism. A
dev70 pass is ~59k steps and the maskable variants add a `get_action_masks` call
per step, which makes them substantially slower single-env. The `eval_run` rule's
original `runtime=60` — and then `runtime=240` — killed the maskable evals
mid-rollout; because the eval loop has no SB3 log table and SIGKILL discards
buffered stdout, the log was **empty**, indistinguishable from a hang. Two ceiling
guesses failed for want of a measured rate, so the loop now prints a `steps/s`
heartbeat every 2k steps (flushed, rule exports `PYTHONUNBUFFERED=1`): the log
reveals the real throughput and a stall is no longer mistaken for a slow pass.

The heartbeat then overturned an earlier assumption. It showed the episodes are
**far larger than the ~59k job count** (≈1M+ decision steps — the env's MDP takes
many advance/no-op steps per scheduled job) and, decisively, that on the *identical*
dev70 environment PPO ran at **~34.6 steps/s while DQN ran at ~18** — a 2× gap that
the shared environment rebuild cannot explain. The per-step cost is therefore
dominated by the **policy forward pass**: a single batch-1 pass streams the
`56,090 × 4096 ≈ 230M`-weight first layer (~920 MB of float32) from memory every
step, which is memory-bandwidth-bound on CPU. This contradicted the initial "no
GPU, the env is the wall" choice. The GPU (L4, ~10× the memory bandwidth) attacks
exactly this cost, and `SB3.load()` uses `device="auto"` so inference moves to CUDA
with no code change. But the cluster has only 4 GPU nodes (≈2–3 typically free,
shared with other projects), so putting *all* eval on GPU would serialise it onto a
few nodes and leave the 6 CPU-only nodes idle. Fix: a **hybrid placement** — only the
**DQN family** (whose ~18 steps/s over a ~1M+ step episode would exceed the wall on
CPU) requests a GPU; **PPO/A2C and their maskable variants** (fast enough to finish
full-trace within the ceiling on CPU) run CPU-only, so they use the otherwise-idle
CPU nodes and do not compete for the scarce GPUs. With ~1M+ step episodes the old
`runtime=480` (8 h) killed evals mid-pass, so the ceiling was raised to the **14 h
partition max (840 min)** — a hard wall (eval has no resume). The heartbeat also emits
jobs-completed and the running `avg_waiting`/`avg_slowdown` so a single full run shows
where the metrics converge, in case a length cap is later needed. The holdout
evaluation has the same per-pass profile; it was split into one job per seed
(parallel) and keeps a GPU request (the winning algorithm is unknown at DAG-build
time and it is only 10 jobs) plus the same heartbeat and 840 ceiling. Each run records
`eval_wall_s` in its metrics file.

## Only the final checkpoint is kept (scratch capacity)

Ceph `/scratch` is 500 GB. Each checkpoint zip is ~2 GB (the policy's first layer
alone is `56,090 × 4096 ≈ 230M` weights), and the callback writes one per
`save_interval` — 10 per run × 60 runs ≈ 1.2 TB, far over the cap. The
intermediate checkpoints are never read: evaluation loads the final `model_path`
from the manifest and nothing globs the `selector/` directory. So training prunes
every zip except the final `{total_timesteps}.zip` immediately after the explicit
final save, holding scratch to ~60 × 2 GB ≈ 120 GB of models. (Equivalent lazier
option, not taken so mid-run checkpoints still exist for inspection: don't write
the intermediates at all.)

## Results interpretation and future options (physical_job, dev70, 10 seeds)

These notes are for the results/discussion and future-work sections. They record
how to read the `baseline_comparison.csv` output and which levers are worth
pulling before the deeplearn sweep is judged.

### Masking is the dominant factor; masked PPO reaches heuristic parity

Across the six DRL treatments, **action masking is the single largest determinant
of quality**. In eval the masked variants finish the trace in ≈76k–91k decision
steps; the non-masked `dqn`/`a2c` take ≈591k — 6–8× more — because without a mask
the policy repeatedly selects invalid actions (negative-reward no-ops that don't
advance a placement), and their episode reward is ≈ −800k vs the masked ≈ −170.
This is not under-training: it is the absence of masking in a large discrete
action space, and more steps do not fix it. The non-masked RL rows therefore
function as a **masking ablation** ("masking is necessary"), not as competitive
baselines. Within the masked variants the ordering is the textbook PPO > DQN > A2C;
`maskable_ppo` is the strongest and the only DRL treatment that is baseline-competitive.

### `maskable_ppo` vs the heuristics: parity-minus on averages, a tail win

Against the strongest baseline (LCFS) `maskable_ppo` lands **within ~9% on average
waiting** (2243 vs 2052), ~27% on `avg_slowdown` (7.49 vs 5.90), ~2% on turnaround,
ties on `max_waiting` (69126 vs 63993, **not** significant, p=0.19), and **beats all
three baselines on `max_slowdown`** (2918 vs 3092, significant). So the headline is
**heuristic-competitive with a worst-case-slowdown advantage**, not an outright win —
a defensible result on its own, especially paired with the ablation showing the other
algorithms collapse.

Two interpretation cautions for the writeup:

- **`p=0.001953` is the *smallest possible* two-sided Wilcoxon signed-rank p at
  n=10** (`(1/2)^9`). It means all 10 seeds fell on the same side — i.e. the gap is
  perfectly *consistent*, not that it is *large*. Effect sizes here are small
  (~9% on waiting). Do not let "significantly worse" read as a rout; report the
  effect size alongside the p-value.
- The **loss on averages but win on the tail** is the signature of a
  **reward-alignment** ceiling: the agent optimizes its shaped reward, which tracks
  worst-case slowdown better than mean waiting. This shapes the future-work options below.

### Is 3M timesteps the limiter? Probably a ceiling, not a budget shortfall

The instinct is to blame the training budget, but three signals point to a
**converged ceiling** rather than under-training: (1) the tight, all-seeds-agree
consistency above (under-training usually shows high seed variance, not a
reproducible small gap); (2) the average-loss/tail-win split, which is a
reward-proxy artifact that more steps only *reinforce*; (3) the policy is already
well-behaved (no starvation, `max_waiting ≈ baseline`), not stuck in a degenerate
basin exploration would escape.

**A config wrinkle makes "just add steps" non-trivial.** `learning_rate` is
`linear_3e-4` — linear decay from 3e-4 **to 0 over the horizon**. So (a) by 3M the
LR is ≈0 and a flat end-of-curve is *partly forced*, making "plateaued" ambiguous
(converged vs. LR exhausted); and (b) resuming the 3M checkpoint to 6M does **not**
behave like more training — the LR is already 0. Testing a larger budget requires
setting the horizon to 6M/10M and **retraining from scratch** (which re-stretches
the schedule, holding a higher LR longer early) — a different, 2–3×-cost run, not
an extension.

**How to decide before spending the compute:** read one representative (median)
seed's TensorBoard — `ep_rew_mean` slope over 2M→3M (still rising ⇒ budget could
help; flat ⇒ it won't), `entropy_loss` (collapsed early ⇒ committed policy),
`explained_variance` (low ⇒ the value function, not steps, is the bottleneck).
Then the cheapest decisive test is **2–3 seeds at 6M**, fresh, compared against the
3M seeds; if the gap doesn't move, budget is ruled out. Expectation given the
signature above: 6M/10M yields *marginal* movement, and the higher-leverage levers
are **reward shaping** (align the reward with mean waiting/slowdown, not just the
tail) and **observation features** — not raw steps.

### Why deeplearn is the more promising regime for DRL

The two traces differ structurally in a way that matters: physical_job baselines
report `gpu_utilization = 0.000` (a **single-resource, CPU-only** placement
problem), whereas deeplearn_job baselines report `gpu_utilization ≈ 0.261` — GPU is
a **real, contended** dimension. LCFS/SJF/UNICEP are myopic single-key heuristics
that cannot reason about joint CPU+GPU packing; on a CPU-only trace they are already
near-optimal, leaving a learned policy almost no headroom (exactly the parity we see).
The hypothesis for the paper: **DRL's advantage should scale with the resource-packing
complexity of the workload**, and deeplearn (CPU+GPU contention) is the
harder-to-heuristic regime where a learned policy has an axis to exploit.

- **Floor case (worst case):** if the GPU dimension buys nothing and DRL keeps the
  *same relative* gap as on physical, projecting the physical gaps onto deeplearn's
  LCFS baseline gives ≈940 avg_waiting (vs 860), ≈4.47 avg_slowdown (vs 3.52), and a
  persistent ~6% `max_slowdown` win — i.e. the identical parity-minus-with-tail-edge
  story at deeplearn's lower absolute scale. This is the *floor*, not the expectation.
- Note deeplearn is absolutely lighter-loaded (baseline slowdown ~3.5 vs 5.9,
  waiting ~860 vs 2052), so comparisons must be **relative**; do not compare
  physical DRL's absolute numbers against deeplearn baselines (different workloads —
  the magnitudes are set by the trace, not the policy).
- If deeplearn DRL *does* win, confirm the win **concentrates on GPU-contended /
  multi-resource jobs** rather than being uniform, so the mechanism is credible
  rather than lucky. Diagnose the budget/ceiling question **separately per trace** —
  the physical headroom finding does not transfer to deeplearn.

## Reproducibility niceties

- `PYTHONUNBUFFERED=1` is exported in the train rule so a hard crash (e.g. an
  OOM kill / SIGKILL) still flushes its traceback to the log instead of leaving
  an empty file.
- Snakemake skips already-completed treatments on rerun, so recovering from a
  partial sweep (e.g. only re-running the failed DQN/A2C jobs) does not redo the
  finished PPO runs.
