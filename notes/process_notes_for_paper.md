# Process Notes for Paper

Practical notes from running the full sweep on the SLURM cluster. Intended for
the methodology / experimental-setup and limitations sections.

## Compute environment

- Nodes: 128 GB RAM, GPU (`gres=gpu:1`), 24 cores. Partition `main`.
- Each `(seed × algorithm)` treatment is one SLURM job (Snakemake fans them out).
- Training: 3M timesteps, `n_envs=20` (SubprocVecEnv) for on-policy algorithms.

## DQN needed substantially more RAM than the on-policy algorithms

DQN (and MaskableDQN) are off-policy and keep an experience **replay buffer**;
PPO/A2C are on-policy and discard each rollout after the update. The replay
buffer is what drives DQN's memory footprint.

The observation is a `gym.spaces.Dict` of `float64` vectors totalling **56,090
values ≈ 438 KB per observation**. It is dominated by the cluster vector
(`5 × max_jobs = 5 × 10,260 = 51,300` values); the queue vector is only 3,585.
Stable-Baselines3 stores both `obs` and `next_obs` for every transition, so each
stored transition costs ≈ **0.88 MB**. Because the observation is a `Dict`, SB3
forbids `optimize_memory_usage`, so the 2× obs/next_obs cost cannot be avoided.

Consequences:

- `buffer_size` maps almost linearly to RAM: `≈ buffer_size × 0.88 MB`.
- A conventional `buffer_size=1,000,000` would need ~880 GB — far beyond any
  node. Even 200k needs ~167 GB, still over the 128 GB nodes.
- Buffer sized to fit: **`buffer_size=75,000` ≈ 63 GB**, requested with
  `mem_mb=98304` (96 GB) to leave headroom for the model, CUDA context, and env.
- On-policy runs (PPO/A2C) used only a fraction of this; the 96 GB request is
  provisioned for the DQN worst case and shared across the rule.

Paper-ready framing: the large per-observation footprint (a direct consequence
of the per-job cluster encoding) makes off-policy replay memory-bound on this
problem, capping the practical replay-buffer size well below values common in
the DQN literature (e.g. 1M for Atari, where each frame is ~7 KB).

## A2C wall-clock (timeouts)

A2C hit the 8 h (`runtime=480`) ceiling on some nodes while PPO finished
comfortably. A2C updates on a very short rollout (SB3 default `n_steps=5`), i.e.
a gradient step roughly every `5 × n_envs` environment steps, giving far more
update passes than PPO for the same 3M steps — it is update-bound, not
rollout-bound. Options if it recurs: raise `runtime` (if the partition's max
wall-time allows), or raise A2C `n_steps` to reduce update frequency (note: this
changes learning dynamics, so keep it consistent across seeds for fairness).

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
observation rebuild that bounds training, but without the 20-env parallelism
(~11 env-steps/s on one worker). A dev70 pass (~59k steps) therefore takes
~90 min, and the maskable variants add a `get_action_masks` call per step. The
`eval_run` rule's original `runtime=60` killed these mid-rollout; because SIGKILL
discards buffered stdout, the log was **empty**, which reads as a hang rather than
a timeout. Fix: `runtime=240`. No GPU is requested — the wall is the environment
rebuild, not the policy forward pass, so a GPU would only idle and block the
scarce training GPUs. The holdout evaluation has the same per-pass cost and
originally bundled all seeds of the winning algorithm into one serial job
(~6–14 h); it was split into one job per seed (parallel), so it scales like
`eval_run`. Each run records `eval_wall_s` in its metrics file for tuning.

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

## Reproducibility niceties

- `PYTHONUNBUFFERED=1` is exported in the train rule so a hard crash (e.g. an
  OOM kill / SIGKILL) still flushes its traceback to the log instead of leaving
  an empty file.
- Snakemake skips already-completed treatments on rerun, so recovering from a
  partial sweep (e.g. only re-running the failed DQN/A2C jobs) does not redo the
  finished PPO runs.
