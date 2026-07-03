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

## Reproducibility niceties

- `PYTHONUNBUFFERED=1` is exported in the train rule so a hard crash (e.g. an
  OOM kill / SIGKILL) still flushes its traceback to the log instead of leaving
  an empty file.
- Snakemake skips already-completed treatments on rerun, so recovering from a
  partial sweep (e.g. only re-running the failed DQN/A2C jobs) does not redo the
  finished PPO runs.
