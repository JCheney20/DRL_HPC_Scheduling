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

## A2C needed an entropy floor to stay numerically stable

A `maskable_a2c` seed crashed at ~20k steps with a `MaskableCategorical`
`Simplex()` violation — the distribution's probabilities no longer summed to one
because the policy logits had gone non-finite. Diagnosis: with the default
`ent_coef=0.0`, the policy saturated to ~zero entropy within 20k steps on the
~230M-parameter `[4096,2048,1024]` first layer (visible as `entropy_loss ≈ 0` and
all losses collapsing to ~1e-7). A2C optimises with RMSprop (`eps=1e-5`), whose
update `lr·g/(√v + eps)` amplifies the resulting near-zero gradients: the running
variance decays, a gradient spike is divided by an almost-zero denominator, and
the weights take a divergent step that pushes the logits to `±inf`/`NaN`. The next
masked softmax then cannot lie on the simplex and PyTorch raises. Only an occasional
seed triggered it, but because training is seeded the failure is deterministic for
that seed (a plain rerun reproduces it), so it needed a fix rather than a retry.
Fix: enable a small entropy bonus, **`ent_coef=0.01`** (the value SB3's own A2C
example uses), in the A2C construction path (`a2c` and `maskable_a2c`), which keeps
entropy up and prevents the collapse. It is a cheap guard against a rare
divergence, not a correction to a pervasive problem. Operationally the A2C cohort
is **mixed by design**: only the run that diverged was retrained under the floor,
because rerunning it was already required (it had crashed); the other A2C seeds,
which never diverged, were left as trained with the default `ent_coef=0.0` rather
than re-burning stable runs. The coefficient is small enough that it does not
materially change A2C's behaviour on the runs that were already stable, so the
floor functions as a targeted stabiliser for the seed that needed it rather than a
cohort-wide objective change. Applied only to A2C — PPO and DQN never exhibited
the divergence. The masking was not implicated: an explicit all-false-mask guard
(`a2c_mask.py`) did not trigger.

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
`runtime` raised to **480** with that headroom; tune from the heartbeat /
`eval_wall_s`. No GPU is requested — the wall is the environment rebuild, not the
policy forward pass, so a GPU would only idle and block the scarce training GPUs.
The holdout evaluation has the same per-pass cost and originally bundled all seeds
of the winning algorithm into one serial job; it was split into one job per seed
(parallel) and carries the same heartbeat and 480 ceiling. Each run records
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

## Reproducibility niceties

- `PYTHONUNBUFFERED=1` is exported in the train rule so a hard crash (e.g. an
  OOM kill / SIGKILL) still flushes its traceback to the log instead of leaving
  an empty file.
- Snakemake skips already-completed treatments on rerun, so recovering from a
  partial sweep (e.g. only re-running the failed DQN/A2C jobs) does not redo the
  finished PPO runs.
