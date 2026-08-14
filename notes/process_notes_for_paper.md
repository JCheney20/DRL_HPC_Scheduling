# Process Notes for Paper

Practical notes from running the full sweep on the SLURM cluster. Intended for
the methodology / experimental-setup and limitations sections.

## Compute environment

- Nodes: 128 GB RAM, GPU (`gres=gpu:1`), 24 cores. Partition `main`.
- Each `(seed × algorithm)` treatment is one SLURM job (Snakemake fans them out).
- Training: 3M timesteps, `n_envs=20` (SubprocVecEnv) for all six treatments (DQN vectorized too).
- Dependency versions are pinned to match Wang et al.'s stack for comparability
  (`flake.lock`, plus hand-pinned `nix/sb3-contrib.nix` at 2.6.0). That choice has a
  cost: it inherits upstream defects fixed in later releases — see
  [A2C "numerical instability"](#a2c-numerical-instability-was-a-dependency-bug-not-divergence)
  for one that cost a training run, and the case for un-pinning in a future iteration.

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

## A2C "numerical instability" was a dependency bug, not divergence

A `maskable_a2c` seed crashed with a `MaskableCategorical` `Simplex()` violation —
`Expected parameter probs (Tensor of shape (20, 513)) ... to satisfy the constraint
Simplex()` — deterministically at 677,540 of 3M timesteps, identically across seven
re-runs on three different nodes. **This section previously recorded a different
diagnosis (entropy collapse plus non-standard advantage normalization). That
diagnosis was wrong and is corrected here**, retained only where it explains why the
wrong fixes appeared to work.

**What it is not.** A diagnostic in `a2c_mask.py`'s rollout loop dumps the policy
state at the moment of failure. At the crash point the policy parameters,
observations, latents and raw logits are **all finite**, and the masked
probabilities recompute to a valid distribution. `value_loss` peaked at 1.2e-4 with
no growth. Nothing had diverged.

**What it is.** `MaskableCategorical.apply_masking` ends by caching
`self.probs = logits_to_probs(logits)`, and it is called *twice* per forward pass:
once from `MaskableCategorical.__init__` with `masks=None` (the distribution is
built unmasked), then again from `MaskableActorCriticPolicy` with the real action
masks. `torch.distributions.Distribution.__init__` validates a parameter only once
it is no longer lazy — `if param not in self.__dict__` — so the first call's cache
is what the second call checks against `Simplex()`. **The tensor torch rejects is
the stale unmasked softmax, not the masked distribution being built.** Whether it
trips is float32 summation error against a 1e-6 tolerance.

This accounts for the observations that defeated every divergence theory:

- **Only `maskable_a2c` fails.** Stock SB3 `a2c` uses a plain `Categorical` whose
  `probs` stays lazy and is never validated, so it can saturate arbitrarily hard
  without crashing — which it does.
- **`ent_coef=0.01` appeared to help.** It held the policy near-uniform, and a
  near-uniform softmax sums cleanly. It suppressed the symptom without fixing
  anything (and never let a run learn — see below).
- **Bit-identical failure across re-runs.** The arithmetic is deterministic, so a
  seeded run fails at the same timestep every time. This reads as a reproducible
  algorithmic defect, and was read as one.

**Version provenance — the part worth stating in the paper.** The environment
deliberately pins `sb3-contrib==2.6.0` (`nix/sb3-contrib.nix`) to match the versions
used by Wang et al., whose HPCSim environment this work builds on, so that results
are comparable against theirs rather than against a differently-versioned stack.
**The bug was fixed upstream in sb3-contrib 2.9.0**, which clears the cached `probs`
before the re-init; its fix comment names this exact failure ("stale float32 probs
deviate from sum=1 by >1e-6 ... many categories"). Pinning for comparability
therefore inherited a defect that had already been repaired upstream. A future
iteration should move to current `sb3-contrib`/`torch` releases and re-establish the
comparison baseline there rather than carrying the pinned stack forward.

**Fix as shipped.** Rather than bump the pin mid-study — three minor versions, a
fresh container, and loss of comparability with completed runs — `src/sb3_compat.py`
backports upstream's one-liner: drop the cached `probs` before `apply_masking`
re-inits. It is imported once from `src/utils.py`. The masked probabilities are
bit-identical to the unpatched intent, so no result changes, and because it is repo
code it deploys with a `git pull` and needs no `.sif` rebuild. Argument validation is
deliberately left ON (rather than the blunter
`Distribution.set_default_validate_args(False)`), so genuinely non-finite logits
would still raise.

**Two earlier fixes, reassessed.** `ent_coef=0.01` is gone and should stay gone, but
for a measured reason rather than a stability one: with the bonus all 20 A2C runs
finish at 99.8–100% of the `ln(513)=6.240` entropy ceiling — still essentially
uniform — because the entropy term (~0.0624) outweighs `policy_loss` (1e-9 to 1e-3)
by roughly 600×. It also applied to A2C only, confounding any "A2C is temperamental"
comparison against PPO and DQN (reviewer item N26a). `a2c_mask.py`'s
`normalize_advantage` default stays `False`: that is canonical A2C and defensible on
its own merits, but it did **not** fix the crash and is no longer claimed to.

**Limitation to state honestly.** The mechanism is established by construction — in
2.6.0 the only route by which `probs` enters `__dict__` is that cache assignment, so
a `Simplex()` error naming `probs` can only be the previous call's cache — and a
regression test reproduces the exact error end-to-end. The *drift itself* was only
reproducible synthetically at 4096 categories; at this project's 513 it measures
~6e-7 on CPU, just under tolerance. Training ran on CUDA, whose softmax and sum
reductions accumulate in a different order, and that crossing point was not
reproduced on CPU hardware.

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

**Later correction to the holdout half of this (M5b).** Two things changed the
calculus. The DQN family was subsequently measured to evaluate acceptably on CPU,
so the reason for its GPU request does not survive; and holdout was extended from
the single Pareto winner to all six treatments, taking it from 10 jobs to 60 per
trace. At that scale a blanket GPU request is actively harmful — it would
serialise the whole holdout stage onto the 2–3 typically-free (and shared) GPU
nodes while the six CPU-only nodes idle. Holdout evaluation is therefore now
CPU-only. Note this does **not** retract the finding above: the per-step cost of a
DRL eval is still dominated by the batch-1 forward pass through the ~230M-weight
first layer, and the 2× PPO-vs-DQN gap on an identical environment is still the
evidence for it. What changed is the *scheduling* conclusion drawn from it, once
the job count grew and CPU throughput turned out to be adequate. The `eval_run`
rule for the dev split still uses the hybrid placement described above.

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

## The random-policy control (reviewer item N27)

This section records a control that was added to the baseline set, why it was
built the way it was, and how to read it. It is a methods addition, but its
purpose is to close a gap in the *results* argument, so it is written to be
usable in both sections.

### The gap it closes

On physical, masking + a *near-uniform* policy (`maskable_a2c`, avg waiting
2,328 ± 90 s — entropy 6.2297 against the `ln(513) = 6.240` ceiling, i.e. a
policy that has barely departed from uniform) lands within **3.6%** of masking +
a *learned* policy (`maskable_ppo`, 2,243 ± 41 s). Two explanations fit that
observation equally well:

1. both policies learned something, and the learned one is slightly better; or
2. the physical benchmark is largely **insensitive to policy quality** — any
   masked policy scores ≈2,300, and neither one learned anything
   schedule-relevant.

Nothing in the existing result set separates them, and reading (2) does not just
weaken the A2C row: it invalidates *every* physical DRL claim, because it would
mean the headline number is produced by the action mask and the environment
rather than by the policy. That is the rejection risk. The fix is a control
whose policy is uniform over the valid actions **by construction** and has
learned nothing at all: it establishes the floor that "MaskablePPO learned to
schedule" has to clear.

### The design decision that matters: which simulator the control runs in

`HPCsim` is effectively two different simulators depending on the entry point,
and conflating them would have quietly produced the wrong experiment:

- **`HPCsim.run()`** — the heuristic loop that LCFS/SJF/UNICEP use. It drives
  `job_schedule_allocation()`, which consults the `Scheduler` strategy object
  **and runs `Scheduler.backfill()`**.
- **`HPCsim.step()`** — the RL MDP the six DRL treatments are evaluated on: a
  513-way discrete action (`window_size=512` queue slots + one forward/no-op),
  an action mask, and **no backfill anywhere**.

The obvious implementation — add a `"random"` strategy to the `Scheduler` class
so `run_baseline.py` picks it up like any other heuristic — would have produced
*random-choice-plus-backfill*: a fourth heuristic, not a control. Its score would
differ from MaskablePPO's for reasons unrelated to policy quality (backfill alone
moves the metric substantially), and the comparison would answer no question
anyone asked.

A control has to hold **everything except the policy** fixed. So the control
rolls out the identical MDP — same wrapper stack
(`Float32Observation(AllocationCommit(HPCsim(...)))`), same 513-action space,
same `action_masks()`, same absence of backfill, same reward accounting, same
metric extraction — and substitutes exactly one thing:
`model.predict(obs, ...)` becomes a uniform draw over the valid entries of the
mask. It is built by calling `evaluate_agents.build_env` directly rather than
reconstructing the env, so the control and the DRL evaluation cannot silently
drift apart if the wrapper stack is ever changed.

The observation is still computed and then ignored. That is the property under
test: a policy that cannot see the state.

Practical consequence for the write-up: the control is **not** a scheduling
heuristic and must not be described as one. It is an ablation of the *policy*
within the DRL evaluation pipeline, reported in the baseline table because that
is where a reader needs to see it.

### Seeds — what they vary, and what they do not

The three heuristics are deterministic and run seedless. This control is the one
stochastic member of the baseline set, so it runs the same 10 seeds as the DRL
treatments and is the only baseline row reported as **mean ± std**.

What the seed varies is worth stating precisely, because it is a question a
reviewer will ask. `random_job=False` and the trace, cluster and topology are
fixed, so **the environment is deterministic and the seed does not perturb it** —
the seed varies only the action draws. That is the correct analogue of the DRL
side, where the seed varies the trained policy and the evaluation environment is
likewise identical across seeds. The spread across the control's 10 seeds is
therefore *pure policy variance*, which is what makes it comparable to the ± on
the DRL rows.

The draws come from a dedicated `np.random.default_rng(seed)` rather than the
env's `np_random`. Nothing on this path currently consumes `np_random`, but if
anything ever did, a shared stream would couple the policy's randomness to the
environment's and quietly stop this being a clean control.

Only the **masked** control is run. The unmasked question is already answered by
the unmasked DRL treatments, which function as the masking ablation; and an
unmasked uniform policy would draw from 513 actions of which typically ~1 is
valid, would essentially never commit a placement, and would hit
`AllocationCommit`'s hang guard rather than produce a comparable number.

### Two pipeline assumptions the control broke

Both were assumptions that "baseline" implies "deterministic, one run", and both
had to be relaxed rather than worked around:

1. **`baseline_aggregate.py` rejected the control outright.** Its duplicate check
   keyed on `(treatment_id, split_id)`, so the 10 seeds looked like 10 collisions
   and raised. The key is now `(treatment_id, split_id, seed)` — which still
   catches a genuine re-run of a seedless heuristic, since their seed field is
   empty and therefore still collides.
2. **`baseline_summary.csv` had nowhere to put a standard deviation.** It carried
   only `{metric}_mean_mean`. It now also carries `{metric}_mean_std` and an
   explicit `n_seeds`, matching `algorithm_summary.csv`'s convention. For the
   deterministic heuristics the grouping is a no-op — the mean of one value is
   that value, and its std is NaN (ddof=1 at n=1), which `fmt_mean_std` already
   renders with no ± term, so **their rows are byte-identical to before**.

The std is load-bearing rather than decorative: a control reported as a point
estimate cannot answer the question N27 asks. If the control's spread overlaps
MaskablePPO's, that overlap *is* the finding.

### How to read the result

The comparison to make is control vs `maskable_a2c` vs `maskable_ppo` on
`avg_waiting`, physical, dev70. Three outcomes, decided before the numbers land
so the reading is not fitted to them:

- **Control is clearly worse than both masked treatments** (well outside the
  seed spread). Reading (2) is dead. Masking plus *learning* beats masking plus
  chance, `maskable_a2c`'s near-uniform entropy notwithstanding — which sharpens
  `N19` into a genuinely interesting result: a policy can sit near the entropy
  ceiling and still have learned a useful action *ordering*, since evaluation is
  deterministic argmax over near-tied logits, not sampling.
- **Control is comparable to `maskable_a2c` but clearly worse than
  `maskable_ppo`.** The cleanest outcome for the paper: it splits the two,
  confirms `maskable_a2c` is effectively unlearned, and leaves MaskablePPO's
  result standing with a measured margin over chance.
- **Control is comparable to both.** Reading (2) holds, and the honest response
  is to say so: the physical benchmark does not discriminate policy quality at
  this scale, the physical DRL numbers cannot support a "learned to schedule"
  claim, and the paper's weight moves to deeplearn (which the existing
  `gpu_utilization ≈ 0.261` vs `0.000` argument already predicts is the more
  discriminating regime). This would be a bad result but a publishable one; it
  is much worse to have a reviewer discover it.

Run it on **both** traces. Physical is where the risk is, but the control is
also the natural check on deeplearn's odd result that unmasked `ppo` wins
operationally with a degenerate critic (`N1`/`N21`) — if chance also does well
there, that finding needs rewording too.

### Cost

Evaluation only: no training, no GPU. The control has no network, so unlike the
DRL evals there is no forward pass to accelerate — a GPU request would only queue
the job behind the 2–3 contended GPU nodes for nothing. It is also why the
control should be *faster* per step than any DRL eval: the measured per-step cost
of a DRL eval is dominated by streaming the `56,090 × 4096 ≈ 230M`-weight first
layer (see the evaluation section above), and the control does none of that.

Budget from the masked DRL evals, which are the right reference for step count
(masking is what keeps the episode at ≈75k–91k decision steps rather than
≈591k): physical masked evals took ≈2,700–3,400 s wall, deeplearn ≈350–630 s.
The control should come in under those, and 10 seeds fan out to one SLURM job
each, so expect well under an hour of wall time per trace rather than the 4–6 h
budgeted in the TODO. The rule still requests the 840-minute partition maximum —
an unused ceiling costs nothing, whereas the two ceilings guessed too low on the
eval rule each cost a full rerun and left empty logs (SIGKILL discards buffered
stdout). The same flushed `steps/s` heartbeat is emitted for the same reason.

One caveat on step count: a worse policy generally churns *more* decision steps
before completing the trace (`maskable_dqn` needs 91k where `maskable_ppo` needs
77k), so the control may run longer than the masked treatments even though each
step is cheaper. `decision_count` is recorded per run and is itself worth
reporting — if the control needs far more steps to clear the same trace, that is
independent evidence the learned policies are doing something.

### Limits to state honestly

- The control bounds **policy quality**, not reward-proxy quality. If the control
  scores well, that is evidence the environment plus mask do most of the work; it
  is not evidence about whether the shaped reward is aligned with `avg_waiting`.
  That remains `N1`/`N21`'s question.
- It is deliberately **excluded from the Friedman/Nemenyi omnibus**. That test
  compares the six DRL treatments; adding a seventh arm would widen the Nemenyi
  critical difference and worsen a power problem `M4` and `N4` already flag,
  which would be a real cost for no gain. The control is reported descriptively
  beside the heuristics, and if a test is wanted the right one is a seed-matched
  Wilcoxon signed-rank against `maskable_ppo` on the shared seed set.
- The comparison is on dev70, so it inherits `M5`'s in-sample caveat. That is not
  a problem here: the control never trained on anything, so if anything the
  in-sample setting *favours* the DRL treatments, and the control's floor is
  conservative.

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
