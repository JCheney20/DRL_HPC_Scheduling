"""random_control.py

Uniform-random-over-valid-actions control policy (reviewer item N27).

WHY THIS EXISTS
---------------
On the physical trace, masking + a *near-uniform* policy (MaskableA2C,
2,328 +/- 90 s avg waiting) lands within 3.6% of masking + a *learned* policy
(MaskablePPO, 2,243 +/- 41 s). Two readings of that fit the data equally well:

  (a) both policies learned something, and the learned one is slightly better;
  (b) the physical benchmark is largely insensitive to policy quality, so
      *any* masked policy scores ~2,300 and neither one learned anything
      schedule-relevant.

Nothing in the current result set distinguishes them, and (b) would invalidate
every physical DRL claim, not just A2C's. This module supplies the missing
control: a policy that is uniform over the valid actions *by construction* and
has learned nothing at all. It sets the floor that "MaskablePPO learned to
schedule" has to clear.

WHY IT RUNS THE MDP PATH, NOT THE HEURISTIC PATH
------------------------------------------------
HPCsim is two different simulators depending on which entry point is used, and
this distinction is the whole design of this module:

  * ``HPCsim.run()`` -- the heuristic loop used by the LCFS/SJF/UNICEP
    baselines. It drives ``job_schedule_allocation()``, which consults the
    ``Scheduler`` strategy object AND runs ``Scheduler.backfill()``.
  * ``HPCsim.step()`` -- the RL MDP the six DRL treatments are evaluated on. A
    513-way discrete action (``window_size=512`` queue slots + one
    forward/no-op), an action mask, and **no backfill anywhere**.

Adding a ``"random"`` strategy to the ``Scheduler`` class would therefore have
produced a random-choice-plus-backfill scheduler: a fourth heuristic, not a
control. Its score would differ from MaskablePPO's for reasons that have
nothing to do with policy quality (backfill alone is worth a large slice of the
metric), and the comparison would answer no question anyone asked.

The control has to hold *everything except the policy* fixed. So this module
rolls out the identical MDP: the same wrapper stack, the same action space, the
same ``action_masks()``, the same reward accounting, the same metric
extraction -- built by calling ``evaluate_agents.build_env`` directly rather
than reconstructing it, so the two cannot silently drift apart. The single
substitution is ``model.predict(obs, ...)`` -> a uniform draw over the valid
entries of the mask. The observation is computed and then ignored, which is
precisely the property being tested: a policy that cannot see the state.

ON SEEDS
--------
The heuristic baselines are deterministic and run seedless. This control is
not: it is the only baseline with a stochastic component, so it carries the
same 10 seeds as the DRL treatments and is reported as mean +/- std.

Note what the seed does and does not vary. ``random_job=False`` and the trace,
cluster and topology are fixed, so **the environment is deterministic and the
seed does not perturb it** -- the seed varies only the action draws. That is
the correct analogue of the DRL side, where the seed varies the policy and the
evaluation environment is likewise identical across seeds. The spread across
these 10 seeds is therefore pure policy variance, which makes it directly
comparable to the DRL rows' +/- and is exactly the quantity N27 needs.

The draws come from a dedicated ``np.random.default_rng(seed)`` rather than the
env's ``np_random``. Nothing in HPCsim currently consumes ``np_random`` on this
path, but if something ever did, sharing the stream would couple the policy's
randomness to the environment's and quietly stop this being a clean control.

ONLY THE MASKED CONTROL IS RUN
------------------------------
There is deliberately no unmasked random arm. N27 asks about "masking + a
uniform policy", and the unmasked question is already answered: the unmasked
DRL treatments function as the masking ablation. An unmasked uniform policy
would draw from 513 actions of which typically ~1 is valid, would essentially
never commit a placement, and would run into ``AllocationCommit``'s hang guard
rather than producing a comparable number.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks

from src.evaluate_agents import build_env
from src.utils import RunSpec, safe_metric_access

# RANDOM_ALGORITHM / RANDOM_TREATMENT_ID live in src/utils.py beside
# TRAD_ALGORITHMS, so the pipeline has one source of truth for the label. Note
# the treatment_id is "random__mask_true", not the "<algo>__mask_false" the
# heuristics use: this control genuinely does use the mask, and the label has
# to survive being read off a results table by someone who was not in the room.


def rollout_random(
    spec: RunSpec,
    seed: int,
    max_steps: int | None = None,
    heartbeat_every: int = 2000,
) -> dict:
    """Roll a uniform-random-over-valid-actions policy over the full trace.

    Returns an ``EvalResult``-shaped dict (same keys, same order) so the caller
    can write it through the existing baseline output contract unchanged.
    """
    env = build_env(spec, seed=seed)
    rng = np.random.default_rng(seed)

    # The observation is built by the env and then never read: this policy
    # cannot see the state, which is the property under test. Bound to `_` so
    # that is visibly deliberate rather than looking like a dropped variable.
    _, _ = env.reset(seed=seed)
    done = False
    truncated = False
    episode_reward = 0.0
    n_steps = 0
    decision_latencies: list[float] = []
    t_start = time.perf_counter()

    while not (done or truncated):
        if max_steps is not None and n_steps >= max_steps:
            break

        t_dec0 = time.perf_counter()

        # Same call the maskable treatments make in evaluate_agents, so the
        # control sees byte-identical masks. reshape(-1) because HPCsim builds
        # the mask as a plain Python list (Queue.get_state), not an ndarray.
        mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            # Should be unreachable: HPCsim.step() forwards system time until
            # at least one job is allocatable before it returns, and the
            # forward action is only masked off (forward_count >= 10) after
            # forwards that each guaranteed a valid job. Fail loudly rather
            # than substituting a fallback action -- a silent fallback would
            # stop the policy being uniform-over-valid and would corrupt the
            # very quantity this control measures. Mirrors a2c_mask.py's
            # all-false-mask guard.
            raise ValueError(
                f"[{spec.run_id}] all-false action mask at step {n_steps} "
                f"(time={env.time}, queue={len(env.queue.job_queue)}, "
                f"forward_count={env.forward_count}) — cannot draw a valid action"
            )

        action = int(rng.choice(valid))

        decision_latencies.append(time.perf_counter() - t_dec0)
        _, reward, done, truncated, _ = env.step(action)
        if not np.isfinite(reward):
            raise ValueError(f"[{spec.run_id}] Non-finite reward encountered: {reward}")
        episode_reward += float(reward)
        n_steps += 1

        # Same heartbeat contract as evaluate_agents: a single-env full-trace
        # pass has no SB3 log table, so without this a SIGKILL at the runtime
        # ceiling is indistinguishable from a hang. Flushed, and the rule
        # exports PYTHONUNBUFFERED=1.
        if n_steps % heartbeat_every == 0:
            el = time.perf_counter() - t_start
            jobs = len(env.evaluator.completed_job)
            msg = (
                f"[{spec.run_id}] {n_steps} steps, {jobs} jobs, {el:.0f}s, "
                f"{n_steps / el:.1f} steps/s"
            )
            if n_steps % (heartbeat_every * 10) == 0:
                wt = env.evaluator.waiting_time() or (0.0, 0.0)
                sd = env.evaluator.bounded_slowdown() or (0.0, 0.0)
                msg += f", avg_wait={wt[1]:.1f}, avg_slowdown={sd[1]:.3f}"
            print(msg, flush=True)

    eval_wall_s = time.perf_counter() - t_start

    # Same guarded accessors evaluate_agents uses: these Evaluator methods
    # return None (not a tuple) when no job has completed, which would other-
    # wise raise on unpack and lose the whole run rather than reporting a zero.
    max_w, avg_w = safe_metric_access(
        env.evaluator.waiting_time, (0.0, 0.0), "waiting_time"
    )
    max_s, avg_s = safe_metric_access(
        env.evaluator.bounded_slowdown, (0.0, 0.0), "bounded_slowdown"
    )
    avg_t = safe_metric_access(
        env.evaluator.average_turnaround, 0.0, "average_turnaround"
    )
    cpu_util, gpu_util = safe_metric_access(
        env.utilization, (0.0, 0.0), "utilization"
    )

    if truncated:
        # AllocationCommit's hang guard fired, or max_steps cut the pass short.
        # The metrics below are then computed over a partial trace and are NOT
        # comparable to a full-trace DRL row, so say so in the log rather than
        # letting a short row pass silently into the summary.
        print(
            f"[WARN] {spec.run_id} truncated after {n_steps} steps — "
            f"metrics cover a partial trace and are not comparable to the "
            f"full-trace treatments.",
            flush=True,
        )

    return {
        "run_id": spec.run_id,
        "treatment_id": spec.treatment_id,
        "algorithm": spec.algorithm,
        "use_masking": spec.use_masking,
        "window_size": spec.window_size,
        "tail_size": spec.tail_size,
        "seed": seed,
        "split_id": spec.split_id,
        "model_path": "",
        "trace_file": spec.trace_file,
        "topology_file": spec.topology_file,
        "node_file": spec.node_file,
        "episode_reward": episode_reward,
        "decision_count": n_steps,
        "decision_latency_mean_ms": (
            float(np.mean(decision_latencies) * 1000.0) if decision_latencies else 0.0
        ),
        "eval_wall_s": round(eval_wall_s, 2),
        "max_waiting": float(max_w),
        "avg_waiting": float(avg_w),
        "max_slowdown": float(max_s),
        "avg_slowdown": float(avg_s),
        "avg_turnaround": float(avg_t),
        "cpu_utilization": float(cpu_util),
        "gpu_utilization": float(gpu_util),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
