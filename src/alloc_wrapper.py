"""alloc_wrapper.py

Allocation-commit wrapper for HPCsim.

HPCsim is a *two-level* scheduler env. ``HPCsim.step()`` is the high-level
"which job to schedule next" policy; a *separate* ``ENV_allocator`` agent is the
low-level "which nodes to place it on" policy that actually commits a placement
(``cluster.allocation`` + ``queue.pop_sched_job`` + schedules the completion
event via ``add_job_completion``).

This project trains a single high-level agent and never wires up the low-level
``ENV_allocator``. So ``HPCsim.step()`` only *checks* that a job could be placed
(``check_allocate_list`` -> ``info['allocation']=True``) and returns without ever
committing it. Consequences, all from the same missing commit:

  * no job is ever removed from ``queue.job_queue`` (it grows monotonically);
  * no ``('complete', id)`` event is ever scheduled, so
    ``evaluator.completed_job`` stays empty -> every metric (``waiting_time``,
    ``bounded_slowdown``, ``average_turnaround``) is 0.0 / None;
  * ``done = (len(event_queue) + len(job_queue) == 0)`` can never become True,
    because ``job_queue`` never empties -> the episode runs until the wall clock.

Passing ``allocator="best_fit"`` does NOT fix this: that only builds the
``Allocator`` strategy object (``self.allocator``). The strategy is only
*invoked* inside ``HPCsim.job_schedule_allocation()`` (the classic non-RL
``run()`` loop), never inside ``HPCsim.step()``. Configuring *how* to place nodes
does nothing when the place-nodes call is never reached on the RL path.

This wrapper closes the loop for the flat single-agent formulation: whenever
``HPCsim.step`` reports ``info['allocation']``, it commits the placement with the
env's own built-in allocator (mirroring ``job_schedule_allocation`` exactly),
pops the job, schedules its completion, and returns a fresh post-commit
observation (so the agent's next action indexes the queue as it now is, not as it
was before the commit). Jobs then run and complete, metrics populate, and the
episode terminates naturally.

Because this changes the observation / transition / action-mask / reward the
agent experiences, it MUST wrap the env in BOTH training and evaluation, or the
two disagree and the trained policy is evaluated out-of-distribution.

Composition: wrap HPCsim directly (HPCsim is read-only), *inside*
``Float32Observation``:

    Float32Observation(AllocationCommit(HPCsim(...)))
"""

from __future__ import annotations

import gymnasium as gym


class AllocationCommit(gym.Wrapper):
    """Commit the placement HPCsim.step only flags but never performs.

    Parameters
    ----------
    env:
        A raw ``HPCsim`` instance (not yet wrapped by ``Float32Observation``).
    hang_guard_steps:
        Safety net. If this many *consecutive* decision steps pass with no new
        job completing, truncate the episode. On a healthy run completions start
        early and recur, so the counter never approaches this bound; it only
        fires if the commit path is (still) broken, turning a silent 14 h
        wall-clock hang into a bounded, visible stop. Set ``None`` to disable.
    """

    def __init__(self, env: gym.Env, hang_guard_steps: int | None = 500_000):
        super().__init__(env)
        self._hang_guard_steps = hang_guard_steps
        self._last_completed = 0
        self._steps_since_completion = 0

    def __getattr__(self, name):
        # This gymnasium version does not auto-forward attributes through a
        # plain Wrapper (see the same note in src/obs_wrapper.py). Eval reaches
        # HPCsim internals (evaluator, utilization, ...) *through* this wrapper,
        # so forward them manually to the base env.
        if name.startswith("_"):
            raise AttributeError(
                f"Attempted to get missing private attribute '{name}'"
            )
        return getattr(self.env, name)

    def action_masks(self):
        # Explicit: sb3-contrib's get_action_masks() / VecEnv.env_method reach
        # the mask through this wrapper. current_valid_job is kept in sync with
        # the post-commit observation in step().
        return self.env.action_masks()

    def reset(self, *, seed=None, options=None):
        # HPCsim.reset predates the gymnasium API (accepts only `seed`).
        obs, info = self.env.reset(seed=seed)
        self._last_completed = len(self.env.evaluator.completed_job)
        self._steps_since_completion = 0
        return obs, info

    def step(self, action):
        env = self.env  # the raw HPCsim
        obs, reward, done, truncated, info = env.step(action)

        # HPCsim flagged a feasible placement but did not commit it. Commit it
        # here, exactly as job_schedule_allocation() would (Cluster/Scheduler
        # are the source of truth; we only call their public methods).
        if info.get("allocation"):
            job = info["selected_job"]
            ok, node_dict = env.cluster.check_allocate_list(job)
            if ok:
                node_list = env.allocator.allocator(
                    job, node_dict, env.cluster.topology, weight=env.allocate_weight
                )
                if env.cluster.allocation(job, node_list, env.time):
                    env.allocated_job_count += 1
                    env.queue.pop_sched_job(job)
                    env.add_job_completion(job)   # schedules ('complete', id)
                    env.sort_event_queue()
                    # Return a fresh post-commit observation so the next action
                    # indexes the queue as it now is (the committed job is gone).
                    # Do NOT forward time here: any other jobs allocatable at the
                    # current instant should still be offered to the agent; when
                    # none are, HPCsim.step forwards on the following step.
                    new_obs, mask = env.get_state()
                    env.current_valid_job = mask
                    obs = new_obs
                    done = (len(env.event_queue) + len(env.queue.job_queue)) == 0

        # Hang guard: bound the episode if nothing is completing.
        completed = len(env.evaluator.completed_job)
        if completed > self._last_completed:
            self._last_completed = completed
            self._steps_since_completion = 0
        else:
            self._steps_since_completion += 1
            if (
                self._hang_guard_steps is not None
                and self._steps_since_completion >= self._hang_guard_steps
            ):
                print(
                    f"[AllocationCommit] hang guard tripped: "
                    f"{self._steps_since_completion} steps with no job completion "
                    f"({completed} completed so far) — truncating.",
                    flush=True,
                )
                truncated = True

        return obs, reward, done, truncated, info
