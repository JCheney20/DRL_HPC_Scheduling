"""verify_reward_fix.py

Deterministic check of the reward/trace fixes that the pipeline smoke cannot
cover. The smoke runs only 200 timesteps, but the reward tick fires at
action_counts >= 512, so smoke never executes the reward path at all — this
script forces that path directly.

Checks:
  1. get_reward with an EMPTY job_queue at the tick returns 0 (finite), not nan
     — this is the regression that NaN-ed the policy net and crashed
     MaskablePPO/MaskableA2C.
  2. The tick fires for action_counts >= 512 (not only == 512) and resets.
  3. A non-empty queue yields an hours-scaled negative reward
     (mean(waits) / 3600), i.e. O(1-30) not O(1e4-1e5).
  4. Trace_Reader rejects requested_node < 1 at load (would otherwise
     ZeroDivisionError deep in a run).

Run from the repo root:
    PYTHONPATH=. python scripts/verify_reward_fix.py

Temporary: delete after the fixes are validated.
"""
from __future__ import annotations

import math
import sys

from src.HPCsim.HPCsim import HPCsim
from src.HPCsim.Trace_Reader import Job


def check_reward() -> None:
    env = HPCsim(
        topology_file="data/topology/physical_topology.txt",
        allocator="best_fit",
        node_file="data/topology/nodes.csv",
        trace_file="data/splits/physical_job_dev70.tsv",
        random_job=False,
        window_size=512,
        tail_size=64,
        seed=0,
    )
    env.reset(seed=0)

    # (1) empty queue at the tick -> 0, finite (was nan before the guard)
    env.queue.job_queue = []
    env.waiting_time_list = []
    env.action_counts = 512
    r = env.get_reward()
    assert math.isfinite(r) and r == 0, f"empty-queue reward should be 0, got {r!r}"
    assert env.action_counts == 0, "tick did not reset action_counts"
    print(f"[1] empty-queue tick -> {r} (finite, 0) OK")

    # (2) overshoot >= 512 still fires and resets
    env.queue.job_queue = []
    env.waiting_time_list = []
    env.action_counts = 517
    r = env.get_reward()
    assert math.isfinite(r) and r == 0 and env.action_counts == 0
    print("[2] overshoot tick (action_counts=517) fired and reset OK")

    # (3) non-empty queue -> hours-scaled negative reward
    class _Job:
        def __init__(self, submit: int) -> None:
            self.system_submit = submit

    env.time = 36000  # 10 hours
    env.queue.job_queue = [_Job(0), _Job(0), _Job(0)]  # each waited 36000 s
    env.waiting_time_list = []
    env.action_counts = 512
    r = env.get_reward()
    expected = -(36000.0 / 3600.0)  # mean(waits) / 3600 = 10.0
    assert math.isfinite(r) and abs(r - expected) < 1e-6, f"expected {expected}, got {r}"
    assert abs(r) < 100, f"reward not hours-scaled (too large): {r}"
    print(f"[3] non-empty tick -> {r} (== -10.0 h, hours-scaled) OK")


def check_trace_validation() -> None:
    good = {
        "JobID": 1, "AllocNodes": 1, "ReqNodes": 1,
        "AllocCPUS": 4, "ReqCPUS": 4, "Allgpu": 0, "Reqgpu": 0,
        "Allmem": "4000M", "ReqMem": "4000M", "TimelimitRaw": 10,
        "ElapsedRaw": 100, "Submit": 0,
    }
    Job(good, random_job=True)  # baseline: a valid row constructs fine

    bad = dict(good, AllocNodes=0, ReqNodes=0)
    try:
        Job(bad, random_job=True)
    except ValueError as e:
        print(f"[4] requested_node=0 rejected at load: {e} OK")
        return
    raise AssertionError("requested_node=0 was NOT rejected")


def main() -> int:
    check_reward()
    check_trace_validation()
    print("ALL REWARD/TRACE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
