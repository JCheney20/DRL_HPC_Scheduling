# Float32 Observation Wrapper

`src/obs_wrapper.py` — `Float32Observation`, a `gymnasium.ObservationWrapper` applied to every
HPCsim env in both training (`train_agents.build_training_env`) and evaluation
(`evaluate_agents.build_env`), for all six algorithms.

## How it works

HPCsim declares its observation as a `gym.spaces.Dict` of `Box(..., dtype=float)` — and `dtype=float`
in NumPy/Gymnasium is **float64** (8 bytes per value). The wrapper does two things:

1. Rebuilds `observation_space` as a Dict of `Box(..., dtype=np.float32)` (casting `low`/`high`,
   keeping shape). **This is the part SB3 reads to size its replay/rollout buffers.**
2. Casts each observation array to float32 in `observation()` before it reaches the agent.

It wraps the env HPCsim *returns*; **HPCsim itself is untouched** (a hard project constraint).

## Why we're doing it

The observation is large: `5*node + 5*max_jobs` (cluster) + `window_size*7+1` (queue) =
**56,090 values**, dominated by the cluster vector (`5*max_jobs = 51,300`). At float64 that is
**438 KB per observation**. Off-policy DQN stores both `obs` and `next_obs` for every transition
(~0.88 MB/transition), so a replay buffer of any useful size reaches hundreds of GB and **OOM-kills**
on the 128 GB nodes. On-policy PPO similarly pays ~18 GB for its rollout buffer. That float64
precision is wasted (see below), so it is free memory to reclaim.

## Direct benefits

- **DQN replay buffer halves**: 438→219 KB/obs, transition 0.88→0.44 MB. This is what lets
  `buffer_size=150,000` fit (~66 GB) inside the 96 GB request — 150k at float64 would need 132 GB
  and OOM.
- **PPO rollout buffer halves**: ~18→9 GB.
- **Less host↔device transfer** every step for all algorithms (smaller obs tensors copied to GPU).

## Why we know it will work (behavior-identical)

Two independent reasons:

1. **SB3 already downcasts to float32 at the network boundary.** Policy weights are float32, and
   SB3's `preprocess_obs` converts observations to float32 before they ever reach the network. So the
   network sees float32 either way — the float64 buffer merely stored precision that was discarded on
   the very next forward pass. Casting at *storage* time instead of *forward* time yields
   **bit-identical** network inputs (float64→float32 truncation is deterministic and independent of
   when it happens).
2. **float32 exactly represents this data's range.** float32 represents all integers exactly up to
   2^24 ≈ 16.7 M. The observation values here — node memory (710000), core/GPU counts, queue times in
   seconds — are all well under that, so no rounding is introduced for the magnitudes involved.

Together: the agent's inputs, gradients, and learned policy are unchanged; only the storage dtype
differs.

## Key details / decisions

- **Applied globally to all six algorithms**, not just DQN (the only one that strictly needs it for
  memory). Rationale: a change that *could* be argued to affect learning is applied uniformly so it
  cannot be seen as skewing the comparison — the same principle as applying action masking
  consistently. Reproducibility parity over a marginal compute saving.
- **It changed the saved `observation_space` dtype**, so previously-trained float64 models no longer
  load against a float32 eval env (SB3's `check_for_correct_spaces` compares dtype). All models were
  therefore **retrained from scratch** under float32 — one clean, uniform sweep.
- **Enabled the DQN buffer bump** 75k→150k (better replay diversity for the most memory-constrained
  algorithm) at no extra RAM.
- **Maskable variants**: the wrapper explicitly forwards `action_masks()` to the base env; Gymnasium's
  wrapper attribute-forwarding also covers the eval metric reads (`env.evaluator`, `env.utilization`).

## Verification

- Standalone self-check in `src/obs_wrapper.py` (`python -m src.obs_wrapper`) — asserts the wrapped
  space and observations are float32 and `action_masks()` resolves, using a dummy Dict env (no HPCsim
  needed).
- End-to-end: `just run_smoke` exercises the wrapper through train → eval (model loads with no
  `check_for_correct_spaces` error) → aggregate → stats.
