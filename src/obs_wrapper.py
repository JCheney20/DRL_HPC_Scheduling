"""obs_wrapper.py

Float32 observation wrapper for HPCsim.

HPCsim declares its Dict observation space with ``dtype=float`` (float64). SB3
already downcasts observations to float32 at the network boundary
(``preprocess_obs``; policy weights are float32), so the float64 storage is pure
waste: it doubles every replay/rollout buffer and host<->device transfer while
the network never sees the extra precision. Storing float32 is therefore
behavior-identical at the network boundary (see notes/obs_wrapper.md for the
full argument) and halves DQN's replay buffer and PPO's rollout buffer.

This wraps the env HPCsim returns; it does NOT modify HPCsim.

References:
  - Gymnasium wrappers: https://gymnasium.farama.org/api/wrappers/
  - SB3 preprocess_obs: stable_baselines3.common.preprocessing
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor


class MaskableMonitor(Monitor):
    """SB3 Monitor that also forwards ``action_masks``.

    Monitor is needed for SB3 to emit ``rollout/ep_rew_mean`` / ``ep_len_mean``
    (it injects the per-episode ``info["episode"]`` block that the logger reads);
    without it the console shows only ``time/*``, which is exactly what hid the
    NaN-reward regression. But this gymnasium version does not auto-forward
    attributes through wrappers (see the note in ``Float32Observation`` and
    ``AllocationCommit``), so a stock Monitor would shadow ``action_masks`` and
    break sb3-contrib's ``get_action_masks``/``env_method("action_masks")``.
    Forward it explicitly. Monitor does not alter observations, rewards, or the
    transition, so wrapping with it does not violate the train/eval env-parity
    contract — it is training-only bookkeeping.
    """

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


class Float32Observation(gym.ObservationWrapper):
    """Cast a Dict of float64 Box observations (and the space) to float32."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                key: gym.spaces.Box(
                    low=np.asarray(space.low, dtype=np.float32),
                    high=np.asarray(space.high, dtype=np.float32),
                    shape=space.shape,
                    dtype=np.float32,
                )
                for key, space in env.observation_space.spaces.items()
            }
        )

    def __getattr__(self, name):
        """
        Manually forward missing attributes to the base environment 
        to restore the older Gym API behavior.
        """
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def observation(self, observation: dict) -> dict:
        return {key: np.asarray(value, dtype=np.float32) for key, value in observation.items()}

    def reset(self, *, seed=None, options=None):
        # HPCsim predates the gymnasium API: its reset() accepts only `seed`, no
        # `options`. Gymnasium's default ObservationWrapper.reset forwards
        # `options=` and would raise "HPCsim.reset() got an unexpected keyword
        # argument 'options'". Drop it (HPCsim never used it).
        obs, info = self.env.reset(seed=seed)
        return self.observation(obs), info

    # Explicit forwarding for the maskable variants: sb3-contrib's
    # get_action_masks() calls env.action_masks() (or env_method in a VecEnv).
    # Gymnasium's Wrapper.__getattr__ would forward this too, but the maskable
    # path is critical for 4 of 6 algorithms, so make it explicit.
    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


if __name__ == "__main__":
    # Self-check on a dummy Dict env — no HPCsim/data needed.
    class _DummyDictEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Dict(
                {
                    "cluster": gym.spaces.Box(0.0, 1e6, shape=(4,), dtype=float),
                    "queue": gym.spaces.Box(0.0, 1e6, shape=(3,), dtype=float),
                }
            )
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None):  # mimic HPCsim: old-gym reset, no `options`
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

        def action_masks(self):
            return np.array([True, False])

    base = _DummyDictEnv()
    assert base.observation_space["cluster"].dtype == np.float64  # HPCsim uses float

    wrapped = Float32Observation(base)
    assert wrapped.observation_space["cluster"].dtype == np.float32
    assert wrapped.observation_space["queue"].dtype == np.float32

    obs, _ = wrapped.reset(seed=0)
    assert obs["cluster"].dtype == np.float32 and obs["queue"].dtype == np.float32
    obs, *_ = wrapped.step(0)
    assert obs["cluster"].dtype == np.float32

    # Attribute forwarding (evaluator/utilization in eval) + explicit mask method.
    assert np.array_equal(wrapped.action_masks(), np.array([True, False]))

    print("Float32Observation self-check OK")
