from typing import Any, NamedTuple
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import DictReplayBuffer 

class MaskableReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    # keep optional for flexibility during rollout wiring
    action_masks: th.Tensor | None
    next_action_masks: th.Tensor | None


class MaskableDictReplayBufferSamples(NamedTuple):
    observations: dict[str, th.Tensor]
    actions: th.Tensor
    next_observations: dict[str, th.Tensor]
    dones: th.Tensor
    rewards: th.Tensor
    action_masks: th.Tensor | None
    next_action_masks: th.Tensor | None

class MaskableReplayBuffer(ReplayBuffer):
    """
    Replay buffer with action mask storage for MaskableDQN.
    Similar pattern to HerReplayBuffer: custom arrays + overridden add/sample path.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        assert isinstance(action_space, spaces.Discrete), "MaskableDQN expects Discrete action space"
        self.n_actions = action_space.n
        # Mirrors HER style: per-transition per-env side arrays
        self.action_masks = np.zeros((self.buffer_size, self.n_envs, self.n_actions), dtype=bool)
        self.next_action_masks = np.zeros((self.buffer_size, self.n_envs, self.n_actions), dtype=bool)
    def add(  # type: ignore[override]
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        action_masks: np.ndarray | None = None,
        next_action_masks: np.ndarray | None = None,
    ) -> None:
        # save index BEFORE parent add() increments pos
        pos = self.pos
        # store main transition first (same as normal ReplayBuffer behavior)
        super().add(obs, next_obs, action, reward, done, infos)
        # optional current mask (for diagnostics)
        if action_masks is not None:
            m = np.asarray(action_masks, dtype=bool)
            if m.ndim == 2 and m.shape == (self.n_envs, self.n_actions):
                self.action_masks[pos] = m
            else:
                raise ValueError(f"action_masks shape must be {(self.n_envs, self.n_actions)}, got {m.shape}")
        # required for masked TD target
        if next_action_masks is None:
            nm = np.ones((self.n_envs, self.n_actions), dtype=bool)
        else:
            nm = np.asarray(next_action_masks, dtype=bool)
        if nm.ndim != 2 or nm.shape != (self.n_envs, self.n_actions):
            raise ValueError(f"next_action_masks shape must be {(self.n_envs, self.n_actions)}, got {nm.shape}")
        self.next_action_masks[pos] = nm
    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableReplayBufferSamples:
        # ReplayBuffer uses random env indices for vectorized envs
        env_indices = np.random.randint(0, self.n_envs, size=(len(batch_inds),))
        
        # --- standard ReplayBuffer extraction logic ---
        obs = self._normalize_obs(self.observations[batch_inds, env_indices, :], env)
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        actions = self.actions[batch_inds, env_indices, :]
        dones = self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])

        rewards = self.rewards[batch_inds, env_indices]
        rewards = self._normalize_reward(rewards.reshape(-1, 1), env)
        
        # --- mask extraction aligned to same (batch_inds, env_indices) ---
        batch_action_masks = self.action_masks[batch_inds, env_indices, :]
        batch_next_action_masks = self.next_action_masks[batch_inds, env_indices, :]

        return MaskableReplayBufferSamples(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(rewards),
            action_masks=self.to_torch(batch_action_masks),
            next_action_masks=self.to_torch(batch_next_action_masks),
        )


class MaskableDictReplayBuffer(DictReplayBuffer):
    """
    Replay buffer with action mask storage for MaskableDQN.
    Similar pattern to HerReplayBuffer: custom arrays + overridden add/sample path.
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        assert isinstance(action_space, spaces.Discrete), "MaskableDQN expects Discrete action space"
        self.n_actions = action_space.n
        # Mirrors HER style: per-transition per-env side arrays
        self.action_masks = np.zeros((self.buffer_size, self.n_envs, self.n_actions), dtype=bool)
        self.next_action_masks = np.zeros((self.buffer_size, self.n_envs, self.n_actions), dtype=bool)
    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        action_masks: np.ndarray | None = None,
        next_action_masks: np.ndarray | None = None,
    ) -> None:
        # save index BEFORE parent add() increments pos
        pos = self.pos
        # store main transition first (same as normal ReplayBuffer behavior)
        super().add(obs, next_obs, action, reward, done, infos)
        # optional current mask (for diagnostics)
        if action_masks is not None:
            m = np.asarray(action_masks, dtype=bool)
            if m.ndim == 2 and m.shape == (self.n_envs, self.n_actions):
                self.action_masks[pos] = m
            else:
                raise ValueError(f"action_masks shape must be {(self.n_envs, self.n_actions)}, got {m.shape}")
        # required for masked TD target
        if next_action_masks is None:
            nm = np.ones((self.n_envs, self.n_actions), dtype=bool)
        else:
            nm = np.asarray(next_action_masks, dtype=bool)
        if nm.ndim != 2 or nm.shape != (self.n_envs, self.n_actions):
            raise ValueError(f"next_action_masks shape must be {(self.n_envs, self.n_actions)}, got {nm.shape}")
        self.next_action_masks[pos] = nm
    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MaskableDictReplayBufferSamples:
        # ReplayBuffer uses random env indices for vectorized envs
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        # --- standard ReplayBuffer extraction logic ---
        
        next_obs = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env,)
        obs = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()},env,)

        assert isinstance(obs, dict)
        assert isinstance(next_obs, dict)

        observations = {key: self.to_torch(val) for key, val in obs.items()}
        next_observations = {key: self.to_torch(val) for key, val in next_obs.items()}

        actions = self.actions[batch_inds, env_indices, :]
        dones = self.to_torch((self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1))
        rewards = self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env))

        # --- mask extraction aligned to same (batch_inds, env_indices) ---
        batch_action_masks = self.action_masks[batch_inds, env_indices, :]
        batch_next_action_masks = self.next_action_masks[batch_inds, env_indices, :]

        return MaskableDictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(actions),
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            action_masks=self.to_torch(batch_action_masks),
            next_action_masks=self.to_torch(batch_next_action_masks),
        )

