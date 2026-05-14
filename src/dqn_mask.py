'''
- class MaskableDQN(...) (off-policy pattern)
- __init__(...): epsilon schedule, target update params, replay settings.
- _setup_model(): online Q-net + target Q-net + optimizer.
- ReplayBuffer contract includes next_action_mask.
- predict(..., action_masks=None, deterministic=False):
  - epsilon-random from valid actions only
  - greedy via argmax(masked_q)
- collect_rollouts(...):
  - step env
  - store transition + action_mask + next_action_mask
- train(gradient_steps, batch_size):
  - compute masked target max on next_obs using next_action_mask
  - TD target and Huber/MSE loss
  - optimize online net
  - periodic target net update
  - log train/loss, train/q_mean, train/td_error_mean
- Guardrails:
  - no invalid random actions
  - no invalid argmax actions
  - all-false next_action_mask handling defined (fail fast preferred).
'''
import warnings
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from mask_replay_buffer import MaskableReplayBuffer, MaskableDictReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update, obs_as_tensor, should_collect_more_steps 
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

SelfMaskableDQN = TypeVar("SelfMaskableDQN", bound="MaskableDQN")


class MaskableDQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`dqn_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: str | type[DQNPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        # n_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            # n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.use_masking = True 
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()


    def _setup_model(self) -> None:
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = MaskableDictReplayBuffer
            else:
                self.replay_buffer_class = MaskableReplayBuffer


        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def collect_rollouts(
    self,
    env,
    callback,
    train_freq,
    replay_buffer,
    action_noise=None,
    learning_starts=0,
    log_interval=None,
    use_masking: bool = True,
    ):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking")
        num_collected_steps, num_collected_episodes = 0, 0
        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            action_masks = None
            if use_masking:
                action_masks = self._validate_action_masks(get_action_masks(env), env)
            if self.num_timesteps < learning_starts:
                if use_masking:
                    actions = self._sample_valid_random_actions(action_masks)
                else:
                    actions = np.array([self.action_space.sample() for _ in range(env.num_envs)], dtype=np.int64)
            else:
                actions, _ = self.predict(
                    self._last_obs,
                    deterministic=False,
                    action_masks=action_masks,
                )
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            num_collected_steps += 1
            callback.update_locals(locals())
            if not callback.on_step():
                continue_training = False
                break
            next_action_masks = None
            if use_masking:
                next_action_masks = self._validate_action_masks(get_action_masks(env), env)
            replay_buffer.add(
                obs=self._last_obs,
                next_obs=new_obs,
                action=actions,
                reward=rewards,
                done=dones,
                infos=infos,
                action_masks=action_masks,
                next_action_masks=next_action_masks,
            )
            self._update_info_buffer(infos, dones)
            self._on_step()  # updates target net + exploration schedule
            # 8) episode accounting
            for done in dones:
                if done:
                    num_collected_episodes += 1
                    self._episode_num += 1
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self.dump_logs()    
            self._last_obs = new_obs
        callback.on_rollout_end()
        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        q_means = []
        td_error_means = []

        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = getattr(replay_data, "discounts", None)
            if discounts is None:
                discounts = self.gamma

            next_action_masks = getattr(replay_data, "next_action_masks", None)
            if self.use_masking and next_action_masks is None:
                raise ValueError("Replay sample missing next_action_masks for MaskableDQN")

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                if self.use_masking:
                    next_q_masked = self._mask_q_values(next_q_values, next_action_masks)
                    next_q_values, _ = next_q_masked.max(dim=1)
                else:
                    next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            td_error = target_q_values - current_q_values
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            losses.append(loss.item())
            q_means.append(current_q_values.mean().item())
            td_error_means.append(td_error.abs().mean().item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()


        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/q_mean", np.mean(q_means))
        self.logger.record("train/td_error_mean", np.mean(td_error_means))

    def _validate_action_masks(self, action_masks: np.ndarray, env: VecEnv) -> np.ndarray:
        masks = np.asarray(action_masks)
        if masks.ndim == 1:
            masks = masks.reshape(env.num_envs, -1)
        if masks.shape[0] != env.num_envs:
            raise ValueError(f"Mask batch mismatch: got {masks.shape[0]}, expected {env.num_envs}")
        masks = masks.astype(bool)
        if isinstance(self.action_space, spaces.Discrete):
            if masks.shape[1] != self.action_space.n:
                raise ValueError(f"Discrete mask width mismatch: got {masks.shape[1]}, expected {self.action_space.n}")
        if masks.ndim != 2:
            raise ValueError(f"Expected 2D action mask [n_envs, n_actions], got shape {masks.shape}")
        if not masks.any(axis=1).all():
            bad = np.where(~masks.any(axis=1))[0].tolist()
            raise ValueError(f"All-false action mask for env indices {bad}")
        return masks
    
    def _mask_q_values(self, q_values: th.Tensor, masks: np.ndarray | th.Tensor) -> th.Tensor:
        mask_t = th.as_tensor(masks, device=q_values.device, dtype=th.bool)
        
        # Handle 1D case (single env, non-vectorized)
        if q_values.dim() == 1:
            mask_t = mask_t.squeeze(0) if mask_t.dim() == 2 else mask_t
            if not mask_t.any():
                raise ValueError("All-false action mask")
            return q_values.masked_fill(~mask_t, -1e9)
        
        # 2D case (batched)
        if mask_t.shape != q_values.shape:
            raise ValueError(f"Mask batch mismatch: got {mask_t.shape}, expected {q_values.shape}")
        if not th.all(mask_t.any(dim=1)):
            bad = th.where(~mask_t.any(dim=1))[0].tolist()
            raise ValueError(f"All-false action mask for env indices {bad}")
        return q_values.masked_fill(~mask_t, -1e9)

    def _sample_valid_random_actions(self, masks: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        m = np.asarray(masks, dtype=bool)
        if m.ndim != 2:
            raise ValueError(f"Expected 2D action mask [n_envs, n_actions], got shape {m.shape}")
        actions = np.empty(m.shape[0], dtype=np.int64)
        for i, row in enumerate(m):
            valid = np.flatnonzero(row)
            if valid.size == 0:
                raise ValueError("No valid choices")
            actions[i] = rng.choice(valid)
        return actions

    def _normalize_predict_masks(self, action_masks, n_batch):
        if action_masks is None:
            return None
        m = np.asarray(action_masks, dtype=bool)
        # accept single row for non-vectorized case
        if m.ndim == 1:
            m = m.reshape(1, -1)
        if m.ndim != 2:
            raise ValueError(f"Expected mask shape [batch, n_actions], got {m.shape}")
        if m.shape[0] != n_batch:
            raise ValueError(f"Mask batch mismatch: got {m.shape[0]}, expected {n_batch}")
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("MaskableDQN currently expects Discrete action space")
        if m.shape[1] != self.action_space.n:
            raise ValueError(f"Mask action width mismatch: got {m.shape[1]}, expected {self.action_space.n}")
        # fail fast on all-false rows
        valid_per_row = m.any(axis=1)
        if not valid_per_row.all():
            bad = np.where(~valid_per_row)[0].tolist()
            raise ValueError(f"All-false action mask for batch rows {bad}")
        return m

    def predict( # type: ignore[override]
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        action_masks: np.ndarray | None = None, 
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # Normalise observation to numpy
        if isinstance(observation, dict):
            observation = {k: np.asarray(v) for k, v in observation.items()}
        else:
            observation = np.asarray(observation)

        is_vectorized = self.policy.is_vectorized_observation(observation)
        if isinstance(observation, dict):
            n_batch = observation[next(iter(observation.keys()))].shape[0] if is_vectorized else 1
        else:
            n_batch = observation.shape[0] if is_vectorized else 1
        masks = self._normalize_predict_masks(action_masks,n_batch) if action_masks is not None else None

        if not deterministic and np.random.rand() < self.exploration_rate:
            if masks is not None:
                actions = self._sample_valid_random_actions(masks)
                return (actions if is_vectorized else actions[0]), state
            else:
                if n_batch == 1:
                    actions = np.array([self.action_space.sample()], dtype=np.int64)
                else:
                    actions = np.array([self.action_space.sample() for _ in range(n_batch)], dtype=np.int64) 
                return (actions if is_vectorized else actions[0]), state
        if masks is not None:
            with th.no_grad():
                obs_tensor = obs_as_tensor(observation, self.device)
                # Add batch dimension if not vectorized
                if not is_vectorized:
                    if isinstance(obs_tensor, dict):
                        obs_tensor = {k: v.unsqueeze(0) for k, v in obs_tensor.items()}
                    else:
                        obs_tensor = obs_tensor.unsqueeze(0)
                q_values = self.q_net(obs_tensor)
                masked_q = self._mask_q_values(q_values, masks)
                actions = masked_q.argmax(dim=-1).cpu().numpy()
            return (actions if is_vectorized else actions[0]), state
        
        actions, state = self.policy.predict(observation, state, episode_start, deterministic)
        return actions, state

    def learn(
        self: SelfMaskableDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "MaskableDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        use_masking: bool = True,
    ) -> SelfMaskableDQN:
        total_timesteps, callback = self._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )
        self.use_masking=use_masking
        callback.on_training_start(locals(), globals())
        assert self.env is not None
        while self.num_timesteps < total_timesteps:
            # 2) custom rollout call (critical: pass use_masking through)
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                use_masking=use_masking,
            )
            if not rollout.continue_training:
                break
            # 3) training condition (same as off-policy pattern)
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)            
            if self.num_timesteps > self.learning_starts:
                gradient_steps = self.gradient_steps
                if gradient_steps < 0:
                    gradient_steps = rollout.episode_timesteps
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
        callback.on_training_end()
        return self

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
