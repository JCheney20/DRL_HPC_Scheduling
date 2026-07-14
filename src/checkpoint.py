"""checkpoint.py

A thin override of SB3's CheckpointCallback that preserves the project's
existing checkpoint filename convention: trained_model/{name}/selector/{step}.zip

SB3's CheckpointCallback hardcodes {name_prefix}_{steps}_steps.{ext}, which
would break evaluate_agents.py's manifest-driven model loading and the
expected paths documented in docs/smoke_evidence_template.md. Overriding
_checkpoint_path() is the only change needed — _on_step(), save_replay_buffer,
and save_vecnormalize all continue to work unchanged since they call this
method rather than building the path themselves.

Ref: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback
"""

import os

from stable_baselines3.common.callbacks import CheckpointCallback


class SelectorCheckpointCallback(CheckpointCallback):
    """CheckpointCallback writing to {save_path}/{checkpoint_type}{steps}.{ext}.

    Also counts completed episodes across the whole run. ``model._episode_num``
    is only maintained by the custom ``MaskableDQN``; stock on-policy algos
    (MaskablePPO/MaskableA2C) never touch it, so reading it in the metadata
    sidecar reports 0 for every PPO/A2C run. The callback, by contrast, sees the
    per-step ``dones`` for every algorithm, so accumulating them here gives a
    correct, algorithm-agnostic episode count.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.episodes_completed = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.episodes_completed += int(sum(bool(d) for d in dones))
        return super()._on_step()

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        return os.path.join(self.save_path, f"{checkpoint_type}{self.num_timesteps}.{extension}")
