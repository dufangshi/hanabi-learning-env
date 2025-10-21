"""
Shared rollout buffer for MAPPO Hanabi.
Implements turn-based storage compatible with the original MAPPO runner.
"""
from __future__ import annotations

from typing import Dict, Generator, Optional

import numpy as np
import torch


class RolloutBuffer:
    """Buffer storing `[time, env, agent, …]` samples for PPO updates."""

    def __init__(
        self,
        T: int,
        n_envs: int,
        n_agents: int,
        obs_shape,
        cent_obs_shape,
        action_shape,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
        store_available_actions: bool = False,
        act_dim: Optional[int] = None,
        store_active_masks: bool = False,
    ) -> None:
        self.episode_length = T
        self.n_rollout_threads = n_envs
        self.num_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        obs_shape = tuple(obs_shape)
        cent_obs_shape = tuple(cent_obs_shape)
        action_shape = (
            tuple(action_shape)
            if isinstance(action_shape, (list, tuple))
            else (action_shape,)
        )

        self.share_obs = np.zeros(
            (T + 1, n_envs, n_agents) + cent_obs_shape, dtype=np.float32
        )
        self.obs = np.zeros((T + 1, n_envs, n_agents) + obs_shape, dtype=np.float32)

        self.value_preds = np.zeros((T + 1, n_envs, n_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros((T, n_envs, n_agents, 1), dtype=np.float32)

        self.actions = np.zeros((T, n_envs, n_agents) + action_shape, dtype=np.float32)
        self.action_log_probs = np.zeros((T, n_envs, n_agents, 1), dtype=np.float32)
        self.rewards = np.zeros((T, n_envs, n_agents, 1), dtype=np.float32)

        self.masks = np.ones((T + 1, n_envs, n_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = (
            np.ones((T, n_envs, n_agents, 1), dtype=np.float32)
            if store_active_masks
            else None
        )

        self.available_actions = None
        if store_available_actions:
            assert act_dim is not None, "act_dim required for available_actions"
            self.available_actions = np.ones(
                (T + 1, n_envs, n_agents, act_dim), dtype=np.float32
            )

        self.step = 0
        self.ptr = 0

        self.advs = torch.zeros((T, n_envs, n_agents, 1), dtype=torch.float32, device=device)

    def to(self, device: torch.device) -> "RolloutBuffer":
        self.device = device
        self.advs = self.advs.to(device)
        return self

    def chooseinsert(
        self,
        share_obs,
        obs,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        active_masks=None,
        available_actions=None,
    ) -> None:
        idx = self.step
        self.share_obs[idx] = share_obs.copy()
        self.obs[idx] = obs.copy()
        self.actions[idx] = actions.copy()
        self.action_log_probs[idx] = action_log_probs.copy()
        self.value_preds[idx] = value_preds.copy()
        self.rewards[idx] = rewards.copy()
        self.masks[idx + 1] = masks.copy()
        if self.active_masks is not None and active_masks is not None:
            self.active_masks[idx] = active_masks.copy()
        if self.available_actions is not None and available_actions is not None:
            self.available_actions[idx] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
        self.ptr = min(self.ptr + 1, self.episode_length)

    def chooseafter_update(self) -> None:
        """Copy last timestep data to index 0 for the next rollout."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.value_preds[0] = self.value_preds[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()
        if self.active_masks is not None:
            self.active_masks[0] = 1.0

    def after_update(self) -> None:
        self.chooseafter_update()
        self.step = 0
        self.ptr = 0
        self.advantages.fill(0.0)
        self.advs.zero_()

    def compute_returns_and_advantages(self, next_values: torch.Tensor) -> None:
        if self.ptr == 0:
            return

        next_values_np = next_values.detach().cpu().numpy()
        if next_values_np.shape[-1] != 1:
            next_values_np = next_values_np[..., None]
        self.value_preds[-1] = next_values_np.copy()

        gae = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        for step in reversed(range(self.ptr)):
            mask = self.masks[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * self.value_preds[step + 1] * mask
                - self.value_preds[step]
            )
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            self.advantages[step] = gae
            self.returns[step] = gae + self.value_preds[step]

        advs_tensor = torch.from_numpy(self.advantages[: self.ptr])
        self.advs.zero_()
        self.advs[: self.ptr] = advs_tensor.to(self.device)

    def feed_forward_generator(
        self, advantages: torch.Tensor, num_mini_batch: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        T = self.ptr
        batch_size = T * self.n_rollout_threads * self.num_agents
        if batch_size == 0:
            return

        def _flatten(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr[:T]).reshape(batch_size, *arr.shape[3:])

        advs = advantages[:T].reshape(batch_size, advantages.shape[-1])
        obs = _flatten(self.obs)
        cent_obs = _flatten(self.share_obs)
        actions = _flatten(self.actions)
        log_probs = _flatten(self.action_log_probs)
        values = _flatten(self.value_preds)[:batch_size]
        returns = _flatten(self.returns)[:batch_size]

        if self.available_actions is not None:
            available = _flatten(self.available_actions)
        else:
            available = None
        active_masks = (
            _flatten(self.active_masks) if self.active_masks is not None else None
        )

        mini_batch_size = batch_size // num_mini_batch
        if mini_batch_size == 0:
            mini_batch_size = batch_size
            num_mini_batch = 1

        permutation = torch.randperm(batch_size)
        for i in range(num_mini_batch):
            start = i * mini_batch_size
            end = batch_size if i == num_mini_batch - 1 else (i + 1) * mini_batch_size
            index = permutation[start:end]

            batch = {
                "obs": obs[index].to(self.device, dtype=torch.float32),
                "cent_obs": cent_obs[index].to(self.device, dtype=torch.float32),
                "actions": actions[index].to(self.device, dtype=torch.long),
                "old_log_probs": log_probs[index].to(self.device, dtype=torch.float32),
                "old_values": values[index].to(self.device, dtype=torch.float32),
                "returns": returns[index].to(self.device, dtype=torch.float32),
                "advs": advs[index].to(self.device, dtype=torch.float32),
            }
            if available is not None:
                batch["available_actions"] = available[index].to(self.device, dtype=torch.float32)
            if active_masks is not None:
                batch["active_masks"] = active_masks[index].to(self.device, dtype=torch.float32)
            yield batch
