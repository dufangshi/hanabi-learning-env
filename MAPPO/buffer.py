"""
buffer must expose a .get_training_samples() that returns a dict with:
          obs, cent_obs, actions, old_log_probs, returns, advs, old_values
        and optionally available_actions, active_masks
"""
import torch

class RolloutBuffer:
    """
    Shared buffer for MAPPO (centralized critic, shared actor), no RNN.
    Shapes use [T, N_env, N_agent, ...] during collection and are flattened to [N] for training.
    """
    def __init__(self, T, n_envs, n_agents, obs_shape, cent_obs_shape, action_shape,
                 gamma=0.99, gae_lambda=0.95, device=torch.device("cpu"),
                 store_available_actions=False, act_dim=None, store_active_masks=False):
        self.T = T
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.gamma = gamma
        self.lam = gae_lambda
        self.device = device
        self.ptr = 0

        obs_shape = tuple(obs_shape) if isinstance(obs_shape, (list, tuple)) else (obs_shape,)
        cent_obs_shape = tuple(cent_obs_shape) if isinstance(cent_obs_shape, (list, tuple)) else (cent_obs_shape,)

        if isinstance(action_shape, (list, tuple)):
            action_shape_tuple = tuple(action_shape)
        elif isinstance(action_shape, int):
            action_shape_tuple = (action_shape,) if action_shape > 0 else ()
        else:
            raise TypeError(f"Unsupported action_shape type: {type(action_shape)}")

        self._discrete_actions = len(action_shape_tuple) == 0 or action_shape_tuple == (1,)
        if self._discrete_actions:
            action_shape_tuple = () if len(action_shape_tuple) == 0 else (1,)

        shp = (T, n_envs, n_agents)
        self.obs = torch.zeros(shp + obs_shape, dtype=torch.float32)
        self.cent_obs = torch.zeros(shp + cent_obs_shape, dtype=torch.float32)
        action_dtype = torch.long if self._discrete_actions else torch.float32
        self.actions = torch.zeros(shp + action_shape_tuple, dtype=action_dtype)
        self.log_probs = torch.zeros(shp + (1,), dtype=torch.float32)
        self.values = torch.zeros(shp + (1,), dtype=torch.float32)
        self.rewards = torch.zeros(shp + (1,), dtype=torch.float32)
        self.dones = torch.zeros(shp + (1,), dtype=torch.float32)  # 1.0 if done at t+1

        self.returns = torch.zeros_like(self.values)
        self.advs = torch.zeros_like(self.values)

        self.available_actions = None
        if store_available_actions:
            assert act_dim is not None, "act_dim required to store available_actions"
            self.available_actions = torch.ones(shp + (act_dim,), dtype=torch.float32)

        self.active_masks = None
        if store_active_masks:
            self.active_masks = torch.ones(shp + (1,), dtype=torch.float32)

        # next observations for bootstrapping if needed (optional)
        self.next_cent_obs = torch.zeros((n_envs, n_agents) + tuple(cent_obs_shape), dtype=torch.float32)

    def to(self, device):
        self.device = device
        for name, tensor in self.__dict__.items():
            if isinstance(tensor, torch.Tensor):
                setattr(self, name, tensor.to(device))
        return self

    @torch.no_grad()
    def insert(self, obs, cent_obs, actions, log_probs, values, rewards, dones,
               available_actions=None, active_masks=None):
        """
        Insert one time step across all envs and agents.
        All inputs should be torch tensors on CPU (we move them in get_training_samples()) or already on device.
        Shapes:
          obs:            [N_env, N_agent, *obs_shape]
          cent_obs:       [N_env, N_agent, *cent_obs_shape]
          actions:        [N_env, N_agent, *action_shape] (discrete: [..,1] or scalar long)
          log_probs:      [N_env, N_agent, 1]
          values:         [N_env, N_agent, 1]
          rewards:        [N_env, N_agent, 1]
          dones:          [N_env, N_agent, 1]  (1.0 if episode ended after this step)
        """
        if self.ptr >= self.T:
            raise IndexError("RolloutBuffer is full. Call after_update() before inserting again.")

        step = self.ptr

        def _copy(target, source, dtype=None):
            tensor = torch.as_tensor(
                source,
                dtype=dtype or target.dtype,
                device=target.device,
            )
            tensor = tensor.view_as(target[step])
            target[step].copy_(tensor)

        _copy(self.obs, obs)
        _copy(self.cent_obs, cent_obs)
        action_dtype = torch.long if self._discrete_actions else self.actions.dtype
        _copy(self.actions, actions, dtype=action_dtype)
        _copy(self.log_probs, log_probs)
        _copy(self.values, values)
        _copy(self.rewards, rewards)
        _copy(self.dones, dones)

        if self.available_actions is not None and available_actions is not None:
            _copy(self.available_actions, available_actions)
        if self.active_masks is not None and active_masks is not None:
            _copy(self.active_masks, active_masks)

        self.ptr += 1

    @torch.no_grad()
    def compute_returns_and_advantages(self, last_values):
        """
        GAE-λ, working backward over time.
        last_values: [N_env, N_agent, 1] critic value at the last next state (after final step)
        """
        if self.ptr == 0:
            return

        last_values = torch.as_tensor(
            last_values,
            dtype=self.values.dtype,
            device=self.values.device,
        )
        if last_values.dim() == 2:  # allow [N_env, N_agent]
            last_values = last_values.unsqueeze(-1)

        adv = torch.zeros(
            (self.n_envs, self.n_agents, 1),
            dtype=self.values.dtype,
            device=self.values.device,
        )

        for step in reversed(range(self.ptr)):
            next_value = last_values if step == self.ptr - 1 else self.values[step + 1]
            mask = 1.0 - self.dones[step]
            delta = self.rewards[step] + self.gamma * next_value * mask - self.values[step]
            adv = delta + self.gamma * self.lam * mask * adv
            self.advs[step] = adv

        self.returns[:self.ptr] = self.advs[:self.ptr] + self.values[:self.ptr]
        

    def after_update(self):
        """Reset write pointer and (optionally) clear tensors for next rollout."""
        self.ptr = 0
        for name, tensor in self.__dict__.items():
            if isinstance(tensor, torch.Tensor):
                tensor.zero_()

    def feed_forward_generator(self, advantages: torch.Tensor, num_mini_batch: int):
        """
        Yield mini-batches with the fields PPO needs.
        Inputs:
          - advantages: [T, N_env, N_agent, 1] tensor (already computed/normalized if you want)
          - num_mini_batch: number of mini-batches per epoch
        Yields dicts with keys:
          obs, cent_obs, actions, old_log_probs, old_values, returns, advs
          (and available_actions, active_masks if stored)
        Shapes yielded are [batch, ...].
        """
        T_filled = self.ptr if self.ptr > 0 else self.T
        batch_size = T_filled * self.n_envs * self.n_agents
        if batch_size == 0:
            return

        def _flatten(tensor):
            return tensor[:T_filled].reshape(batch_size, *tensor.shape[3:])

        obs = _flatten(self.obs)
        cent_obs = _flatten(self.cent_obs)
        actions = _flatten(self.actions)
        old_log_probs = _flatten(self.log_probs)
        old_values = _flatten(self.values)
        returns = _flatten(self.returns)
        advs = advantages[:T_filled].reshape(batch_size, advantages.shape[-1])

        available_actions = (
            _flatten(self.available_actions) if self.available_actions is not None else None
        )
        active_masks = (
            _flatten(self.active_masks) if self.active_masks is not None else None
        )

        mini_batch_size = batch_size // num_mini_batch
        if mini_batch_size == 0:
            mini_batch_size = batch_size
            num_mini_batch = 1

        permutation = torch.randperm(batch_size, device=obs.device)
        for i in range(num_mini_batch):
            start = i * mini_batch_size
            end = batch_size if i == num_mini_batch - 1 else (i + 1) * mini_batch_size
            index = permutation[start:end]

            batch = {
                "obs": obs[index],
                "cent_obs": cent_obs[index],
                "actions": actions[index],
                "old_log_probs": old_log_probs[index],
                "old_values": old_values[index],
                "returns": returns[index],
                "advs": advs[index],
            }

            if available_actions is not None:
                batch["available_actions"] = available_actions[index]
            if active_masks is not None:
                batch["active_masks"] = active_masks[index]

            yield batch
