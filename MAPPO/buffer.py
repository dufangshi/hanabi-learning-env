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

        shp = (T, n_envs, n_agents)
        self.obs = torch.zeros(shp + tuple(obs_shape), dtype=torch.float32)
        self.cent_obs = torch.zeros(shp + tuple(cent_obs_shape), dtype=torch.float32)
        self.actions = torch.zeros(shp + tuple(action_shape), dtype=torch.long if len(action_shape)==0 else torch.float32)
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
        #TODO

    @torch.no_grad()
    def compute_returns_and_advantages(self, last_values):
        """
        GAE-λ, working backward over time.
        last_values: [N_env, N_agent, 1] critic value at the last next state (after final step)
        """
    #TODO
        

    def after_update(self):
        """Reset write pointer and (optionally) clear tensors for next rollout."""
        #TODO

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
        #TODO