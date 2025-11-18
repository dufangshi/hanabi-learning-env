import copy
import argparse
import os
from typing import Dict
import numpy as np
import torch
import torch.nn as nn

try:  # Support running as package/module
    from .networks import ActorNetwork, CriticNetwork
except ImportError:  # pragma: no cover - fallback when executed directly
    from networks import ActorNetwork, CriticNetwork

# #CITAITON: inspired by  https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/utils/valuenorm.py 
class ValueNorm(nn.Module):
    """Maintains running mean and variance for value normalization."""

    def __init__(self, shape, norm_axes=1, beta=0.99999, epsilon=1e-5, per_element=False):
        super().__init__()
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.norm_axes = norm_axes
        self.beta = beta
        self.epsilon = epsilon
        self.per_element = per_element

        self.register_buffer("mean", torch.zeros(self.shape))
        self.register_buffer("mean_sq", torch.zeros(self.shape))
        self.register_buffer("count", torch.tensor(0.0))

    def reset_parameters(self):
        self.mean.zero_()
        self.mean_sq.zero_()
        self.count.zero_()

    def _debiased_stats(self):
        debias = self.count.clamp(min=self.epsilon)
        mean = self.mean / debias
        mean_sq = self.mean_sq / debias
        var = (mean_sq - mean ** 2).clamp(min=1e-2)
        return mean, var

    @torch.no_grad()
    def update(self, x):
        """Update running meaning and variance with a new batch."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.mean.device, dtype=torch.float32)

        axes = tuple(range(self.norm_axes))
        batch_mean = x.mean(dim=axes)
        batch_mean_sq = (x ** 2).mean(dim=axes)

        weight = self.beta ** np.prod(x.shape[:self.norm_axes]) if self.per_element else self.beta
        self.mean.mul_(weight).add_(batch_mean * (1 - weight))
        self.mean_sq.mul_(weight).add_(batch_mean_sq * (1 - weight))
        self.count.mul_(weight).add_(1 - weight)

    def normalize(self, x):
        """Normalize input using stored statistics"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.mean.device, dtype=torch.float32)

        mean, var = self._debiased_stats()
        return (x - mean[(None,) * self.norm_axes]) / (torch.sqrt(var)[(None,) * self.norm_axes])

    def denormalize(self, x):
        """Reverse normalization."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.mean.device, dtype=torch.float32)

        mean, var = self._debiased_stats()
        return x * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]




def check(input):
    """Convert numpy array to tensor if needed"""
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def update_linear_schedule(optimizer, episode, episodes, initial_lr: float):
    """Linearly decay LearningRate to 0 by the last episode."""
    lr = initial_lr * (1 - (episode / max(1, episodes - 1)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr

def _to_tpdv(x, tpdv):
    """
    It converts any NumPy array or Tensor to the right device (CPU/GPU) and
    dtype (float32, etc.), based on a pre-defined config dict called tpdv.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(**tpdv)

# Combines the following two class
#  R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
#  R_MAPPOPolicy https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py

class MAPPOAgent:
    """ Combines R_MAPPO and R_MAPPOPolicy from the guide repo
    """
    #notice policy removed from inputs
    def __init__(self, args,obs_space,cent_obs_space,act_space, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        

        self.clip_epsilon = getattr(args, "clip_epsilon", getattr(args, "clip_param", 0.2))
        self.ppo_epoch = getattr(args, "ppo_epoch", 4)
        self.num_mini_batch = getattr(args, "num_mini_batch", 1)
        self.data_chunk_length = getattr(args, "data_chunk_length", None)
        self.value_loss_coef = getattr(args, "value_loss_coef", 0.5)
        self.entropy_coef = getattr(args, "entropy_coef", 0.01)
        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)
        self.huber_delta = getattr(args, "huber_delta", 1.0)
        self.use_clipped_value_loss = getattr(args, "use_clipped_value_loss", True)
        self.use_huber_loss = getattr(args, "use_huber_loss", True)
        # self._use_recurrent_policy = args.use_recurrent_policy if hasattr(args, "use_recurrent_policy") else False
        # self._use_naive_recurrent = args.use_naive_recurrent_policy if hasattr(args, "use_naive_recurrent_policy") else False
        
        
        self._use_valuenorm = args.use_valuenorm
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_normalizer = None


        #TODO: not sure if we need these
        # self._use_value_active_masks = args.use_value_active_masks
        # self._use_policy_active_masks = args.use_policy_active_masks


        # setup policy within class below
        # note since policy is not a separate class like R_MAPPOPolicy we can't do calls
        # like self.R_MAPPO.get_actions. We instead do something like x = MAPPOAgent(...) then x.get_actions(...) directly
       
        # spaces & nets
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = ActorNetwork(args, self.obs_space, self.act_space, self.device)
        self.critic = CriticNetwork(args, self.share_obs_space, self.device)

        #TODO add to args instead of hardcoding
        self.lr = getattr(args, "lr", 3e-4)
        self.opti_eps = getattr(args, "opti_eps", 1e-5)
        self.weight_decay = getattr(args, "weight_decay", 0.0)
        self.critic_lr = getattr(args, "critic_lr", 3e-4)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
    
    # see guide R_R_MAPPOPolicy  https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
    
    
    # ------------------API used by Runner-----------------------------
    # see guide R_R_MAPPOPolicy  https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py
    @torch.no_grad()
    def get_actions(self, cent_obs, obs, available_actions=None):
        """Returns (values, actions, action_log_prob) for data collection."""
        actions, action_log_prob = self.actor(obs, available_actions)
        values = self.critic(cent_obs)
        return values, actions, action_log_prob
    
    # see guide R_R_MAPPOPolicy  https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py
    @torch.no_grad()
    def get_values(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        values = self.critic(cent_obs)
        return values
    
    #-----------------------Training-----------------
    def _value_loss(self, values, returns, old_values=None):
        if self.use_huber_loss:
            loss_fn = nn.SmoothL1Loss(beta=self.huber_delta, reduction="mean")
        else:
            loss_fn = nn.MSELoss(reduction="mean")

        if self.use_clipped_value_loss and old_values is not None:
            # clip values like PPO clip for critic
            clipped_values = old_values + (values - old_values).clamp(-self.clip_epsilon, self.clip_epsilon)
            loss_unclipped = loss_fn(values, returns)
            loss_clipped = loss_fn(clipped_values, returns)
            value_loss = torch.max(loss_unclipped, loss_clipped).mean() 
        else:
            value_loss = loss_fn(values, returns)

        return value_loss
    
    def _actor_critic_step(self, batch: Dict[str, torch.Tensor], 
                           update_actor: bool = True) -> Dict[str, float]:
        """
        One gradient step on a mini-batch.
        batch contain:
          obs, cent_obs, actions, old_log_probs, advs, returns, old_values,
          optionally: available_actions, active_masks (1/0 per sample)
        Shapes are [B, ...] where B = mini-batch size * (n_agents) if flat.
        """
        obs = _to_tpdv(batch["obs"], self.tpdv)
        cent_obs = _to_tpdv(batch["cent_obs"], self.tpdv)
        actions = _to_tpdv(batch["actions"], self.tpdv)
        old_log_probs = _to_tpdv(batch["old_log_probs"], self.tpdv)
        advs = _to_tpdv(batch["advs"], self.tpdv)
        returns = _to_tpdv(batch["returns"], self.tpdv)
        old_values = _to_tpdv(batch["old_values"], self.tpdv)
        available_actions = _to_tpdv(batch["available_actions"], self.tpdv) if "available_actions" in batch and batch["available_actions"] is not None else None
        active_masks = _to_tpdv(batch["active_masks"], self.tpdv) if "active_masks" in batch and batch["active_masks"] is not None else None

        # Normalize advantages (stable)
        # advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8) #TODO:double normalization

        # ---- Actor ----
        new_log_probs, entropy = None, None
        policy_loss = torch.tensor(0.0, **self.tpdv)
        if update_actor:
            # new_log_probs, entropy = self.actor.evaluate_actions(obs, action, available_actions) TODO
            new_log_probs, entropy = self.actor.evaluate_actions(obs, actions, available_actions)
            approx_kl = (old_log_probs - new_log_probs).mean()
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advs
            unclipped_actor_loss = -torch.min(surr1, surr2)

            if active_masks is not None:
                policy_loss = (unclipped_actor_loss * active_masks).sum() / (active_masks.sum() + 1e-8)
                entropy_loss = -(entropy * active_masks).sum() / (active_masks.sum() + 1e-8)
            else:
                policy_loss = unclipped_actor_loss.mean()
                entropy_loss = entropy.mean() #TODO: changed 

        # ---- Critic ----
        values = self.critic(cent_obs)
        # TODO: add normalization integration
        if self.value_normalizer is not None:
            self.value_normalizer.update(returns.cpu().numpy())
            returns = torch.as_tensor(
                self.value_normalizer.normalize(returns), **self.tpdv
            )
            values = self.value_normalizer.normalize(values)

        value_loss = self._value_loss(values, returns, old_values)

        # ---- Optimize ----
        stats = {}
        if update_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
            (policy_loss + self.entropy_coef * entropy_loss).backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            stats.update({
                "policy_loss": float(policy_loss.detach().cpu()),
                "entropy": float(entropy.mean().detach().cpu()) if entropy is not None else 0.0,
                "kl": float(approx_kl.detach().cpu()),
                "grad_norm_actor": float(actor_grad_norm),
            })

        self.critic_optimizer.zero_grad(set_to_none=True)
        (self.value_loss_coef * value_loss).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        stats.update({
            "value_loss": float(value_loss.detach().cpu()),
            "value_pred_mean": float(values.mean().detach().cpu()),
            "grad_norm_critic": float(critic_grad_norm),
        })
        return stats
    
    # see guide R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
    # def ppo_update(self, samples: Dict[str, torch.Tensor], update_actor: bool = True) -> Dict[str, float]:
    #     """
    #     Run multiple epochs of PPO over mini-batches.
    #     Expect samples to already be flattened to [N, ...] where N = T * n_env * n_agent.
    #     sample keys: obs, cent_obs, actions, old_log_probs, returns, advs, old_values
    #         Optional: available_actions, active_masks
    #     """
    
    
    #  see guide R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
    def train(self, buffer, update_actor=True):
        """
        Use the buffer's feed-forward mini-batch generator.
        """
        # --- normalize advantages (mask-aware if provided) ---
        advs = buffer.advs  # shape [T, N_env, N_agent, 1]
        mean = advs.mean()
        std  = advs.std(unbiased=False)
        advs = (advs - mean) / (std + 1e-8)

        # --- accumulate stats across epochs × minibatches ---
        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "value_pred_mean": 0.0,
            "kl": 0.0,
            "grad_norm_actor": 0.0,
            "grad_norm_critic": 0.0,
        }
        updates = 0

        for _ in range(self.ppo_epoch):
            for sample in buffer.feed_forward_generator(advs, self.num_mini_batch):
                # move tensors to device once here (feed_forward_generator can also emit on-device)
                sample = {k: _to_tpdv(v, self.tpdv) for k, v in sample.items()}
                stats = self._actor_critic_step(sample, update_actor=update_actor)

                train_info["value_loss"]      += stats.get("value_loss", 0.0)
                train_info["policy_loss"]     += stats.get("policy_loss", 0.0)
                train_info["dist_entropy"]    += stats.get("entropy", 0.0)
                train_info["value_pred_mean"] += stats.get("value_pred_mean", 0.0)
                train_info["kl"]              += stats.get("kl", 0.0)
                train_info["grad_norm_actor"] += stats.get("grad_norm_actor", 0.0)
                train_info["grad_norm_critic"]+= stats.get("grad_norm_critic", 0.0)
                updates += 1

        if updates > 0:
            for k in train_info:
                train_info[k] /= updates

        return train_info


    # --------------------Modes-----------------------
    #  see guide R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
    def prep_training(self):
        """
        keep all behaviors needed for training active:
            Dropout layers will randomly drop activations.
            BatchNorm layers will update running statistics.
        So during training, the model behaves stochastically and updates its internal stats.
        """
        self.actor.train()
        self.critic.train()
        self._is_training = True
        
    #  see guide R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
    @torch.no_grad()
    def prep_rollout(self):
        """
        evaluating (doing inference), don’t apply training randomness
            Dropout is disabled (no random masking).
            BatchNorm uses fixed running mean/variance (no updates).
        This makes predictions deterministic and stable for rollout or testing.
        """
        self.actor.eval()
        self.critic.eval()
        self._is_training = False
