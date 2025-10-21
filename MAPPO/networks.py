import copy
import argparse
import os
import numpy as np
import torch
import torch.nn as nn



#notice no weight decay in the original paper
#notice paper uses huber loss 

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

def init_(m, gain=0.01):
    nn.init.orthogonal_(m.weight, gain=gain)
    nn.init.constant_(m.bias, 0)
    return m


#TODO: we may be able to remove this
def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


#This combines MLPBase and MLPlayer from https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/utils/mlp.py
class MLPLayer(nn.Module):
    """
    Fully-connected MLP backbone with optional input normalization.
    """
    def __init__(self, args, obs_shape):
        super().__init__()
        input_dim = obs_shape[0] #TODO:double check the indexing obs_dim = obs_shape[0][0] 

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        # self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_layer_dim = args.hidden_layer_dim

        # TODO: implement
        # # Optional normalization on input features
        # if use_feature_normalization:
        #     self.feature_norm = nn.LayerNorm(input_dim)

        # Choose activation and init methods
        activation = nn.ReLU() if self._use_ReLU else nn.Tanh()
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain('relu' if self._use_ReLU else 'tanh')

        # First layer
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, self.hidden_layer_dim)),
            activation,
            nn.LayerNorm(self.hidden_layer_dim)
        )

        # Hidden layers (cloned)
        fc_hidden = nn.Sequential(
            init_(nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim)),
            activation,
            nn.LayerNorm(self.hidden_layer_dim)
        )
        self.fc_layers = get_clones(fc_hidden, self._layer_N )

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc_layers[i](x)
        return x
#This is based on ACTLayer https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/utils/act.py    
class ActionLayer(nn.Module):
    """ 
    """
    def __init__(self, action_space, inputs_dim, gain=0.01):
        super(ActionLayer, self).__init__()
        action_dim = action_space.n
        self.linear = init_(nn.Linear(inputs_dim, action_dim))
        self._large_neg = -1e10
    
    def forward(self, x, available_actions=None, deterministic=False):
        logits = self.linear(x)
        logits = self._mask_logits(logits, available_actions)
        dist = FixedCategorical(logits=logits)
        actions = dist.mode() if deterministic else dist.sample()
        action_log_probs = dist.log_probs(actions)
        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        logits = self.linear(x)
        logits = self._mask_logits(logits, available_actions)
        return torch.softmax(logits, dim=-1)

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        logits = self.linear(x)
        logits = self._mask_logits(logits, available_actions)
        dist = FixedCategorical(logits=logits)
        log_probs = dist.log_probs(action)
        entropy = dist.entropy().unsqueeze(-1)
        return log_probs, entropy

    def _mask_logits(self, logits, available_actions):
        if available_actions is None:
            return logits
        if isinstance(available_actions, np.ndarray):
            available_actions = torch.from_numpy(available_actions)
        available_actions = available_actions.to(device=logits.device, dtype=torch.float32)
        if available_actions.dim() == 1:
            available_actions = available_actions.unsqueeze(0)
        if available_actions.shape != logits.shape:
            available_actions = available_actions.expand_as(logits)
        mask = available_actions <= 0
        if mask.any():
            logits = logits.masked_fill(mask, self._large_neg)
        return logits
    
# Should be similar to R_Actor class in https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py
class ActorNetwork(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(ActorNetwork, self).__init__()
        self.hidden_layer_dim = args.hidden_layer_dim #hidden layer dimension, simlar to hidden_size default is 512

        obs_shape = get_shape_from_obs_space(obs_space)

        self.mlp = MLPLayer(args, obs_shape)
        self.act = ActionLayer(action_space, self.hidden_layer_dim)
        self.to(device)

    def forward(self, obs, available_actions=None, deterministic=False):
        if available_actions is not None:
            available_actions = check(available_actions).to(device=next(self.parameters()).device, dtype=torch.float32)

        actor_features = self.mlp(obs)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic=deterministic)

        return actions, action_log_probs
    def evaluate_actions(self, obs, action, available_actions=None, active_masks=None):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        obs = obs.to(device=next(self.parameters()).device, dtype=torch.float32)

        if action is not None and isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(device=next(self.parameters()).device, dtype=torch.long)

        if available_actions is not None:
            available_actions = check(available_actions).to(
                device=next(self.parameters()).device, dtype=torch.float32
            )

        actor_features = self.mlp(obs)
        return self.act.evaluate_actions(actor_features, action, available_actions, active_masks)
# Should be similar to R_Critic class in https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py
class CriticNetwork(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(CriticNetwork, self).__init__()

        self.hidden_layer_dim = args.hidden_layer_dim   
        self._use_popart = False

        # Get the flattened input dimension from the observation space
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        input_dim = cent_obs_shape[0] if isinstance(cent_obs_shape, tuple) else cent_obs_shape
        self.mlp = MLPLayer(args,cent_obs_shape)

        #TODO need Popart implementation to make tis work
        # if self._use_popart:
        #     self.v_out = init_(
        #         PopArt(self.hidden_layer_dim, 1, device=device),
        #         gain=1.0
        #     )
        # else:
        #     self.v_out = init_(
        #         nn.Linear(self.hidden_layer_dim, 1),
        #         gain=1.0
        #     )
        self.v_out = init_(nn.Linear(self.hidden_layer_dim, 1),gain=1.0)
        self.to(device)
    def forward(self, cent_obs):
        """
        """
        critic_features = self.mlp(cent_obs)

        values = self.v_out(critic_features)

        return values


if __name__ == "__main__":
    pass
