import copy
import numpy as np
import torch
import torch.nn as nn

#citation: code from https://github.com/zoeyuchao/mappo/tree/79f6591882088a0f583f7a4bcba44041141f25f5

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

class NewCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(NewCategorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            if available_actions.dim() == 1:
                available_actions = available_actions.unsqueeze(0)
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, action_space, inputs_dim, use_orthogonal=1, gain=0.01):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False

        action_dim = action_space.n
        self.action_out = NewCategorical(inputs_dim, action_dim)
    
    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x, available_actions)
        actions = action_logits.mode() if deterministic else action_logits.sample() 
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs


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


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()
        print("DEBUG obs_shape:", obs_shape)

        obs_dim = obs_shape[0][0]  # get the observation dimension
        
        # Use hidden_size from args, not hardcoded 64
        self.hidden_size = args.hidden_size
        self.mlp = MLPLayer(obs_dim, self.hidden_size, 1, 1, 1)

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x)
        x = torch.as_tensor(x, dtype=torch.float32)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.mlp(x)
        return x
    

class ActorNetwork(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(ActorNetwork, self).__init__()
        self.hidden_size = args.hidden_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)
        self.act = ACTLayer(action_space, self.hidden_size)
        self.to(device)

    def forward(self, obs, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features, available_actions)

        return actions, action_log_probs
    

class R_MAPPOPolicy:
    """
    MAPPO Policy class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = ActorNetwork(args, self.obs_space, self.act_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

    def get_actions(self, obs, available_actions=None):

        actions, action_log_probs = self.actor(obs, available_actions)
        return actions, action_log_probs 


class R_MAPPO():

    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

if __name__ == "__main__":
    pass