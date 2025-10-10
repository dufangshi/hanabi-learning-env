import copy
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from networks import ActorNetwork, CriticNetwork



def check(input):
    """Convert numpy array to tensor if needed"""
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

# Combines the following two class
#  R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
#  R_MAPPOPolicy https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py

class MAPPOAgent:
    """ Combines R_MAPPO and R_MAPPOPolicy from the guide repo
    """
    #notice policy removed from inputs
    def __init__(self, args,obs_space,cent_obs_space,act_space, devices=torch.device("cpu")):
        self.device = devices
        self.tpdv = dict(dtype=torch.float32, device=self.device)
        

        # self.clip_epislon = args.clip_epislon #changed this name from clip_param
        # self.ppo_epoch = args.ppo_epoch #
        # self.num_mini_batch = args.num_mini_batch
        # self.data_chunk_length = args.data_chunk_length 
        # self.value_loss_coef = args.value_loss_coef
        # self.entropy_coef = args.entropy_coef
        # self.max_grad_norm = args.max_grad_norm       
        # self.huber_delta = args.huber_delta
        # # self._use_recurrent_policy = args.use_recurrent_policy
        # # self._use_naive_recurrent = args.use_naive_recurrent_policy
        # self._use_max_grad_norm = args.use_max_grad_norm
        # self._use_clipped_value_loss = args.use_clipped_value_loss
        # self._use_huber_loss = args.use_huber_loss
        # self._use_popart = args.use_popart
        # self._use_valuenorm = args.use_valuenorm

        #TODO: not sure if we need these
        # self._use_value_active_masks = args.use_value_active_masks
        # self._use_policy_active_masks = args.use_policy_active_masks


        # setup policy within class below
        # note since policy is not a separate class like R_MAPPOPolicy we can't do calls
        # like self.R_MAPPO.get_actions. We instead do something like x = MAPPOAgent(...) then x.get_actions(...) directly
       
       
       
        # self.lr = args.lr
        # self.critic_lr = args.critic_lr
        # self.opti_eps = args.opti_eps
        # self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = ActorNetwork(args, self.obs_space, self.act_space, self.device)
        self.critic = CriticNetwork(args, self.share_obs_space, self.device)

        #TODO add to args instead of hardcoding
        self.lr = 1 
        self.opti_eps = 1 
        self.weight_decay = 1
        self.critic_lr = 1

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
        #TODO: implement for training 
        # update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        # update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        pass 
    # see guide R_R_MAPPOPolicy  https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py
    def get_actions(self, cent_obs, obs, available_actions=None):
        actions, action_log_prob = self.actor(obs, available_actions)
        #TODO:get values
        values = self.critic(cent_obs)
        return values, actions, action_log_prob
    # see guide R_R_MAPPOPolicy  https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py
    def get_values(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        values, _ = self.critic(cent_obs)
        return values
    #  see guide R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
    def ppo_update(self, sample, update_actor=True):
        #TODO: implement for training
        pass
    #  see guide R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
    def train(self, buffer, update_actor=True):
         #TODO: implement for training
        pass 
    #  see guide R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py

    def prep_training(self):
         #TODO: implement for training
        pass 
    #  see guide R_MAPPO https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/algorithms/r_mappo/r_mappo.py
    def prep_rollout(self):
         #TODO: implement for training
        pass 
