"""
TODO: This should combine 
https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/scripts/train/train_hanabi_forward.py
https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/runner/shared/base_runner.py
https://github.com/zoeyuchao/mappo/blob/79f6591882088a0f583f7a4bcba44041141f25f5/onpolicy/runner/shared/hanabi_runner_forward.py

As an example of how to get the interactions going see testinginteraction.py

"""

import sys
import numpy as np
import torch
import time
from utils.util import _t2n, get_shape_from_obs_space, get_shape_from_act_space
from config import HanabiConfig

def make_train_env(all_args):
    raise NotImplementedError  #TODO

def make_eval_env(all_args):
    raise NotImplementedError  #TODO

def main(args):
    # will make a runner and call runner.run()
    raise NotImplementedError  #TODO


class HanabiRunner:
    def __init__(self, config):
        # ----------BASE RUNNER CONFIG----------
        self.all_args: HanabiConfig = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.hanabi_name = self.all_args.hanabi_name
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads

        self.episode_length = self.all_args.episode_length
        self.gamma = self.all_args.gamma
        self.gae_lambda = self.all_args.gae_lambda

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        
        # TODO other args from base_runner.py/Runner.__init__ based on usage

        # ----------HANABI RUNNER CONFIG----------
        self.true_total_num_steps = 0

        # ----------POLICY NETWORK AND TRAINER----------
        from agent import MAPPOAgent
        obs_space = self.envs.observation_space[0]
        shared_obs_space = self.envs.share_observation_space[0]
        act_space = self.envs.action_space[0]

        self.agent = MAPPOAgent(args=self.all_args,
                                 obs_space=obs_space,
                                 cent_obs_space=shared_obs_space,
                                 act_space=act_space,
                                 device=self.device
        )

        # if self.model_dir is not None:
        #     self.restore()  TODO later when we want to save/restore models
        
        # ----------BUFFER----------
        from buffer import RolloutBuffer
        # TODO check this
        self.buffer = RolloutBuffer(T=self.episode_length,
                                    n_envs=self.n_rollout_threads,  #TODO: this is only for training, for eval use n_eval_rollout_threads
                                    n_agent=self.num_agents,
                                    obs_shape=get_shape_from_obs_space(obs_space),
                                    cent_obs_shape=get_shape_from_obs_space(shared_obs_space),
                                    action_shape=get_shape_from_act_space(act_space),
                                    gamma=self.gamma,
                                    gae_lambda=self.gae_lambda,
                                    device=self.device,
                                    store_available_actions=None,  #TODO
                                    act_dim=act_space.n,
                                    store_active_masks=None  #TODO
        ) 

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        self.turn_obs = np.zeros((self.n_rollout_threads,*self.buffer.obs.shape[2:]), dtype=np.float32)
        self.turn_share_obs = np.zeros((self.n_rollout_threads,*self.buffer.share_obs.shape[2:]), dtype=np.float32)
        self.turn_available_actions = np.zeros((self.n_rollout_threads,*self.buffer.available_actions.shape[2:]), dtype=np.float32)
        self.turn_values = np.zeros((self.n_rollout_threads,*self.buffer.value_preds.shape[2:]), dtype=np.float32)
        self.turn_actions = np.zeros((self.n_rollout_threads,*self.buffer.actions.shape[2:]), dtype=np.float32)       
        self.turn_action_log_probs = np.zeros((self.n_rollout_threads,*self.buffer.action_log_probs.shape[2:]), dtype=np.float32)
        self.turn_rnn_states = np.zeros((self.n_rollout_threads,*self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        self.turn_rnn_states_critic = np.zeros_like(self.turn_rnn_states)
        self.turn_masks = np.ones((self.n_rollout_threads,*self.buffer.masks.shape[2:]), dtype=np.float32)
        self.turn_active_masks = np.ones_like(self.turn_masks)
        self.turn_bad_masks = np.ones_like(self.turn_masks)
        self.turn_rewards = np.zeros((self.n_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        self.turn_rewards_since_last_action = np.zeros_like(self.turn_rewards)

        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            # if self.use_linear_lr_decay: TODO self.use_linear_lr_decay always false?
            #     self.agent.policy.lr_decay(episode, episodes)

            self.scores = []
            for step in range(self.episode_length):
                self.reset_choose = np.zeros(self.n_rollout_threads) == 1.0
                # Sample actions
                self.collect(step) 

                if step == 0 and episode > 0:
                    # deal with the data of the last index in buffer
                    self.buffer.share_obs[-1] = self.turn_share_obs.copy()
                    self.buffer.obs[-1] = self.turn_obs.copy()
                    self.buffer.available_actions[-1] = self.turn_available_actions.copy()
                    self.buffer.active_masks[-1] = self.turn_active_masks.copy()

                    # deal with rewards
                    # 1. shift all rewards
                    self.buffer.rewards[0:self.episode_length-1] = self.buffer.rewards[1:]
                    # 2. last step rewards
                    self.buffer.rewards[-1] = self.turn_rewards.copy()

                    # compute return and update network
                    self.compute()
                    train_infos = self.train()

                # insert turn data into buffer
                self.buffer.chooseinsert(self.turn_share_obs,
                                        self.turn_obs,
                                        self.turn_rnn_states,
                                        self.turn_rnn_states_critic,
                                        self.turn_actions,
                                        self.turn_action_log_probs,
                                        self.turn_values,
                                        self.turn_rewards,
                                        self.turn_masks,
                                        self.turn_bad_masks,
                                        self.turn_active_masks,
                                        self.turn_available_actions)
                # env reset
                obs, share_obs, available_actions = self.envs.reset(self.reset_choose)
                # share_obs = share_obs if self.use_centralized_V else obs TODO self.use_centralized_V always true?

                self.use_obs[self.reset_choose] = obs[self.reset_choose]
                self.use_share_obs[self.reset_choose] = share_obs[self.reset_choose]
                self.use_available_actions[self.reset_choose] = available_actions[self.reset_choose]
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     self.save()  TODO later when we want to save/restore models

            # log information
            # if episode % self.log_interval == 0 and episode > 0:
            #     end = time.time()
            #     print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
            #             .format(self.hanabi_name,
            #                     self.algorithm_name,
            #                     self.experiment_name,
            #                     episode,
            #                     episodes,
            #                     total_num_steps,
            #                     self.num_env_steps,
            #                     int(total_num_steps / (end - start))))

            #     average_score = np.mean(self.scores) if len(self.scores) > 0 else 0.0
            #     print("average score is {}.".format(average_score))
            #     if self.use_wandb:
            #         wandb.log({'average_score': average_score}, step = self.true_total_num_steps)
            #     else:
            #         self.writter.add_scalars('average_score', {'average_score': average_score}, self.true_total_num_steps)

            #     train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
                
            #     self.log_train(train_infos, self.true_total_num_steps) TODO later when we want to log training info

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.true_total_num_steps)

    def warmup(self):
        """Collect warmup pre-training data."""
        # reset env
        self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
        obs, share_obs, available_actions = self.envs.reset(self.reset_choose)

        # share_obs = share_obs if self.use_centralized_V else obs  TODO self.use_centralized_V always true?

        # replay buffer
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect rollouts for training."""
        for current_agent_id in range(self.num_agents):
            env_actions = np.ones((self.n_rollout_threads, *self.buffer.actions.shape[3:]), dtype=np.float32)*(-1.0)
            choose = np.any(self.use_available_actions == 1, axis=1)
            if ~np.any(choose):
                self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
                break
            
            self.agent.prep_rollout()
            #TODO continue checking here
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.agent.trainer.policy.get_actions(self.use_share_obs[choose],
                                                self.use_obs[choose],
                                                self.turn_rnn_states[choose, current_agent_id],
                                                self.turn_rnn_states_critic[choose, current_agent_id],
                                                self.turn_masks[choose, current_agent_id],
                                                self.use_available_actions[choose])
            
            self.turn_obs[choose, current_agent_id] = self.use_obs[choose].copy()
            self.turn_share_obs[choose, current_agent_id] = self.use_share_obs[choose].copy()
            self.turn_available_actions[choose, current_agent_id] = self.use_available_actions[choose].copy()
            self.turn_values[choose, current_agent_id] = _t2n(value)
            self.turn_actions[choose, current_agent_id] = _t2n(action)
            env_actions[choose] = _t2n(action)
            self.turn_action_log_probs[choose, current_agent_id] = _t2n(action_log_prob)
            self.turn_rnn_states[choose, current_agent_id] = _t2n(rnn_state)
            self.turn_rnn_states_critic[choose, current_agent_id] = _t2n(rnn_state_critic)

            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(env_actions)
            
            self.true_total_num_steps += (choose==True).sum()
            share_obs = share_obs if self.use_centralized_V else obs

            # truly used value
            self.use_obs = obs.copy()
            self.use_share_obs = share_obs.copy()
            self.use_available_actions = available_actions.copy()

            # rearrange reward
            # reward of step 0 will be thrown away.
            self.turn_rewards[choose, current_agent_id] = self.turn_rewards_since_last_action[choose, current_agent_id].copy()
            self.turn_rewards_since_last_action[choose, current_agent_id] = 0.0
            self.turn_rewards_since_last_action[choose] += rewards[choose]

            # done==True env

            # deal with reset_choose
            self.reset_choose[dones == True] = np.ones((dones == True).sum(), dtype=bool)

            # deal with all agents
            self.use_available_actions[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)
            self.turn_masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)
            self.turn_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            self.turn_rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

            # deal with the current agent
            self.turn_active_masks[dones == True, current_agent_id] = np.ones(((dones == True).sum(), 1), dtype=np.float32)

            # deal with the left agents
            left_agent_id = current_agent_id + 1
            left_agents_num = self.num_agents - left_agent_id
            self.turn_active_masks[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            
            self.turn_rewards[dones == True, left_agent_id:] = self.turn_rewards_since_last_action[dones == True, left_agent_id:]
            self.turn_rewards_since_last_action[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            
            # other variables use what at last time, action will be useless.
            self.turn_values[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
            self.turn_obs[dones == True, left_agent_id:] = 0
            self.turn_share_obs[dones == True, left_agent_id:] = 0

            # done==False env
            # deal with current agent
            self.turn_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)
            self.turn_active_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)

            # done==None
            # pass

            for done, info in zip(dones, infos):
                if done:
                    if 'score' in info.keys():
                        self.scores.append(info['score'])

    # def insert(self, data):
    #     """
    #     Insert data into buffer.
    #     :param data: (Tuple) data to insert into training buffer.
    #     """
    #     raise NotImplementedError  #TODO not needed, use insert from buffer directly?

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        raise NotImplementedError  #TODO

    def train(self):
        """Train policies with data in buffer. """
        raise NotImplementedError  #TODO
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        """Evaluates policy across environments and logs the average episode score for performance tracking."""
        raise NotImplementedError  #TODO
    
    # @torch.no_grad()
    # def eval_100k(self, eval_games=100000):
    #     """Runs large-scale evaluation over 100k games and reports the mean score across all trials."""
    #     raise NotImplementedError  #TODO later when we want to implement evaluation

    # def save(self):
    #     """Save policy's actor and critic networks."""
    #     raise NotImplementedError  #TODO later when we want to save/restore models

    # def restore(self):
    #     """Restore policy's networks from a saved model."""
    #     raise NotImplementedError  #TODO later when we want to save/restore models

    # def log_train(self, train_infos, total_num_steps):
    #     """
    #     Log training info.
    #     :param train_infos: (dict) information about training update.
    #     :param total_num_steps: (int) total number of training env steps.
    #     """
    #     raise NotImplementedError  #TODO later when we want to log training info

    # def log_env(self, env_infos, total_num_steps):
    #     """
    #     Log env info.
    #     :param env_infos: (dict) information about env state.
    #     :param total_num_steps: (int) total number of training env steps.
    #     """
    #     raise NotImplementedError  #TODO not needed for hanabi


if __name__ == "__main__":
    main(sys.argv[1:])