"""
This file shows an example interaction between the MAPPO agent implementation.
This should serve a starting point for building our runner.py module


"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANABI_SRC = ROOT  / "third_party" / "hanabi"
HEADER_DIR = HANABI_SRC
LIB_DIR = HANABI_SRC


if str(HANABI_SRC) not in sys.path:
    sys.path.insert(0, str(HANABI_SRC))

import pyhanabi
from Hanabi_Env import HanabiEnv
import argparse
import os
import socket
import numpy as np
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from envwrappers import ChooseDummyVecEnv, ChooseSubprocVecEnv
from agent import MAPPOAgent


def _ensure_pyhanabi_loaded() -> None:
    """Make sure the cffi definitions and native library are available."""
    if not pyhanabi.cdef_loaded():
        if not pyhanabi.try_cdef(prefixes=[str(HEADER_DIR)]):
            raise RuntimeError("Failed to load pyhanabi.h definitions")

    if not pyhanabi.lib_loaded():
        if not pyhanabi.try_load(prefixes=[str(LIB_DIR), str(HEADER_DIR)]):
            raise RuntimeError("Failed to load libpyhanabi shared library")


def make_train_env(game_config, n_rollout_threads):
    def get_env_fn(rank):
        _ensure_pyhanabi_loaded()
        def init_env():
            env = HanabiEnv(game_config, 42)
            return env
        return init_env
    if n_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


if __name__ == "__main__":
    _ensure_pyhanabi_loaded()
    #hardcoding rather than parsing terminal inputs just for testing
    args = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--hanabi_name", type=str, default="Hanabi-Full-Minimal")
    args.add_argument("--num_agents", type=int, default=2)
    args.add_argument("--use_obs_instead_of_state", action='store_true', default=False, help="Whether to use global state or concatenated obs")
    args.add_argument("--hidden_layer_dim", type=int, default=64) 
    args.add_argument("--use_feature_normalization", type=bool, default=False) 
    args.add_argument("--use_orthogonal", type=int, default=1) 
    args.add_argument("--use_ReLU", type=bool, default=True) 
    args.add_argument("--layer_N", type=int, default=1) 
    args.add_argument("--clip_epsilon", type=float, default=0.2)
    args.add_argument("--use_clipped_value_loss", type=bool, default=True)
    args.add_argument("--use_huber_loss", type=bool, default=True)
    argsconfig = args.parse_args()


    E = make_train_env(argsconfig, 1)
    print(E)
    print(E.envs)
    print(E.envs[0])
    print(E.envs[0].observation_space[0])
    
    obs, share_obs, available_actions = E.envs[0].reset()
    print("Observation:", obs)
    print("Share observation space:", E.envs[0].share_observation_space[0])
    print("Action space:", E.envs[0].action_space[0])
    print("Available actions:", available_actions)


    agent = MAPPOAgent(argsconfig,E.envs[0].observation_space[0], E.envs[0].share_observation_space[0], E.envs[0].action_space[0])
    
    values, actions, action_log_probs = agent.get_actions(share_obs, obs, available_actions)

    print("\nActions:", actions)
    print("Action log probs:", action_log_probs)
    eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions =E.envs[0].step(actions)
    print('Observation:', eval_obs)
    print('Rewards', eval_rewards)
    print('Available actions', eval_available_actions)