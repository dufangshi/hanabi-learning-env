"""train file guide: https://github.com/zoeyuchao/mappo/blob/main/onpolicy/scripts/train/train_hanabi_forward.py"""

#!/usr/bin/env python
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
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

# Now import
from envwrappers import ChooseDummyVecEnv
from exampleagent import R_MAPPOPolicy
import importlib.util


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
    args = argparse.ArgumentParser(description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--hanabi_name", type=str, default="Hanabi-Full-Minimal")
    args.add_argument("--num_agents", type=int, default=2)
    args.add_argument("--use_obs_instead_of_state", action='store_true', default=False, help="Whether to use global state or concatenated obs")
    args.add_argument("--hidden_size", type=int, default=64) 
    argsconfig = args.parse_args()

    print("-----")
    print(argsconfig.hanabi_name)
    print(argsconfig.num_agents)
    print(f"Hidden size: {argsconfig.hidden_size}")

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
    
    policy = R_MAPPOPolicy(argsconfig, E.envs[0].observation_space, E.envs[0].share_observation_space[0], E.envs[0].action_space[0])
    
    actions, action_log_probs = policy.get_actions(obs, available_actions)
    print("\nActions:", actions)
    print("Action log probs:", action_log_probs)
    eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions =E.envs[0].step(actions)
    print('Observation:', eval_obs)
    print('Rewards', eval_rewards)
    print('Available actions', eval_available_actions)