"""Simple demo using Hanabi_env to play random moves and get vectorized observations."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HANABI_SRC = ROOT / "third_party" / "hanabi"
HEADER_DIR = HANABI_SRC
LIB_DIR = HANABI_SRC  # Library is in the root hanabi directory

if str(HANABI_SRC) not in sys.path:
    sys.path.insert(0, str(HANABI_SRC))

import pyhanabi
from Hanabi_Env import HanabiEnv


def _ensure_pyhanabi_loaded() -> None:
    """Make sure the cffi definitions and native library are available."""
    if not pyhanabi.cdef_loaded():
        if not pyhanabi.try_cdef(prefixes=[str(HEADER_DIR)]):
            raise RuntimeError("Failed to load pyhanabi.h definitions")

    if not pyhanabi.lib_loaded():
        if not pyhanabi.try_load(prefixes=[str(LIB_DIR), str(HEADER_DIR)]):
            raise RuntimeError("Failed to load libpyhanabi shared library")


def main() -> None:
    _ensure_pyhanabi_loaded()

    # Create a simple args object for HanabiEnv
    class Args:
        hanabi_name = "Hanabi-Full"
        num_agents = 2
        use_obs_instead_of_state = True

    # Create Hanabi environment
    env = HanabiEnv(Args(), seed=0)

    print(f"Vectorized observation shape: {env.vectorized_observation_shape()}")
    print(f"Vectorized share observation shape: {env.vectorized_share_observation_shape()}")
    print()

    # Reset environment
    obs, share_obs, available_actions = env.reset()
    print(f"Initial state - Current player: {env.state.cur_player()}")

    print(f"Observation length: {len(obs)}")
    print(f"Share observation length: {len(share_obs)}")
    print(f"Available actions length: {len(available_actions)}")
    print(f"Available actions: {available_actions}")
    print(f"Observation (first 50 elements): {obs[:50]}")
    print()

    # Take a random action from available actions
    legal_action_indices = [i for i, valid in enumerate(available_actions) if valid == 1]
    if not legal_action_indices:
        raise RuntimeError("No legal moves available")

    action = random.choice(legal_action_indices)
    print(f"Taking random action: {action}")

    # Step environment (HanabiEnv expects action as array/list)
    obs, share_obs, rewards, done, infos, available_actions = env.step([action])

    print(f"\nAfter action - Current player: {env.state.cur_player()}")
    print(f"Rewards: {rewards}, Done: {done}")
    print(f"Observation length: {len(obs)}")
    print(f"Share observation length: {len(share_obs)}")
    print(f"Available actions: {available_actions}")
    print(f"Observation (first 50 elements): {obs[:50]}")


if __name__ == "__main__":
    main()
