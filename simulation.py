"""Simple demo using pyhanabi to play a random move."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HANABI_SRC = ROOT / "third_party" / "hanabi-learning-environment"
HEADER_DIR = HANABI_SRC / "hanabi_learning_environment"
LIB_DIR = HANABI_SRC / "build" / "hanabi_learning_environment"

if str(HANABI_SRC) not in sys.path:
    sys.path.insert(0, str(HANABI_SRC))

from hanabi_learning_environment import pyhanabi
from hanabi_learning_environment.pyhanabi import HanabiMove


def _ensure_pyhanabi_loaded() -> None:
    """Make sure the cffi definitions and native library are available."""
    if not pyhanabi.cdef_loaded():
        if not pyhanabi.try_cdef(prefixes=[str(HEADER_DIR)]):
            raise RuntimeError("Failed to load pyhanabi.h definitions")

    if not pyhanabi.lib_loaded():
        if not pyhanabi.try_load(prefixes=[str(LIB_DIR), str(HEADER_DIR)]):
            raise RuntimeError("Failed to load libpyhanabi shared library")


def _advance_chance(state: pyhanabi.HanabiState) -> None:
    """Resolve chance events (card deals) so the next player can act."""
    while not state.is_terminal() and state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()


def main() -> None:
    _ensure_pyhanabi_loaded()

    # 1. Game Initialization
    game = pyhanabi.HanabiGame({
        "players": 2,
        "colors": 5,
        "ranks": 5,
        "hand_size": 5,
        "max_information_tokens": 8,
        "max_life_tokens": 3,
        "random_start_player": False,
        "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE,
    })
    state = game.new_initial_state()
    encoder = pyhanabi.ObservationEncoder(game)

    while not state.is_terminal():
        if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
            _advance_chance(state)
            continue
        print(f"Current game state:\n{state}")
        print(f"\nPlayer {state.cur_player()}'s turn")

        legal_moves = state.legal_moves()
        print(f"Legal moves: {legal_moves}")

        obs = state.observation(state.cur_player())
        obs_encoded = encoder.encode(obs)

        # placeholder for choosing a move with input obs_encoded
        move = random.choice(legal_moves)
        print(f"\nPlayer {state.cur_player()} selected move: {move}")

        state.apply_move(move)

        print("-" * 200)
    
    reward = state.score()
    print(f"GAME OVER! Final score: {reward}")
    print(f"Final state:\n{state}")

    print((f"Cards successfully played: {sum(state.fireworks())}"))

if __name__ == "__main__":
    main()
