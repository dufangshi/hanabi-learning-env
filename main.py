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

    game = pyhanabi.HanabiGame({"players": 2, "random_start_player": False})
    state = game.new_initial_state()

    _advance_chance(state)

    print("Initial state:")
    print(state)

    legal_moves = state.legal_moves()
    if not legal_moves:
        raise RuntimeError("No legal moves available in the initial state")

    move = random.choice(legal_moves)
    print(f"Randomly selected move: {move}")

    state.apply_move(move)
    _advance_chance(state)

    print("\nState after applying the move:")
    print(state)


if __name__ == "__main__":
    main()
