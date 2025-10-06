# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a wrapper repository for the Hanabi Learning Environment, which is a research platform for Hanabi experiments. The core Hanabi environment is a C++ library with Python bindings accessed via CFFI. This repository automates the setup and provides a convenient entry point.

## Environment Setup

**Initial setup:**
```bash
# 1. (Linux only) Edit environment.yaml to uncomment gcc_linux-64 and gxx_linux-64
# 2. Create conda environment
conda env create -f environment.yaml
# 3. Activate environment
conda activate hanabi_learning_env
# 4. Run main.py (first run automatically builds C++ extension)
python main.py
```

The setup.py script automatically:
- Clones the official Hanabi Learning Environment from DeepMind's GitHub
- Builds the C++ extension using CMake and Make
- Places it in `third_party/hanabi-learning-environment/`

## Architecture

### Repository Structure

- **Root level**: Wrapper/helper code for easy setup
  - `main.py`: Entry point demonstrating basic usage
  - `setup.py`: Automates cloning and building the C++ Hanabi library
  - `environment.yaml`: Conda environment specification

- **third_party/hanabi-learning-environment/**: The actual Hanabi environment (auto-cloned)
  - `hanabi_learning_environment/pyhanabi.py`: Low-level Python interface to C++ code via CFFI
  - `hanabi_learning_environment/rl_env.py`: High-level RL environment with OpenAI Gym-like API
  - `hanabi_learning_environment/agents/`: Example agents (random, simple, rainbow DQN)
  - `build/`: CMake build output containing compiled `libpyhanabi.so` or `libpyhanabi.dylib`

### Key Components

**C++ Library Loading (pyhanabi.py)**:
- Uses CFFI to bind to the C++ Hanabi implementation
- Must load definitions from `pyhanabi.h` header file
- Must load compiled library (`libpyhanabi.so`/`.dylib`) from build directory
- The `try_cdef()` and `try_load()` functions handle finding these files

**Two API Levels**:
1. **Low-level (pyhanabi.py)**: Direct access to game state, moves, observations
   - `HanabiGame`: Game configuration and parameters
   - `HanabiState`: Current game state with full information
   - `HanabiObservation`: Player's observed view (no own cards)
   - `HanabiMove`: Represents actions (play, discard, reveal)

2. **High-level (rl_env.py)**: OpenAI Gym-style interface for RL
   - `HanabiEnv`: Main environment class with `reset()` and `step()` methods
   - `make()`: Factory function creating preset game variants
   - Returns observations as nested dicts with vectorized encodings

**main.py Pattern**:
- Ensures CFFI definitions and library are loaded before use
- Adds `third_party/hanabi-learning-environment` to sys.path
- Provides paths to header and library directories
- Handles chance events (card deals) between player actions

## Common Commands

```bash
# Run the basic example
python main.py

# Run RL environment example
python third_party/hanabi-learning-environment/examples/rl_env_example.py

# Run low-level game example
python third_party/hanabi-learning-environment/examples/game_example.py

# Rebuild the C++ extension (from third_party/hanabi-learning-environment/build)
cd third_party/hanabi-learning-environment/build
cmake ..
make -j
```

## Important Details

**CFFI Loading**: When working with the Hanabi environment, always ensure the library is loaded before use. Follow the pattern in main.py:
```python
from hanabi_learning_environment import pyhanabi

if not pyhanabi.cdef_loaded():
    if not pyhanabi.try_cdef(prefixes=[str(HEADER_DIR)]):
        raise RuntimeError("Failed to load pyhanabi.h definitions")

if not pyhanabi.lib_loaded():
    if not pyhanabi.try_load(prefixes=[str(LIB_DIR), str(HEADER_DIR)]):
        raise RuntimeError("Failed to load libpyhanabi shared library")
```

**Chance Events**: The Hanabi game includes "chance" moves (card dealing) represented by a special player ID (`CHANCE_PLAYER_ID = -1`). After player actions, always resolve chance events before the next player acts:
```python
while not state.is_terminal() and state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
    state.deal_random_card()
```

**Observation Types**: The game supports three observation types (configured via `observation_type` parameter):
- `MINIMAL`: Basic human-like view without hint memory
- `CARD_KNOWLEDGE`: Includes per-card hint knowledge and simple inferences
- `SEER`: Shows all cards including player's own (for debugging)