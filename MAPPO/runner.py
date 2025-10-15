"""
Entry point for running MAPPO agents in the Hanabi learning environment.

This module provides lightweight helpers to build vectorised Hanabi environments
and a minimal runner that can roll out the current MAPPO agent implementation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

try:  # Allow running as a package or as a script.
    from .config import HanabiConfig, get_config
    from .envwrappers import ChooseDummyVecEnv, ChooseSubprocVecEnv
    from .buffer import RolloutBuffer
except ImportError:  # pragma: no cover - fallback for direct execution
    from config import HanabiConfig, get_config
    from envwrappers import ChooseDummyVecEnv, ChooseSubprocVecEnv
    from buffer import RolloutBuffer


ROOT = Path(__file__).resolve().parent.parent
HANABI_SRC = ROOT / "third_party" / "hanabi"
HEADER_DIR = HANABI_SRC
LIB_DIR = HANABI_SRC

if str(HANABI_SRC) not in sys.path:
    sys.path.insert(0, str(HANABI_SRC))

try:
    import pyhanabi  # type: ignore
    from Hanabi_Env import HanabiEnv  # type: ignore
except ImportError as exc:  # pragma: no cover - loaded dynamically at runtime
    pyhanabi = None  # type: ignore
    HanabiEnv = None  # type: ignore
    _PYHANABI_IMPORT_ERROR = exc
else:
    _PYHANABI_IMPORT_ERROR = None


def _ensure_pyhanabi_loaded() -> None:
    """Load pyhanabi's C definitions and shared library if necessary."""
    if pyhanabi is None or HanabiEnv is None:
        raise ImportError(
            "pyhanabi is not available. Ensure third_party/hanabi is built."
        ) from _PYHANABI_IMPORT_ERROR

    if not pyhanabi.cdef_loaded():
        if not pyhanabi.try_cdef(prefixes=[str(HEADER_DIR)]):
            raise RuntimeError("Failed to load pyhanabi headers via cdef().")

    if not pyhanabi.lib_loaded():
        if not pyhanabi.try_load(prefixes=[str(LIB_DIR), str(HEADER_DIR)]):
            raise RuntimeError("Failed to load the pyhanabi shared library.")


def _build_vec_env(all_args: HanabiConfig, n_envs: int):
    """Create a vectorised Hanabi environment matching the MAPPO runner API."""
    if n_envs <= 0:
        return None

    def get_env_fn(rank: int):
        _ensure_pyhanabi_loaded()

        def init_env():
            # Use a time-based seed offset so that parallel workers are decorrelated.
            seed = int(time.time()) + rank
            return HanabiEnv(all_args, seed)

        return init_env

    if n_envs == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    env_fns = [get_env_fn(i) for i in range(n_envs)]
    return ChooseSubprocVecEnv(env_fns)


def make_train_env(all_args: HanabiConfig):
    """Factory for training environments. Returns a vectorised env wrapper."""
    return _build_vec_env(all_args, all_args.n_rollout_threads)


def make_eval_env(all_args: HanabiConfig):
    """Factory for evaluation environments (if evaluation enabled)."""
    return _build_vec_env(all_args, all_args.n_eval_rollout_threads)


def main(argv: Optional[Iterable[str]] = None):
    """CLI entry-point (e.g. `python -m MAPPO.runner`)."""
    if argv is not None:
        argv = list(argv)
        sys.argv = [sys.argv[0], *argv]

    all_args = get_config()
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runner = HanabiRunner(
        dict(
            all_args=all_args,
            envs=envs,
            eval_envs=eval_envs,
            device=device,
            num_agents=all_args.num_agents,
        )
    )
    runner.run()


class HanabiRunner:
    """Minimal runner that steps Hanabi environments with the MAPPO agent."""

    def __init__(self, config):
        self.all_args: HanabiConfig = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.hanabi_name = self.all_args.hanabi_name
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.episode_length = self.all_args.episode_length
        self.num_env_steps = int(self.all_args.num_env_steps)

        self.save_interval = self.all_args.save_interval
        self.eval_interval = self.all_args.eval_interval
        self.use_eval = self.all_args.use_eval
        self.log_interval = self.all_args.log_interval

        try:
            from .agent import MAPPOAgent
        except ImportError:  # pragma: no cover
            from agent import MAPPOAgent

        obs_space = self.envs.observation_space[0]
        shared_obs_space = self.envs.share_observation_space[0]
        act_space = self.envs.action_space[0]

        self.agent = MAPPOAgent(
            args=self.all_args,
            obs_space=obs_space,
            cent_obs_space=shared_obs_space,
            act_space=act_space,
            device=self.device,
        )

        obs_shape = tuple(obs_space.shape) if hasattr(obs_space, "shape") else tuple(obs_space)
        cent_obs_shape = (
            tuple(shared_obs_space.shape)
            if hasattr(shared_obs_space, "shape")
            else tuple(shared_obs_space)
        )

        self.buffer = RolloutBuffer(
            T=self.episode_length,
            n_envs=self.n_rollout_threads,
            n_agents=1,
            obs_shape=obs_shape,
            cent_obs_shape=cent_obs_shape,
            action_shape=1,
            gamma=self.all_args.gamma,
            gae_lambda=self.all_args.gae_lambda,
            device=torch.device("cpu"),
            store_available_actions=True,
            act_dim=act_space.n,
            store_active_masks=False,
        )
        self.last_score: Optional[float] = None

    def _reset_envs(self, mask: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones(self.n_rollout_threads, dtype=bool)
        obs, share_obs, available_actions = self.envs.reset(mask)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(share_obs, dtype=np.float32),
            np.asarray(available_actions, dtype=np.float32),
        )

    def run(self):
        if self.envs is None:
            raise RuntimeError("Training environments are not initialised.")

        obs, share_obs, available_actions = self._reset_envs()

        total_env_steps = 0
        finished_episodes = 0

        steps_per_update = max(1, self.episode_length)
        num_updates = max(
            1, self.num_env_steps // (steps_per_update * max(1, self.n_rollout_threads))
        )

        print(
            f"[HanabiRunner] Starting rollout for up to {self.num_env_steps} environment steps "
            f"with {self.n_rollout_threads} parallel env(s) over {num_updates} update(s)."
        )

        last_train_info = {}
        for update in range(num_updates):
            steps_collected = 0
            while (
                steps_collected < steps_per_update
                and total_env_steps < self.num_env_steps
            ):
                self.agent.prep_rollout()

                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                share_tensor = torch.as_tensor(share_obs, dtype=torch.float32, device=self.device)
                avail_tensor = torch.as_tensor(available_actions, dtype=torch.float32, device=self.device)

                values, actions, action_log_probs = self.agent.get_actions(
                    share_tensor, obs_tensor, avail_tensor
                )
                actions_np = actions.detach().cpu().numpy()

                obs_next, share_next, rewards, dones, infos, available_next = self.envs.step(actions_np)

                rewards_np = np.asarray(rewards, dtype=np.float32)[:, :1, :]
                dones_np = np.asarray(dones, dtype=np.float32).reshape(
                    self.n_rollout_threads, 1, 1
                )

                self.buffer.insert(
                    obs=torch.as_tensor(obs, dtype=torch.float32).unsqueeze(1),
                    cent_obs=torch.as_tensor(share_obs, dtype=torch.float32).unsqueeze(1),
                    actions=actions.detach().cpu().long().unsqueeze(1),
                    log_probs=action_log_probs.detach().cpu().unsqueeze(1),
                    values=values.detach().cpu().unsqueeze(1),
                    rewards=torch.as_tensor(rewards_np, dtype=torch.float32),
                    dones=torch.as_tensor(dones_np, dtype=torch.float32),
                    available_actions=torch.as_tensor(
                        available_actions, dtype=torch.float32
                    ).unsqueeze(1),
                )

                total_env_steps += self.n_rollout_threads
                steps_collected += 1

                info_list = infos if isinstance(infos, (list, tuple)) else [infos] * self.n_rollout_threads
                for info in info_list:
                    if isinstance(info, dict) and "score" in info:
                        self.last_score = info["score"]

                obs = np.asarray(obs_next, dtype=np.float32)
                share_obs = np.asarray(share_next, dtype=np.float32)
                available_actions = np.asarray(available_next, dtype=np.float32)

                done_mask = np.asarray(dones, dtype=bool).reshape(self.n_rollout_threads)
                if done_mask.any():
                    reset_obs, reset_share, reset_available = self._reset_envs(done_mask)
                    reset_obs = np.asarray(reset_obs, dtype=np.float32)
                    reset_share = np.asarray(reset_share, dtype=np.float32)
                    reset_available = np.asarray(reset_available, dtype=np.float32)
                    obs[done_mask] = reset_obs[done_mask]
                    share_obs[done_mask] = reset_share[done_mask]
                    available_actions[done_mask] = reset_available[done_mask]
                    finished_episodes += int(done_mask.sum())

            if self.buffer.ptr == 0:
                break

            with torch.no_grad():
                next_values = self.agent.get_values(
                    torch.as_tensor(share_obs, dtype=torch.float32, device=self.device)
                )
            self.buffer.compute_returns_and_advantages(
                next_values.detach().cpu().unsqueeze(1)
            )

            last_train_info = self.agent.train(self.buffer)
            self.buffer.after_update()

            if total_env_steps >= self.num_env_steps:
                break

        print(
            f"[HanabiRunner] Completed {total_env_steps} environment steps across {finished_episodes} finished episode(s). "
            f"Last observed score: {self.last_score}."
        )
        if last_train_info:
            print(
                "[HanabiRunner] Training stats: "
                + ", ".join(f"{k}={v:.4f}" for k, v in last_train_info.items())
            )

        return {
            "total_env_steps": total_env_steps,
            "episodes_completed": finished_episodes,
            "last_score": self.last_score,
            "train_info": last_train_info,
        }


if __name__ == "__main__":
    main(sys.argv[1:])
