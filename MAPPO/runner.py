"""
Hanabi MAPPO runner with per-agent turn-based data collection.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

import logging
from datetime import datetime

import csv

try:
    from .config import HanabiConfig, get_config
    from .envwrappers import ChooseDummyVecEnv, ChooseSubprocVecEnv
    from .buffer import RolloutBuffer
    from .utils.util import get_shape_from_obs_space, get_shape_from_act_space
except ImportError:  # pragma: no cover
    from config import HanabiConfig, get_config
    from envwrappers import ChooseDummyVecEnv, ChooseSubprocVecEnv
    from buffer import RolloutBuffer
    from utils.util import get_shape_from_obs_space, get_shape_from_act_space

ROOT = Path(__file__).resolve().parent.parent
HANABI_SRC = ROOT / "third_party" / "hanabi"
HEADER_DIR = HANABI_SRC
LIB_DIR = HANABI_SRC

if str(HANABI_SRC) not in sys.path:
    sys.path.insert(0, str(HANABI_SRC))

try:
    import pyhanabi  # type: ignore
    from Hanabi_Env import HanabiEnv  # type: ignore
except ImportError as exc:  # pragma: no cover
    pyhanabi = None  # type: ignore
    HanabiEnv = None  # type: ignore
    _PYHANABI_IMPORT_ERROR = exc
else:
    _PYHANABI_IMPORT_ERROR = None

def setup_logger(log_dir: Path, name: str = "hanabi") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Avoid adding multiple handlers if re-run in the same process
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()   # also show in console
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to {log_path}")
    return logger

def _tile_agents(x: np.ndarray, n_agents: int):
    # x: (n_envs, ... ) -> (n_envs, n_agents, ... )
    return np.repeat(x[:, None, ...], n_agents, axis=1)

def _ensure_pyhanabi_loaded() -> None:
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
    if n_envs <= 0:
        return None

    def get_env_fn(rank: int):
        _ensure_pyhanabi_loaded()

        def init_env():
            seed = int(time.time()) + rank
            return HanabiEnv(all_args, seed)

        return init_env

    if n_envs == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    env_fns = [get_env_fn(i) for i in range(n_envs)]
    return ChooseSubprocVecEnv(env_fns)


def make_train_env(all_args: HanabiConfig):
    return _build_vec_env(all_args, all_args.n_rollout_threads)


def make_eval_env(all_args: HanabiConfig):
    return _build_vec_env(all_args, all_args.n_eval_rollout_threads)


def main(argv: Optional[Iterable[str]] = None):
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

        self.use_centralized_V = True
        self.true_total_num_steps = 0

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

        obs_shape = tuple(get_shape_from_obs_space(obs_space))
        share_obs_shape = tuple(get_shape_from_obs_space(shared_obs_space))
        action_shape = (get_shape_from_act_space(act_space),)

        self.act_dim = act_space.n

        self.buffer = RolloutBuffer(
            T=self.episode_length,
            n_envs=self.n_rollout_threads,
            n_agents=self.num_agents,
            obs_shape=obs_shape,
            cent_obs_shape=share_obs_shape,
            action_shape=action_shape,
            gamma=self.all_args.gamma,
            gae_lambda=self.all_args.gae_lambda,
            device=torch.device("cpu"),
            store_available_actions=True,
            act_dim=self.act_dim,
            store_active_masks=True,
        )

        self._init_turn_buffers(obs_shape, share_obs_shape, act_space.n)
        self.episode_scores: list[float] = []
        self.env_live_mask = np.ones(self.n_rollout_threads, dtype=bool)
        self.current_episode_reward = np.zeros(self.n_rollout_threads, dtype=np.float32)
        self.current_episode_length = np.zeros(self.n_rollout_threads, dtype=np.int32)
        self.episode_reward_history: list[float] = []
        self.episode_length_history: list[int] = []
        
        # game log/ metrics
        self.log_dir = Path(getattr(self.all_args, "log_dir", "logs"))
        self.log = setup_logger(self.log_dir, name="hanabi")
        
        self.metrics_path = self.log_dir / "metrics.csv"
        self.metrics_fp = open(self.metrics_path, "a", newline="")
        self.metrics_csv = csv.writer(self.metrics_fp)

        # write header only if file is empty
        if self.metrics_path.stat().st_size == 0:
            self.metrics_csv.writerow([
                "episode","total_env_steps",
                "mean_score_last10",
                "reward_mean_last50","reward_max_last50","reward_min_last50","avg_len_last50",
                "adv_mean","adv_std"
            ])
            self.metrics_fp.flush()

    def _init_turn_buffers(self, obs_shape, share_obs_shape, act_dim):
        self.turn_obs = np.zeros(
            (self.n_rollout_threads, self.num_agents) + obs_shape, dtype=np.float32
        )
        self.turn_share_obs = np.zeros(
            (self.n_rollout_threads, self.num_agents) + share_obs_shape, dtype=np.float32
        )
        self.turn_available_actions = np.zeros(
            (self.n_rollout_threads, self.num_agents, act_dim), dtype=np.float32
        )
        self.turn_values = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        self.turn_actions = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        self.turn_action_log_probs = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        self.turn_masks = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        self.turn_active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        self.turn_bad_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        self.turn_rewards = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        self.turn_rewards_since_last_action = np.zeros_like(self.turn_rewards)

    def warmup(self):
        reset_choose = np.ones(self.n_rollout_threads, dtype=bool)
        obs, share_obs, available_actions = self.envs.reset(reset_choose)
        share_obs = share_obs if self.use_centralized_V else obs

        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()

        # Tile across agents when writing into the buffer
        obs_b  = _tile_agents(obs, self.num_agents)                       # (n_envs, n_agents, obs_dim)
        share_b = _tile_agents(share_obs, self.num_agents)                # (n_envs, n_agents, cent_obs_dim)
        avail_b = _tile_agents(available_actions, self.num_agents)        # (n_envs, n_agents, act_dim)
    
        self.buffer.share_obs[0] = share_b
        self.buffer.obs[0] = obs_b
        self.buffer.value_preds[0] = np.zeros(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        self.buffer.masks[0] = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        if self.buffer.available_actions is not None:
            self.buffer.available_actions[0] = avail_b
        self.env_live_mask[:] = True
        self.current_episode_reward.fill(0.0)
        self.current_episode_length.fill(0)

    def run(self):
        if self.envs is None:
            raise RuntimeError("Training environments are not initialised.")

        self.warmup()

        episodes = (
            self.num_env_steps // self.episode_length // self.n_rollout_threads
        )

        total_env_steps = 0
        finished_episodes = 0

        for episode in range(episodes):
            for step in range(self.episode_length):
                self.reset_choose = np.zeros(self.n_rollout_threads, dtype=bool)
                finished_episodes += self._collect(step)

                self.buffer.chooseinsert(
                    self.turn_share_obs,
                    self.turn_obs,
                    self.turn_actions,
                    self.turn_action_log_probs,
                    self.turn_values,
                    self.turn_rewards,
                    self.turn_masks,
                    active_masks=self.turn_active_masks,
                    available_actions=self.turn_available_actions,
                )

                if self.reset_choose.any():
                    reset_obs, reset_share, reset_available = self.envs.reset(
                        self.reset_choose
                    )
                    reset_share = reset_share if self.use_centralized_V else reset_obs

                    self.use_obs[self.reset_choose] = reset_obs[self.reset_choose]
                    self.use_share_obs[self.reset_choose] = reset_share[self.reset_choose]
                    self.use_available_actions[
                        self.reset_choose
                    ] = reset_available[self.reset_choose]
                    self.env_live_mask[self.reset_choose] = True

                # self.buffer.share_obs[self.buffer.step] = self.use_share_obs.copy()
                # self.buffer.obs[self.buffer.step] = self.use_obs.copy()
                
                share_b = _tile_agents(self.use_share_obs, self.num_agents)
                obs_b   = _tile_agents(self.use_obs, self.num_agents)    
                self.buffer.share_obs[self.buffer.step] = share_b
                self.buffer.obs[self.buffer.step] = obs_b
                if self.buffer.available_actions is not None:
                    avail_b = _tile_agents(self.use_available_actions, self.num_agents)
                    self.buffer.available_actions[self.buffer.step] = avail_b

                total_env_steps += self.n_rollout_threads

            with torch.no_grad():
                next_values = self.agent.get_values(
                    torch.as_tensor(
                        self.use_share_obs, dtype=torch.float32, device=self.device
                    )
                )
            next_values = next_values.detach().cpu().numpy()
            next_values = np.repeat(next_values[:, None, :], self.num_agents, axis=1)
            self.buffer.compute_returns_and_advantages(
                torch.as_tensor(next_values, dtype=torch.float32),self.agent.value_normalizer
            )

            last_train_info = self.agent.train(self.buffer)

            if self.buffer.ptr > 0:
                actions_np = self.buffer.actions[: self.buffer.ptr].astype(np.int64).flatten()
                counts = np.bincount(actions_np, minlength=self.act_dim)
                top_actions = counts.argsort()[-3:][::-1]
                debug_actions = [(int(a), int(counts[a])) for a in top_actions]
                adv_tensor = self.buffer.advs[: self.buffer.ptr].cpu()
                adv_mean = float(adv_tensor.mean().item())
                adv_std = float(adv_tensor.std(unbiased=False).item())
                # print(
                #     f"[HanabiRunner][DEBUG] Action top-3 {debug_actions}, advantages mean={adv_mean:.4f}, std={adv_std:.4f}"
                # )
                self.log.info(f"[HanabiRunner][DEBUG] Action top-3 {debug_actions}, advantages mean={adv_mean:.4f}, std={adv_std:.4f}")

            if self.episode_reward_history:
                recent_rewards = self.episode_reward_history[-50:]
                recent_lengths = self.episode_length_history[-50:] or [0]
                # print(
                #     "[HanabiRunner][DEBUG] Reward stats (last 50): mean={:.2f}, max={:.2f}, min={:.2f}; avg length={:.2f}".format(
                #         float(np.mean(recent_rewards)),
                #         float(np.max(recent_rewards)),
                #         float(np.min(recent_rewards)),
                #         float(np.mean(recent_lengths)),
                #     )
                # )
                self.log.info(
                    "[HanabiRunner][DEBUG] Reward stats (last 50): mean={:.2f}, max={:.2f}, min={:.2f}; avg length={:.2f}".format(
                        float(np.mean(recent_rewards)),
                        float(np.max(recent_rewards)),
                        float(np.min(recent_rewards)),
                        float(np.mean(recent_lengths)),
                    )
                )

            # === Append metrics row ===
            # mean score over last 10 finished episodes (authoritative 0–5 for 1-color)
            mean_score = float(np.mean(self.episode_scores[-10:])) if self.episode_scores else 0.0

            # advantage stats
            if self.buffer.ptr > 0:
                adv_tensor = self.buffer.advs[: self.buffer.ptr].cpu()
                adv_mean = float(adv_tensor.mean().item())
                adv_std  = float(adv_tensor.std(unbiased=False).item())
            else:
                adv_mean = adv_std = 0.0

            # reward/length stats (these are your debug aggregates; may be agent-summed)
            if self.episode_reward_history:
                recent_rewards = self.episode_reward_history[-50:]
                recent_lengths = self.episode_length_history[-50:] or [0]
                r_mean = float(np.mean(recent_rewards))
                r_max  = float(np.max(recent_rewards))
                r_min  = float(np.min(recent_rewards))
                l_mean = float(np.mean(recent_lengths))
            else:
                r_mean = r_max = r_min = l_mean = 0.0

            # write a CSV row (episode-level)
            self.metrics_csv.writerow([
                int(episode), int(total_env_steps),
                mean_score,
                r_mean, r_max, r_min, l_mean,
                adv_mean, adv_std
            ])
            self.metrics_fp.flush()
            # === end append metrics ===
            self.buffer.after_update()

            if episode % self.log_interval == 0 and self.episode_scores:
                mean_score = float(np.mean(self.episode_scores[-10:]))
                # print(
                #     f"[HanabiRunner] Episode {episode} mean score (last 10): {mean_score:.2f}"
                # )
                self.log.info(
                    f"[HanabiRunner] Episode {episode} mean score (last 10): {mean_score:.2f}"
                )

        # print(
        #     f"[HanabiRunner] Completed {total_env_steps} environment steps across {finished_episodes} episodes."
        # )
        self.log.info(
            f"[HanabiRunner] Completed {total_env_steps} environment steps across {finished_episodes} episodes."
        )
        
        self.metrics_fp.close()
        

    def _collect(self, step: int) -> None:
        self.turn_obs.fill(0.0)
        self.turn_share_obs.fill(0.0)
        self.turn_available_actions.fill(0.0)
        self.turn_values.fill(0.0)
        self.turn_actions.fill(0.0)
        self.turn_action_log_probs.fill(0.0)
        self.turn_masks.fill(0.0)
        self.turn_active_masks.fill(0.0)
        self.turn_rewards.fill(0.0)

        finished_this_step = 0

        for current_agent_id in range(self.num_agents):
            env_actions = np.ones((self.n_rollout_threads, 1), dtype=np.float32) * (-1.0)
            choose = self.env_live_mask & np.any(self.use_available_actions == 1, axis=1)

            if not np.any(choose):
                self.reset_choose[:] = True
                break

            self.agent.prep_rollout()
            values, actions, action_log_probs = self.agent.get_actions(
                torch.as_tensor(
                    self.use_share_obs[choose],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(
                    self.use_obs[choose], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    self.use_available_actions[choose],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )

            values_np = values.detach().cpu().numpy().reshape(-1, 1)
            actions_np = actions.detach().cpu().numpy().reshape(-1)
            action_log_probs_np = action_log_probs.detach().cpu().numpy().reshape(-1, 1)

            env_actions[choose, 0] = actions_np

            self.turn_obs[choose, current_agent_id] = self.use_obs[choose].copy()
            self.turn_share_obs[choose, current_agent_id] = self.use_share_obs[choose].copy()
            self.turn_available_actions[choose, current_agent_id] = self.use_available_actions[choose].copy()
            self.turn_values[choose, current_agent_id, 0] = values_np[:, 0]
            self.turn_actions[choose, current_agent_id, 0] = actions_np
            self.turn_action_log_probs[choose, current_agent_id, 0] = action_log_probs_np[:, 0]
            self.turn_masks[choose, current_agent_id, 0] = 1.0
            self.turn_active_masks[choose, current_agent_id, 0] = 1.0

            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(env_actions.astype(np.int32))
            share_obs = share_obs if self.use_centralized_V else obs

            rewards = np.asarray(rewards, dtype=np.float32)
            env_reward = rewards.reshape(self.n_rollout_threads, -1).sum(axis=1)
            self.current_episode_reward[choose] += env_reward[choose]
            self.current_episode_length[choose] += 1
            self.turn_rewards[choose, current_agent_id] = (
                self.turn_rewards_since_last_action[choose, current_agent_id].copy()
            )
            self.turn_rewards_since_last_action[choose, current_agent_id] = 0.0
            self.turn_rewards_since_last_action[choose] += rewards[choose]

            dones = np.asarray(dones, dtype=bool)
            if dones.any():
                self.reset_choose[dones] = True
                self.turn_masks[dones] = 0.0
                self.turn_active_masks[dones] = 0.0
                self.turn_active_masks[dones, current_agent_id, 0] = 1.0
                self.env_live_mask[dones] = False

                left_agent_id = current_agent_id + 1
                if left_agent_id < self.num_agents:
                    self.turn_active_masks[dones, left_agent_id:, 0] = 0.0
                    self.turn_rewards[dones, left_agent_id:] = (
                        self.turn_rewards_since_last_action[dones, left_agent_id:]
                    )
                    self.turn_rewards_since_last_action[dones, left_agent_id:] = 0.0
                    self.turn_values[dones, left_agent_id:] = 0.0
                    self.turn_obs[dones, left_agent_id:] = 0.0
                    self.turn_share_obs[dones, left_agent_id:] = 0.0
                    self.turn_available_actions[dones, left_agent_id:] = 0.0

            info_list = (
                infos.tolist()
                if isinstance(infos, np.ndarray)
                else list(infos)
                if isinstance(infos, (list, tuple))
                else [infos] * self.n_rollout_threads
            )
            for env_idx, done_flag in enumerate(dones):
                if done_flag:
                    info = (
                        info_list[env_idx]
                        if env_idx < len(info_list)
                        else info_list[-1]
                    )
                    if isinstance(info, dict) and "score" in info:
                        self.episode_scores.append(info["score"])
                    self.episode_reward_history.append(float(self.current_episode_reward[env_idx]))
                    self.episode_length_history.append(int(self.current_episode_length[env_idx]))
                    self.current_episode_reward[env_idx] = 0.0
                    self.current_episode_length[env_idx] = 0
                    finished_this_step += 1

            self.use_obs = obs.copy()
            self.use_share_obs = share_obs.copy()
            self.use_available_actions = available_actions.copy()

        return finished_this_step


if __name__ == "__main__":
    main(sys.argv[1:])
