"""
Hanabi MAPPO runner with per-agent turn-based data collection.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
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

        # Setup save directories
        self._setup_directories()

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

        # Load pretrained model if specified
        if self.all_args.model_dir is not None:
            self.restore(self.all_args.model_dir)

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
        self.current_episode_positive_reward = np.zeros(self.n_rollout_threads, dtype=np.float32)
        self.current_episode_length = np.zeros(self.n_rollout_threads, dtype=np.int32)
        self.episode_reward_history: list[float] = []
        self.episode_positive_reward_history: list[float] = []
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
                "iteration","total_env_steps",
                "mean_score_last10",
                "reward_mean_last50","reward_max_last50","reward_min_last50","avg_len_last50",
                "adv_mean","adv_std"
            ])
            self.metrics_fp.flush()

    def _setup_directories(self):
        """Create directory structure for saving models and logs."""
        # Create run directory: results/Hanabi/{hanabi_name}/{algorithm_name}/{experiment_name}/run{N}/
        base_dir = Path(self.all_args.result_dir) / "Hanabi" / self.hanabi_name / self.algorithm_name / self.experiment_name

        if not base_dir.exists():
            run_id = 1
        else:
            # Find existing run directories
            existing_runs = [int(d.name.replace("run", "")) for d in base_dir.iterdir()
                           if d.is_dir() and d.name.startswith("run") and d.name.replace("run", "").isdigit()]
            run_id = max(existing_runs) + 1 if existing_runs else 1

        self.run_dir = base_dir / f"run{run_id}"
        self.save_dir = self.run_dir / "models"
        self.log_dir = self.run_dir / "logs"

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"[HanabiRunner] Saving models to: {self.save_dir}")
        print(f"[HanabiRunner] Saving logs to: {self.log_dir}")

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

    def save(self, iteration=0):
        """Save actor and critic networks along with training state and config."""
        save_path_actor = self.save_dir / f"actor_iter{iteration}.pt"
        save_path_critic = self.save_dir / f"critic_iter{iteration}.pt"
        save_path_checkpoint = self.save_dir / f"checkpoint_iter{iteration}.pt"

        # Save actor and critic networks
        torch.save(self.agent.actor.state_dict(), str(save_path_actor))
        torch.save(self.agent.critic.state_dict(), str(save_path_critic))

        # Save training state
        checkpoint = {
            'iteration': iteration,
            'episode_scores': self.episode_scores,
            'episode_reward_history': self.episode_reward_history,
            'episode_length_history': self.episode_length_history,
        }
        torch.save(checkpoint, str(save_path_checkpoint))

        # Save training configuration (once per run, not per iteration)
        config_path = self.save_dir / "config.json"
        if not config_path.exists():
            import json
            from dataclasses import asdict
            config_dict = asdict(self.all_args)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"[HanabiRunner] Saved config to {config_path}")

        print(f"[HanabiRunner] Saved model at iteration {iteration} to {self.save_dir}")

        # Also save as latest for easy resuming
        torch.save(self.agent.actor.state_dict(), str(self.save_dir / "actor_latest.pt"))
        torch.save(self.agent.critic.state_dict(), str(self.save_dir / "critic_latest.pt"))
        torch.save(checkpoint, str(self.save_dir / "checkpoint_latest.pt"))

    def restore(self, model_dir):
        """Restore actor and critic networks from saved model."""
        model_path = Path(model_dir)

        # Try to load latest models first, then fall back to specific iteration
        actor_path = model_path / "actor_latest.pt"
        critic_path = model_path / "critic_latest.pt"
        checkpoint_path = model_path / "checkpoint_latest.pt"

        if not actor_path.exists():
            # Try to find the latest iteration checkpoint (support both old 'ep' and new 'iter' naming)
            actor_files = list(model_path.glob("actor_iter*.pt")) + list(model_path.glob("actor_ep*.pt"))
            if not actor_files:
                raise FileNotFoundError(f"No actor model found in {model_dir}")
            # Sort by iteration/episode number
            def extract_num(p):
                stem = p.stem
                if "_iter" in stem:
                    return int(stem.split("_iter")[-1])
                else:
                    return int(stem.split("_ep")[-1])
            actor_path = sorted(actor_files, key=extract_num)[-1]
            iter_num = extract_num(actor_path)

            # Check which naming convention to use
            if "_iter" in actor_path.stem:
                critic_path = model_path / f"critic_iter{iter_num}.pt"
                checkpoint_path = model_path / f"checkpoint_iter{iter_num}.pt"
            else:
                critic_path = model_path / f"critic_ep{iter_num}.pt"
                checkpoint_path = model_path / f"checkpoint_ep{iter_num}.pt"

        print(f"[HanabiRunner] Loading model from {actor_path}")

        # Load networks
        self.agent.actor.load_state_dict(torch.load(str(actor_path), map_location=self.device))
        self.agent.critic.load_state_dict(torch.load(str(critic_path), map_location=self.device))

        # Load training state if available
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            self.episode_scores = checkpoint.get('episode_scores', [])
            self.episode_reward_history = checkpoint.get('episode_reward_history', [])
            self.episode_length_history = checkpoint.get('episode_length_history', [])
            # Support both old 'episode' and new 'iteration' keys
            iter_or_ep = checkpoint.get('iteration', checkpoint.get('episode', 0))
            print(f"[HanabiRunner] Resumed from iteration {iter_or_ep}")
        else:
            print("[HanabiRunner] No checkpoint file found, starting fresh training state")

    def evaluate(self, num_eval_episodes: int = 500, current_iteration: int = 0) -> dict:
        """
        Run evaluation episodes and collect statistics.

        Args:
            num_eval_episodes: Number of episodes to run for evaluation
            current_iteration: Current training iteration number

        Returns:
            Dictionary with evaluation statistics (mean, std, max, min scores, distribution)
        """
        if self.eval_envs is None:
            print("[HanabiRunner][EVAL] No evaluation environments available, skipping evaluation")
            return {}

        eval_start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[HanabiRunner][EVAL] Starting evaluation: {num_eval_episodes} episodes at iteration {current_iteration}")

        self.agent.prep_rollout()  # Set networks to eval mode

        eval_episode_scores = []
        eval_episode_positive_rewards = []
        episodes_completed = 0

        n_eval_envs = self.eval_envs.num_envs

        # Initialize environments - reset all environments
        reset_choose = np.ones(n_eval_envs, dtype=bool)
        obs, share_obs, available_actions = self.eval_envs.reset(reset_choose)
        eval_obs = obs.copy()
        eval_share_obs = share_obs if self.use_centralized_V else obs.copy()
        eval_available_actions = available_actions.copy()

        # Track which environments should continue collecting episodes
        envs_active = np.ones(n_eval_envs, dtype=bool)
        # Track cumulative positive reward for each active episode
        episode_positive_rewards = np.zeros(n_eval_envs, dtype=np.float32)

        max_iterations = num_eval_episodes * 100  # Safety limit to avoid infinite loops
        iteration_count = 0

        while episodes_completed < num_eval_episodes and iteration_count < max_iterations:
            iteration_count += 1

            # Check for environments that have no available actions BEFORE trying to step
            # This prevents trying to act on environments that are already done
            no_actions_before_step = ~np.any(eval_available_actions == 1, axis=1)

            # Process environments with no actions as done BEFORE stepping
            envs_to_reset = np.zeros(n_eval_envs, dtype=bool)
            for env_idx in range(n_eval_envs):
                if no_actions_before_step[env_idx] and envs_active[env_idx]:
                    # Mark for reset without collecting score (should have been collected last iter)
                    if episodes_completed < num_eval_episodes:
                        envs_to_reset[env_idx] = True
                        envs_active[env_idx] = False

            # Reset environments that had no actions
            if np.any(envs_to_reset):
                obs_reset, share_obs_reset, available_reset = self.eval_envs.reset(envs_to_reset)
                eval_obs[envs_to_reset] = obs_reset[envs_to_reset]
                eval_share_obs[envs_to_reset] = share_obs_reset[envs_to_reset]
                eval_available_actions[envs_to_reset] = available_reset[envs_to_reset]
                envs_active[envs_to_reset] = True

            # Find environments that have available actions and are still active
            env_has_actions = envs_active & (np.any(eval_available_actions == 1, axis=1))

            if not np.any(env_has_actions):
                # No environments ready - all must be done
                break

            # Get actions for active environments
            with torch.no_grad():
                _, actions, _ = self.agent.get_actions(
                    torch.as_tensor(eval_share_obs[env_has_actions], dtype=torch.float32, device=self.device),
                    torch.as_tensor(eval_obs[env_has_actions], dtype=torch.float32, device=self.device),
                    torch.as_tensor(eval_available_actions[env_has_actions], dtype=torch.float32, device=self.device),
                )

            actions = actions.detach().cpu().numpy()

            # Create full action array (-1 for inactive envs)
            eval_actions = np.ones((n_eval_envs, 1), dtype=np.float32) * (-1.0)
            eval_actions[env_has_actions] = actions

            # Step environment - the environment handles turn management
            obs, share_obs, rewards, dones, infos, available_actions = self.eval_envs.step(eval_actions)

            # Convert dones to boolean array (handles None values from invalid actions)
            dones = np.asarray(dones, dtype=bool)

            # Accumulate positive rewards for each environment
            rewards_array = np.asarray(rewards, dtype=np.float32)
            env_rewards = rewards_array.reshape(n_eval_envs, -1).sum(axis=1)
            positive_rewards = np.maximum(env_rewards, 0.0)
            episode_positive_rewards += positive_rewards

            eval_obs = obs.copy()
            eval_share_obs = share_obs if self.use_centralized_V else obs.copy()
            eval_available_actions = available_actions.copy()

            # Process done episodes and reset them for next episode
            envs_to_reset_after_done = np.zeros(n_eval_envs, dtype=bool)
            for env_idx in range(n_eval_envs):
                # Check if explicitly done
                if dones[env_idx] and envs_active[env_idx]:
                    # Extract score from info
                    info = infos[env_idx] if isinstance(infos, (list, tuple)) else infos
                    if isinstance(info, dict) and "score" in info:
                        score = info["score"]
                        eval_episode_scores.append(float(score))
                        # Record the cumulative positive reward for this completed episode
                        eval_episode_positive_rewards.append(float(episode_positive_rewards[env_idx]))
                        episodes_completed += 1

                        # If we still need more episodes, reset this environment for the next episode
                        if episodes_completed < num_eval_episodes:
                            envs_to_reset_after_done[env_idx] = True
                            # Reset the positive reward counter for the new episode
                            episode_positive_rewards[env_idx] = 0.0
                        else:
                            # We've reached the target, deactivate this environment
                            envs_active[env_idx] = False

            # Reset environments that completed episodes and need to continue
            if np.any(envs_to_reset_after_done):
                obs_reset, share_obs_reset, available_reset = self.eval_envs.reset(envs_to_reset_after_done)
                eval_obs[envs_to_reset_after_done] = obs_reset[envs_to_reset_after_done]
                eval_share_obs[envs_to_reset_after_done] = share_obs_reset[envs_to_reset_after_done]
                eval_available_actions[envs_to_reset_after_done] = available_reset[envs_to_reset_after_done]
                # Keep these environments active for the next episode
                # (they're already active, no need to set envs_active[envs_to_reset_after_done] = True)

        eval_duration = time.time() - eval_start_time

        if iteration_count >= max_iterations:
            print(f"[HanabiRunner][EVAL] Warning: Hit iteration limit ({max_iterations})")

        # Compute statistics
        if len(eval_episode_scores) == 0:
            print("[HanabiRunner][EVAL] Warning: No episodes completed during evaluation")
            return {'mean': 0.0, 'std': 0.0, 'variance': 0.0, 'max': 0.0, 'min': 0.0, 'count': 0}

        # Statistics for final scores (official metric)
        eval_scores_array = np.array(eval_episode_scores)
        mean_score = float(np.mean(eval_scores_array))
        std_score = float(np.std(eval_scores_array))
        variance_score = float(np.var(eval_scores_array))
        max_score = float(np.max(eval_scores_array))
        min_score = float(np.min(eval_scores_array))
        unique_scores, counts = np.unique(eval_scores_array, return_counts=True)
        score_distribution = ", ".join([f"{int(score)}:{int(count)}" for score, count in zip(unique_scores, counts)])

        # Statistics for cumulative positive rewards (learning signal)
        eval_positive_rewards_array = np.array(eval_episode_positive_rewards)
        mean_positive_reward = float(np.mean(eval_positive_rewards_array))
        std_positive_reward = float(np.std(eval_positive_rewards_array))
        variance_positive_reward = float(np.var(eval_positive_rewards_array))
        max_positive_reward = float(np.max(eval_positive_rewards_array))
        min_positive_reward = float(np.min(eval_positive_rewards_array))
        # Convert to cards played (divide by num_agents since each card gives reward to all players)
        cards_played_array = eval_positive_rewards_array / self.num_agents
        mean_cards_played = float(np.mean(cards_played_array))
        unique_cards, counts_cards = np.unique(cards_played_array.astype(int), return_counts=True)
        cards_distribution = ", ".join([f"{int(cards)}cards:{int(count)}" for cards, count in zip(unique_cards, counts_cards)])

        eval_stats = {
            'mean': mean_score,
            'std': std_score,
            'variance': variance_score,
            'max': max_score,
            'min': min_score,
            'count': len(eval_episode_scores),
            'distribution': score_distribution,
            'mean_positive_reward': mean_positive_reward,
            'mean_cards_played': mean_cards_played,
            'cards_distribution': cards_distribution
        }

        # Create comprehensive report
        report_lines = [
            "=" * 80,
            f"EVALUATION REPORT - Iteration {current_iteration}",
            f"Timestamp: {timestamp}",
            f"Duration: {eval_duration:.2f} seconds",
            "=" * 80,
            f"Episodes completed: {len(eval_episode_scores)}",
            "",
            "FINAL SCORE (Official Metric - 0 if bombed out):",
            f"  Mean: {mean_score:.4f}",
            f"  Variance: {variance_score:.4f}",
            f"  Std deviation: {std_score:.4f}",
            f"  Max: {max_score:.2f}",
            f"  Min: {min_score:.2f}",
            f"  Distribution: {score_distribution}",
            "",
            "CUMULATIVE POSITIVE REWARD (Learning Progress - cards played before death):",
            f"  Mean Reward: {mean_positive_reward:.4f} (≈{mean_cards_played:.2f} cards played per game)",
            f"  Variance: {variance_positive_reward:.4f}",
            f"  Std deviation: {std_positive_reward:.4f}",
            f"  Max: {max_positive_reward:.2f} (≈{max_positive_reward/self.num_agents:.0f} cards)",
            f"  Min: {min_positive_reward:.2f} (≈{min_positive_reward/self.num_agents:.0f} cards)",
            f"  Distribution by cards played: {cards_distribution}",
            "=" * 80,
        ]
        report = "\n".join(report_lines)

        # Print report to console
        print(report)

        # Write report to log file
        log_dir = self.save_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"eval_iter_{current_iteration}.txt"

        with open(log_file, 'w') as f:
            f.write(report + "\n")

        print(f"[HanabiRunner][EVAL] Results saved to {log_file}")

        return eval_stats

    def run(self):
        if self.envs is None:
            raise RuntimeError("Training environments are not initialised.")

        self.warmup()

        num_iterations = (
            self.num_env_steps // self.episode_length // self.n_rollout_threads
        )

        total_env_steps = 0
        finished_episodes = 0

        for iteration in range(num_iterations):
            # Track scores and positive rewards for this iteration
            iteration_start_score_count = len(self.episode_scores)
            iteration_start_positive_reward_count = len(self.episode_positive_reward_history)

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

                # Tile across agents when writing into the buffer
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
                torch.as_tensor(next_values, dtype=torch.float32)
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

            # write a CSV row (iteration-level)
            self.metrics_csv.writerow([
                int(iteration), int(total_env_steps),
                mean_score,
                r_mean, r_max, r_min, l_mean,
                adv_mean, adv_std
            ])
            self.metrics_fp.flush()
            # === end append metrics ===
            self.buffer.after_update()

            if iteration % self.log_interval == 0:
                # Calculate statistics for episodes completed during this iteration
                iteration_scores = self.episode_scores[iteration_start_score_count:]
                iteration_positive_rewards = self.episode_positive_reward_history[iteration_start_positive_reward_count:]

                if len(iteration_scores) > 0:
                    # Calculate mean final score (official metric)
                    mean_score = float(np.mean(iteration_scores))
                    scores_array = np.array(iteration_scores)
                    unique_scores, counts = np.unique(scores_array, return_counts=True)
                    score_distribution = ", ".join([f"{int(score)}:{int(count)}" for score, count in zip(unique_scores, counts)])

                    # Calculate mean cumulative positive reward (learning signal)
                    mean_positive_reward = float(np.mean(iteration_positive_rewards))
                    positive_rewards_array = np.array(iteration_positive_rewards)
                    # Divide by num_agents to get cards played (each card gives reward to all players)
                    cards_played_array = positive_rewards_array / self.num_agents
                    unique_cards, counts_cards = np.unique(cards_played_array.astype(int), return_counts=True)
                    cards_distribution = ", ".join([f"{int(cards)}cards:{int(count)}" for cards, count in zip(unique_cards, counts_cards)])

                    print(
                        f"[HanabiRunner] Iteration {iteration}/{num_iterations} (n={len(iteration_scores)} games)\n"
                        f"  Final Score: {mean_score:.2f} (distribution: {score_distribution})\n"
                        f"  Cumulative +Reward: {mean_positive_reward:.2f} (≈{mean_positive_reward/self.num_agents:.2f} cards played, distribution: {cards_distribution})"
                    )
                else:
                    print(
                        f"[HanabiRunner] Iteration {iteration}/{num_iterations} "
                        f"(no games completed in this iteration)"
                    )

            # Run evaluation periodically
            if self.use_eval and iteration % self.eval_interval == 0 and iteration > 0:
                eval_stats = self.evaluate(num_eval_episodes=self.all_args.eval_episodes, current_iteration=iteration)
                # Switch back to training mode after evaluation
                self.agent.prep_training()

            # Save model periodically
            if iteration % self.save_interval == 0:
                self.save(iteration)

        # Save final model
        self.save(num_iterations)

        # Run final evaluation
        if self.use_eval:
            print("[HanabiRunner] Running final evaluation...")
            eval_stats = self.evaluate(num_eval_episodes=self.all_args.eval_episodes, current_iteration=num_iterations)

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
            # Track cumulative positive rewards (ignore negative rewards from bombing out)
            positive_reward = np.maximum(env_reward, 0.0)
            self.current_episode_positive_reward[choose] += positive_reward[choose]
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
                    self.episode_positive_reward_history.append(float(self.current_episode_positive_reward[env_idx]))
                    self.episode_length_history.append(int(self.current_episode_length[env_idx]))
                    self.current_episode_reward[env_idx] = 0.0
                    self.current_episode_positive_reward[env_idx] = 0.0
                    self.current_episode_length[env_idx] = 0
                    finished_this_step += 1

            self.use_obs = obs.copy()
            self.use_share_obs = share_obs.copy()
            self.use_available_actions = available_actions.copy()

        return finished_this_step


if __name__ == "__main__":
    main(sys.argv[1:])
