#!/usr/bin/env python3
"""
Standalone evaluation script for trained MAPPO Hanabi models.

Usage:
    python evaluate_model.py --model_dir results/.../models --iteration latest
    python evaluate_model.py --model_dir results/.../models --iteration 8000 --num_episodes 1000
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch

# Add MAPPO to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "MAPPO"))
sys.path.insert(0, str(ROOT / "third_party" / "hanabi"))

from MAPPO.config import HanabiConfig
from MAPPO.agent import MAPPOAgent
from MAPPO.envwrappers import ChooseDummyVecEnv, ChooseSubprocVecEnv
from MAPPO.utils.util import get_shape_from_obs_space
from MAPPO.game_statistics import GameStatisticsTracker, GameStatisticsCollector
from MAPPO.gameplay_trace import GameplayTraceLogger

import pyhanabi
from Hanabi_Env import HanabiEnv


def load_config_from_json(config_path: Path) -> Optional[HanabiConfig]:
    """Load training configuration from JSON file."""
    if not config_path.exists():
        return None

    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return HanabiConfig(**config_dict)
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return None


def infer_config_from_state_dict(state_dict: Dict, args: argparse.Namespace) -> HanabiConfig:
    """Infer network architecture from state_dict when config.json is not available."""
    print("[Eval] Config file not found, inferring from state_dict...")

    # Infer hidden_layer_dim from first layer
    fc1_weight_key = 'mlp.fc1.0.weight'
    if fc1_weight_key in state_dict:
        hidden_dim = state_dict[fc1_weight_key].shape[0]
    else:
        hidden_dim = 64  # Default fallback
        print(f"Warning: Could not infer hidden_dim, using default {hidden_dim}")

    # Count number of hidden layers
    layer_n = 0
    while f'mlp.fc_layers.{layer_n}.0.weight' in state_dict:
        layer_n += 1

    print(f"[Eval] Inferred architecture: hidden_dim={hidden_dim}, layer_N={layer_n}")

    # Create config with inferred and default values
    return HanabiConfig(
        algorithm_name='mappo',
        experiment_name='eval',
        n_rollout_threads=args.num_envs,
        n_eval_rollout_threads=args.num_envs,
        num_env_steps=int(1e6),
        hanabi_name=args.hanabi_name,
        hanabi_mode=args.hanabi_mode,
        num_agents=args.num_agents,
        use_obs_instead_of_state=False,
        episode_length=100,
        hidden_layer_dim=hidden_dim,
        use_feature_normalization=False,
        use_orthogonal=True,
        use_ReLU=True,
        layer_N=layer_n,
        clip_param=0.2,
        clip_epsilon=None,
        use_clipped_value_loss=True,
        gamma=0.99,
        gae_lambda=0.95,
        use_huber_loss=True,
        huber_delta=1.0,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        ppo_epoch=4,
        num_mini_batch=1,
        lr=3e-4,
        critic_lr=3e-4,
        opti_eps=1e-5,
        weight_decay=0.0,
        save_interval=1,
        model_dir=None,
        result_dir='./results',
        log_interval=1,
        use_eval=False,
        eval_interval=100,
        eval_episodes=500,
    )


def load_model(model_dir: Path, iteration: str, config: HanabiConfig, device: torch.device):
    """Load actor and critic networks from checkpoint."""
    # Determine which checkpoint to load
    if iteration.lower() == 'latest':
        actor_path = model_dir / "actor_latest.pt"
        critic_path = model_dir / "critic_latest.pt"
        if not actor_path.exists():
            # Find the latest numbered checkpoint
            actor_files = sorted(model_dir.glob("actor_iter*.pt"),
                               key=lambda p: int(p.stem.split('_iter')[-1]))
            if not actor_files:
                raise FileNotFoundError(f"No model checkpoints found in {model_dir}")
            actor_path = actor_files[-1]
            iter_num = int(actor_path.stem.split('_iter')[-1])
            critic_path = model_dir / f"critic_iter{iter_num}.pt"
    else:
        iter_num = int(iteration)
        actor_path = model_dir / f"actor_iter{iter_num}.pt"
        critic_path = model_dir / f"critic_iter{iter_num}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {actor_path}")

    print(f"[Eval] Loading model from: {actor_path.name}")

    # Create dummy environment to get obs/action spaces
    # Ensure pyhanabi is loaded
    HANABI_SRC = ROOT / "third_party" / "hanabi"
    if not pyhanabi.cdef_loaded():
        if not pyhanabi.try_cdef(prefixes=[str(HANABI_SRC)]):
            raise RuntimeError("Failed to load pyhanabi headers")
    if not pyhanabi.lib_loaded():
        if not pyhanabi.try_load(prefixes=[str(HANABI_SRC)]):
            raise RuntimeError("Failed to load pyhanabi library")

    dummy_env = HanabiEnv(config, seed=0)
    obs_space = dummy_env.observation_space[0]
    share_obs_space = dummy_env.share_observation_space[0]
    act_space = dummy_env.action_space[0]

    # Create agent
    agent = MAPPOAgent(
        args=config,
        obs_space=obs_space,
        cent_obs_space=share_obs_space,
        act_space=act_space,
        device=device,
    )

    # Load weights
    agent.actor.load_state_dict(torch.load(str(actor_path), map_location=device))
    agent.critic.load_state_dict(torch.load(str(critic_path), map_location=device))
    agent.prep_rollout()  # Set to eval mode

    return agent, dummy_env.players


def create_eval_env(config: HanabiConfig, num_envs: int):
    """Create vectorized evaluation environments."""
    def get_env_fn(rank: int):
        def init_env():
            seed = int(time.time()) + rank
            return HanabiEnv(config, seed)
        return init_env

    if num_envs == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        env_fns = [get_env_fn(i) for i in range(num_envs)]
        return ChooseSubprocVecEnv(env_fns)


def evaluate(agent, eval_envs, num_agents: int, num_episodes: int, device: torch.device,
             use_centralized_V: bool = True, collect_statistics: bool = False,
             trace_gameplay: bool = False) -> Dict:
    """
    Run evaluation episodes and collect statistics.
    This follows the same logic as HanabiRunner.evaluate() in runner.py.
    """
    eval_start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[Eval] Starting evaluation: {num_episodes} episodes")

    agent.prep_rollout()  # Ensure eval mode

    eval_episode_scores = []
    eval_episode_positive_rewards = []
    episodes_completed = 0

    n_eval_envs = eval_envs.num_envs

    # Statistics collection
    statistics_collector = GameStatisticsCollector() if collect_statistics else None

    # Gameplay trace logger
    trace_logger = GameplayTraceLogger(num_agents) if trace_gameplay else None

    # Initialize environments
    reset_choose = np.ones(n_eval_envs, dtype=bool)
    obs, share_obs, available_actions = eval_envs.reset(reset_choose)
    eval_obs = obs.copy()
    eval_share_obs = share_obs if use_centralized_V else obs.copy()
    eval_available_actions = available_actions.copy()

    # Track active environments and rewards
    envs_active = np.ones(n_eval_envs, dtype=bool)
    episode_positive_rewards = np.zeros(n_eval_envs, dtype=np.float32)

    max_iterations = num_episodes * 100
    iteration_count = 0

    while episodes_completed < num_episodes and iteration_count < max_iterations:
        iteration_count += 1

        # Check for environments with no available actions
        no_actions_before_step = ~np.any(eval_available_actions == 1, axis=1)

        envs_to_reset = np.zeros(n_eval_envs, dtype=bool)
        for env_idx in range(n_eval_envs):
            if no_actions_before_step[env_idx] and envs_active[env_idx]:
                if episodes_completed < num_episodes:
                    envs_to_reset[env_idx] = True
                    envs_active[env_idx] = False

        if np.any(envs_to_reset):
            obs_reset, share_obs_reset, available_reset = eval_envs.reset(envs_to_reset)
            eval_obs[envs_to_reset] = obs_reset[envs_to_reset]
            eval_share_obs[envs_to_reset] = share_obs_reset[envs_to_reset]
            eval_available_actions[envs_to_reset] = available_reset[envs_to_reset]
            envs_active[envs_to_reset] = True

        # Find active environments
        env_has_actions = envs_active & (np.any(eval_available_actions == 1, axis=1))

        if not np.any(env_has_actions):
            break

        # Get actions
        with torch.no_grad():
            _, actions, _ = agent.get_actions(
                torch.as_tensor(eval_share_obs[env_has_actions], dtype=torch.float32, device=device),
                torch.as_tensor(eval_obs[env_has_actions], dtype=torch.float32, device=device),
                torch.as_tensor(eval_available_actions[env_has_actions], dtype=torch.float32, device=device),
            )

        actions = actions.detach().cpu().numpy()

        # Create full action array
        eval_actions = np.ones((n_eval_envs, 1), dtype=np.float32) * (-1.0)
        eval_actions[env_has_actions] = actions

        # Trace gameplay if enabled (before step)
        if trace_logger is not None:
            # Get environment 0 (the only one when tracing)
            env_state = eval_envs.get_state(0)
            env_game = eval_envs.get_game(0)

            if env_state is not None and not env_state.is_terminal():
                # Get full observation from environment
                env = eval_envs.envs[0]  # Access the underlying environment
                full_observation = env._make_observation_all_players()
                current_player = env_state.cur_player()

                # Log turn start with state
                trace_logger.log_turn_start(current_player, full_observation, env_state)

                # Convert action int to dict
                action_int = int(eval_actions[0, 0])
                if action_int >= 0:
                    action_move = env_game.get_move(action_int)
                    action_dict = action_move.to_dict()
                    trace_logger.log_action(current_player, action_dict)

        # Step environment
        obs, share_obs, rewards, dones, infos, available_actions = eval_envs.step(eval_actions)

        # Trace gameplay if enabled (after step)
        if trace_logger is not None and not dones[0]:
            # Get new observation after step
            env = eval_envs.envs[0]
            new_observation = env._make_observation_all_players()

            # Log action result - extract scalar reward
            reward_val = rewards[0]
            if isinstance(reward_val, (list, np.ndarray)):
                if isinstance(reward_val, np.ndarray) and reward_val.size > 0:
                    reward_val = reward_val.item() if reward_val.size == 1 else reward_val.flatten()[0]
                else:
                    reward_val = reward_val[0]
            reward_val = float(reward_val)

            info = infos[0] if isinstance(infos, (list, tuple, np.ndarray)) else infos
            trace_logger.log_action_result(new_observation, reward_val, info)

        dones = np.asarray(dones, dtype=bool)

        # Accumulate rewards
        rewards_array = np.asarray(rewards, dtype=np.float32)
        env_rewards = rewards_array.reshape(n_eval_envs, -1).sum(axis=1)
        positive_rewards = np.maximum(env_rewards, 0.0)
        episode_positive_rewards += positive_rewards

        eval_obs = obs.copy()
        eval_share_obs = share_obs if use_centralized_V else obs.copy()
        eval_available_actions = available_actions.copy()

        # Process done episodes
        envs_to_reset_after_done = np.zeros(n_eval_envs, dtype=bool)
        for env_idx in range(n_eval_envs):
            if dones[env_idx] and envs_active[env_idx]:
                # Handle infos as array-like (list, tuple, or numpy.ndarray) or single dict
                info = infos[env_idx] if isinstance(infos, (list, tuple, np.ndarray)) else infos
                if isinstance(info, dict) and "score" in info:
                    score = info["score"]
                    eval_episode_scores.append(float(score))
                    eval_episode_positive_rewards.append(float(episode_positive_rewards[env_idx]))
                    episodes_completed += 1

                    # Collect statistics if enabled
                    if collect_statistics and statistics_collector is not None:
                        try:
                            # Get the final state and game from the environment
                            final_state = eval_envs.get_state(env_idx)
                            final_game = eval_envs.get_game(env_idx)
                            if final_state is not None and final_game is not None:
                                # Disable debug mode by default for cleaner output
                                debug_mode = False
                                tracker = GameStatisticsTracker(final_state, final_game, num_agents)
                                episode_stats = tracker.analyze(debug=debug_mode)
                                statistics_collector.add_episode(episode_stats)
                        except Exception as e:
                            print(f"[Eval] Warning: Failed to collect statistics for episode {episodes_completed}: {e}")

                    # Log game end if tracing enabled
                    if trace_logger is not None and env_idx == 0:
                        try:
                            final_state = eval_envs.get_state(env_idx)
                            if final_state is not None:
                                trace_logger.log_game_end(final_state, trace_logger.turn_number)
                        except Exception as e:
                            print(f"[Eval] Warning: Failed to log game end: {e}")

                    if episodes_completed < num_episodes:
                        envs_to_reset_after_done[env_idx] = True
                        episode_positive_rewards[env_idx] = 0.0
                    else:
                        envs_active[env_idx] = False

                    # Print progress
                    if episodes_completed % 50 == 0:
                        progress_msg = f"[Eval] Progress: {episodes_completed}/{num_episodes} episodes completed"
                        if collect_statistics:
                            progress_msg += " (collecting statistics)"
                        print(progress_msg)

        if np.any(envs_to_reset_after_done):
            obs_reset, share_obs_reset, available_reset = eval_envs.reset(envs_to_reset_after_done)
            eval_obs[envs_to_reset_after_done] = obs_reset[envs_to_reset_after_done]
            eval_share_obs[envs_to_reset_after_done] = share_obs_reset[envs_to_reset_after_done]
            eval_available_actions[envs_to_reset_after_done] = available_reset[envs_to_reset_after_done]

    eval_duration = time.time() - eval_start_time

    if iteration_count >= max_iterations:
        print(f"[Eval] Warning: Hit iteration limit ({max_iterations})")

    # Compute statistics
    if len(eval_episode_scores) == 0:
        print("[Eval] Warning: No episodes completed")
        return {}

    # Final score statistics
    eval_scores_array = np.array(eval_episode_scores)
    mean_score = float(np.mean(eval_scores_array))
    std_score = float(np.std(eval_scores_array))
    variance_score = float(np.var(eval_scores_array))
    max_score = float(np.max(eval_scores_array))
    min_score = float(np.min(eval_scores_array))
    unique_scores, counts = np.unique(eval_scores_array, return_counts=True)
    score_distribution = ", ".join([f"{int(score)}:{int(count)}" for score, count in zip(unique_scores, counts)])

    # Positive reward statistics
    eval_positive_rewards_array = np.array(eval_episode_positive_rewards)
    mean_positive_reward = float(np.mean(eval_positive_rewards_array))
    std_positive_reward = float(np.std(eval_positive_rewards_array))
    variance_positive_reward = float(np.var(eval_positive_rewards_array))
    max_positive_reward = float(np.max(eval_positive_rewards_array))
    min_positive_reward = float(np.min(eval_positive_rewards_array))
    cards_played_array = eval_positive_rewards_array / num_agents
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
        'cards_distribution': cards_distribution,
        'duration': eval_duration,
        'timestamp': timestamp,
    }

    # Create report
    report_lines = [
        "=" * 80,
        "STANDALONE EVALUATION REPORT",
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
        f"  Max: {max_positive_reward:.2f} (≈{max_positive_reward/num_agents:.0f} cards)",
        f"  Min: {min_positive_reward:.2f} (≈{min_positive_reward/num_agents:.0f} cards)",
        f"  Distribution by cards played: {cards_distribution}",
        "=" * 80,
    ]

    eval_stats['report'] = "\n".join(report_lines)

    # Add statistics report if collected
    if collect_statistics and statistics_collector is not None:
        stats_report = statistics_collector.generate_report()
        eval_stats['statistics_report'] = stats_report
        eval_stats['statistics_collector'] = statistics_collector

    # Add gameplay trace if collected
    if trace_logger is not None:
        eval_stats['gameplay_trace'] = trace_logger.get_trace()

    return eval_stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MAPPO Hanabi model')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing saved models (e.g., results/.../run14/models)')
    parser.add_argument('--iteration', type=str, default='latest',
                       help='Model iteration to load ("latest" or specific number like "8000")')
    parser.add_argument('--num_episodes', type=int, default=500,
                       help='Number of episodes to evaluate (default: 500)')
    parser.add_argument('--num_envs', type=int, default=10,
                       help='Number of parallel environments (default: 10)')
    parser.add_argument('--save_log', action='store_true',
                       help='Save evaluation report to log file')
    parser.add_argument('--collect_statistics', action='store_true',
                       help='Collect detailed game statistics (play decisions, discards, hints)')
    parser.add_argument('--trace_gameplay', action='store_true',
                       help='Trace detailed gameplay turn-by-turn (requires --num_episodes 1 --num_envs 1)')

    # Optional overrides (used when config.json is not available)
    parser.add_argument('--hanabi_name', type=str, default='Hanabi-Full',
                       help='Hanabi environment name (default: Hanabi-Full)')
    parser.add_argument('--hanabi_mode', type=str, default='full',
                       help='Hanabi mode (default: full)')
    parser.add_argument('--num_agents', type=int, default=2,
                       help='Number of agents/players (default: 2)')

    args = parser.parse_args()

    # Validate trace_gameplay requirements
    if args.trace_gameplay:
        if args.num_episodes != 1:
            print("Error: --trace_gameplay requires --num_episodes 1")
            sys.exit(1)
        if args.num_envs != 1:
            print("Error: --trace_gameplay requires --num_envs 1")
            sys.exit(1)
        # Auto-enable save_log when tracing
        args.save_log = True

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    # Load or infer configuration
    config_path = model_dir / "config.json"
    config = load_config_from_json(config_path)

    if config is None:
        print("[Eval] No config.json found, will infer from model weights...")
        # Load actor to infer architecture
        if args.iteration.lower() == 'latest':
            actor_path = model_dir / "actor_latest.pt"
            if not actor_path.exists():
                actor_files = sorted(model_dir.glob("actor_iter*.pt"))
                if actor_files:
                    actor_path = actor_files[-1]
        else:
            actor_path = model_dir / f"actor_iter{args.iteration}.pt"

        if actor_path.exists():
            state_dict = torch.load(actor_path, map_location='cpu')
            config = infer_config_from_state_dict(state_dict, args)
        else:
            print(f"Error: Cannot find model checkpoint: {actor_path}")
            sys.exit(1)
    else:
        print(f"[Eval] Loaded config from {config_path}")

    # Override num_envs for evaluation
    config.n_eval_rollout_threads = args.num_envs

    print(f"\n[Eval] Configuration:")
    print(f"  Hanabi variant: {config.hanabi_name}")
    print(f"  Num agents: {config.num_agents}")
    print(f"  Hidden dim: {config.hidden_layer_dim}")
    print(f"  Layer N: {config.layer_N}")
    print(f"  Num eval envs: {args.num_envs}")
    print(f"  Num episodes: {args.num_episodes}")
    print()

    # Load model
    agent, num_agents = load_model(model_dir, args.iteration, config, device)

    # Create evaluation environments
    print(f"[Eval] Creating {args.num_envs} evaluation environments...")
    eval_envs = create_eval_env(config, args.num_envs)

    # Run evaluation
    eval_stats = evaluate(agent, eval_envs, num_agents, args.num_episodes, device,
                         collect_statistics=args.collect_statistics,
                         trace_gameplay=args.trace_gameplay)

    # Print report
    if eval_stats and 'report' in eval_stats:
        print("\n" + eval_stats['report'])

    # Print statistics report if collected
    if args.collect_statistics and eval_stats and 'statistics_report' in eval_stats:
        print("\n" + eval_stats['statistics_report'])

    # Save log file if requested
    if args.save_log and eval_stats:
        log_dir = model_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        iter_str = args.iteration if args.iteration != 'latest' else 'latest'
        log_file = log_dir / f"standalone_eval_iter_{iter_str}.txt"

        with open(log_file, 'w') as f:
            f.write(eval_stats['report'] + "\n")

        print(f"\n[Eval] Results saved to: {log_file}")

        # Save statistics report if collected
        if args.collect_statistics and 'statistics_report' in eval_stats:
            stats_file = log_dir / f"statistics_iter_{iter_str}.txt"
            with open(stats_file, 'w') as f:
                f.write(eval_stats['statistics_report'] + "\n")
            print(f"[Eval] Statistics saved to: {stats_file}")

        # Save gameplay trace if collected
        if args.trace_gameplay and 'gameplay_trace' in eval_stats:
            trace_file = log_dir / f"gameplay_trace_iter_{iter_str}.txt"
            with open(trace_file, 'w') as f:
                f.write(eval_stats['gameplay_trace'] + "\n")
            print(f"[Eval] Gameplay trace saved to: {trace_file}")

    # Clean up
    eval_envs.close()

    print("\n[Eval] Evaluation complete!")

    # Return mean score as exit code for scripting (clipped to 0-255)
    if eval_stats:
        sys.exit(int(min(eval_stats['mean'], 255)))
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
