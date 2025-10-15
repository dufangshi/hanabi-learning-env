import argparse
from dataclasses import dataclass

@dataclass
class HanabiConfig:
    algorithm_name: str
    experiment_name: str
    n_rollout_threads: int
    n_eval_rollout_threads: int
    num_env_steps: int
    hanabi_name: str
    num_agents: int
    use_obs_instead_of_state: bool
    episode_length: int
    hidden_layer_dim: int
    use_feature_normalization: bool
    use_orthogonal: bool
    use_ReLU: bool
    layer_N: int
    clip_param: float
    use_clipped_value_loss: bool
    gamma: float
    gae_lambda: float
    use_huber_loss: bool
    save_interval: int
    log_interval: int
    use_eval: bool
    eval_interval: int

def get_config() -> HanabiConfig:
    """
    The configuration parser for hyperparameters of Hanabi Learning Environment.

    Note: Only cover arguments listed in testinginteraction.py for now.
    TODO: should we move this to runner.py instead?
    """
    parser = argparse.ArgumentParser(
        description="mappo_hanabi", formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default='mappo')
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')    

    # env parameters
    parser.add_argument("--hanabi_name", type=str, default="Hanabi-Full-Minimal")  # choice details can be seen at ~/third_party/hanabi/Hanabi_Env.py
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--use_obs_instead_of_state", action='store_true', default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--hidden_layer_dim", type=int, default=64) 
    parser.add_argument("--use_feature_normalization", type=bool, default=False) 
    parser.add_argument("--use_orthogonal", type=int, default=1) 
    parser.add_argument("--use_ReLU", type=bool, default=True) 
    parser.add_argument("--layer_N", type=int, default=1)

    # ppo parameters
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")

    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between continuous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")

    args = parser.parse_args()  #TODO might need to change to parser.parse_known_args() if using framework that adds its own unknown args e.g. wandb
    return HanabiConfig(**vars(args))