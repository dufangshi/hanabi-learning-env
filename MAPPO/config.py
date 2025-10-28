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
    hanabi_mode: str
    num_agents: int
    use_obs_instead_of_state: bool
    episode_length: int
    hidden_layer_dim: int
    use_feature_normalization: bool
    use_orthogonal: bool
    use_ReLU: bool
    layer_N: int
    clip_param: float
    clip_epsilon: float
    use_clipped_value_loss: bool
    gamma: float
    gae_lambda: float
    use_huber_loss: bool
    huber_delta: float
    entropy_coef: float
    value_loss_coef: float
    max_grad_norm: float
    ppo_epoch: int
    num_mini_batch: int
    lr: float
    critic_lr: float
    opti_eps: float
    weight_decay: float
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
    parser.add_argument("--n_rollout_threads", type=int, default=512,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')    

    # env parameters
    parser.add_argument("--hanabi_name", type=str, default="Hanabi-Full-Minimal")  # choice details can be seen at ~/third_party/hanabi/Hanabi_Env.py
    parser.add_argument(
        "--hanabi_mode",
        type=str,
        choices=["minimal", "full", "full-minimal", "small", "very-small"],
        default="minimal",
        help="Shortcut for common Hanabi variants (maps to hanabi_name).",
    )
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--use_obs_instead_of_state", action='store_true', default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--hidden_layer_dim", type=int, default=128) 
    parser.add_argument("--use_feature_normalization", type=bool, default=False) 
    parser.add_argument("--use_orthogonal", type=int, default=1) 
    parser.add_argument("--use_ReLU", type=bool, default=True) 
    parser.add_argument("--layer_N", type=int, default=2)

    # ppo parameters
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--clip_epsilon", type=float, default=None,
                        help="override clip_param for PPO clip range; defaults to clip_param")
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_huber_loss", action='store_false', default=True,
                        help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--huber_delta", type=float, default=1.0,
                        help="delta value for SmoothL1 (Huber) loss.")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="entropy regularisation coefficient.")
    parser.add_argument("--value_loss_coef", type=float, default=0.5,
                        help="value loss coefficient.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="gradient norm clipping value.")
    parser.add_argument("--ppo_epoch", type=int, default=4,
                        help="number of PPO epochs per update.")
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help="number of mini batches per epoch.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="actor learning rate.")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                        help="critic learning rate.")
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help="optimizer epsilon.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="optional L2 weight decay.")

    # save parameters
    parser.add_argument("--save_interval", type=int, default=1, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between continuous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")

    args = parser.parse_args()  #TODO might need to change to parser.parse_known_args() if using framework that adds its own unknown args e.g. wandb
    if args.clip_epsilon is None:
        args.clip_epsilon = args.clip_param

    mode_to_name = {
        "minimal": "Hanabi-Very-Small",
        "full": "Hanabi-Full",
        "full-minimal": "Hanabi-Full-Minimal",
        "small": "Hanabi-Small",
        "very-small": "Hanabi-Very-Small",
    }
    args.hanabi_name = mode_to_name.get(args.hanabi_mode, args.hanabi_name)
    return HanabiConfig(**vars(args))
