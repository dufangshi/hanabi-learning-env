#!/usr/bin/env python3
"""
Visualize MAPPO training progress by parsing evaluation logs across multiple runs.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


def parse_eval_log(log_path: Path) -> Dict:
    """Parse a single evaluation log file and extract metrics.

    Args:
        log_path: Path to the evaluation log file

    Returns:
        Dictionary with parsed metrics or None if parsing failed
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Extract iteration number from filename
        filename = log_path.stem
        iter_match = re.search(r'eval_iter_(\d+)', filename)
        if not iter_match:
            return None
        iteration = int(iter_match.group(1))

        # Extract mean final score
        score_match = re.search(r'Mean:\s+([\d.]+)', content)
        if not score_match:
            return None
        mean_score = float(score_match.group(1))

        # Extract death count (score=0)
        dist_match = re.search(r'Distribution:\s+(.+)', content)
        death_count = 0
        if dist_match:
            dist_str = dist_match.group(1)
            zero_match = re.search(r'0:(\d+)', dist_str)
            if zero_match:
                death_count = int(zero_match.group(1))

        # Extract cumulative positive reward mean
        pos_reward_match = re.search(r'Mean Reward:\s+([\d.]+)', content)
        mean_positive_reward = 0.0
        if pos_reward_match:
            mean_positive_reward = float(pos_reward_match.group(1))

        # Extract mean cards played
        cards_match = re.search(r'≈([\d.]+)\s+cards played per game', content)
        mean_cards_played = 0.0
        if cards_match:
            mean_cards_played = float(cards_match.group(1))

        # Extract episode count
        episodes_match = re.search(r'Episodes completed:\s+(\d+)', content)
        num_episodes = 0
        if episodes_match:
            num_episodes = int(episodes_match.group(1))

        return {
            'iteration': iteration,
            'mean_score': mean_score,
            'death_count': death_count,
            'num_episodes': num_episodes,
            'death_rate': death_count / num_episodes if num_episodes > 0 else 0,
            'mean_positive_reward': mean_positive_reward,
            'mean_cards_played': mean_cards_played,
        }
    except Exception as e:
        print(f"Warning: Failed to parse {log_path}: {e}")
        return None


def collect_all_logs(base_dir: Path, select_runs: Optional[List[str]] = None) -> List[Dict]:
    """Collect and parse all evaluation logs from all run directories, merging runs with cumulative iteration numbers.

    Args:
        base_dir: Base directory containing run folders (e.g., .../check/)
        select_runs: Optional list of run names to process (e.g., ["run14", "run15", "run16"]).
                    If None, processes all runs in base_dir.

    Returns:
        List of parsed log dictionaries with cumulative iteration numbers, sorted by iteration
    """
    # Helper function to extract run number for sorting
    def get_run_number(run_path: Path) -> int:
        """Extract run number from directory name for sorting."""
        match = re.search(r'run(\d+)', run_path.name)
        return int(match.group(1)) if match else 999999  # Put non-numeric runs at the end

    # Determine which runs to process
    if select_runs is not None:
        # Filter to only specified runs
        run_dirs = []
        for run_name in select_runs:
            run_path = base_dir / run_name
            if run_path.exists() and run_path.is_dir():
                run_dirs.append(run_path)
            else:
                print(f"Warning: Run directory not found: {run_path}")
        # Sort by run number
        run_dirs = sorted(run_dirs, key=get_run_number)
    else:
        # Process all runs
        run_dirs = sorted(base_dir.glob('run*'), key=get_run_number)

    all_data = []
    cumulative_iter_offset = 0  # Track the maximum iteration from previous runs

    for run_dir in run_dirs:
        run_name = run_dir.name
        logs_dir = run_dir / 'logs'
        if not logs_dir.exists():
            continue

        # Parse all eval_iter_*.txt files for this run
        log_files = sorted(logs_dir.glob('eval_iter_*.txt'))
        run_data = []

        for log_file in log_files:
            parsed = parse_eval_log(log_file)
            if parsed:
                run_data.append(parsed)

        if not run_data:
            continue

        # Sort by iteration for this run
        run_data.sort(key=lambda x: x['iteration'])

        # Find the maximum iteration in this run
        max_iter_in_run = max(d['iteration'] for d in run_data)

        # Adjust iteration numbers by adding the cumulative offset
        for data_point in run_data:
            # Create a copy to avoid modifying the original
            adjusted_data = data_point.copy()
            adjusted_data['iteration'] = data_point['iteration'] + cumulative_iter_offset
            adjusted_data['original_run'] = run_name
            adjusted_data['original_iteration'] = data_point['iteration']
            all_data.append(adjusted_data)

        # Update cumulative offset for next run
        cumulative_iter_offset += max_iter_in_run

        print(f"  {run_name}: {len(run_data)} logs, iteration range {min(d['iteration'] for d in run_data)}-{max_iter_in_run} "
              f"(adjusted to {min(d['iteration'] for d in run_data) + cumulative_iter_offset - max_iter_in_run}-{cumulative_iter_offset})")

    # Sort all data by adjusted iteration
    all_data.sort(key=lambda x: x['iteration'])

    return all_data


def plot_training_curves(data: List[Dict], output_path: Path):
    """Create comprehensive training curves visualization.

    Args:
        data: List of parsed log dictionaries
        output_path: Path to save the plot
    """
    if not data:
        print("No data to plot!")
        return

    # Extract data arrays
    iterations = np.array([d['iteration'] for d in data])
    mean_scores = np.array([d['mean_score'] for d in data])
    death_rates = np.array([d['death_rate'] * 100 for d in data])  # Convert to percentage
    mean_positive_rewards = np.array([d['mean_positive_reward'] for d in data])
    mean_cards_played = np.array([d['mean_cards_played'] for d in data])

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MAPPO Training Progress on Hanabi-Full', fontsize=16, fontweight='bold')

    # Plot 1: Mean Final Score vs Iteration
    ax1 = axes[0, 0]
    ax1.plot(iterations, mean_scores, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Mean Final Score', fontsize=12)
    ax1.set_title('Final Score (Official Metric)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Add trend line
    if len(iterations) > 1:
        z = np.polyfit(iterations, mean_scores, 2)
        p = np.poly1d(z)
        ax1.plot(iterations, p(iterations), 'r--', alpha=0.5, linewidth=1.5, label='Trend')
        ax1.legend()

    # Plot 2: Death Rate vs Iteration
    ax2 = axes[0, 1]
    ax2.plot(iterations, death_rates, 'r-', linewidth=2, marker='s', markersize=4, alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Death Rate (%)', fontsize=12)
    ax2.set_title('Death Rate (Life Tokens = 0)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    # Add horizontal reference lines
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.3, label='50% death rate')
    ax2.axhline(y=25, color='green', linestyle='--', alpha=0.3, label='25% death rate')
    ax2.legend()

    # Plot 3: Cumulative Positive Reward vs Iteration
    ax3 = axes[1, 0]
    ax3.plot(iterations, mean_positive_rewards, 'g-', linewidth=2, marker='^', markersize=4, alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Mean Cumulative Positive Reward', fontsize=12)
    ax3.set_title('Cumulative Positive Reward (Learning Signal)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    # Plot 4: Mean Cards Played vs Iteration
    ax4 = axes[1, 1]
    ax4.plot(iterations, mean_cards_played, 'm-', linewidth=2, marker='D', markersize=4, alpha=0.7)
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Mean Cards Played', fontsize=12)
    ax4.set_title('Average Cards Played Per Game', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    # Add max possible score reference line
    ax4.axhline(y=25, color='orange', linestyle='--', alpha=0.3, label='Perfect game (25 cards)')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to: {output_path}")

    # Also create a focused plot on Final Score and Death Rate
    fig2, (ax_score, ax_death) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig2.suptitle('MAPPO Training: Key Metrics', fontsize=16, fontweight='bold')

    # Score plot
    ax_score.plot(iterations, mean_scores, 'b-', linewidth=2.5, marker='o', markersize=5, alpha=0.8, label='Mean Score')
    ax_score.fill_between(iterations, 0, mean_scores, alpha=0.2, color='blue')
    ax_score.set_ylabel('Mean Final Score', fontsize=13, fontweight='bold')
    ax_score.set_title('Final Score Progression', fontsize=13)
    ax_score.grid(True, alpha=0.3)
    ax_score.legend(fontsize=11)
    ax_score.set_ylim(bottom=0)

    # Death rate plot
    ax_death.plot(iterations, death_rates, 'r-', linewidth=2.5, marker='s', markersize=5, alpha=0.8, label='Death Rate')
    ax_death.fill_between(iterations, 0, death_rates, alpha=0.2, color='red')
    ax_death.axhline(y=50, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='50% threshold')
    ax_death.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax_death.set_ylabel('Death Rate (%)', fontsize=13, fontweight='bold')
    ax_death.set_title('Death Rate Progression', fontsize=13)
    ax_death.grid(True, alpha=0.3)
    ax_death.legend(fontsize=11)
    ax_death.set_ylim([0, 100])

    plt.tight_layout()
    focused_output = output_path.parent / (output_path.stem + '_focused' + output_path.suffix)
    plt.savefig(focused_output, dpi=150, bbox_inches='tight')
    print(f"✓ Focused training curves saved to: {focused_output}")


def print_statistics(data: List[Dict]):
    """Print summary statistics of the training.

    Args:
        data: List of parsed log dictionaries
    """
    if not data:
        print("No data to analyze!")
        return

    print("\n" + "="*70)
    print("TRAINING STATISTICS SUMMARY")
    print("="*70)

    iterations = [d['iteration'] for d in data]
    scores = [d['mean_score'] for d in data]
    death_rates = [d['death_rate'] * 100 for d in data]

    print(f"\nTotal evaluation points: {len(data)}")
    print(f"Iteration range: {min(iterations)} - {max(iterations)}")

    print(f"\n--- Final Score Statistics ---")
    print(f"  Best mean score: {max(scores):.2f} (iteration {iterations[scores.index(max(scores))]})")
    print(f"  Worst mean score: {min(scores):.2f} (iteration {iterations[scores.index(min(scores))]})")
    print(f"  Overall average: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

    print(f"\n--- Death Rate Statistics ---")
    print(f"  Lowest death rate: {min(death_rates):.1f}% (iteration {iterations[death_rates.index(min(death_rates))]})")
    print(f"  Highest death rate: {max(death_rates):.1f}% (iteration {iterations[death_rates.index(max(death_rates))]})")
    print(f"  Overall average: {np.mean(death_rates):.1f}% ± {np.std(death_rates):.1f}%")

    # Check for trend (comparing first 25% vs last 25% of data)
    n = len(scores)
    early_scores = scores[:n//4]
    late_scores = scores[-n//4:]
    if early_scores and late_scores:
        score_change = np.mean(late_scores) - np.mean(early_scores)
        print(f"\n--- Training Trend ---")
        print(f"  Early phase avg score: {np.mean(early_scores):.2f}")
        print(f"  Late phase avg score: {np.mean(late_scores):.2f}")
        print(f"  Change: {score_change:+.2f} ({score_change/np.mean(early_scores)*100:+.1f}%)")

    print("="*70 + "\n")


def main():
    """Main function to generate training visualizations."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize MAPPO training progress')
    parser.add_argument('--base_dir', type=str,
                       default='/home/u/dev/hanabi/hanabi-learning-env/results/Hanabi/Hanabi-Full/mappo/check',
                       help='Base directory containing run folders (default: check directory)')
    parser.add_argument('--select_runs', type=str, nargs='*', default=None,
                       help='Specific run names to visualize (e.g., run14 run15 run16). If not specified, all runs are used.')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the visualization (default: base_dir/training_curves.png)')

    args = parser.parse_args()

    # Define paths
    base_dir = Path(args.base_dir)
    output_path = Path(args.output) if args.output else (base_dir / 'training_curves.png')

    print("="*70)
    print("MAPPO Training Visualization")
    print("="*70)
    print(f"\nScanning directory: {base_dir}")

    if args.select_runs:
        print(f"Selected runs: {', '.join(args.select_runs)}")
    else:
        print("Processing all runs in directory")

    # Collect all logs
    print("\nCollecting evaluation logs...")
    if args.select_runs:
        print(f"Merging selected runs with cumulative iteration numbers...")
    else:
        print("Merging all runs with cumulative iteration numbers...")

    data = collect_all_logs(base_dir, select_runs=args.select_runs)

    if not data:
        print("ERROR: No valid evaluation logs found!")
        return

    print(f"\n✓ Successfully parsed {len(data)} evaluation logs")

    # Print statistics
    print_statistics(data)

    # Generate plots
    print("Generating training curves...")
    plot_training_curves(data, output_path)

    print("\n✓ Visualization complete!")
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {output_path.parent / (output_path.stem + '_focused' + output_path.suffix)}")


if __name__ == '__main__':
    main()
