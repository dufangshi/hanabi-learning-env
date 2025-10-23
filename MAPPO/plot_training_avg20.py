import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main(csv_path="logs/metrics.csv", out_path="logs/training_curve.png"):
    csv_path = Path(csv_path)
    episodes, mean10, eval_mean = [], [], []

    # --- 1. read the csv ---
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            mean10.append(float(row["mean_score_last10"]))
            # if you later add an "eval_mean_score" column, also collect it:
            eval_mean.append(float(row.get("eval_mean_score", "nan")))

    episodes = np.array(episodes)
    mean10 = np.array(mean10)
    eval_mean = np.array(eval_mean)

    # --- 2. optional smoothing (moving average) ---
    def smooth(y, window=20):
        if len(y) < window:
            return y
        return np.convolve(y, np.ones(window)/window, mode="valid")

    smooth_mean10 = smooth(mean10, window=20)

    # --- 3. plot ---
    plt.figure(figsize=(9,5))
    plt.plot(episodes, mean10, alpha=0.3, label="Mean score (last 10)")
    plt.plot(episodes[:len(smooth_mean10)], smooth_mean10, linewidth=2, label="Smoothed (×20)")
    if not np.isnan(eval_mean).all():
        plt.plot(episodes, eval_mean, color="orange", linewidth=2, label="Eval mean score")

    plt.xlabel("Episode")
    plt.ylabel("Score (0–5 for 1-color)")
    plt.title("Hanabi MAPPO — Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
