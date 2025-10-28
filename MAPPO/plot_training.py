import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main(csv_path="logs/metrics.csv", out_path="logs/training_curve.png"):
    csv_path = Path(csv_path)
    episodes, mean10 = [], []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            episodes.append(int(row["episode"]))
            mean10.append(float(row["mean_score_last10"]))
    episodes = np.array(episodes)
    mean10   = np.array(mean10)

    plt.figure(figsize=(9,5))
    plt.plot(episodes, mean10, linewidth=2, label="Mean score (last 10)")
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
