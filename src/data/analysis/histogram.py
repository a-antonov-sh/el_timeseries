import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.config import PREPARED_DIR

BINS = 100
BAR_WIDTH = 60
DIAGRAMS_DIR = os.path.join("data", "diagrams")


def _load(series_id=None):
    path = os.path.join(PREPARED_DIR, "train_context.parquet")
    if not os.path.exists(path):
        path = os.path.join(PREPARED_DIR, "context.parquet")
    df = pd.read_parquet(path)
    if series_id is not None:
        df = df[df["id"] == str(series_id)]
    return df["target"].values


def _save_histogram(s, title, filename):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(s, bins=BINS, color="steelblue", edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    out = os.path.join(DIAGRAMS_DIR, filename)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def histogram(series_id="0"):
    s = _load(series_id)
    if len(s) == 0:
        print(f"Series {series_id} not found.")
        return
    print(f"Series {series_id}  n={len(s)}  min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}")
    counts, edges = np.histogram(s, bins=BINS)
    max_count = counts.max()
    for i, count in enumerate(counts):
        bar = "█" * int(count / max_count * BAR_WIDTH)
        print(f"{edges[i]:8.2f} |{bar} {count}")
    _save_histogram(s, f"Series {series_id} distribution", f"histogram_series{series_id}.png")


def histogram_all():
    s = _load()
    print(f"All series  n={len(s)}  min={s.min():.2f}  max={s.max():.2f}  mean={s.mean():.2f}")
    counts, edges = np.histogram(s, bins=BINS)
    max_count = counts.max()
    for i, count in enumerate(counts):
        bar = "█" * int(count / max_count * BAR_WIDTH)
        print(f"{edges[i]:8.2f} |{bar} {count}")
    _save_histogram(s, "All series distribution", "histogram_all.png")


def std_per_slot():
    path = os.path.join(PREPARED_DIR, "train_context.parquet")
    if not os.path.exists(path):
        path = os.path.join(PREPARED_DIR, "context.parquet")
    df = pd.read_parquet(path)

    for col in ("hour", "day_of_week"):
        grouped = df.groupby(col)["target"].std()
        max_std = grouped.max()
        print(f"\nStd by {col}:")
        for slot, std in grouped.items():
            bar = "█" * int(std / max_std * BAR_WIDTH)
            print(f"  {slot:2d} |{bar} {std:.4f}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "all":
        histogram_all()
    elif arg == "std":
        std_per_slot()
    else:
        histogram(arg or "0")