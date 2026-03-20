from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def _moving_average(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=np.float64)
    window = max(1, min(window, len(arr)))
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def plot_metrics(log_dir: str | Path) -> tuple[Path, Path]:
    log_dir = Path(log_dir)
    online_csv = log_dir / "online_metrics.csv"
    online_summary_json = log_dir / "online_summary.json"

    if not online_csv.exists():
        raise FileNotFoundError(f"Missing file: {online_csv}")
    if not online_summary_json.exists():
        raise FileNotFoundError(f"Missing file: {online_summary_json}")

    mpl_dir = log_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt

    rows_by_variant: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with online_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_variant[row["variant"]].append(
                (int(row["global_episode"]), float(row["accuracy"]))
            )

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for variant, points in sorted(rows_by_variant.items()):
        points = sorted(points, key=lambda x: x[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        y_smooth = _moving_average(y, window=10)
        ax1.plot(x, y_smooth, label=variant)

    ax1.set_title("Online Adaptation Curve (Smoothed Accuracy)")
    ax1.set_xlabel("Global Episode")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.2)
    ax1.legend()

    adaptation_path = log_dir / "adaptation_curve.png"
    fig1.tight_layout()
    fig1.savefig(adaptation_path, dpi=160)
    plt.close(fig1)

    with online_summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    variants = sorted(summary.keys())
    gains = [float(summary[v].get("mean_cold_start_gain", 0.0)) for v in variants]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(variants, gains)
    ax2.set_title("Memory Consolidation Gain (Cold-Start)")
    ax2.set_ylabel("Gain vs First Exposure")
    ax2.axhline(0.0, color="black", linewidth=1)
    ax2.grid(axis="y", alpha=0.2)

    consolidation_path = log_dir / "consolidation_gain.png"
    fig2.tight_layout()
    fig2.savefig(consolidation_path, dpi=160)
    plt.close(fig2)

    return adaptation_path, consolidation_path
