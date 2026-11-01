"""Summary figure: tracking and shape error of every tuning experiment on the real robot.

The runs are listed explicitly, in the order the experiments were made, with a
label saying what changed. For each run:
  - tracking error: distance between the real gripper and where it should be
    at that instant (includes the lag);
  - shape error: distance between the real gripper and the closest point of the
    lemniscate (how well the figure is drawn, whatever the timing).

Usage (from the project root):
    python lemniscate/summarize_runs.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lemniscate.compare_sim_real import AXIS, GRID, REAL, SURFACE, TEXT, TEXT_2, ee_positions  # noqa: E402
from lemniscate.trajectory import load_model  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs"
FIGURES = Path(__file__).resolve().parent / "figures"

# (log file suffix, label). Order = order of the experiments.
RUNS = [
    ("153920", "30 Hz, 16 s/loop (LeRobot defaults)"),
    ("154232", "30 Hz, 8 s/loop (LeRobot defaults)"),
    ("155011", "acceleration 50"),
    ("155101", "acceleration 20"),
    ("160044", "I = 2"),
    ("160132", "I = 5"),
    ("160218", "I = 10"),
    ("162140", "I = 10, dead zone 0"),
    ("163620", "I = 10, D = 16"),
    ("163751", "I = 10, D = 0"),
    ("163704", "I = 10, P = 24, D = 48"),
    ("164445", "I = 10, 60 Hz"),
    ("164600", "I = 10, 100 Hz"),
    ("164915", "100 Hz"),
    ("165541", "100 Hz, acceleration 50"),
    ("170217", "100 Hz, acc. 50, I = 10, P = 24, D = 48"),
    ("170356", "100 Hz, acc. 50, I = 10  (final)"),
]


def errors(model, path: Path) -> tuple[float, float]:
    log = np.load(path)
    desired = ee_positions(model, log["cmd_deg"]) * 1000  # mm
    real = ee_positions(model, log["meas_deg"]) * 1000
    tracking = np.linalg.norm(real - desired, axis=1).mean()
    # Closest point of the desired curve (subsampled: plenty of points per loop).
    shape = np.sqrt(((real[:, None, :] - desired[None, ::4, :]) ** 2).sum(-1)).min(1).mean()
    return tracking, shape


def main() -> None:
    model = load_model()
    labels, tracking, shape = [], [], []
    for suffix, label in RUNS:
        t, s = errors(model, OUTPUTS / f"real_run_20261101_{suffix}.npz")
        labels.append(label)
        tracking.append(t)
        shape.append(s)
        print(f"{label:<42} tracking {t:5.1f} mm   shape {s:4.1f} mm")

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True, facecolor=SURFACE)
    y = np.arange(len(labels))[::-1]
    panels = [
        (axes[0], tracking, "Tracking error (includes lag)"),
        (axes[1], shape, "Shape error (distance from the curve)"),
    ]
    for ax, values, title in panels:
        ax.set_facecolor(SURFACE)
        ax.barh(y, values, height=0.6, color=REAL)
        for yi, v in zip(y, values, strict=True):
            ax.text(v + 0.15, yi, f"{v:.1f}", va="center", fontsize=8.5, color=TEXT_2)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("mean error [mm]", color=TEXT_2)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=TEXT_2, labelsize=9)
        ax.set_xlim(0, max(values) * 1.15)
    axes[0].set_yticks(y, labels)
    fig.suptitle(
        "SO-101 lemniscate on the real robot: every tuning experiment (simulation: 0.15 mm)",
        x=0.02, ha="left", fontsize=13, color=TEXT,
    )
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "tuning_summary.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
