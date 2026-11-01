"""Compare the lemniscate drawn by the real SO-101 with the simulated one.

1. Loads a real run log (`outputs/real_run_*.npz`, the latest by default).
2. Real: the measured joint angles go through MuJoCo forward kinematics to get
   where the gripper actually was.
3. Simulation: the same commands, at the same rate, are replayed in MuJoCo with
   physics (position actuators), like `sim_lemniscate.py`.
4. Desired: forward kinematics of the commanded joint angles (the lemniscate).
5. Plots the three paths in the y-z plane, the tracking error over time and a
   close-up of one joint, saves the figure next to the log and prints the errors.

Usage (from the project root):
    python lemniscate/compare_sim_real.py
    python lemniscate/compare_sim_real.py --run lemniscate/outputs/real_run_20261101_154232.npz
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lemniscate.trajectory import EE_SITE, load_model  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs"

# Colors (validated categorical palette, light mode) and text/axis tokens.
DESIRED = "#898781"
SIM = "#2a78d6"
REAL = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def ee_positions(model: mujoco.MjModel, q_deg: np.ndarray) -> np.ndarray:
    """Forward kinematics: (N, 5) arm joint angles in degrees -> (N, 3) gripper positions."""
    data = mujoco.MjData(model)
    out = np.empty((len(q_deg), 3))
    for i, q in enumerate(q_deg):
        data.qpos[:5] = np.radians(q)
        mujoco.mj_kinematics(model, data)
        out[i] = data.site(EE_SITE).xpos
    return out


def simulate(model: mujoco.MjModel, cmd_deg: np.ndarray, fps: int) -> np.ndarray:
    """Replay the commands with physics; returns the simulated joint angles (N, 5) in degrees."""
    data = mujoco.MjData(model)
    data.qpos[:5] = np.radians(cmd_deg[0])
    data.ctrl[:5] = np.radians(cmd_deg[0])
    mujoco.mj_forward(model, data)
    substeps = round(1 / (fps * model.opt.timestep))
    out = np.empty_like(cmd_deg)
    for i, cmd in enumerate(cmd_deg):
        data.ctrl[:5] = np.radians(cmd)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        out[i] = np.degrees(data.qpos[:5])
    return out


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)
    ax.title.set_color(TEXT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None, help="real run log (default: latest)")
    parser.add_argument("--joint", default="elbow_flex", help="joint shown in the close-up")
    args = parser.parse_args()

    run_path = args.run or max(OUTPUTS.glob("real_run_*.npz"))
    log = np.load(run_path)
    t, cmd, meas = log["t"], log["cmd_deg"], log["meas_deg"]
    joints = list(log["joints"])
    fps, period = int(log["fps"]), float(log["period"])
    settings = f"P={int(log['p_gain']) if 'p_gain' in log else 16}"
    settings += f", I={int(log['i_gain']) if 'i_gain' in log else 0}"
    settings += f", D={int(log['d_gain']) if 'd_gain' in log else 32}"
    settings += f", acceleration={int(log['acceleration']) if 'acceleration' in log else 254}"
    print(f"Run: {run_path.name} ({period:.0f} s per loop, {settings})")

    model = load_model()
    sim = simulate(model, cmd, fps)
    p_des, p_sim, p_real = (ee_positions(model, q) * 100 for q in (cmd, sim, meas))  # cm
    err_sim = np.linalg.norm(p_sim - p_des, axis=1) * 10  # mm
    err_real = np.linalg.norm(p_real - p_des, axis=1) * 10

    print("Tracking error [mm]   mean    max")
    print(f"  simulation        {err_sim.mean():6.2f} {err_sim.max():6.2f}")
    print(f"  real robot        {err_real.mean():6.2f} {err_real.max():6.2f}")

    plt.rcParams["font.family"] = "sans-serif"
    # The lemniscate is wide and short: full width on top, the two time plots below.
    fig = plt.figure(figsize=(12, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.22)
    ax_path = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_joint = fig.add_subplot(gs[1, 1])
    for ax in (ax_path, ax_err, ax_joint):
        style(ax)

    # 1. Paths in the y-z plane (front view), last loop only.
    n_loop = int(period * fps)
    last = slice(len(t) - n_loop, len(t))
    ax_path.plot(p_des[last, 1], p_des[last, 2], color=DESIRED, lw=1.5, ls="--", label="Desired", zorder=3)
    ax_path.plot(p_sim[last, 1], p_sim[last, 2], color=SIM, lw=2, label="Simulation")
    ax_path.plot(p_real[last, 1], p_real[last, 2], color=REAL, lw=2, label="Real robot")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.set_xlabel("y [cm]")
    ax_path.set_ylabel("z [cm]")
    ax_path.set_title("Gripper path, front view (last loop)", loc="left", fontsize=11)
    ax_path.legend(frameon=False, labelcolor=TEXT_2, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    # 2. Tracking error over time.
    ax_err.plot(t, err_sim, color=SIM, lw=1.5, label="Simulation")
    ax_err.plot(t, err_real, color=REAL, lw=1.5, label="Real robot")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("distance from desired [mm]")
    ax_err.set_ylim(bottom=0)
    ax_err.set_title(
        f"Tracking error (mean: sim {err_sim.mean():.2f} mm, real {err_real.mean():.1f} mm)", loc="left", fontsize=11
    )
    ax_err.legend(frameon=False, labelcolor=TEXT_2, fontsize=9, loc="upper left")

    # 3. Close-up of one joint over 2 s: the real servo moves in steps.
    j = joints.index(args.joint)
    win = (t >= t[-1] - period / 2) & (t < t[-1] - period / 2 + 2.0)
    ax_joint.plot(t[win], cmd[win, j], color=DESIRED, lw=1.5, ls="--", label="Commanded", zorder=3)
    ax_joint.plot(t[win], sim[win, j], color=SIM, lw=2, label="Simulation")
    ax_joint.step(t[win], meas[win, j], where="post", color=REAL, lw=2, label="Real robot (measured)")
    ax_joint.set_xlabel("time [s]")
    ax_joint.set_ylabel(f"{args.joint} [deg]")
    ax_joint.set_title(f"{args.joint}, 2 s close-up", loc="left", fontsize=11)
    ax_joint.legend(frameon=False, labelcolor=TEXT_2, fontsize=9, loc="best")

    fig.suptitle(
        f"SO-101 lemniscate: simulation vs real robot   ({period:.0f} s per loop, {settings})",
        x=0.07, ha="left", fontsize=13, color=TEXT,
    )
    out = run_path.with_name(run_path.stem.replace("real_run", "compare") + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
