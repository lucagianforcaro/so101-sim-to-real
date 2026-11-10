"""Measure the quality of joystick teleoperation logs (teleop_joystick.py --log).

For every log, only the stretches where a stick was moving the gripper are
used, resampled at 30 Hz (the dataset rate) so that sessions run at different
control rates are compared on the same grid. Gripper positions come from
forward kinematics of the commanded and of the measured joint angles.

Metrics:
  - cycle time (median, 95th percentile): is the requested rate kept?
  - lag: time shift that best aligns the measured gripper path with the commanded one
  - shape error: distance of the measured gripper from the commanded path, lag removed
  - jerkiness: RMS acceleration of the measured gripper / RMS acceleration of the
    commanded gripper (1 = as smooth as the command, higher = jerkier)
  - stalled intervals: 33 ms intervals where the gripper moved less than 20 % of
    what was commanded (the "staircase")

Usage (from the project root):
    python joystick/analyze_teleop.py                     # every log in joystick/outputs
    python joystick/analyze_teleop.py joystick/outputs/teleop_20261110_115016.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joystick.teleop_joystick import AXIS_X, AXIS_Y, AXIS_Z, DEADZONE, ArmKinematics  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs"
RATE = 30  # analysis grid [Hz]


def segments(t: np.ndarray, moving: np.ndarray, min_len: float = 0.5) -> list[tuple[float, float]]:
    """Time intervals where `moving` holds continuously for at least min_len seconds."""
    out, start = [], None
    for ti, m in zip(t, moving, strict=True):
        if m and start is None:
            start = ti
        elif not m and start is not None:
            if ti - start >= min_len:
                out.append((start, ti))
            start = None
    if start is not None and t[-1] - start >= min_len:
        out.append((start, t[-1]))
    return out


def analyse(path: Path, kin: ArmKinematics) -> dict:
    log = np.load(path)
    t, cmd, meas, axes = log["t"], log["cmd"], log["meas"], log["axes"]
    dt = np.diff(t)
    # Gaps (ready-pose moves, pauses) are not control cycles.
    cycles = dt[dt < 0.2]

    p_cmd = np.array([kin.fk(np.radians(c[:4])) for c in cmd]) * 1000  # mm
    p_meas = np.array([kin.fk(np.radians(m[:4])) for m in meas]) * 1000
    moving = np.any(np.abs(axes[:, [AXIS_X, AXIS_Y, AXIS_Z]]) > DEADZONE, axis=1)

    lags, shape_errs, acc_m, acc_c, stalled, n_int = [], [], [], [], 0, 0
    for a, b in segments(t, moving):
        grid = np.arange(a, b, 1 / RATE)
        if len(grid) < 10:
            continue
        pc = np.column_stack([np.interp(grid, t, p_cmd[:, i]) for i in range(3)])
        pm = np.column_stack([np.interp(grid, t, p_meas[:, i]) for i in range(3)])
        # Lag: shift (0-0.5 s) that best aligns measured with commanded.
        best = min(
            range(0, min(16, len(grid) - 5)),
            key=lambda k: np.mean(np.sum((pm[k:] - pc[: len(pc) - k]) ** 2, 1)),
        )
        lags.append(best / RATE)
        shape_errs.extend(np.linalg.norm(pm[best:] - pc[: len(pc) - best], axis=1))
        acc_m.extend(np.linalg.norm(np.diff(pm, 2, axis=0), axis=1))
        acc_c.extend(np.linalg.norm(np.diff(pc, 2, axis=0), axis=1))
        step_c = np.linalg.norm(np.diff(pc[: len(pc) - best], axis=0), axis=1)
        step_m = np.linalg.norm(np.diff(pm[best:], axis=0), axis=1)
        active = step_c > 0.5  # commanded motion of at least 0.5 mm in 33 ms
        stalled += int(np.sum(step_m[active] < 0.2 * step_c[active]))
        n_int += int(np.sum(active))

    acc_m, acc_c = np.array(acc_m), np.array(acc_c)
    return {
        "file": path.name,
        "fps": int(log["fps"]),
        "cycle_median_ms": float(np.median(cycles) * 1000),
        "cycle_p95_ms": float(np.percentile(cycles, 95) * 1000),
        "moving_s": float(sum(b - a for a, b in segments(t, moving))),
        "lag_ms": float(np.median(lags) * 1000) if lags else float("nan"),
        "shape_mm": float(np.mean(shape_errs)) if shape_errs else float("nan"),
        "jerk_ratio": float(np.sqrt(np.mean(acc_m**2)) / np.sqrt(np.mean(acc_c**2)))
        if len(acc_c)
        else float("nan"),
        "stalled_pct": 100 * stalled / n_int if n_int else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Motion-quality metrics of teleop_joystick.py --log files.")
    parser.add_argument(
        "logs", nargs="*", type=Path, help="log files (default: every log in joystick/outputs)"
    )
    files = parser.parse_args().logs or sorted(OUTPUTS.glob("teleop_*.npz"))
    kin = ArmKinematics()
    rows = [analyse(f, kin) for f in files]
    print(
        f"{'log':<28}{'fps':>5}{'cycle med/p95 [ms]':>21}{'moving [s]':>12}{'lag [ms]':>10}"
        f"{'shape [mm]':>12}{'jerk ratio':>12}{'stalled %':>11}"
    )
    for r in rows:
        print(
            f"{r['file']:<28}{r['fps']:>5}{r['cycle_median_ms']:>12.1f} / {r['cycle_p95_ms']:<6.1f}{r['moving_s']:>12.1f}"
            f"{r['lag_ms']:>10.0f}{r['shape_mm']:>12.2f}{r['jerk_ratio']:>12.2f}{r['stalled_pct']:>11.1f}"
        )


if __name__ == "__main__":
    main()
