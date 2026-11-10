"""Draw a lemniscate with the simulated SO-101 in MuJoCo.

1. Builds the lemniscate path and solves the IK for every point (offline).
2. Saves the joint trajectory to `lemniscate/outputs/trajectory.npz`, the file
   the real-robot script will replay.
3. Plays the trajectory in the MuJoCo viewer with physics: the joint targets go
   to the position actuators (like the real servos), at the same 30 Hz rate
   used on the real robot. Yellow dots: the desired path; red trail: where the
   gripper actually went.

Usage (from the project root):
    python lemniscate/sim_lemniscate.py
    python lemniscate/sim_lemniscate.py --period 6 --width 0.06 --no-viewer
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lemniscate.trajectory import EE_SITE, JOINTS, build_path, load_model, solve_trajectory  # noqa: E402

FPS = 30
OUTPUT = Path(__file__).resolve().parent / "outputs" / "trajectory.npz"


def draw_points(scn: mujoco.MjvScene, points, radius: float, rgba, start: int) -> int:
    """Add spheres to the viewer's user scene from index `start`; returns the next free index."""
    n = start
    for p in points:
        if n >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[n],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius),
            np.asarray(p, dtype=np.float64),
            np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        n += 1
    scn.ngeom = n
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=float, default=8.0, help="seconds per lemniscate loop")
    parser.add_argument("--width", type=float, default=0.08, help="lemniscate half-width [m]")
    parser.add_argument("--x", type=float, default=0.25, help="distance in front of the base [m]")
    parser.add_argument("--z", type=float, default=0.18, help="height of the centre [m]")
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--no-viewer", action="store_true", help="only compute and save the trajectory")
    args = parser.parse_args()

    model = load_model()
    n_points = int(args.period * FPS)
    path = build_path(n_points, width=args.width, x=args.x, z_center=args.z)
    qs, errs = solve_trajectory(model, path, q_start=np.zeros(model.nq))
    print(f"{n_points} points, max IK error {errs.max() * 1000:.2f} mm")
    print("Joint range used [deg]:")
    for name, lo, hi in zip(JOINTS, np.degrees(qs.min(0)), np.degrees(qs.max(0)), strict=True):
        print(f"  {name:<14} {lo:7.1f} .. {hi:7.1f}")

    OUTPUT.parent.mkdir(exist_ok=True)
    np.savez(OUTPUT, q_rad=qs, path=path, fps=FPS, joints=JOINTS)
    print(f"Saved {OUTPUT}")
    if args.no_viewer:
        return

    data = mujoco.MjData(model)
    # Start the simulation on the first point of the trajectory.
    data.qpos[:] = qs[0]
    data.ctrl[:] = qs[0]
    mujoco.mj_forward(model, data)
    substeps = round(1 / (FPS * model.opt.timestep))  # physics steps per command
    trail: list[np.ndarray] = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(args.loops):
            for q in qs:
                if not viewer.is_running():
                    return
                t0 = time.perf_counter()
                data.ctrl[:] = q  # joint targets to the position actuators
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                trail.append(data.site(EE_SITE).xpos.copy())
                n = draw_points(viewer.user_scn, path, 0.003, (1, 1, 0, 1), 0)
                draw_points(viewer.user_scn, trail[-300:], 0.002, (1, 0, 0, 1), n)
                viewer.sync()
                time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))

        tracking = np.linalg.norm(np.array(trail[-n_points:]) - path, axis=1)
        print(
            f"Tracking error last loop: mean {tracking.mean() * 1000:.1f} mm, max {tracking.max() * 1000:.1f} mm"
        )
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
