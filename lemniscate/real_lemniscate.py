"""Draw the lemniscate with the real SO-101 follower.

Replays the joint trajectory computed in simulation (`outputs/trajectory.npz`,
written by `sim_lemniscate.py`), so simulation and real robot receive exactly
the same commands. Joint zeros and directions were checked with
`check_joint_signs.py`: the trajectory is used without corrections.

Sequence:
  1. load the trajectory, convert radians -> LeRobot degrees, resample it to
     the requested period (slower = safer); the gripper stays where it is
  2. connect (every command is clipped to `--max-step` degrees from the
     present position)
  3. move slowly to the first point, then wait for ENTER
  4. run the loops at 30 Hz, logging commanded and measured positions
  5. at the end or on Ctrl+C: move back to the start pose, wait for ENTER,
     release torque
  6. save the log to `outputs/real_run_<date-time>.npz`

Usage (from the project root):
    python lemniscate/real_lemniscate.py                  # 16 s per loop, 3 loops
    python lemniscate/real_lemniscate.py --period 8 --loops 5
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# Command rate. 30 Hz is LeRobot's standard, but the servo reaches each new goal
# in a few ms and then waits: a 30 Hz stream of goals becomes a staircase. The
# bus allows ~5 ms per write+read cycle, so 100 Hz (--fps) is possible.
FPS = 30
OUTPUTS = Path(__file__).resolve().parent / "outputs"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def load_trajectory(path: Path, period_s: float, fps: int = FPS) -> np.ndarray:
    """Trajectory in LeRobot degrees, resampled to `period_s * fps` points per loop.

    Returns (n_points, 5): the five arm joints (the gripper is not commanded).
    """
    traj = np.load(path)
    q_deg = np.degrees(traj["q_rad"][:, :5])
    n_in = len(q_deg)
    n_out = int(period_s * fps)
    # Periodic interpolation: the loop closes on itself (point n_in == point 0).
    x_in = np.arange(n_in + 1)
    x_out = np.linspace(0, n_in, n_out, endpoint=False)
    closed = np.vstack([q_deg, q_deg[:1]])
    return np.column_stack([np.interp(x_out, x_in, closed[:, j]) for j in range(5)])


def to_action(q_deg: np.ndarray, gripper: float) -> dict:
    action = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, q_deg, strict=True)}
    action["gripper.pos"] = gripper
    return action


def read_arm(robot) -> np.ndarray:
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in ARM_JOINTS])


def move_to(
    robot, start: np.ndarray, target: np.ndarray, gripper: float, duration_s: float, fps: int = FPS
) -> None:
    """Linear interpolation in joint space from `start` to `target`."""
    steps = max(1, int(duration_s * fps))
    for i in range(1, steps + 1):
        t0 = time.perf_counter()
        robot.send_action(to_action(start + i / steps * (target - start), gripper))
        time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="so101_follower")
    parser.add_argument("--trajectory", type=Path, default=OUTPUTS / "trajectory.npz")
    parser.add_argument("--period", type=float, default=16.0, help="seconds per loop")
    parser.add_argument("--fps", type=int, default=FPS, help="commands per second (LeRobot standard 30)")
    parser.add_argument("--loops", type=int, default=3)
    parser.add_argument("--max-step", type=float, default=5.0, help="max degrees per command (safety)")
    parser.add_argument(
        "--p-gain", type=int, default=16, help="servo position P gain (LeRobot default 16, Feetech default 32)"
    )
    parser.add_argument(
        "--i-gain", type=int, default=0, help="servo position I gain (LeRobot default 0 = P-only control)"
    )
    parser.add_argument("--d-gain", type=int, default=32, help="servo position D gain (LeRobot default 32)")
    parser.add_argument(
        "--acceleration", type=int, default=254, help="servo acceleration 0-254 (LeRobot default 254 = max)"
    )
    args = parser.parse_args()

    q_traj = load_trajectory(args.trajectory, args.period, args.fps)
    print(f"Trajectory: {len(q_traj)} points per loop ({args.period:.0f} s), {args.loops} loops")

    # A higher P gain gives more torque for the same small error, so the joints
    # overcome static friction sooner (less stick-slip), at the risk of vibrating.
    # connect() writes it to the motors; it is not stored across connections.
    # The I gain accumulates the error over time, removing the steady-state error
    # caused by gravity and friction (P-only control leaves the arm sagging), at
    # the risk of overshoot and oscillation.
    # The D gain reacts to how fast the error changes: it damps oscillations,
    # but on a tick-quantised encoder it also brakes every small movement.
    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        max_relative_target=args.max_step,
        position_p_coefficient=args.p_gain,
        position_i_coefficient=args.i_gain,
        position_d_coefficient=args.d_gain,
    )
    robot = SO101Follower(config)
    robot.connect()
    # connect() sets the servos' acceleration to the maximum (254): every new
    # goal is reached with an abrupt start and stop, which turns a stream of
    # small goals into a staircase. A lower value smooths it, with more lag.
    # The register is in the servo's RAM: written with torque on, not persistent.
    for motor in robot.bus.motors:
        robot.bus.write("Acceleration", motor, args.acceleration)
    print(f"Servo P: {args.p_gain}, I: {args.i_gain}, D: {args.d_gain}, acceleration: {args.acceleration}")

    obs = robot.get_observation()
    home = read_arm(robot)
    gripper = obs["gripper.pos"]
    print("Start pose [deg]:", dict(zip(ARM_JOINTS, home.round(1), strict=True)))

    log_t, log_cmd, log_meas = [], [], []
    try:
        print("Moving to the first point of the lemniscate (5 s)...")
        move_to(robot, home, q_traj[0], gripper, duration_s=5.0, fps=args.fps)
        input("At the start point. Press ENTER to draw the lemniscate (Ctrl+C to stop)...")

        t_start = time.perf_counter()
        for k in range(args.loops * len(q_traj)):
            # Command k is due at t_start + k / fps: keeps the rate on average
            # even if a single cycle takes longer than 1 / fps.
            t_due = t_start + k / args.fps
            time.sleep(max(0.0, t_due - time.perf_counter()))
            cmd = q_traj[k % len(q_traj)]
            robot.send_action(to_action(cmd, gripper))
            meas = read_arm(robot)
            log_t.append(time.perf_counter() - t_start)
            log_cmd.append(cmd)
            log_meas.append(meas)
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        print("Back to the start pose (5 s)...")
        move_to(robot, read_arm(robot), home, gripper, duration_s=5.0, fps=args.fps)
        input("Hold the arm (torque will be released) and press ENTER...")
        robot.disconnect()

    if log_t:
        OUTPUTS.mkdir(exist_ok=True)
        out = OUTPUTS / f"real_run_{datetime.now():%Y%m%d_%H%M%S}.npz"
        t, cmd, meas = np.array(log_t), np.array(log_cmd), np.array(log_meas)
        np.savez(out, t=t, cmd_deg=cmd, meas_deg=meas, joints=ARM_JOINTS, period=args.period, fps=args.fps, p_gain=args.p_gain, i_gain=args.i_gain, d_gain=args.d_gain, acceleration=args.acceleration)
        err = np.abs(cmd - meas)
        print(f"Saved {out}")
        print(f"Achieved rate: {len(t) / t[-1]:.1f} Hz")
        print("Mean |commanded - measured| [deg]:", dict(zip(ARM_JOINTS, err.mean(0).round(2), strict=True)))


if __name__ == "__main__":
    main()
