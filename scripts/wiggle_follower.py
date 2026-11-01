"""Move the SO-101 follower without a leader arm.

Each joint in turn oscillates smoothly around its current position (sine
wave), then the arm goes back to where it started. Positions are in degrees
(0 = middle of the range set during calibration); the gripper is in 0-100
(% open).

Safety: keep the workspace clear and one hand near the power supply.
`max_relative_target` caps how far any joint can be commanded per step.

Usage:
    python scripts/wiggle_follower.py                      # all joints
    python scripts/wiggle_follower.py --joints elbow_flex gripper --amplitude 20

A detailed explanation of how the script works is at the bottom of this file.
"""

import argparse
import math
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# Joint names, as defined by LeRobot for the SO-101 (motor IDs 1 to 6).
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Command rate: one position command every 1/30 s (~33 ms).
FPS = 30


def move_to(robot, start: dict, target: dict, duration_s: float) -> None:
    """Linear interpolation from `start` to `target` over `duration_s`."""
    steps = max(1, int(duration_s * FPS))
    for i in range(1, steps + 1):
        # `a` goes from 0 to 1: each step moves a bit closer to the target.
        a = i / steps
        robot.send_action({k: start[k] + a * (target[k] - start[k]) for k in target})
        time.sleep(1 / FPS)


def main() -> None:
    # Command-line options, with defaults matching this project's setup.
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="so101_follower")
    parser.add_argument("--joints", nargs="+", default=JOINTS, choices=JOINTS)
    parser.add_argument("--amplitude", type=float, default=15.0, help="degrees (gripper: percent open)")
    parser.add_argument("--period", type=float, default=3.0, help="seconds per oscillation")
    args = parser.parse_args()

    # `id` selects the calibration file, so positions are in calibrated degrees.
    # `max_relative_target` clips every command to at most 10 degrees away from
    # the present position, which prevents sudden jumps.
    config = SO101FollowerConfig(port=args.port, id=args.id, max_relative_target=10.0)
    robot = SO101Follower(config)
    # Opens the serial port, checks the motors, writes the calibration to them
    # and enables torque: from now on the arm holds its position.
    robot.connect()

    # Read the present position of every joint and keep it as the reference
    # ("home") around which the joints oscillate.
    obs = robot.get_observation()
    home = {f"{j}.pos": obs[f"{j}.pos"] for j in JOINTS}
    print("Start position:", {k: round(v, 1) for k, v in home.items()})

    try:
        for joint in args.joints:
            key = f"{joint}.pos"
            print(f"Moving {joint} (±{args.amplitude})")
            # Number of commands for one full oscillation (3 s x 30 fps = 90).
            steps = int(args.period * FPS)
            for i in range(steps + 1):
                # Every command contains all six joints: the others stay at home.
                action = dict(home)
                # One full sine period: 0 -> +amplitude -> 0 -> -amplitude -> 0.
                offset = args.amplitude * math.sin(2 * math.pi * i / steps)
                # The gripper is in 0-100: keep it within range.
                action[key] = min(100.0, max(0.0, home[key] + offset)) if joint == "gripper" else home[key] + offset
                # Send the six goal positions in a single bus message.
                robot.send_action(action)
                time.sleep(1 / FPS)
            # Short pause between joints.
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        # Always runs (normal end, Ctrl+C or error): bring the arm back
        # smoothly to the start position before releasing it.
        obs = robot.get_observation()
        current = {k: obs[k] for k in home}
        print("Back to start position")
        move_to(robot, current, home, duration_s=2.0)
        # Disconnecting disables torque: the arm goes limp and falls under
        # gravity, so wait until it is held.
        input("Hold the arm (torque will be released) and press ENTER...")
        robot.disconnect()


if __name__ == "__main__":
    main()


# =============================================================================
# How this script works
# =============================================================================
#
# Key idea: position control
# --------------------------
# The SO-101 uses Feetech STS3215 servos in position mode. You do not tell a
# motor how fast to turn: you tell it which angle to reach (the "goal
# position"), and its internal controller (a PID loop reading the motor's own
# encoder) drives the shaft there.
#
# Moving the arm therefore means sending a stream of goal positions that are
# close to each other and frequent enough (30 per second here) to produce a
# smooth motion. This is exactly what the leader arm does during
# teleoperation, and what a trained policy such as ACT does at inference time:
# it outputs goal positions.
#
# Units
# -----
# Thanks to the calibration file (selected by `--id`), positions are not raw
# encoder ticks (0-4095 per turn) but degrees, with 0 at the middle of the
# range recorded during calibration. The gripper is the exception: 0-100,
# i.e. percent open.
#
# Flow
# ----
#   connect -> read start position ("home")
#     -> for each selected joint, 90 times:
#          compute position (sine wave) -> send -> wait 33 ms
#     -> move back to home in 2 s -> wait for ENTER -> release the motors
#
# 1. Connection. `SO101Follower` is the same class used by `lerobot-calibrate`
#    and `lerobot-record`. `connect()` opens the serial port, checks that all
#    motors answer, writes the calibration to them and enables torque.
#
# 2. Start position. `get_observation()` returns the present position of each
#    joint, e.g. {"shoulder_pan.pos": 2.3, "elbow_flex.pos": -10.5, ...}. It
#    is stored as `home`, the center of every oscillation.
#
# 3. Oscillation loop. For each joint, `i / steps` goes from 0 to 1, so
#    `2 * pi * i / steps` covers one full sine period. Multiplied by the
#    amplitude, the offset goes 0 -> +15 -> 0 -> -15 -> 0 degrees: the joint
#    starts at rest, swings one way, then the other, and ends where it
#    started. A sine is smoother than a step because it slows down near the
#    turning points. Each command carries all six joints: the ones not being
#    moved simply receive their home position. `send_action` writes all
#    goals in one `sync_write` message, so the motors move together.
#
# 4. Safety net. With `max_relative_target=10.0`, LeRobot reads the present
#    position before each command and clips any goal more than 10 degrees
#    away. The `finally` block guarantees that, whatever happens (normal end,
#    Ctrl+C, error), the arm is brought back to `home` with `move_to`, a
#    linear interpolation: position = start + a * (target - start), with `a`
#    going from 0 to 1 over 2 seconds (60 steps).
#
# 5. Release. `disconnect()` disables torque: the motors become free and the
#    arm falls under gravity unless it is held, hence the ENTER prompt.
#
# Link to the rest of the project
# -------------------------------
# The same "read -> decide -> send goal position -> wait" loop runs when a
# policy controls the robot. The only difference is the "decide" step: here a
# sine wave, there a neural network that looks at the webcam images and the
# joint positions.
