"""Diagnose the wrist roll: move it slowly and log position, goal and load.

Only wrist_roll moves (the other joints hold their position). It goes from
its present position down by --range degrees and back, at --speed deg/s,
while printing the commanded and measured angle and the motor's load and
current. Where the joint stops:

  - load/current rise while the position stays still -> the motor is pushing
    against something (e.g. the gripper motor cable): a mechanical resistance
  - load/current stay low while the position stays still -> the motor is not
    executing the command

Usage (from the project root; watch the gripper and its cable, Ctrl+C to stop):
    python scripts/test_wrist_roll.py
    python scripts/test_wrist_roll.py --range 60 --direction up
"""

import argparse
import time

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042597-if00"
FPS = 30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--range", type=float, default=60.0, help="degrees to travel")
    parser.add_argument("--speed", type=float, default=5.0, help="deg/s")
    parser.add_argument("--direction", choices=["down", "up"], default="down")
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id="so101_follower"))
    robot.connect()
    obs = robot.get_observation()
    hold = {k: v for k, v in obs.items() if k.endswith(".pos")}
    start = hold["wrist_roll.pos"]
    sign = -1 if args.direction == "down" else 1
    n = int(args.range / args.speed * FPS)
    goals = (
        np.concatenate([np.linspace(0, sign * args.range, n), np.linspace(sign * args.range, 0, n)]) + start
    )
    print(f"wrist_roll from {start:.1f} to {start + sign * args.range:.1f} deg and back, {args.speed} deg/s")
    print("   goal   measured   error   load   current")

    try:
        for i, goal in enumerate(goals):
            t0 = time.perf_counter()
            robot.send_action({**hold, "wrist_roll.pos": float(goal)})
            if i % (FPS // 2) == 0:
                pos = robot.get_observation()["wrist_roll.pos"]
                load = robot.bus.read("Present_Load", "wrist_roll", normalize=False)
                current = robot.bus.read("Present_Current", "wrist_roll", normalize=False)
                print(f"{goal:7.1f}   {pos:7.1f}   {goal - pos:6.1f}   {load:5d}   {current:5d}")
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        input("Hold the arm (torque will be released) and press ENTER...")
        robot.disconnect()


if __name__ == "__main__":
    main()
