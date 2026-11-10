"""Measure the table height in the robot's frame, by touching it with the gripper.

The model's z = 0 is the bottom of the robot base. If the robot does not stand
directly on the surface the objects are on (clamp, plate, support), the real
table is lower or higher than that, and a fixed lower limit for the gripper is
wrong. This script measures it with the robot itself:

  1. torque is released: the arm can be moved by hand
  2. bring the gripper, pointing down (vertical), with the fingertips just
     touching the table; press ENTER (3 points spread over the work area)
  3. the gripper point height at each touch is computed with forward
     kinematics; the lowest allowed gripper height is the highest touch + 3 mm
  4. saved to joystick/table.json, read by teleop_joystick.py

Re-run it whenever the robot or the table changes.

Usage (from the project root):
    python joystick/calibrate_table.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joystick.teleop_joystick import ARM, FOLLOWER_PORT, ArmKinematics  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "table.json"
N_POINTS = 3
MARGIN = 0.003  # fingertips this far above the table at the lowest allowed height


def main() -> None:
    kin = ArmKinematics()
    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="so101_follower"))
    robot.connect()
    input("Hold the arm: torque will be released. Press ENTER...")
    robot.bus.disable_torque()

    touches = []
    try:
        for i in range(N_POINTS):
            input(
                f"\nPoint {i + 1}/{N_POINTS}: gripper vertical, fingertips just touching the table "
                "(spread the points over the work area). Press ENTER..."
            )
            obs = robot.get_observation()
            q = np.radians([obs[f"{n}.pos"] for n in ARM])
            pos, pitch = kin.fk(q), np.degrees(kin.pitch(q))
            print(
                f"  gripper point at x {pos[0] * 100:.1f}  y {pos[1] * 100:.1f}  z {pos[2] * 100:.2f} cm, pitch {pitch:.0f} deg"
            )
            if abs(pitch + 90) > 10:
                print(
                    "  The gripper is not vertical (pitch should be about -90): the fingertips are not the lowest "
                    "point. Repeat this point."
                )
                continue
            touches.append(pos)
    finally:
        robot.disconnect()

    if not touches:
        print("No valid point: nothing saved.")
        return
    z = np.array([p[2] for p in touches])
    z_min = float(z.max() + MARGIN)
    print(f"\nTouch heights [cm]: {np.round(z * 100, 2)}  (spread {np.ptp(z) * 100:.2f} cm)")
    if np.ptp(z) > 0.01:
        print("The points differ by more than 1 cm: is the table tilted, or was one touch not on the table?")
    print(f"Lowest allowed gripper height: {z_min * 100:.2f} cm (model default was 1.00 cm)")
    OUTPUT.write_text(
        json.dumps(
            {
                "z_min": z_min,
                "touches": [p.tolist() for p in touches],
                "margin": MARGIN,
                "date": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
