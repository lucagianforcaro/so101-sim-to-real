"""Teach the ready pose (triangle in teleop_joystick.py) by moving the arm by hand.

  1. torque is released: the arm can be moved by hand
  2. bring the arm to the pose you want to start every teleoperation from,
     and press ENTER
  3. the joint angles are read and checked (gripper inside the workspace box,
     wrist roll inside its free range, elbow and wrist away from their limits)
  4. saved to joystick/ready.json, read by teleop_joystick.py

Usage (from the project root; hold the arm, it goes limp):
    python joystick/teach_ready_pose.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joystick.teleop_joystick import (  # noqa: E402
    ARM,
    ELBOW_MAX,
    FOLLOWER_PORT,
    READY,
    ROLL_MAX,
    ROLL_MIN,
    WORKSPACE_MAX,
    WORKSPACE_MIN,
    ArmKinematics,
)

OUTPUT = Path(__file__).resolve().parent / "ready.json"
TABLE = Path(__file__).resolve().parent / "table.json"


def main() -> None:
    kin = ArmKinematics()
    z_min = json.loads(TABLE.read_text())["z_min"] if TABLE.exists() else WORKSPACE_MIN[2]
    box_min = np.array([WORKSPACE_MIN[0], WORKSPACE_MIN[1], z_min])

    robot = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="so101_follower"))
    robot.connect()
    input("Hold the arm: torque will be released. Press ENTER...")
    robot.bus.disable_torque()
    try:
        input("Move the arm to the ready pose you want, then press ENTER...")
        obs = robot.get_observation()
    finally:
        print("Hold the arm: releasing the connection.")
        robot.disconnect()

    pose = {name: float(obs[f"{name}.pos"]) for name in READY}
    q = np.radians([pose[n] for n in ARM])
    pos, pitch = kin.fk(q), np.degrees(kin.pitch(q))
    print("Joints [deg]:", {k: round(v, 1) for k, v in pose.items()})
    print(
        f"Gripper at x {pos[0] * 100:.1f}  y {pos[1] * 100:.1f}  z {pos[2] * 100:.1f} cm, pitch {pitch:.0f} deg"
    )

    problems = []
    if np.any(pos < box_min) or np.any(pos > WORKSPACE_MAX):
        problems.append(
            f"gripper outside the workspace box {np.round(box_min * 100, 1)} .. {np.round(WORKSPACE_MAX * 100, 1)} cm"
        )
    if not ROLL_MIN <= pose["wrist_roll"] <= ROLL_MAX:
        problems.append(
            f"wrist roll {pose['wrist_roll']:.0f} outside its free range {ROLL_MIN:.0f}..{ROLL_MAX:.0f}"
        )
    if q[2] >= ELBOW_MAX:
        problems.append(f"elbow {np.degrees(q[2]):.0f} deg beyond {np.degrees(ELBOW_MAX):.0f} (arm folded)")
    margin = np.degrees(kin.hi[3] - q[3])
    if margin < 3:
        problems.append(f"wrist_flex only {margin:.1f} deg from its limit")
    if problems:
        print("Not saved:\n  - " + "\n  - ".join(problems))
        return

    OUTPUT.write_text(
        json.dumps({"pose": pose, "date": datetime.now().isoformat(timespec="seconds")}, indent=2)
    )
    print(f"Saved {OUTPUT}: triangle will now go to this pose.")


if __name__ == "__main__":
    main()
