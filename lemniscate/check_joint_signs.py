"""Digital twin: mirror the real SO-101 follower in the MuJoCo model.

Torque is disabled, so the arm can be moved by hand; the MuJoCo model shows
the same joint angles. Used to check, before sending the simulated trajectory
to the real robot, that every joint has the same zero and the same direction
in simulation and in LeRobot's calibration.

The robot receives no motion commands: this script only reads.

Usage (from the project root, holding the arm: it goes limp):
    python lemniscate/check_joint_signs.py
    python lemniscate/check_joint_signs.py --port $FOLLOWER_PORT
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lemniscate.trajectory import JOINTS, load_model  # noqa: E402

RATE_HZ = 20
PRINT_EVERY_S = 0.5


def lerobot_to_sim(obs: dict, model: mujoco.MjModel) -> np.ndarray:
    """LeRobot observation (degrees, gripper 0-100) -> MuJoCo qpos (radians)."""
    q = np.radians([obs[f"{j}.pos"] for j in JOINTS[:5]])
    # Gripper: linear map of 0-100 % onto the model's joint range (visual only).
    lo, hi = model.jnt_range[JOINTS.index("gripper")]
    gripper = lo + obs["gripper.pos"] / 100 * (hi - lo)
    return np.append(q, gripper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="so101_follower")
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    # connect() ends with torque enabled: disable it so the arm moves by hand.
    robot.connect()
    robot.bus.disable_torque()

    model = load_model()
    data = mujoco.MjData(model)
    last_print = 0.0

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.perf_counter()
                obs = robot.get_observation()
                # Kinematics only: set the angles and update the poses, no physics.
                data.qpos[:] = lerobot_to_sim(obs, model)
                mujoco.mj_forward(model, data)
                viewer.sync()

                if t0 - last_print > PRINT_EVERY_S:
                    print("  ".join(f"{j[:11]:>11} {obs[f'{j}.pos']:6.1f}" for j in JOINTS))
                    last_print = t0
                time.sleep(max(0.0, 1 / RATE_HZ - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
