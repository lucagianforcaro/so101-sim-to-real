"""Teleoperate the SO-101 follower with a USB gamepad (Trust Predator).

The sticks command the gripper's VELOCITY in Cartesian space; the script
integrates it into a target position, solves the inverse kinematics in MuJoCo
and streams the joint angles to the follower through LeRobot.

Controls (mapping found with joystick/js_test.py):
    left stick  up/down      gripper forward / back      (axis 1)
    left stick  left/right   gripper left / right        (axis 0)
    right stick up/down      gripper up / down           (axis 2)
    d-pad       up/down      gripper pitch (tilt)        (axis 5)
    R1 / L1     hold         close / open the gripper    (buttons 7 / 6)
    R2 / L2     hold         wrist roll, one direction each (buttons 5 / 4)
    Start                    precision mode on / off (slow) (button 9)
    triangle                 go to the ready pose        (button 0)
                             (arm straight ahead, gripper 20 cm in front, 12 cm high, open)
    square                   straighten the gripper to vertical, gradually (button 3)
    Ctrl+C                   stop

Grasping from above: keep the gripper vertical. Going down with a tilted
gripper folds the arm (shoulder swings, elbow at its limit): such motions are
stopped by the elbow guard.

Inverse kinematics: 4 joints (pan, lift, elbow, wrist_flex) for 4 targets
(x, y, z and the gripper's pitch). On the SO-101, lift, elbow and wrist_flex
rotate about parallel axes, so the gripper's pitch is exactly
-(lift + elbow + wrist_flex). With as many targets as joints there is one
solution near the current pose: no drifting posture and no flips between
elbow-up / elbow-down configurations.

Safety: target clipped to a workspace box; a target the IK cannot reach is
not followed (the gripper stops at the edge); every command is clipped to
10 degrees from the present position; a joystick disconnection stops the
motion; on exit the arm is released only after ENTER.

Usage (from the project root; arm and gamepad attached to WSL):
    python joystick/teleop_joystick.py
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from joystick.js_reader import Joystick  # noqa: E402
from lemniscate.trajectory import EE_SITE, load_model  # noqa: E402

FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042597-if00"
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]  # IK joints (MuJoCo indices 0-3)

# Gamepad mapping (Trust Predator, from js_test.py).
AXIS_X, AXIS_Y, AXIS_Z, AXIS_PITCH = 1, 0, 2, 5
BTN_CLOSE, BTN_OPEN, BTN_HOME, BTN_VERTICAL = 7, 6, 0, 3
BTN_ROLL_NEG, BTN_ROLL_POS = (
    4,
    5,
)  # L2 / R2: wrist roll, one direction each (not on a stick, to avoid mixing it up)
BTN_PRECISION = 9  # Start: toggles the slow precision mode
DEADZONE = 0.12

# Speeds.
SPEED = 0.08  # m/s at full stick
SLOW_FACTOR = 0.3  # precision mode (Start)
PITCH_SPEED = np.radians(40)  # rad/s
ROLL_SPEED = 60.0  # deg/s (buttons: fixed speed)
# Wrist roll range, measured with scripts/test_wrist_roll.py: a mechanical stop
# (by design) at about +14 deg; free for 150 deg above. LeRobot maps the full
# turn to [-180, 180]: stay away from both the stop and the wrap-around.
ROLL_MIN, ROLL_MAX = 20.0, 170.0
ROLL_SIGN = -1.0  # R2 -> roll direction the operator expects (L2 the other way)
GRIPPER_SPEED = 60.0  # %/s

# Box the gripper must stay in [m], robot base frame (x forward, y left, z up).
# The gripper point is ~0.7 cm above the fingertips: z = 1 cm leaves the
# fingertips ~3 mm above the table, enough to grasp a 2.5 cm cube at mid height.
WORKSPACE_MIN = np.array([0.10, -0.25, 0.01])
WORKSPACE_MAX = np.array([0.40, 0.25, 0.30])  # above ~30 cm the arm is nearly stretched (singular)
PITCH_MIN, PITCH_MAX = np.radians(-95), np.radians(30)
# Normal motion moves a joint by < 2 deg per cycle; near a singularity a small
# gripper step needs a big joint rotation: such IK solutions are not followed.
MAX_JOINT_STEP = np.radians(3)
# Keeping the gripper tilted while going down to the table folds the arm (the
# shoulder swings ~100 deg and the elbow reaches its 90 deg limit): motions
# that bring the elbow beyond this angle are not followed.
ELBOW_MAX = np.radians(85)
VERTICAL = np.radians(-90)  # gripper pointing down: the posture for grasping from above

# Ready pose: arm straight ahead (pan 0), gripper 20 cm in front of the base and
# 12 cm above the table, open. At this height the gripper cannot be vertical:
# it is tilted to -70 deg (wrist_flex 88.6 deg, 6.4 deg from its limit) and
# straightens by itself going down (the desired pitch stays vertical).
READY = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -35.3,
    "elbow_flex": 16.8,
    "wrist_flex": 88.6,
    "wrist_roll": None,
    "gripper": 70.0,
}  # wrist_roll None: keep the present roll (0 deg is beyond the stop)


def stick(value: float) -> float:
    """Dead zone, then rescale so the output still goes smoothly from 0 to 1."""
    if abs(value) < DEADZONE:
        return 0.0
    return np.sign(value) * (abs(value) - DEADZONE) / (1 - DEADZONE)


class ArmKinematics:
    """Forward and inverse kinematics of the 4 IK joints, in the MuJoCo model."""

    def __init__(self):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.lo = self.model.jnt_range[:4, 0]
        self.hi = self.model.jnt_range[:4, 1]
        self.jacp = np.zeros((3, self.model.nv))
        self.site = self.model.site(EE_SITE).id

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Gripper position for the 4 arm joints q [rad] (wrist_roll and gripper at 0)."""
        self.data.qpos[:] = 0
        self.data.qpos[:4] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy()

    @staticmethod
    def pitch(q: np.ndarray) -> float:
        return -(q[1] + q[2] + q[3])

    def ik(
        self, q0: np.ndarray, pos: np.ndarray, pitch: float, iters: int = 60, w_pitch: float = 0.1
    ) -> tuple[np.ndarray, float, float]:
        """Damped least squares on [x, y, z, pitch] from q0. Returns (q, position error [m], pitch error [rad]).

        w_pitch: weight of the pitch target. 0.1 = 1 rad of pitch error weighs like
        10 cm of position error (pitch effectively enforced); a small value lets the
        pitch give way when the position cannot be reached with it.
        """
        q = q0.copy()
        for _ in range(iters):
            err_pos = pos - self.fk(q)
            err_pitch = pitch - self.pitch(q)
            if np.linalg.norm(err_pos) < 2e-4 and abs(err_pitch) < 2e-3:
                break
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_jacSite(self.model, self.data, self.jacp, None, self.site)
            J = np.vstack([self.jacp[:, :4], w_pitch * np.array([0.0, -1.0, -1.0, -1.0])])  # noqa: N806 (math name)
            e = np.append(err_pos, w_pitch * err_pitch)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(4), e)
            q = np.clip(q + 0.8 * dq, self.lo, self.hi)
        err_pos = pos - self.fk(q)
        return q, float(np.linalg.norm(err_pos)), float(abs(pitch - self.pitch(q)))


def solve_step(kin: ArmKinematics, q: np.ndarray, target: np.ndarray, pitch: float) -> np.ndarray | None:
    """Joint angles for the next gripper target, or None if it must not be followed.

    Position has priority over pitch: first try to reach the target with the
    desired pitch; if that is impossible (e.g. gripper vertical too high up),
    try again with a much lower pitch weight, so the gripper tilts only as much
    as needed. The desired pitch is not changed, so the gripper goes back to it
    by itself where it is reachable again (e.g. down near the table).
    The solution is not followed if it misses the target, jerks a joint or folds
    the elbow beyond ELBOW_MAX.
    """
    for w_pitch, max_pitch_err in ((0.1, np.radians(2)), (0.0005, np.inf)):
        q_new, err_pos, err_pitch = kin.ik(q, target, pitch, w_pitch=w_pitch)
        if (
            err_pos < 0.003
            and err_pitch < max_pitch_err
            and np.max(np.abs(q_new - q)) < MAX_JOINT_STEP
            and q_new[2] < ELBOW_MAX
        ):
            return q_new
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=FOLLOWER_PORT)
    parser.add_argument("--id", default="so101_follower")
    parser.add_argument(
        "--fps",
        type=int,
        default=100,
        help="control rate; 100 Hz is much smoother than 30 (see analyze_teleop.py)",
    )
    parser.add_argument("--acceleration", type=int, default=50, help="servo acceleration (lemniscate: 50)")
    parser.add_argument("--i-gain", type=int, default=10, help="servo I gain (lemniscate: 10, no sag)")
    parser.add_argument(
        "--log", action="store_true", help="log every cycle to joystick/outputs/teleop_<date>.npz"
    )
    args = parser.parse_args()
    dt = 1.0 / args.fps

    # Lowest gripper height: measured on the real table (calibrate_table.py) if available.
    table_file = Path(__file__).resolve().parent / "table.json"
    if table_file.exists():
        WORKSPACE_MIN[2] = json.loads(table_file.read_text())["z_min"]
        print(f"Table calibration: lowest gripper height {WORKSPACE_MIN[2] * 100:.2f} cm")
    else:
        print(
            f"No table calibration (run joystick/calibrate_table.py): lowest gripper height "
            f"{WORKSPACE_MIN[2] * 100:.2f} cm from the model"
        )

    # Ready pose (triangle): taught by hand with teach_ready_pose.py if available.
    ready_file = Path(__file__).resolve().parent / "ready.json"
    ready_pose = dict(READY)
    if ready_file.exists():
        ready_pose = json.loads(ready_file.read_text())["pose"]
        print("Ready pose: taught (ready.json)")
    else:
        print("Ready pose: default (teach one with joystick/teach_ready_pose.py)")

    kin = ArmKinematics()
    js = Joystick().open()
    print(f"Gamepad: {len(js.axes)} axes, {len(js.buttons)} buttons")

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port, id=args.id, max_relative_target=10.0, position_i_coefficient=args.i_gain
        )
    )
    robot.connect()
    for motor in robot.bus.motors:
        robot.bus.write("Acceleration", motor, args.acceleration)

    def read_joints() -> dict:
        obs = robot.get_observation()
        return {name: obs[f"{name}.pos"] for name in READY}

    def send(cmd: dict) -> None:
        robot.send_action({f"{name}.pos": float(v) for name, v in cmd.items()})

    def move_to(target: dict, duration: float = 2.5) -> None:
        start = read_joints()
        steps = int(duration / dt)
        for i in range(1, steps + 1):
            send({n: start[n] + i / steps * (target[n] - start[n]) for n in target})
            time.sleep(dt)

    # Start from where the arm is: no jump when the loop begins (the roll is
    # brought inside its free range if it is outside).
    cmd = read_joints()
    cmd["wrist_roll"] = float(np.clip(cmd["wrist_roll"], ROLL_MIN, ROLL_MAX))
    q = np.radians([cmd[n] for n in ARM])
    target = kin.fk(q)
    pitch = kin.pitch(q)
    print(
        "Ready. Press triangle first (grasp-ready pose). Sticks: move; square: gripper vertical; Ctrl+C: stop."
    )

    last_print = 0.0
    straighten = False  # square pressed: bring the gripper to vertical
    precision, precision_prev = False, False
    log = {k: [] for k in ("t", "cmd", "meas", "target", "pitch_desired", "axes", "buttons")}
    t_start = time.perf_counter()
    try:
        while True:
            t0 = time.perf_counter()
            js.poll()
            if not js.connected:
                print("\nGamepad disconnected: motion stopped.")
                break
            ax = lambda n: js.axes.get(n, 0.0)  # noqa: E731
            btn = lambda n: js.buttons.get(n, False)  # noqa: E731

            if btn(BTN_HOME):
                print("\nGoing to the ready pose...")
                roll = ready_pose["wrist_roll"]
                if roll is None:  # default pose: keep the present roll
                    roll = read_joints()["wrist_roll"]
                move_to({**ready_pose, "wrist_roll": float(np.clip(roll, ROLL_MIN, ROLL_MAX))})
                cmd = read_joints()
                q = np.radians([cmd[n] for n in ARM])
                # Desired pitch: vertical. Not reachable at the ready height (the
                # gripper stays at about -70), it is reached by itself going down.
                target, pitch = kin.fk(q), VERTICAL
                straighten = False
                continue

            # Start toggles the precision mode (on the press, not while held).
            if btn(BTN_PRECISION) and not precision_prev:
                precision = not precision
                print(f"\nPrecision mode {'ON' if precision else 'OFF'}")
            precision_prev = btn(BTN_PRECISION)
            scale = SLOW_FACTOR if precision else 1.0
            # Stick up gives a negative value: invert so that "up" = forward / up.
            velocity = scale * SPEED * np.array([-stick(ax(AXIS_X)), -stick(ax(AXIS_Y)), -stick(ax(AXIS_Z))])
            new_target = np.clip(target + velocity * dt, WORKSPACE_MIN, WORKSPACE_MAX)

            # Pitch: d-pad, or square to straighten the gripper (pointing down) gradually.
            if btn(BTN_VERTICAL):
                straighten = True
            if ax(AXIS_PITCH) != 0:
                straighten = False  # the d-pad takes over
            if straighten:
                step = 2 * PITCH_SPEED * dt
                new_pitch = pitch + float(np.clip(VERTICAL - pitch, -step, step))
                straighten = abs(VERTICAL - new_pitch) > 1e-3
            else:
                new_pitch = pitch - scale * PITCH_SPEED * ax(AXIS_PITCH) * dt
            new_pitch = float(np.clip(new_pitch, PITCH_MIN, PITCH_MAX))

            # `pitch` is the DESIRED pitch: the actual one gives way where it cannot be kept (see solve_step).
            # Nothing commanded (sticks at rest, pitch unchanged): no IK, the arm holds exactly
            # where it is (otherwise a soft-pitch solution would slowly drift towards vertical).
            if np.any(new_target != target) or new_pitch != pitch:
                pitch = new_pitch
                q_new = solve_step(kin, q, new_target, pitch)
                if q_new is not None:
                    q, target = q_new, new_target
                # else: target not reachable, only with a jerk, or by folding the arm
                # -> keep the previous one (the gripper stops).

            cmd.update({n: float(np.degrees(v)) for n, v in zip(ARM, q, strict=True)})
            roll_dir = (1.0 if btn(BTN_ROLL_POS) else 0.0) - (1.0 if btn(BTN_ROLL_NEG) else 0.0)
            roll = cmd["wrist_roll"] + ROLL_SIGN * scale * ROLL_SPEED * roll_dir * dt
            cmd["wrist_roll"] = float(np.clip(roll, ROLL_MIN, ROLL_MAX))
            grip = (1.0 if btn(BTN_OPEN) else 0.0) - (1.0 if btn(BTN_CLOSE) else 0.0)
            cmd["gripper"] = float(np.clip(cmd["gripper"] + GRIPPER_SPEED * grip * dt, 0, 100))
            send(cmd)

            if args.log:
                meas = read_joints()
                log["t"].append(t0 - t_start)
                log["cmd"].append([cmd[n] for n in READY])
                log["meas"].append([meas[n] for n in READY])
                log["target"].append(target.copy())
                log["pitch_desired"].append(pitch)
                log["axes"].append([ax(n) for n in range(6)])
                log["buttons"].append([btn(n) for n in range(12)])

            if t0 - last_print > 0.5:
                print(
                    f"\rgripper [cm] x {target[0] * 100:5.1f}  y {target[1] * 100:5.1f}  z {target[2] * 100:5.1f}   "
                    f"pitch {np.degrees(kin.pitch(q)):6.1f}  roll {cmd['wrist_roll']:6.1f}  grip {cmd['gripper']:5.1f}"
                    + ("  [slow]" if scale < 1 else "       "),
                    end="",
                    flush=True,
                )
                last_print = t0
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        js.close()
        if args.log and log["t"]:
            out_dir = Path(__file__).resolve().parent / "outputs"
            out_dir.mkdir(exist_ok=True)
            out = out_dir / f"teleop_{datetime.now():%Y%m%d_%H%M%S}.npz"
            np.savez(
                out,
                **{k: np.array(v) for k, v in log.items()},
                joints=list(READY),
                fps=args.fps,
                acceleration=args.acceleration,
                i_gain=args.i_gain,
            )
            print(f"Saved {out} ({len(log['t'])} cycles)")
        input("Hold the arm (torque will be released) and press ENTER...")
        robot.disconnect()


if __name__ == "__main__":
    main()
