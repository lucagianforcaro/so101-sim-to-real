"""Lemniscate trajectory and inverse kinematics for the SO-101 (MuJoCo model).

Shared by the simulation script and the real-robot script, so that both run
exactly the same joint trajectory.

Adapted from my ETH robot-learning course HW2 (SO-100): same lemniscate of
Bernoulli and Damped Least Squares IK, ported to the SO-101 model from
TheRobotStudio/SO-ARM100 (`so101_new_calib`, whose joint zeros sit at the
middle of each joint's range like LeRobot's calibration).
"""

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "third_party/SO-ARM100/Simulation/SO101/scene.xml"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Joints used to reach a 3D position. wrist_roll only changes the gripper
# orientation and the gripper does not move the end-effector site: both stay fixed.
IK_JOINTS = [0, 1, 2, 3]
EE_SITE = "gripperframe"


def lemniscate(t: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Lemniscate of Bernoulli (infinity sign), parametrised by t in [0, 2*pi)."""
    y = a * np.cos(t) / (1 + np.sin(t) ** 2)
    z = a * np.cos(t) * np.sin(t) / (1 + np.sin(t) ** 2)
    return y, z


def build_path(n_points: int, width: float = 0.08, x: float = 0.25, z_center: float = 0.18) -> np.ndarray:
    """End-effector positions (n_points, 3) along a lemniscate in a vertical plane.

    The robot faces +x: the figure lies in the y-z plane, `x` metres in front
    of the base, centred at height `z_center`.
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    y, z = lemniscate(t, a=width)
    return np.column_stack([np.full_like(y, x), y, z + z_center])


def ik_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    q_init: np.ndarray,
    q_ref: np.ndarray | None = None,
    damping: float = 1e-4,
    gain: float = 0.5,
    null_gain: float = 0.1,
    tol: float = 1e-4,
    max_iters: int = 500,
) -> tuple[np.ndarray, float]:
    """Damped Least Squares IK on the end-effector position only.

    Starts from `q_init` (the previous solution along the path, which keeps the
    trajectory continuous) and returns (q, final position error in metres).
    Uses a scratch copy of `data`'s state: the caller's state is restored.

    The 4 IK joints for a 3D position leave one redundant degree of freedom.
    If `q_ref` is given, a null-space term uses it to pull the joints towards
    `q_ref` without moving the end effector, so the posture depends only on the
    target, not on the path taken to reach it (a closed path gives a closed
    joint trajectory).
    """
    saved_qpos = data.qpos.copy()
    q = q_init.copy()
    lo, hi = model.jnt_range[:, 0], model.jnt_range[:, 1]
    jacp = np.zeros((3, model.nv))

    for _ in range(max_iters):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        err = target_pos - data.site(EE_SITE).xpos
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, jacp, None, model.site(EE_SITE).id)
        J = jacp[:, IK_JOINTS]  # noqa: N806 (math name)
        # Damped pseudo-inverse: J+ = J^T (J J^T + lambda I)^-1
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))  # noqa: N806 (math name)
        dq = J_pinv @ err
        # Null-space term: (I - J+ J) only keeps joint motions that do not move
        # the end effector, and uses them to get closer to q_ref.
        dq_null = np.zeros(len(IK_JOINTS))
        if q_ref is not None:
            dq_null = (np.eye(len(IK_JOINTS)) - J_pinv @ J) @ (null_gain * (q_ref - q)[IK_JOINTS])
        # Converged when the end effector is on target and the posture has settled.
        if np.linalg.norm(err) < tol and np.linalg.norm(dq_null) < 1e-6:
            break
        q[IK_JOINTS] += gain * dq + dq_null
        q = np.clip(q, lo, hi)

    data.qpos[:] = saved_qpos
    mujoco.mj_kinematics(model, data)
    return q, float(np.linalg.norm(err))


def solve_trajectory(
    model: mujoco.MjModel, path: np.ndarray, q_start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """IK for every point of `path`, warm-started from the previous solution.

    Returns joint positions (n_points, 6) in radians and the IK errors (n_points,).
    """
    data = mujoco.MjData(model)
    q = q_start.copy()
    # Converge on the first point from the start pose (may be far away).
    q, _ = ik_position(model, data, path[0], q, max_iters=5000)
    # The posture at the first point is the preferred posture for the whole path.
    q_ref = q.copy()
    qs, errs = [], []
    for p in path:
        q, err = ik_position(model, data, p, q, q_ref=q_ref, max_iters=2000)
        qs.append(q.copy())
        errs.append(err)
    return np.array(qs), np.array(errs)


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))
