"""Render a still image of the simulated SO-101 on the lemniscate (for the README).

Usage (from the project root; d3d12 uses the integrated GPU under WSL):
    GALLIUM_DRIVER=d3d12 MUJOCO_GL=egl python lemniscate/render_sim.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lemniscate.trajectory import load_model  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent / "outputs"
FIGURES = Path(__file__).resolve().parent / "figures"
PATH_RGBA = np.array([0x2A / 255, 0x78 / 255, 0xD6 / 255, 1.0], dtype=np.float32)  # simulation blue


def main() -> None:
    traj = np.load(OUTPUTS / "trajectory.npz")
    q, path = traj["q_rad"], traj["path"]

    model = load_model()
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 1000
    data = mujoco.MjData(model)
    data.qpos[:] = q[len(q) // 8]  # gripper on the upper-right lobe
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.15, 0.0, 0.15]
    camera.distance, camera.azimuth, camera.elevation = 0.5, 135, -15

    with mujoco.Renderer(model, height=1000, width=1600) as renderer:
        renderer.update_scene(data, camera=camera)
        scn = renderer.scene
        for p in path[::4]:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.0025),
                p, np.eye(3).flatten(), PATH_RGBA,
            )
            scn.ngeom += 1
        image = renderer.render()

    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "sim_render.png"
    Image.fromarray(image).save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
