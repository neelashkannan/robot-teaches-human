"""The G1 MuJoCo model: loading, joint tables, forward kinematics and keypoints.

Menagerie's ``unitree_g1`` is the 29-dof rev 1.0 G1. qpos layout (nq = 36):
``[x, y, z, qw, qx, qy, qz]`` for the floating base, then 29 hinge joints in
``JOINT_NAMES`` order. The robot faces +x with its left side towards +y.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from rth.config import RobotSettings
from rth.contracts import Joint

ROOT_NQ = 7  # free joint: position (3) + quaternion wxyz (4)

JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Bodies whose origin sits at the human-equivalent joint centre. The hip roll
# origin is where the thigh starts (the pitch origin is further in); the wrist
# pitch origin is the flexing wrist, past the forearm roll joint.
KEYPOINT_BODIES: dict[Joint, str] = {
    Joint.LEFT_SHOULDER: "left_shoulder_roll_link",
    Joint.RIGHT_SHOULDER: "right_shoulder_roll_link",
    Joint.LEFT_ELBOW: "left_elbow_link",
    Joint.RIGHT_ELBOW: "right_elbow_link",
    Joint.LEFT_WRIST: "left_wrist_pitch_link",
    Joint.RIGHT_WRIST: "right_wrist_pitch_link",
    Joint.LEFT_HIP: "left_hip_roll_link",
    Joint.RIGHT_HIP: "right_hip_roll_link",
    Joint.LEFT_KNEE: "left_knee_link",
    Joint.RIGHT_KNEE: "right_knee_link",
    Joint.LEFT_ANKLE: "left_ankle_pitch_link",
    Joint.RIGHT_ANKLE: "right_ankle_pitch_link",
}

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def configure_offscreen(spec: mujoco.MjSpec, offscreen: tuple[int, int] = (1280, 1280), samples: int = 4) -> None:
    """Make the offscreen buffer big enough for booth frames, with ``samples`` MSAA."""
    g = spec.visual.global_
    g.offwidth = max(g.offwidth, offscreen[0])
    g.offheight = max(g.offheight, offscreen[1])
    spec.visual.quality.offsamples = samples


def brighten_headlight(model: mujoco.MjModel) -> None:
    """The headlight sits at the camera, so this lights the robot's front from any view."""
    model.vis.headlight.diffuse[:] = 0.75
    model.vis.headlight.ambient[:] = 0.25


def build_model(
    path: str | Path, offscreen: tuple[int, int] = (1280, 1280), samples: int = 4
) -> mujoco.MjModel:
    """Compile the scene with a big enough offscreen buffer and a brighter headlight.

    Extra scene lights are deliberately not added: the headlight sits at the
    camera, so the robot's front is always lit, and every extra light costs
    render time.
    """
    spec = mujoco.MjSpec.from_file(str(path))
    configure_offscreen(spec, offscreen, samples)
    model = spec.compile()
    brighten_headlight(model)
    return model


def booth_camera(rs: RobotSettings) -> mujoco.MjvCamera:
    """The free camera the booth shows the robot through (RobotSettings camera fields)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, rs.camera_lookat_z)
    cam.distance = rs.camera_distance
    # 180 looks along -x at the robot's front; a few degrees off shows limb depth.
    cam.azimuth = rs.camera_azimuth_deg
    cam.elevation = rs.camera_elevation_deg
    return cam


def render_bgr(renderer: mujoco.Renderer, data: mujoco.MjData, camera: mujoco.MjvCamera,
               *, shadows: bool, out: np.ndarray | None = None) -> np.ndarray:
    """One BGR uint8 frame; shadows also switch the floor reflection (both are costly)."""
    renderer.update_scene(data, camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadows)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(shadows)
    rgb = renderer.render(out=out)
    return np.ascontiguousarray(rgb[:, :, ::-1])


@lru_cache(maxsize=4)
def _cached_model(path: str) -> mujoco.MjModel:
    return build_model(path)


def load_model(path: str | Path) -> mujoco.MjModel:
    """Shared compiled model per path (treat as read-only)."""
    return _cached_model(str(Path(path).resolve()))


@dataclass(frozen=True)
class G1Index:
    """Name -> index tables for one compiled model."""

    qpos_adr: dict[str, int]  # joint name -> qpos address
    lo: np.ndarray  # (nq,) lower limits; -inf/+inf for the root
    hi: np.ndarray
    keypoint_body: dict[Joint, int]
    foot_spheres: tuple[int, ...]  # geom ids of the sole contact spheres

    @staticmethod
    def of(model: mujoco.MjModel) -> G1Index:
        adr = {n: int(model.jnt_qposadr[model.joint(n).id]) for n in JOINT_NAMES}
        lo = np.full(model.nq, -np.inf)
        hi = np.full(model.nq, np.inf)
        for n, a in adr.items():
            lo[a], hi[a] = model.jnt_range[model.joint(n).id]
        kp = {j: model.body(b).id for j, b in KEYPOINT_BODIES.items()}
        feet = {model.body(b).id for b in FOOT_BODIES}
        spheres = tuple(
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] in feet and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        )
        return G1Index(adr, lo, hi, kp, spheres)


def neutral_qpos(model: mujoco.MjModel) -> np.ndarray:
    """All joints zero, upright, at the model's default height."""
    q = model.qpos0.copy()
    q[3:7] = (1.0, 0.0, 0.0, 0.0)
    return q


def keypoints_from_data(data: mujoco.MjData, index: G1Index) -> dict[Joint, np.ndarray]:
    return {j: data.xpos[b].copy() for j, b in index.keypoint_body.items()}


def lowest_foot_z(model: mujoco.MjModel, data: mujoco.MjData, index: G1Index) -> float:
    """Lowest point of the sole spheres (call after kinematics)."""
    g = list(index.foot_spheres)
    return float(np.min(data.geom_xpos[g, 2] - model.geom_size[g, 0]))


class Kinematics:
    """Scratch MjData for forward kinematics on arbitrary qpos vectors."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.index = G1Index.of(model)

    def run(self, qpos: np.ndarray) -> mujoco.MjData:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self.data

    def keypoints(self, qpos: np.ndarray) -> dict[Joint, np.ndarray]:
        return keypoints_from_data(self.run(qpos), self.index)

    def foot_z(self, qpos: np.ndarray) -> float:
        self.run(qpos)
        return lowest_foot_z(self.model, self.data, self.index)

    def ground(self, qpos: np.ndarray) -> np.ndarray:
        """Shift the root vertically so the lowest sole point touches z = 0."""
        q = qpos.copy()
        q[2] -= self.foot_z(q)
        return q
