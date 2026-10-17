"""Stance poses: parse a stance's ``pose:`` spec and build a normalised G1 qpos.

Two sources (see config/stances.yaml)::

    pose: {source: joints, degrees: true, joints: {left_knee_joint: 60, ...}}
    pose: {source: clip, file: fight1_subject2, frame: 1234}

Optional keys for both: ``overrides`` (joint -> angle, applied last; for a clip
these are in radians unless ``degrees``), ``mirror`` (store the other-side
version). ``joints`` also takes ``root_rpy`` (roll, pitch, yaw of the pelvis).
Unlisted joints are 0. Every pose is then normalised: clamped to joint limits,
turned to face the viewer camera (+x), root x/y at the origin, lowest sole on z = 0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from rth.limbs import torso_frame
from rth.robot.clips import load_clip
from rth.robot.model import JOINT_NAMES, ROOT_NQ, Kinematics
from rth.stances import Stance

_JOINT_SET = frozenset(JOINT_NAMES)


def _check_joint_names(d: dict[str, float]) -> dict[str, float]:
    unknown = sorted(set(d) - _JOINT_SET)
    if unknown:
        raise ValueError(f"unknown G1 joints {unknown}; known: {list(JOINT_NAMES)}")
    return d


class _PoseBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degrees: bool = False  # angles in this spec are degrees instead of radians
    overrides: dict[str, float] = Field(default_factory=dict)
    mirror: bool = False

    check_overrides = field_validator("overrides")(_check_joint_names)

    def scale(self) -> float:
        return math.pi / 180.0 if self.degrees else 1.0


class JointsPose(_PoseBase):
    source: Literal["joints"]
    joints: dict[str, float] = Field(default_factory=dict)
    root_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    check_joints = field_validator("joints")(_check_joint_names)


class ClipPose(_PoseBase):
    source: Literal["clip"]
    file: str
    frame: int = Field(ge=0)


PoseSpec = Annotated[JointsPose | ClipPose, Field(discriminator="source")]
_ADAPTER: TypeAdapter[JointsPose | ClipPose] = TypeAdapter(PoseSpec)


def parse_pose_spec(spec: dict[str, Any]) -> JointsPose | ClipPose:
    """Validate a ``pose:`` mapping (raises pydantic.ValidationError when malformed)."""
    return _ADAPTER.validate_python(spec)


# --------------------------------------------------------------------------- quaternions (wxyz)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def axis_quat(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4)
    q[0] = math.cos(angle / 2)
    q[1 + axis] = math.sin(angle / 2)
    return q


def rpy_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic x-y-z (yaw applied last, about world z)."""
    return quat_mul(axis_quat(2, yaw), quat_mul(axis_quat(1, pitch), axis_quat(0, roll)))


def quat_rpy(q: np.ndarray) -> tuple[float, float, float]:
    """Inverse of ``rpy_quat``."""
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, float))
    r = r.reshape(3, 3)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = -math.asin(float(np.clip(r[2, 0], -1.0, 1.0)))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


# --------------------------------------------------------------------------- mirroring

# Roll and yaw axes point along x and z, so reflecting the body through its
# sagittal (x-z) plane negates them; pitch axes lie along y and keep their sign.
_FLIP = ("_roll_joint", "_yaw_joint")


def _mirror_tables() -> tuple[np.ndarray, np.ndarray]:
    n = ROOT_NQ + len(JOINT_NAMES)
    src = np.arange(n)
    sign = np.ones(n)
    pos = {name: ROOT_NQ + i for i, name in enumerate(JOINT_NAMES)}
    for name, i in pos.items():
        if name.startswith("left_"):
            other = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            other = "left_" + name[len("right_") :]
        else:
            other = name
        src[i] = pos[other]
        if name.endswith(_FLIP):
            sign[i] = -1.0
    # Root: y -> -y; reflecting a rotation through the x-z plane negates qx and qz.
    sign[1] = -1.0
    sign[4] = -1.0
    sign[6] = -1.0
    return src, sign


_MIRROR_SRC, _MIRROR_SIGN = _mirror_tables()


def mirror_qpos(qpos: np.ndarray) -> np.ndarray:
    """The left/right mirror image of a G1 qpos. ``mirror_qpos(mirror_qpos(q)) == q``."""
    q = np.asarray(qpos, float)
    return q[_MIRROR_SRC] * _MIRROR_SIGN


# --------------------------------------------------------------------------- building


def heading(kin: Kinematics, qpos: np.ndarray) -> float:
    """Yaw of the torso-frame forward axis (0 = facing +x, towards the camera)."""
    frame = torso_frame(kin.keypoints(qpos))
    if frame is None:
        return 0.0
    f = frame[0]
    return math.atan2(f[1], f[0])


def normalise(kin: Kinematics, qpos: np.ndarray) -> np.ndarray:
    """Clamp to joint limits, face +x, root x/y = 0, lowest sole on the floor."""
    idx = kin.index
    q = np.clip(np.asarray(qpos, float), idx.lo, idx.hi)
    q[3:7] /= np.linalg.norm(q[3:7])
    q[3:7] = quat_mul(axis_quat(2, -heading(kin, q)), q[3:7])
    q[0:2] = 0.0
    return kin.ground(q)


class PoseBuilder:
    """Builds normalised qpos vectors from pose specs."""

    def __init__(self, model: mujoco.MjModel, clips_dir: str | Path) -> None:
        self.kin = Kinematics(model)
        self.clips_dir = clips_dir

    @property
    def nq(self) -> int:
        return self.kin.model.nq

    def raw(self, spec: JointsPose | ClipPose) -> np.ndarray:
        """qpos straight from the spec, before normalisation."""
        adr = self.kin.index.qpos_adr
        k = spec.scale()
        if isinstance(spec, ClipPose):
            clip = load_clip(spec.file, self.clips_dir, adr, self.nq)
            if spec.frame >= len(clip):
                raise ValueError(f"frame {spec.frame} out of range for {spec.file} ({len(clip)})")
            q = clip[spec.frame].copy()
        else:
            q = np.zeros(self.nq)
            q[2] = 0.8
            q[3:7] = rpy_quat(*(a * k for a in spec.root_rpy))
            for name, a in spec.joints.items():
                q[adr[name]] = a * k
        for name, a in spec.overrides.items():
            q[adr[name]] = a * k
        return mirror_qpos(q) if spec.mirror else q

    def build(self, spec: JointsPose | ClipPose | dict[str, Any]) -> np.ndarray:
        if isinstance(spec, dict):
            spec = parse_pose_spec(spec)
        return normalise(self.kin, self.raw(spec))

    def stance(self, stance: Stance) -> np.ndarray:
        """Normalised qpos for a stance; raises if its pose is missing or won't load."""
        if not stance.pose:
            raise ValueError(f"stance {stance.id!r} has no pose")
        return self.build(stance.pose)

    def clip_frame(self, clip: str, frame: int) -> np.ndarray:
        return self.build(ClipPose(source="clip", file=clip, frame=frame))


def clip_pose_snippet(clip: str, frame: int, *, mirror: bool = False) -> str:
    """YAML ``pose:`` line for a clip frame, ready to paste into config/stances.yaml."""
    extra = ", mirror: true" if mirror else ""
    return f"pose: {{source: clip, file: {clip}, frame: {frame}{extra}}}"


def joints_pose_snippet(qpos: np.ndarray, *, min_abs_deg: float = 0.5) -> str:
    """YAML ``pose:`` block (degrees) for a qpos, listing only non-trivial joints."""
    lines = ["pose:", "  source: joints", "  degrees: true"]
    rpy = [round(math.degrees(a), 1) for a in quat_rpy(qpos[3:7])]
    rpy[2] = 0.0  # heading is normalised away anyway
    if any(abs(a) >= min_abs_deg for a in rpy):
        lines.append(f"  root_rpy: {rpy}")
    lines.append("  joints:")
    for i, name in enumerate(JOINT_NAMES):
        deg = math.degrees(float(qpos[ROOT_NQ + i]))
        if abs(deg) >= min_abs_deg:
            lines.append(f"    {name}: {deg:.1f}")
    if lines[-1] == "  joints:":
        lines[-1] = "  joints: {}"
    return "\n".join(lines)
