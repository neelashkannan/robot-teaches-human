"""``G1Teacher``: the simulated Unitree G1 that shows stances (implements ``RobotTeacher``).

The robot is posed kinematically (no physics): each update blends joints, root
height and root orientation from the start pose to the target with a min-jerk
profile, then re-grounds so a sole always rests on the floor mid-transition.
"""

from __future__ import annotations

import math
from typing import Self

import mujoco
import numpy as np

from rth.config import Settings, project_path
from rth.contracts import Joint
from rth.limbs import limb_directions
from rth.robot.model import (
    ROOT_NQ,
    booth_camera,
    build_model,
    keypoints_from_data,
    lowest_foot_z,
    render_bgr,
)
from rth.robot.poses import PoseBuilder, mirror_qpos
from rth.stances import StanceLibrary

NEUTRAL_STANCE = "ready"


def min_jerk(tau: float) -> float:
    """Min-jerk position profile on [0, 1]: zero velocity and acceleration at both ends."""
    t = min(max(tau, 0.0), 1.0)
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def slerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0.0:  # take the short way round
        b, d = -b, -d
    if d > 0.9995:
        q = a + s * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * a + math.sin(s * th) * b) / math.sin(th)


def blend_qpos(start: np.ndarray, target: np.ndarray, s: float) -> np.ndarray:
    """Pose a fraction ``s`` of the way from ``start`` to ``target`` (root quat slerped)."""
    q = start + s * (target - start)
    q[3:7] = slerp(start[3:7], target[3:7], s)
    return q


class G1Teacher:
    """Kinematic G1 in MuJoCo that moves between the stances in the library."""

    def __init__(self, settings: Settings, stances: StanceLibrary) -> None:
        self.settings = settings
        rs = settings.robot
        self.model = build_model(project_path(rs.model_path), samples=rs.render_samples)
        self.data = mujoco.MjData(self.model)
        self.builder = PoseBuilder(self.model, rs.clips_dir)
        self._poses: dict[str, np.ndarray] = {}
        self.load_errors: dict[str, str] = {}
        for s in stances.stances:
            try:
                self._poses[s.id] = self.builder.stance(s)
            except Exception as e:  # noqa: BLE001 - one bad stance mustn't stop the booth
                self.load_errors[s.id] = f"{type(e).__name__}: {e}"
                print(f"[robot] stance {s.id!r} unavailable: {self.load_errors[s.id]}")
        neutral = self._poses.get(NEUTRAL_STANCE)
        if neutral is None:
            neutral = self.builder.build({"source": "joints"})
        self._neutral = neutral
        self._start = neutral.copy()
        self._target = neutral.copy()
        self._t = 0.0
        self._duration = 0.0
        self.stance_id: str | None = NEUTRAL_STANCE if NEUTRAL_STANCE in self._poses else None
        self.mirrored = False
        self._renderer: mujoco.Renderer | None = None
        self._rgb: np.ndarray | None = None
        self._cached: tuple[tuple, np.ndarray] | None = None
        self._camera = self._make_camera()
        self._apply(neutral)

    # ------------------------------------------------------------------ poses

    def stance_ids(self) -> list[str]:
        return list(self._poses)

    def pose(self, stance_id: str, *, mirrored: bool = False) -> np.ndarray:
        """Target qpos of a stance (a copy)."""
        if stance_id not in self._poses:
            why = self.load_errors.get(stance_id, "unknown stance")
            raise KeyError(f"robot can't show {stance_id!r}: {why}")
        q = self._poses[stance_id]
        return mirror_qpos(q) if mirrored else q.copy()

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    # ------------------------------------------------------------------ motion

    def set_target(
        self, stance_id: str, *, mirrored: bool = False, duration_s: float = 2.0
    ) -> None:
        self.move_to(self.pose(stance_id, mirrored=mirrored), duration_s)
        self.stance_id, self.mirrored = stance_id, mirrored

    def go_neutral(self, duration_s: float = 1.5) -> None:
        self.move_to(self._neutral, duration_s)
        self.stance_id = NEUTRAL_STANCE if NEUTRAL_STANCE in self._poses else None
        self.mirrored = False

    def move_to(self, qpos: np.ndarray, duration_s: float) -> None:
        """Start a smooth move from wherever the robot is now to an arbitrary pose."""
        self._start = self.data.qpos.copy()
        self._target = np.asarray(qpos, float).copy()
        self._t = 0.0
        self._duration = max(float(duration_s), 0.0)
        if self._duration == 0.0:
            self._apply(self._target)

    def snap_to(self, qpos: np.ndarray) -> None:
        """Show a pose immediately (used by the viewer's clip scrubber)."""
        self.move_to(qpos, 0.0)

    def update(self, dt: float) -> None:
        self._t += max(float(dt), 0.0)
        if self._duration <= 0.0 or self._t >= self._duration:
            self._apply(self._target)
            return
        self._apply(blend_qpos(self._start, self._target, min_jerk(self._t / self._duration)))

    def at_target(self) -> bool:
        return self._duration <= 0.0 or self._t >= self._duration

    def _apply(self, qpos: np.ndarray) -> None:
        d = self.data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, d)
        dz = lowest_foot_z(self.model, d, self.builder.kin.index)
        if abs(dz) > 1e-6:
            d.qpos[2] -= dz
            mujoco.mj_kinematics(self.model, d)

    # ------------------------------------------------------------------ outputs

    def keypoints(self) -> dict[Joint, np.ndarray]:
        return keypoints_from_data(self.data, self.builder.kin.index)

    def limb_directions(self) -> dict:
        """Current limb directions in the robot's torso frame (convenience for UIs)."""
        return limb_directions(self.keypoints())

    def _make_camera(self) -> mujoco.MjvCamera:
        return booth_camera(self.settings.robot)

    def render(self, width: int, height: int) -> np.ndarray:
        """BGR uint8 (height, width, 3) image of the robot facing the viewer.

        While the robot holds still the last frame is reused, so holds cost nothing.
        """
        c = self._camera
        key = (
            width,
            height,
            self.data.qpos.tobytes(),
            c.distance,
            c.azimuth,
            c.elevation,
            tuple(c.lookat),
        )
        if self._cached is not None and self._cached[0] == key:
            return self._cached[1].copy()
        img = self._render(width, height)
        self._cached = (key, img)
        return img.copy()

    def _render(self, width: int, height: int) -> np.ndarray:
        r = self._renderer
        if r is None or (r.width, r.height) != (width, height):
            if r is not None:
                r.close()
            g = self.model.vis.global_
            if width > g.offwidth or height > g.offheight:
                raise ValueError(
                    f"render size {width}x{height} exceeds offscreen {g.offwidth}x{g.offheight}"
                )
            r = self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        return render_bgr(
            r,
            self.data,
            self._camera,
            shadows=self.settings.robot.render_shadows,
            out=self._rgb_buffer(width, height),
        )

    def _rgb_buffer(self, width: int, height: int) -> np.ndarray:
        if self._rgb is None or self._rgb.shape[:2] != (height, width):
            self._rgb = np.empty((height, width, 3), np.uint8)
        return self._rgb

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["NEUTRAL_STANCE", "ROOT_NQ", "G1Teacher", "blend_qpos", "min_jerk", "slerp"]
