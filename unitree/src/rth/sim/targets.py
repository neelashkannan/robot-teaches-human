"""Physical stance targets: the kinematic stance pose, made statically stable.

``PoseBuilder`` poses are grounded and face the camera, but nothing makes their
COM sit over the feet. ``stabilise`` checks the COM projection against the
support polygon and, if the margin is too small, finds the smallest joint change
that fixes it: an IK with the planted feet held, the COM moved to the nearest
point with enough margin, and a posture cost that makes the ankles by far the
cheapest joints. The result leans the whole body over the ankles, which leaves
every limb direction in the torso frame (what the visitor is scored on) intact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rth.limbs import limb_directions
from rth.robot.model import JOINT_NAMES, ROOT_NQ
from rth.robot.poses import mirror_qpos
from rth.sim.ik import Pose, Task, WholeBodyIK
from rth.sim.support import convex_hull, margin, nearest_inside

SUPPORT_HEIGHT_M = 0.01  # a sole this close to the floor counts as standing on it
MARGIN_DOUBLE_M = 0.03  # COM margin the adjusted pose keeps in double support
MARGIN_SINGLE_M = 0.02  # a G1 foot is only ~6 cm wide


def joint_costs(ankle: float, leg: float, hip_yaw: float, waist: float, arm: float) -> np.ndarray:
    """(29,) per-joint cost vector from per-group values."""
    c = np.empty(len(JOINT_NAMES))
    for i, n in enumerate(JOINT_NAMES):
        if "ankle" in n:
            c[i] = ankle
        elif "hip_yaw" in n:
            c[i] = hip_yaw
        elif "hip" in n or "knee" in n:
            c[i] = leg
        elif "waist" in n:
            c[i] = waist
        else:
            c[i] = arm
    return c


# Ankles cheap, arms nearly frozen: the adjustment is an ankle lean wherever it can be.
STABILISE_COST = joint_costs(ankle=0.05, leg=0.5, hip_yaw=1.0, waist=2.0, arm=10.0)


def yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def level(pose: Pose, floor_z: float) -> Pose:
    """The same foot flat on the floor, keeping only its heading."""
    h = yaw_of(pose.quat)
    return Pose(np.array([pose.pos[0], pose.pos[1], floor_z]), np.array([math.cos(h / 2), 0, 0, math.sin(h / 2)]))


class SoleGeometry:
    """The sole contact spheres of each foot, from the model."""

    def __init__(self, ik: WholeBodyIK) -> None:
        m = ik.model
        self.ik = ik
        self.local: list[np.ndarray] = []
        self.radius = 0.0
        for side, body in enumerate(m.site_bodyid[list(ik.sites)]):
            g = [i for i in ik.index.foot_spheres if m.geom_bodyid[i] == body]
            site_pos = m.site_pos[ik.sites[side]]
            self.local.append(m.geom_pos[g] - site_pos)
            self.radius = float(m.geom_size[g[0], 0])
        # Height of the foot site above a flat sole on the floor.
        self.site_height = float(-self.local[0][:, 2].min() + self.radius)

    def points(self, side: int) -> np.ndarray:
        """(4, 3) world sphere centres of one foot (after ``ik.run``)."""
        d, s = self.ik.data, self.ik.sites[side]
        return d.site_xpos[s] + self.local[side] @ d.site_xmat[s].reshape(3, 3).T

    def sole_points(self, pose: Pose, side: int) -> np.ndarray:
        """(4, 2) floor points of a foot placed at ``pose``."""
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, pose.quat)
        return (pose.pos + self.local[side] @ r.reshape(3, 3).T)[:, :2]

    def lowest(self, side: int) -> float:
        return float(self.points(side)[:, 2].min() - self.radius)


@dataclass
class StanceTarget:
    """Where the robot should end up for one stance (world frame, robot at the origin)."""

    stance_id: str
    mirrored: bool
    qpos: np.ndarray  # stabilised G1 qpos (36,)
    kinematic: np.ndarray  # the PoseBuilder pose it came from
    support: tuple[bool, bool]  # (left, right) feet on the floor
    feet: tuple[Pose, Pose]  # foot site poses at qpos
    com: np.ndarray  # (3,)
    pelvis: Pose
    margin_before: float
    margin_after: float
    adjustment_deg: dict[str, float] = field(default_factory=dict)

    @property
    def single_support(self) -> bool:
        return self.support[0] != self.support[1]

    @property
    def stance_side(self) -> int:
        """The supporting foot in single support (0 = left)."""
        return 0 if self.support[0] else 1


def support_hull(soles: SoleGeometry, feet: tuple[Pose | None, Pose | None]) -> np.ndarray:
    pts = [soles.sole_points(p, s) for s, p in enumerate(feet) if p is not None]
    return convex_hull(np.vstack(pts)) if pts else np.zeros((0, 2))


def _describe(ik: WholeBodyIK, soles: SoleGeometry, q: np.ndarray) -> tuple[tuple[bool, bool], tuple[Pose, Pose]]:
    ik.run(q)
    support = (soles.lowest(0) < SUPPORT_HEIGHT_M, soles.lowest(1) < SUPPORT_HEIGHT_M)
    return support, (ik.foot_pose(0), ik.foot_pose(1))


def narrowed(feet: tuple[Pose | None, Pose | None], max_width: float) -> tuple[Pose | None, Pose | None]:
    """Both planted feet pulled towards their midpoint until they are ``max_width`` apart."""
    if feet[0] is None or feet[1] is None:
        return feet
    gap = float(np.linalg.norm(feet[0].pos[:2] - feet[1].pos[:2]))
    if gap <= max_width:
        return feet
    mid = (feet[0].pos + feet[1].pos) / 2
    k = max_width / gap
    return tuple(Pose(mid + k * (f.pos - mid), f.quat.copy()) for f in feet)


def stabilise(
    ik: WholeBodyIK,
    soles: SoleGeometry,
    kin_q: np.ndarray,
    stance_id: str,
    *,
    max_width: float = np.inf,
) -> StanceTarget:
    """The smallest change to ``kin_q`` whose COM sits safely over its feet.

    "Smallest" is measured where it matters: limb directions in the torso
    frame, with a weak joint-angle cost to pick among equally good answers.
    Feet wider than ``max_width`` apart are first narrowed (see ``narrowed``).
    """
    support, feet = _describe(ik, soles, kin_q)
    planted = tuple(level(f, soles.site_height) if s else None for f, s in zip(feet, support))
    com = ik.com()
    before = margin(support_hull(soles, planted), com[:2])
    planted = narrowed(planted, max_width)
    hull = support_hull(soles, planted)
    want = MARGIN_DOUBLE_M if all(support) else MARGIN_SINGLE_M
    moved = any(
        float(np.linalg.norm(p.pos - f.pos)) > 1e-4 or abs(yaw_of(p.quat) - yaw_of(f.quat)) > 1e-4
        for p, f in zip(planted, feet)
        if p is not None
    )
    q = kin_q.copy()
    if before < want or moved:
        kp = {j: ik.data.xpos[b].copy() for j, b in ik.keypoint_bodies.items()}
        task = Task(
            posture=kin_q[ROOT_NQ:].copy(),
            posture_weight=STABILISE_COST * 0.02,
            feet=planted,
            com_xy=nearest_inside(hull, com[:2], want),
            limbs={k: v for k, v in limb_directions(kp).items() if v is not None},
            limb_weight=1.0,
        )
        q = ik.solve(kin_q, task, iters=400, tol=1e-7)
    _, feet = _describe(ik, soles, q)
    com = ik.com()
    delta = np.degrees(q[ROOT_NQ:] - kin_q[ROOT_NQ:])
    adjust = {JOINT_NAMES[i]: round(float(delta[i]), 1) for i in np.flatnonzero(np.abs(delta) >= 0.5)}
    return StanceTarget(
        stance_id=stance_id,
        mirrored=False,
        qpos=q,
        kinematic=kin_q.copy(),
        support=support,
        feet=feet,
        com=com,
        pelvis=Pose(q[:3].copy(), q[3:7].copy()),
        margin_before=before,
        margin_after=margin(hull, com[:2]),
        adjustment_deg=adjust,
    )


def mirror_target(ik: WholeBodyIK, soles: SoleGeometry, t: StanceTarget) -> StanceTarget:
    """The other-side version of a target (exactly mirrored, so equally stable)."""
    q = mirror_qpos(t.qpos)
    support, feet = _describe(ik, soles, q)
    com = ik.com()
    planted = tuple(level(f, soles.site_height) if s else None for f, s in zip(feet, support))
    hull = support_hull(soles, planted)
    adjust = {}
    for name, deg in t.adjustment_deg.items():
        other = name.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
        adjust[other] = -deg if name.endswith(("_roll_joint", "_yaw_joint")) else deg
    return StanceTarget(
        stance_id=t.stance_id,
        mirrored=not t.mirrored,
        qpos=q,
        kinematic=mirror_qpos(t.kinematic),
        support=support,
        feet=feet,
        com=com,
        pelvis=Pose(q[:3].copy(), q[3:7].copy()),
        margin_before=t.margin_before,
        margin_after=margin(hull, com[:2]),
        adjustment_deg=adjust,
    )
