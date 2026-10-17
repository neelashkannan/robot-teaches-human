"""Whole-body inverse kinematics for the G1 (damped least squares on a scratch MjData).

Works on the G1-only model and its 36-long qpos (floating base + 29 joints), so it
doesn't care what else the dojo scene contains. Tasks: foot site poses, the COM
over the floor, pelvis height and orientation, limb directions in the torso frame
(what the visitor is scored on), and a posture to stay close to. Feet and COM are
weighted far above the rest, so they behave like constraints while the other
tasks decide everything they leave free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from rth.contracts import Limb
from rth.limbs import torso_frame
from rth.robot.model import KEYPOINT_BODIES, ROOT_NQ, G1Index

FOOT_SITES = ("left_foot", "right_foot")
NJ = 29  # actuated joints
BASE_NV = 6

_W_FOOT_POS = 1e3
_W_FOOT_ROT = 3e2
_W_COM = 1e3
_DAMPING = 1e-4
_MAX_STEP = 0.25  # rad (or m) per iteration, keeps Gauss-Newton from overshooting


@dataclass
class Pose:
    """A frame in the world: position (3,) and quaternion wxyz (4,)."""

    pos: np.ndarray
    quat: np.ndarray

    def copy(self) -> Pose:
        return Pose(self.pos.copy(), self.quat.copy())


@dataclass
class Task:
    """What the IK should achieve. ``None`` entries are left free."""

    posture: np.ndarray  # (29,) preferred joint angles
    posture_weight: np.ndarray  # (29,)
    feet: tuple[Pose | None, Pose | None] = (None, None)  # left, right foot site
    com_xy: np.ndarray | None = None
    pelvis_z: float | None = None
    pelvis_quat: np.ndarray | None = None
    pelvis_weight: tuple[float, float] = (0.0, 0.0)  # height, orientation
    foot_weight: tuple[float, float] = field(default=(1.0, 1.0))
    limbs: dict[Limb, np.ndarray] | None = None  # unit directions in the torso frame
    limb_weight: float = 0.0


def quat_error(target: np.ndarray, current: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``current`` to ``target``."""
    v = np.zeros(3)
    mujoco.mju_subQuat(v, np.asarray(target, float), np.asarray(current, float))
    return current_mat @ v


def mat_quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, float).reshape(9))
    return q


class WholeBodyIK:
    """IK and static linearisation for one G1-only model."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.index = G1Index.of(model)
        self.sites = tuple(model.site(s).id for s in FOOT_SITES)
        self.pelvis = model.body("pelvis").id
        self.keypoint_bodies = {j: model.body(b).id for j, b in KEYPOINT_BODIES.items()}
        self.torso = model.body("torso_link").id
        self.lo = self.index.lo[ROOT_NQ:] + 1e-3
        self.hi = self.index.hi[ROOT_NQ:] - 1e-3
        nv = model.nv
        self._jp = np.zeros((3, nv))
        self._jr = np.zeros((3, nv))

    # ------------------------------------------------------------------ kinematics

    def run(self, qpos: np.ndarray) -> mujoco.MjData:
        """Positions, COM and motion subspaces for ``qpos`` (needed by every Jacobian)."""
        d = self.data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d

    def foot_pose(self, side: int) -> Pose:
        """World pose of a foot site after ``run``."""
        s = self.sites[side]
        return Pose(self.data.site_xpos[s].copy(), mat_quat(self.data.site_xmat[s]))

    def com(self) -> np.ndarray:
        return self.data.subtree_com[self.pelvis].copy()

    def foot_jac(self, side: int) -> np.ndarray:
        """(6, nv) linear then angular Jacobian of a foot site."""
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.sites[side])
        return np.vstack([self._jp, self._jr])

    def com_jac(self) -> np.ndarray:
        mujoco.mj_jacSubtreeCom(self.model, self.data, self._jp, self.pelvis)
        return self._jp.copy()

    def gravity(self) -> np.ndarray:
        """Generalised gravity force (nv,) at the current pose, for zero velocity."""
        out = np.zeros(self.model.nv)
        self.data.qvel[:] = 0.0
        mujoco.mj_rne(self.model, self.data, 0, out)
        return out

    # ------------------------------------------------------------------ solving

    def solve(self, qpos: np.ndarray, task: Task, iters: int = 10, tol: float = 1e-5) -> np.ndarray:
        """Refine ``qpos`` towards the task; returns the new qpos (36,)."""
        q = np.asarray(qpos, float).copy()
        m = self.model
        for _ in range(iters):
            d = self.run(q)
            hess = np.eye(m.nv) * _DAMPING
            grad = np.zeros(m.nv)
            worst = 0.0
            for side, target in enumerate(task.feet):
                if target is None:
                    continue
                s = self.sites[side]
                jac = self.foot_jac(side)
                err = np.concatenate(
                    [target.pos - d.site_xpos[s], quat_error(target.quat, mat_quat(d.site_xmat[s]), d.site_xmat[s].reshape(3, 3))]
                )
                k = task.foot_weight[side]
                w = np.array([_W_FOOT_POS] * 3 + [_W_FOOT_ROT] * 3) * k
                hess += jac.T @ (w[:, None] * jac)
                grad += jac.T @ (w * err)
                worst = max(worst, float(np.abs(err[:3]).max()))
            if task.com_xy is not None:
                jac = self.com_jac()[:2]
                err = task.com_xy - d.subtree_com[self.pelvis][:2]
                hess += _W_COM * jac.T @ jac
                grad += _W_COM * jac.T @ err
                worst = max(worst, float(np.abs(err).max()))
            wz, wr = task.pelvis_weight
            if task.pelvis_z is not None and wz > 0.0:
                # The free joint's first three dofs are the pelvis's world linear velocity.
                hess[2, 2] += wz
                grad[2] += wz * (task.pelvis_z - q[2])
            if task.pelvis_quat is not None and wr > 0.0:
                mujoco.mj_jacBody(m, d, None, self._jr, self.pelvis)
                err = quat_error(task.pelvis_quat, q[3:7], d.xmat[self.pelvis].reshape(3, 3))
                hess += wr * self._jr.T @ self._jr
                grad += wr * self._jr.T @ err
            if task.limbs and task.limb_weight > 0.0:
                self._limb_terms(task, hess, grad)
            pw = task.posture_weight
            hess[BASE_NV:, BASE_NV:] += np.diag(pw)
            grad[BASE_NV:] += pw * (task.posture - q[ROOT_NQ:])
            dq = np.linalg.solve(hess, grad)
            big = float(np.abs(dq).max())
            if big > _MAX_STEP:
                dq *= _MAX_STEP / big
            mujoco.mj_integratePos(m, q, dq, 1.0)
            q[ROOT_NQ:] = np.clip(q[ROOT_NQ:], self.lo, self.hi)
            if worst < tol and big < tol:
                break
        return q

    def _limb_terms(self, task: Task, hess: np.ndarray, grad: np.ndarray) -> None:
        """Gauss-Newton terms pulling each limb towards its torso-frame direction.

        The torso frame turns with the trunk, so each term also carries the
        trunk's angular Jacobian (mean of pelvis and torso, as the frame spans
        hips to shoulders); without it the solve settles on a rotated answer.
        """
        m, d = self.model, self.data
        kp = {j: d.xpos[b] for j, b in self.keypoint_bodies.items()}
        frame = torso_frame(kp)
        if frame is None:
            return
        jac_a = np.zeros((3, m.nv))
        jac_b = np.zeros((3, m.nv))
        trunk = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, None, trunk, self.pelvis)
        mujoco.mj_jacBody(m, d, None, self._jr, self.torso)
        trunk = (trunk + self._jr) / 2
        w = task.limb_weight
        for limb, want in task.limbs.items():
            a, b = limb.joints
            v = kp[b] - kp[a]
            n = float(np.linalg.norm(v))
            u = v / n
            mujoco.mj_jacBody(m, d, jac_a, None, self.keypoint_bodies[a])
            mujoco.mj_jacBody(m, d, jac_b, None, self.keypoint_bodies[b])
            skew = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
            jac = (np.eye(3) - np.outer(u, u)) @ (jac_b - jac_a) / n + skew @ trunk
            err = frame.T @ want - u
            hess += w * jac.T @ jac
            grad += w * jac.T @ err

    # ------------------------------------------------------------------ linearisation

    def com_shift_map(self, qpos: np.ndarray, support: tuple[bool, bool], cost: np.ndarray) -> np.ndarray:
        """(29, 2) joint change per metre of COM shift (x, y) with the support feet held still.

        The cheapest change under ``cost`` (29 joint costs; the base is free) that
        moves the COM over the floor while the planted feet keep their pose.
        """
        self.run(qpos)
        rows = [self.foot_jac(s) for s in (0, 1) if support[s]]
        rows.append(self.com_jac()[:2])
        a = np.vstack(rows)
        cost = np.asarray(cost, float).copy()
        for s in (0, 1):
            if not support[s]:
                cost[6 * s : 6 * s + 6] *= 1e3  # leave a swinging leg to its own path
        winv = np.concatenate([np.full(BASE_NV, 1e3), 1.0 / cost])
        aw = a * winv
        gram = aw @ a.T + 1e-8 * np.eye(len(a))
        rhs = np.zeros((len(a), 2))
        rhs[-2:] = np.eye(2)
        dq = aw.T @ np.linalg.solve(gram, rhs)
        return dq[BASE_NV:]

    def static_torques(self, qpos: np.ndarray, support: tuple[bool, bool], left_share: float) -> np.ndarray:
        """(29,) joint torques that hold ``qpos`` still under gravity on the support feet.

        The floor's wrench on each foot balances the floating base; in double
        support the weight is split by ``left_share`` (0..1).
        """
        self.run(qpos)
        g = self.gravity()
        sides = [s for s in (0, 1) if support[s]]
        if not sides:
            return np.zeros(NJ)
        jc = np.vstack([self.foot_jac(s) for s in sides])
        a = jc[:, :BASE_NV].T  # wrenches -> base generalised forces
        share = {0: left_share, 1: 1.0 - left_share}
        w = np.concatenate([np.full(6, max(share[s], 0.02)) for s in sides]) if len(sides) == 2 else np.ones(6)
        aw = a * w
        f = aw.T @ np.linalg.solve(aw @ a.T + 1e-9 * np.eye(6), g[:BASE_NV])
        return g[BASE_NV:] - jc[:, BASE_NV:].T @ f
