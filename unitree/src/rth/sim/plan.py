"""Motion plans: how the robot gets from where it stands to a stance target.

A plan is a queue of segments. Each one produces whole-body references (feet,
COM, pelvis, posture) that the IK turns into joint targets:

* ``shift``: planted feet stay put while the COM glides over the stance foot,
  along a capture-point plan whose centre of pressure never leaves the support
  (a plain min-jerk COM would need it beyond the foot's edge to stop);
* ``swing``: the unloaded foot lifts with its sole level, travels to its new
  place and lowers until it touches;
* ``settle``: planted feet stay put while the COM and the posture blend into the
  stance (a foot the stance lifts, as in crane, follows the posture up);
* ``hold``: stay there.

Each segment starts from the measured state (where the feet really are), so a
slip or a push never accumulates. Arms, waist and pelvis follow one smooth
blend over the whole plan, so they move while the feet step.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from rth.config import SimSettings
from rth.robot.g1 import min_jerk, slerp
from rth.robot.model import JOINT_NAMES, ROOT_NQ
from rth.robot.poses import axis_quat, quat_mul
from rth.sim.ik import Pose, Task
from rth.sim.targets import SoleGeometry, StanceTarget, level, yaw_of

MOVING, STEPPING, HOLDING = "moving", "stepping", "holding"

STEP_POS_M = 0.06  # a planted foot further than this from its target place steps
STEP_YAW_RAD = math.radians(10.0)
SHIFT_TOL_M = 0.02  # COM this close to its goal (and nearly still) ends a weight shift...
SHIFT_STILL = 0.06  # m/s
SHIFT_TIMEOUT_S = 0.8  # ...or this long after its nominal end
TOUCH_TIMEOUT_S = 1.0
LOAD_WAIT_S = 1.5  # a weight shift waits at most this long for its feet to be down  # a swing foot keeps lowering this long waiting for contact
TOUCH_SETTLE_S = 0.15  # a foot touching on an edge this long counts as down
TOUCH_SPEED = 0.06  # m/s, how fast it keeps lowering
TOUCH_DEPTH_M = 0.03  # and how far below the floor it may aim
SETTLE_EXTRA_S = 0.3
SWING_S_PER_M = 1.5
LEVEL_S = 0.3  # how long a freshly planted foot takes to settle flat
GRAVITY = 9.81
PELVIS_SPEED = 0.15  # m/s the pelvis height reference may glide
PELVIS_TURN = 0.6  # rad/s for its orientation
DCM_DT = 0.01
DIP_ONE_FOOT_M = 0.025  # the dip must stay on the foot's outer half (plus the inboard offset)
DIP_TWO_FEET_M = 0.06
INBOARD_M = 0.012  # single-support COM sits this far from the foot centre towards the other foot
DCM_TAIL = 2.0  # time constants (1/omega) the COM gets to catch up after the pressure centre arrives

_ARM = np.array([not n.startswith(("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")) for n in JOINT_NAMES])
_LEG = (np.arange(0, 6), np.arange(6, 12))  # left, right leg joint indices
W_UPPER = 2.0  # arms and waist follow their blend closely
W_LEG_FREE = 2.0  # a leg whose foot is not held (crane's lifted leg)
W_LEG_HELD = 0.02  # a leg whose foot pose decides its joints


@dataclass
class Measured:
    """What the plan needs from the physical robot."""

    feet: tuple[Pose, Pose]  # foot site poses
    contact: tuple[bool, bool]
    com: np.ndarray  # (3,)
    com_vel: np.ndarray  # (3,)
    soles_down: tuple[int, int] = (0, 0)  # sole spheres touching the floor, per foot


@dataclass
class Reference:
    task: Task
    planted: tuple[bool, bool]  # feet holding the robot up (for balance and gravity torques)
    phase: str
    done: bool  # in the final hold


def support_frame(feet: tuple[Pose | None, Pose | None]) -> tuple[np.ndarray, float]:
    """(origin xy, heading) of the planted feet: their midpoint, or the one foot."""
    held = [f for f in feet if f is not None]
    if len(held) == 1:
        return held[0].pos[:2].copy(), yaw_of(held[0].quat)
    left, right = feet
    across = left.pos[:2] - right.pos[:2]
    return (left.pos[:2] + right.pos[:2]) / 2, math.atan2(across[1], across[0]) - math.pi / 2


def dcm_path(c0: np.ndarray, goal: np.ndarray, move_s: float, total_s: float, omega: float,
             dt: float = 0.01) -> np.ndarray:
    """(N, 2) COM path at ``dt`` spacing from ``c0`` at rest to ``goal`` at rest.

    The capture point xi = c + c'/omega glides min-jerk from c0 to goal over
    ``move_s`` and the COM follows it (c' = omega (xi - c)). The centre of
    pressure this needs, p = xi - xi'/omega, never passes the goal: it only
    dips behind the start, by about ``zmp_dip`` (so the planner picks
    ``move_s`` to fit that dip inside the foot it starts on).
    """
    n = max(math.ceil(total_s / dt), 1) + 1
    s = np.array([min_jerk(k * dt / move_s) if move_s > 0 else 1.0 for k in range(n)])
    xi = c0[None, :] + (goal - c0)[None, :] * s[:, None]
    a = math.exp(-omega * dt)
    c = np.empty_like(xi)
    c[0] = c0
    for k in range(n - 1):
        c[k + 1] = xi[k] + (c[k] - xi[k]) * a
    return c


def zmp_dip(distance: float, move_s: float, omega: float) -> float:
    """How far the pressure centre of ``dcm_path`` dips behind its start (fit of the exact value)."""
    return 0.186 * distance * (3.8 / omega) ** 2 / max(move_s, 1e-3) ** 2


def shift_time(distance: float, room: float, omega: float, at_least: float) -> float:
    """Shortest DCM move over ``distance`` whose pressure-centre dip fits in ``room``."""
    return max(at_least, math.sqrt(0.186 * distance * (3.8 / omega) ** 2 / max(room, 1e-3)))


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _far(a: Pose, b: Pose) -> bool:
    d_yaw = abs((yaw_of(a.quat) - yaw_of(b.quat) + math.pi) % (2 * math.pi) - math.pi)
    return float(np.linalg.norm(a.pos[:2] - b.pos[:2])) > STEP_POS_M or d_yaw > STEP_YAW_RAD


@dataclass
class _Segment:
    kind: str  # shift | swing | settle | hold
    duration: float
    side: int = -1  # swing foot (swing) or stance foot (shift)
    goal: Pose | None = None  # swing foot target
    stepping: bool = True
    # Filled in when the segment starts:
    t0: float = 0.0
    planted: tuple[Pose | None, Pose | None] = (None, None)
    com0: np.ndarray | None = None
    com1: np.ndarray | None = None
    swing0: Pose | None = None
    path: np.ndarray | None = None  # COM path (shift, settle), DCM_DT apart
    touched: float | None = None  # when the swing foot first touched
    landed: tuple[Pose | None, Pose | None] = (None, None)  # planted feet as measured at the start
    waited: float = 0.0  # time spent waiting for a planted foot to touch down
    press: float = 0.0  # how far below the floor a waiting foot is reached for


class MotionPlan:
    """References over time for one move into ``target``."""

    def __init__(
        self,
        target: StanceTarget,
        ref_q: np.ndarray,
        ref_com: np.ndarray,
        now: Measured,
        soles: SoleGeometry,
        sim: SimSettings,
        duration_s: float,
    ) -> None:
        self.target = target
        self.soles = soles
        self.sim = sim
        self.start_q = np.asarray(ref_q, float).copy()
        self.com_ref = np.asarray(ref_com, float)[:2].copy()
        self.omega = math.sqrt(GRAVITY / max(float(np.asarray(ref_com)[2]), 0.3))
        self.tail = DCM_TAIL / self.omega
        self.queue: deque[_Segment] = deque(self._segments(now, duration_s))
        self.steps = sum(1 for s in self.queue if s.kind == "swing")
        self.total = sum(s.duration for s in self.queue)
        self.seg: _Segment | None = None
        self._last: Reference | None = None
        self._yaw_offset = 0.0  # heading of the real feet against the target's
        # Pelvis references start where the IK really put the pelvis, then glide
        # (a jump between a soft and a stiff pelvis task would jolt the robot).
        self.pelvis_z = float(self.start_q[2])
        self.pelvis_q = self.start_q[3:7].copy()
        self._solved = self.start_q.copy()
        self.started = False  # a segment began on the last reference() call
        self._t_last = 0.0

    # ------------------------------------------------------------------ planning

    def _segments(self, now: Measured, duration_s: float) -> list[_Segment]:
        sim, tgt = self.sim, self.target
        contact = list(now.contact)
        if not any(contact):  # airborne (being caught): plan as if standing where it is
            contact = [True, True]

        def must_place(side: int) -> bool:
            if not tgt.support[side]:
                return False  # this foot is lifted in the stance (or stays up)
            if not contact[side]:
                return True  # a lifted foot always comes down
            return sim.stepping and _far(now.feet[side], tgt.feet[side])

        airborne = [s for s in (0, 1) if must_place(s) and not contact[s]]
        planted = [s for s in (0, 1) if must_place(s) and contact[s]]
        planted.sort(key=lambda s: -float(np.linalg.norm(now.feet[s].pos[:2] - tgt.feet[s].pos[:2])))
        segs: list[_Segment] = []
        for side in airborne + planted:
            stance = 1 - side
            if not contact[stance]:
                continue  # never lift the only foot holding the robot
            shift = (sim.weight_shift_s if contact[side] else 0.2) + self.tail
            segs.append(_Segment("shift", shift, side=stance))
            reach = float(np.linalg.norm(now.feet[side].pos[:2] - tgt.feet[side].pos[:2]))
            swing = sim.step_swing_s + SWING_S_PER_M * reach  # long steps go slower: a leg is 6 kg
            segs.append(_Segment("swing", swing, side=side, goal=tgt.feet[side]))
            contact[side] = True
        stepping = bool(segs)
        if tgt.single_support and contact[1 - tgt.stance_side]:
            segs.append(_Segment("shift", sim.weight_shift_s + self.tail, side=tgt.stance_side, stepping=stepping))
        settle = sim.weight_shift_s + self.tail + SETTLE_EXTRA_S if stepping else max(float(duration_s), 0.05)
        segs.append(_Segment("settle", settle, stepping=False))
        return segs

    # ------------------------------------------------------------------ references

    def reseed(self, qpos: np.ndarray) -> None:
        """The robot's real configuration at a segment start (pelvis references begin there)."""
        self._solved = np.asarray(qpos, float).copy()
        self.pelvis_z = float(qpos[2])
        self.pelvis_q = np.asarray(qpos[3:7], float).copy()

    def note_solution(self, qpos: np.ndarray) -> None:
        """Tell the plan where the IK put the robot for the last reference."""
        self._solved = np.asarray(qpos, float).copy()

    def _start(self, seg: _Segment, t: float, now: Measured) -> None:
        seg.t0 = t
        self.pelvis_z = float(self._solved[2])
        self.pelvis_q = self._solved[3:7].copy()
        held = list(now.contact)
        if seg.kind == "swing":
            held[seg.side] = False
            held[1 - seg.side] = True
        elif seg.kind == "settle":
            held = [now.contact[s] and self.target.support[s] for s in (0, 1)]
            if not any(held):
                held = list(self.target.support)
        elif seg.kind == "shift":
            held[seg.side] = True
        # Plant feet where they really are, then level them gently: snapping a foot that
        # landed a few mm high or on its edge would make the stiff leg shove the body.
        seg.landed = tuple(now.feet[s].copy() if held[s] else None for s in (0, 1))
        seg.planted = tuple(level(now.feet[s], self.soles.site_height) if held[s] else None for s in (0, 1))
        # Start from the measured COM: after a step the real body is the truth.
        seg.com0 = now.com[:2].copy()
        self.started = True
        if seg.kind == "shift":
            seg.com1 = self._over_foot(seg.planted, seg.side, now)
        elif seg.kind == "swing":
            seg.com1 = seg.com0
            seg.swing0 = now.feet[seg.side].copy()
        else:
            seg.com1 = self._target_com(seg.planted)
        if seg.kind in ("shift", "settle"):
            nominal = max(seg.duration - self.tail - (SETTLE_EXTRA_S if seg.kind == "settle" else 0.0), 0.05)
            if seg.kind == "settle" and self.steps == 0:
                nominal = max(seg.duration * 0.6, 0.05)  # a plain move: the COM shift fits in duration_s
            # Room behind the start for the pressure centre's dip: little if the COM
            # starts over a single foot, plenty if it starts between both.
            room = DIP_ONE_FOOT_M if self._over_one_foot(seg.com0, now) else DIP_TWO_FEET_M
            dist = float(np.linalg.norm(seg.com1 - seg.com0))
            move = shift_time(dist, room, self.omega, nominal)
            extra = SETTLE_EXTRA_S if seg.kind == "settle" else 0.0
            seg.duration = move + self.tail + extra
            seg.path = dcm_path(seg.com0, seg.com1, move, seg.duration, self.omega, DCM_DT)

    def _over_one_foot(self, com: np.ndarray, now: Measured) -> bool:
        """The COM sits above one foot rather than between the two."""
        for side in (0, 1):
            centre = self.soles.sole_points(now.feet[side], side).mean(axis=0)
            if float(np.linalg.norm(com - centre)) < 0.06:
                return True
        return False

    def _over_foot(self, planted: tuple[Pose | None, Pose | None], side: int, now: Measured) -> np.ndarray:
        """COM goal over a stance foot: its sole centre, nudged towards the other foot.

        Leaving room on the outer half lets the next shift start with a
        counter-movement without rolling the foot onto its edge.
        """
        centre = self.soles.sole_points(planted[side], side).mean(axis=0)
        other = now.feet[1 - side].pos[:2]
        toward = other - centre
        n = float(np.linalg.norm(toward))
        return centre + (toward / n) * INBOARD_M if n > 1e-6 else centre

    def _target_com(self, planted: tuple[Pose | None, Pose | None]) -> np.ndarray:
        """The target's COM, carried from its own feet over to where the feet really are."""
        tgt = self.target
        ideal = tuple(tgt.feet[s] if planted[s] is not None else None for s in (0, 1))
        o_t, h_t = support_frame(ideal)
        o_a, h_a = support_frame(planted)
        self._yaw_offset = h_a - h_t
        return o_a + _rot2(h_a - h_t) @ (tgt.com[:2] - o_t)

    @staticmethod
    def _receiving(seg: _Segment) -> list[int]:
        """Feet about to take weight: a shift's stance foot, or every planted foot for settle."""
        if seg.kind == "shift":
            return [seg.side]
        return [s for s in (0, 1) if seg.planted[s] is not None]

    def _feet_down(self, seg: _Segment, now: Measured) -> bool:
        return all(now.soles_down[s] >= 2 for s in self._receiving(seg))

    def _seg_done(self, seg: _Segment, t: float, now: Measured) -> bool:
        el = t - seg.t0 - seg.waited
        if el < seg.duration:
            return False
        if seg.kind == "shift":
            close = float(np.linalg.norm(now.com[:2] - seg.com1)) < SHIFT_TOL_M
            still = float(np.linalg.norm(now.com_vel[:2])) < SHIFT_STILL
            return (close and still) or el >= seg.duration + SHIFT_TIMEOUT_S
        if seg.kind == "swing":
            # Down means flat: three of the four sole spheres. One touching is an edge.
            down = now.soles_down[seg.side]
            if down and seg.touched is None:
                seg.touched = t
            settled = seg.touched is not None and t - seg.touched >= TOUCH_SETTLE_S
            return down >= 3 or settled or el >= seg.duration + TOUCH_TIMEOUT_S
        return seg.kind == "settle"

    def reference(self, t: float, now: Measured) -> Reference:
        """References at plan time ``t`` (seconds since the plan began)."""
        if self.seg is None or (self.seg.kind != "hold" and self._seg_done(self.seg, t, now)):
            if self.seg is not None:
                self.com_ref = self.seg.com1.copy()
            if self.queue:
                self.seg = self.queue.popleft()
            else:
                hold = _Segment("hold", 0.0, stepping=False)
                hold.planted, hold.com1 = self.seg.planted, self.seg.com1
                self.seg = hold
            if self.seg.kind != "hold":
                self._start(self.seg, t, now)
        seg = self.seg
        if (seg.kind in ("shift", "settle") and t - seg.t0 - seg.waited < 0.05
                and not self._feet_down(seg, now) and seg.waited < LOAD_WAIT_S):
            seg.waited += max(t - self._t_last, 0.0)  # never start loading a foot that isn't on the floor
            if not any(now.soles_down[side] for side in self._receiving(seg)):
                # Reach down only for a foot that hovers: pressing one that touches on an
                # edge would push the body away.
                seg.press = min(seg.press + TOUCH_SPEED * max(t - self._t_last, 0.0), TOUCH_DEPTH_M)
        el = max(t - seg.t0 - seg.waited, 0.0)
        tau = 1.0 if seg.duration <= 0 else el / seg.duration
        feet = list(seg.planted)
        if seg.kind != "hold" and el < LEVEL_S:
            k = min_jerk(el / LEVEL_S)
            for side in (0, 1):
                a, b = seg.landed[side], seg.planted[side]
                if a is not None and b is not None:
                    feet[side] = Pose(a.pos + (b.pos - a.pos) * k, slerp(a.quat, b.quat, k))
        if seg.press > 0.0:
            if self._feet_down(seg, now):
                seg.press = max(seg.press - TOUCH_SPEED * max(t - self._t_last, 0.0), 0.0)
            for side in self._receiving(seg):
                f = feet[side]
                feet[side] = Pose(f.pos - np.array([0.0, 0.0, seg.press]), f.quat)
        if seg.path is not None:
            com = seg.path[min(round(el / DCM_DT), len(seg.path) - 1)]
        else:
            com = seg.com1
        if seg.kind == "swing":
            feet[seg.side] = self._swing_pose(seg, el)
        s = min_jerk(t / self.total) if self.total > 0 else 1.0
        if seg.kind in ("settle", "hold"):
            s = 1.0 if seg.kind == "hold" else max(s, min_jerk(tau))
        dt = max(t - self._t_last, 0.0)
        self._t_last = t
        task = self._task(feet, com, s, seg, dt)
        self.com_ref = com.copy() if seg.kind != "hold" else self.com_ref
        planted = (seg.planted[0] is not None, seg.planted[1] is not None)
        phase = HOLDING if seg.kind == "hold" else (STEPPING if seg.stepping else MOVING)
        self._last = Reference(task, planted, phase, seg.kind == "hold")
        return self._last

    def _swing_pose(self, seg: _Segment, el: float) -> Pose:
        """Lift (sole level), travel, lower; past the nominal end keep lowering until contact."""
        p0, g = seg.swing0, seg.goal
        z0 = float(p0.pos[2])
        z1 = self.soles.site_height
        top = max(z0, z1 + self.sim.step_height_m)
        tau = el / seg.duration
        lift = min_jerk(tau / 0.4)
        lower = min_jerk((tau - 0.65) / 0.35)
        move = min_jerk((tau - 0.15) / 0.7)
        z = z0 + (top - z0) * lift - (top - z1) * lower
        if el > seg.duration and seg.touched is None:
            z -= min(TOUCH_SPEED * (el - seg.duration), TOUCH_DEPTH_M)
        xy = p0.pos[:2] + (g.pos[:2] - p0.pos[:2]) * move
        flat = level(p0, z0).quat
        quat = slerp(flat, g.quat, move) if lift >= 1.0 else slerp(p0.quat, flat, lift)
        return Pose(np.array([xy[0], xy[1], z]), quat)

    def _task(self, feet: list[Pose | None], com: np.ndarray, s: float, seg: _Segment, dt: float) -> Task:
        tgt = self.target
        q0, q1 = self.start_q, tgt.qpos
        posture = q0[ROOT_NQ:] + (q1[ROOT_NQ:] - q0[ROOT_NQ:]) * s
        weight = np.where(_ARM, W_UPPER, W_LEG_HELD)
        for side in (0, 1):
            if feet[side] is None:
                weight[_LEG[side]] = W_LEG_FREE
        pelvis_goal = quat_mul(axis_quat(2, self._yaw_offset), tgt.pelvis.quat) if seg.kind in ("settle", "hold") else tgt.pelvis.quat
        want_q = slerp(q0[3:7], pelvis_goal, s)
        want_z = float(q0[2] + (tgt.pelvis.pos[2] - q0[2]) * s)
        self.pelvis_z += float(np.clip(want_z - self.pelvis_z, -PELVIS_SPEED * dt, PELVIS_SPEED * dt))
        ang = 2 * math.acos(min(1.0, abs(float(np.dot(self.pelvis_q, want_q)))))
        if ang > 1e-6:
            self.pelvis_q = slerp(self.pelvis_q, want_q, min(1.0, PELVIS_TURN * dt / ang))
        pelvis_q, pelvis_z = self.pelvis_q, self.pelvis_z
        # While a foot swings, the stance leg may need to sink to reach: keep the pelvis soft.
        wz, wr = (2.0, 2.0) if seg.kind == "swing" else (20.0, 10.0)
        return Task(
            posture=posture,
            posture_weight=weight,
            feet=(feet[0], feet[1]),
            com_xy=com,
            pelvis_z=pelvis_z,
            pelvis_quat=pelvis_q,
            pelvis_weight=(wz, wr),
        )
