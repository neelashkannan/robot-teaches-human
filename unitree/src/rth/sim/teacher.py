"""``PhysicsTeacher``: the G1 as a simulated robot (implements ``RobotTeacher``).

Gravity, contacts and motor limits are MuJoCo's; the robot moves only through
its 29 position-controlled motors. Nothing writes qpos or qvel once it runs,
except ``reset()``, which re-spawns it standing.

Control, every substep of ``update``:

* 50 Hz plan tick: the motion plan (``rth.sim.plan``) gives whole-body
  references, the IK turns them into joint targets, and the static gravity
  torques for that pose (split over the planted feet) become a feed-forward
  offset on the position targets. Each tick plans 20 ms ahead and the targets
  are interpolated in between.
* 250 Hz balance tick: capture-point feedback (``rth.sim.balance``) shifts the
  COM target, which a precomputed joint map turns into ankle (then hip)
  corrections.
* Fall watch: once fallen the motors go to damping only, the harness holds the
  robot for ``recover_wait_s``, then it re-spawns at ready.
"""

from __future__ import annotations

import json
import math
from typing import Self

import mujoco
import numpy as np

from rth.config import Settings, project_path
from rth.contracts import Joint
from rth.limbs import limb_directions
from rth.robot.model import ROOT_NQ, booth_camera, load_model, render_bgr
from rth.robot.poses import PoseBuilder
from rth.sim.balance import Balance
from rth.sim.ik import FOOT_SITES, Pose, WholeBodyIK, mat_quat
from rth.sim.plan import HOLDING, Measured, MotionPlan
from rth.sim.support import convex_hull, margin
from rth.sim.targets import SoleGeometry, StanceTarget, mirror_target, stabilise
from rth.sim.world import HARNESS_EYE_POS, G1Map, build_world
from rth.stances import StanceLibrary

NEUTRAL_STANCE = "ready"
RECOVERING = "recovering"
PLAN_EVERY = 10  # substeps per plan tick (50 Hz at the G1's 2 ms step)
BALANCE_EVERY = 2  # substeps per balance tick (250 Hz)
SETTLED_S = 0.3  # holding this long with a still COM counts as arrived
SETTLED_SPEED = 0.05  # m/s
WINCH_SPEED = 0.3  # m/s the gantry winch pays rope out or in
TAUT_TOL_M = 0.002
LEG_DAMPING = 20.0  # N m s/rad on the 12 leg motors

_TARGET_CACHE: dict[str, dict[tuple[str, bool], StanceTarget]] = {}


def _targets(builder: PoseBuilder, ik: WholeBodyIK, soles: SoleGeometry, stances: StanceLibrary,
             max_width: float, errors: dict[str, str]) -> dict[tuple[str, bool], StanceTarget]:
    """Stabilised targets for every stance, both sides (cached: the IK takes about a second)."""
    key = json.dumps([[s.id, s.pose] for s in stances.stances] + [max_width], sort_keys=True, default=str)
    cached = _TARGET_CACHE.get(key)
    if cached is not None:
        return cached
    out: dict[tuple[str, bool], StanceTarget] = {}
    for s in stances.stances:
        try:
            t = stabilise(ik, soles, builder.stance(s), s.id, max_width=max_width)
        except Exception as e:  # noqa: BLE001 - one bad stance mustn't stop the booth
            errors[s.id] = f"{type(e).__name__}: {e}"
            print(f"[sim] stance {s.id!r} unavailable: {errors[s.id]}")
            continue
        out[(s.id, False)] = t
        out[(s.id, True)] = mirror_target(ik, soles, t)
    if not errors:
        _TARGET_CACHE[key] = out
    return out


class PhysicsTeacher:
    """Physically simulated G1 that steps and balances between the stances in the library."""

    def __init__(self, settings: Settings, stances: StanceLibrary) -> None:
        self.settings = settings
        sim, rs = settings.sim, settings.robot
        g1 = load_model(project_path(rs.model_path))  # bare G1: poses and IK scratch
        self.builder = PoseBuilder(g1, rs.clips_dir)
        self.ik = WholeBodyIK(g1)
        self.soles = SoleGeometry(self.ik)
        self.load_errors: dict[str, str] = {}
        self._targets = _targets(self.builder, self.ik, self.soles, stances, sim.max_stance_width_m,
                                 self.load_errors)
        if (NEUTRAL_STANCE, False) not in self._targets:
            raise RuntimeError(f"the {NEUTRAL_STANCE!r} stance is needed to stand the robot up")
        self.model, self.scene = build_world(settings, 10.0 if sim.harness else None)
        self.data = mujoco.MjData(self.model)
        self.map = G1Map.of(self.model)
        # Menagerie's dampratio=1 is tuned to each joint's own small inertia; the whole
        # body swaying on its legs is barely damped. More leg damping (a motor gain,
        # still only the motors act) calms that sway.
        legs = self.map.actuators[:12]
        self.model.actuator_biasprm[legs, 2] = np.minimum(self.model.actuator_biasprm[legs, 2], -LEG_DAMPING)
        self._sites = tuple(self.model.site(n).id for n in FOOT_SITES)
        self._ts = float(self.model.opt.timestep)
        self.balance = Balance(self.ik)
        self._renderer: mujoco.Renderer | None = None
        self._rgb: np.ndarray | None = None
        self._cached: tuple[tuple, np.ndarray] | None = None
        self._camera = booth_camera(rs)
        self._steps = 0
        self._carry = 0.0
        self._push: tuple[int, np.ndarray, float] | None = None
        self.stance_id = NEUTRAL_STANCE
        self.mirrored = False
        self._wanted: tuple[str, bool, float] | None = None
        self.reset(NEUTRAL_STANCE)

    # ------------------------------------------------------------------ RobotTeacher

    def stance_ids(self) -> list[str]:
        return [sid for sid, m in self._targets if not m]

    def target(self, stance_id: str, *, mirrored: bool = False) -> StanceTarget:
        """The stabilised target of a stance (raises KeyError for unknown ones)."""
        t = self._targets.get((stance_id, mirrored))
        if t is None:
            why = self.load_errors.get(stance_id, "unknown stance")
            raise KeyError(f"robot can't show {stance_id!r}: {why}")
        return t

    def set_target(self, stance_id: str, *, mirrored: bool = False, duration_s: float = 2.0) -> None:
        target = self.target(stance_id, mirrored=mirrored)
        self._wanted = (stance_id, mirrored, float(duration_s))
        if self._recover_t is not None:
            return  # picked up again once it is standing
        self.stance_id, self.mirrored = stance_id, mirrored
        self._start_plan(target, duration_s)

    def go_neutral(self, duration_s: float = 1.5) -> None:
        self.set_target(NEUTRAL_STANCE, duration_s=duration_s)

    def update(self, dt: float) -> None:
        """Advance physics by ``dt`` of wall-clock time (scaled by ``realtime_factor``)."""
        self._carry += max(float(dt), 0.0) * self.settings.sim.realtime_factor
        n = int(self._carry / self._ts + 1e-9)
        cap = self.settings.sim.max_substeps_per_frame
        if n > cap:
            n, self._carry = cap, 0.0  # drop the backlog instead of spiralling
        else:
            self._carry -= n * self._ts
        for _ in range(n):
            self._substep()

    def at_target(self) -> bool:
        if self._recover_t is not None or not self._ref_done:
            return False
        speed = float(np.linalg.norm(self._com_vel()[:2]))
        return self.data.time - self._hold_since >= SETTLED_S and speed < SETTLED_SPEED

    def keypoints(self) -> dict[Joint, np.ndarray]:
        return {j: self.data.xpos[b].copy() for j, b in self.map.keypoint_bodies.items()}

    def limb_directions(self) -> dict:
        """Current limb directions in the robot's torso frame (convenience for UIs)."""
        return limb_directions(self.keypoints())

    def render(self, width: int, height: int) -> np.ndarray:
        """BGR uint8 (height, width, 3) frame from the booth camera.

        A frame is reused only if physics hasn't stepped since it was drawn.
        """
        c = self._camera
        key = (width, height, self._steps, c.distance, c.azimuth, c.elevation, tuple(c.lookat))
        if self._cached is not None and self._cached[0] == key:
            return self._cached[1].copy()
        r = self._renderer
        if r is None or (r.width, r.height) != (width, height):
            if r is not None:
                r.close()
            g = self.model.vis.global_
            if width > g.offwidth or height > g.offheight:
                raise ValueError(f"render size {width}x{height} exceeds offscreen {g.offwidth}x{g.offheight}")
            r = self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        if self._rgb is None or self._rgb.shape[:2] != (height, width):
            self._rgb = np.empty((height, width, 3), np.uint8)
        img = render_bgr(r, self.data, c, shadows=self.settings.robot.render_shadows, out=self._rgb)
        self._cached = (key, img)
        return img.copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ extras

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    def push(self, force_n: tuple[float, float, float], *, body: str = "torso_link", duration_s: float = 0.15) -> None:
        """Shove the robot: a world-frame force at ``body``'s centre of mass for ``duration_s``."""
        bid = self.model.body(body).id
        self._push = (bid, np.asarray(force_n, float), self.data.time + float(duration_s))

    def reset(self, stance_id: str = NEUTRAL_STANCE) -> None:
        """Re-spawn the robot standing still in a stance (the dojo props are left as they are)."""
        target = self.target(stance_id)
        m, d, g = self.model, self.data, self.map
        d.qpos[g.qpos] = target.qpos
        d.qvel[g.qvel] = 0.0
        d.qacc_warmstart[g.qvel] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        self.stance_id, self.mirrored = stance_id, False
        self._push = None
        self._recover_t: float | None = None
        self._ref_q = target.qpos.copy()
        self._ik_q = target.qpos.copy()
        self._ref_from = self._ref_q.copy()
        self._tick_t = d.time
        self._com_from = self._com_to = target.com[:2].copy()
        self._planted = target.support
        self._ref_done = False
        self._hold_since = d.time
        self.balance.reset()
        self._rope_len = self._rope_goal() if g.rope >= 0 else 0.0
        self._set_rope(self._rope_len)
        self._start_plan(target, 0.05)
        self._plan_tick()
        self._ref_from = self._ref_q.copy()
        self._com_from = self._com_to.copy()
        self._balance_tick()

    @property
    def fallen(self) -> bool:
        d, g = self.data, self.map
        z_up = float(d.xmat[g.torso][8])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, z_up))))
        floor = min(self.settings.sim.fall_pelvis_z, float(self._ref_q[2]) - self.settings.sim.harness_slack_m)
        return float(d.xpos[g.pelvis][2]) < floor or tilt > self.settings.sim.fall_tilt_deg

    @property
    def harness_engaged(self) -> bool:
        g = self.map
        if g.rope < 0:
            return False
        return float(self.data.ten_length[g.rope]) >= float(self.model.tendon_range[g.rope, 1]) - TAUT_TOL_M

    @property
    def status(self) -> dict:
        contact, points = self._contacts()
        support = {(True, True): "double", (True, False): "left", (False, True): "right"}.get(contact, "none")
        com = self.data.subtree_com[self.map.pelvis]
        hull = convex_hull(np.array(points)) if points else np.zeros((0, 2))
        cm = margin(hull, com[:2]) if len(points) else -math.inf
        phase = RECOVERING if self._recover_t is not None else self._phase
        return {
            "phase": phase,
            "support": support,
            "fallen": self.fallen,
            "harness_engaged": self.harness_engaged,
            "assist": self.assisted,
            "com_margin_m": float(cm),
            "stance_id": self.stance_id,
            "mirrored": self.mirrored,
            "at_target": self.at_target(),
            "sim_time": self.sim_time,
            "steps_planned": self._plan.steps,
            "pelvis_z": float(self.data.xpos[self.map.pelvis][2]),
            "rope_slack_m": self._rope_slack(),
        }

    # ------------------------------------------------------------------ measurement

    def _contacts(self) -> tuple[tuple[bool, bool], list[np.ndarray]]:
        """Which soles touch something outside the robot, and where (xy)."""
        counts, points = self._sole_contacts()
        return (counts[0] > 0, counts[1] > 0), points

    def _sole_contacts(self) -> tuple[tuple[int, int], list[np.ndarray]]:
        """Sole spheres per foot touching something outside the robot, and the contact points."""
        d, g = self.data, self.map
        spheres: list[set[int]] = [set(), set()]
        points: list[np.ndarray] = []
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if self.model.geom_bodyid[b] in g.robot_bodies:
                    continue
                for side in (0, 1):
                    if a in g.foot_geoms[side]:
                        spheres[side].add(int(a))
                        points.append(c.pos[:2].copy())
        return (len(spheres[0]), len(spheres[1])), points

    def _measured(self) -> Measured:
        d = self.data
        feet = tuple(Pose(d.site_xpos[s].copy(), mat_quat(d.site_xmat[s])) for s in self._sites)
        counts, _ = self._sole_contacts()
        return Measured(feet=(feet[0], feet[1]), contact=(counts[0] > 0, counts[1] > 0),
                        com=d.subtree_com[self.map.pelvis].copy(), com_vel=self._com_vel(), soles_down=counts)

    def _com_vel(self) -> np.ndarray:
        mujoco.mj_subtreeVel(self.model, self.data)
        return self.data.subtree_linvel[self.map.pelvis].copy()

    def _joints(self) -> np.ndarray:
        return self.data.qpos[self.map.qpos[ROOT_NQ:]]

    # ------------------------------------------------------------------ control

    def _start_plan(self, target: StanceTarget, duration_s: float) -> None:
        self.ik.run(self._ik_q)
        com = np.array([*self._com_to, self.ik.com()[2]])
        self._plan = MotionPlan(target, self._ik_q, com, self._measured(), self.soles,
                                self.settings.sim, duration_s)
        self._plan_t0 = self.data.time
        self._phase = self._plan_phase = HOLDING
        self._ref_done = False
        self._fresh = True

    def _substep(self) -> None:
        if self._steps % PLAN_EVERY == 0 and self._recover_t is None:
            self._plan_tick()
        if self._steps % BALANCE_EVERY == 0:
            self._balance_tick()
        d = self.data
        if self._push is not None:
            bid, force, t_end = self._push
            d.xfrc_applied[bid, :3] = force if d.time < t_end else 0.0
            if d.time >= t_end:
                self._push = None
        mujoco.mj_step(self.model, d)
        self._steps += 1

    def _plan_tick(self) -> None:
        """Plan 20 ms ahead: IK targets, balance maps and gravity torques for the next tick."""
        t_next = self.data.time - self._plan_t0 + PLAN_EVERY * self._ts
        ref = self._plan.reference(t_next, self._measured())
        self._phase = ref.phase
        self._ref_from = self._ref_q.copy()
        self._com_from = self._com_to.copy()
        self._tick_t = self.data.time
        self._wind_rope()
        if ref.done and self._ref_done:
            return  # holding: the targets don't change
        if self._plan.started:
            # Seed the IK with the real robot: measured pelvis, commanded joints.
            self._plan.started = False
            seed = self.data.qpos[self.map.qpos].copy()
            seed[ROOT_NQ:] = self._ref_q[ROOT_NQ:]
            self._ik_q = seed
            self._plan.reseed(seed)
            ref = self._plan.reference(t_next, self._measured())
            self._fresh = True
        iters = 25 if self._fresh else 4
        self._fresh = False
        q = self.ik.solve(self._ik_q, ref.task, iters=iters)
        self._plan.note_solution(q)
        self._ik_q = q
        self.ik.run(q)
        self._com_to = self.ik.com()[:2]
        self._ref_q = q
        self._planted = ref.planted
        self.balance.linearise(q, ref.planted, self._left_share(ref))
        if ref.done and not self._ref_done:
            self._hold_since = self.data.time
        self._ref_done = ref.done

    def _wind_rope(self) -> None:
        """The gantry winch follows the planned torso height at a walking pace."""
        if self.map.rope >= 0 and self._recover_t is None:
            goal = self._rope_goal()
            step = WINCH_SPEED * PLAN_EVERY * self._ts
            self._rope_len += float(np.clip(goal - self._rope_len, -step, step))
            self._set_rope(self._rope_len)

    def _left_share(self, ref) -> float:
        """How much of the weight the left foot carries, from where the COM sits between the feet."""
        if ref.planted != (True, True):
            return 1.0 if ref.planted[0] else 0.0
        left, right = (f.pos[:2] for f in ref.task.feet)
        across = left - right
        s = float(np.dot(self._com_to - right, across) / max(float(np.dot(across, across)), 1e-9))
        return float(np.clip(s, 0.0, 1.0))

    def _balance_tick(self) -> None:
        d, g = self.data, self.map
        if self._recover_t is None and self.fallen:
            self._recover_t = d.time
            self.balance.reset()
        if self._recover_t is not None:
            # Damping only: a fallen robot goes limp in the harness instead of thrashing.
            d.ctrl[g.actuators] = np.clip(self._joints(), g.ctrl_lo, g.ctrl_hi)
            if d.time - self._recover_t >= self.settings.sim.recover_wait_s:
                self._recover()
            return
        a = float(np.clip((d.time - self._tick_t) / (PLAN_EVERY * self._ts), 0.0, 1.0))
        q_cmd = self._ref_from[ROOT_NQ:] + a * (self._ref_q[ROOT_NQ:] - self._ref_from[ROOT_NQ:])
        com_ref = self._com_from + a * (self._com_to - self._com_from)
        vel_ref = (self._com_to - self._com_from) / (PLAN_EVERY * self._ts)
        com = d.subtree_com[g.pelvis]
        dq = self.balance.correction(com, self._com_vel(), com_ref, vel_ref, BALANCE_EVERY * self._ts)
        tau = np.clip(self.balance.torques, -g.force_limit, g.force_limit)
        d.ctrl[g.actuators] = np.clip(q_cmd + dq + tau / g.kp, g.ctrl_lo, g.ctrl_hi)

    def _recover(self) -> None:
        wanted = self._wanted
        self.reset(NEUTRAL_STANCE)
        if wanted is not None and wanted[:2] != (NEUTRAL_STANCE, False):
            self.set_target(wanted[0], mirrored=wanted[1], duration_s=wanted[2])

    # ------------------------------------------------------------------ harness

    def _rope_goal(self) -> float:
        """Rope length that leaves ``harness_slack_m`` of drop below the planned torso pose."""
        d = self.ik.run(self._ref_q)
        t = self.ik.torso
        eye = d.xpos[t] + d.xmat[t].reshape(3, 3) @ np.asarray(HARNESS_EYE_POS)
        anchor = self.data.site_xpos[self.map.anchor_site]
        sim = self.settings.sim
        extra = sim.assist_slack_m if self.assisted else sim.harness_slack_m
        return float(np.linalg.norm(anchor - eye)) + extra

    @property
    def assisted(self) -> bool:
        """The rope is deliberately carrying part of the weight for this stance."""
        plan = getattr(self, "_plan", None)
        return self.map.rope >= 0 and plan is not None and plan.target.stance_id in self.settings.sim.assist_stances

    def _set_rope(self, length: float) -> None:
        if self.map.rope >= 0:
            self.model.tendon_range[self.map.rope] = (0.0, length)

    def _rope_slack(self) -> float:
        g = self.map
        if g.rope < 0:
            return math.inf
        return float(self.model.tendon_range[g.rope, 1] - self.data.ten_length[g.rope])


__all__ = ["NEUTRAL_STANCE", "RECOVERING", "PhysicsTeacher"]
