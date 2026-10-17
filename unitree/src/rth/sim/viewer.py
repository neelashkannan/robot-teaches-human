"""`rth sim`: MuJoCo's interactive viewer on the G1 physics sim, in the dojo.

Push the robot with the mouse or the keyboard, switch stances and watch it
react. The controller is ``rth.sim.teacher.PhysicsTeacher``; without it (or
with ``--passive``) the robot just holds Menagerie's "stand" pose on its
position servos, which is still real physics: a push knocks it over.

On macOS MuJoCo's passive viewer must run under ``mjpython``, so ``main``
re-executes itself under it when needed. The key handling (``action_for_key``,
``ViewerState``) is pure, so it is tested without a window.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import queue
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np

from rth.config import Settings, load_settings
from rth.sim.scene import SCENES, build_model

HELP = """\
rth sim: the G1 in the dojo, real physics
  1-7      switch stance             M  mirrored (left and right swapped)
  P        random push               B  big push
  R        reset (back to ready)     N  tour: next stance on a timer
  Space    pause / run
Mouse (MuJoCo viewer)
  double-click a body to select it, then
  Ctrl + right-drag   push it (force)      Ctrl + left-drag   twist it (torque)
  left-drag rotate view, right-drag move view, scroll zoom, Tab / Shift+Tab hide the panels
  F1 MuJoCo help, Esc free camera; the Camera list on the left has 'booth' and 'wide'"""

REEXEC_ENV = "RTH_SIM_UNDER_MJPYTHON"  # set before re-exec, so a failed re-exec can't loop
PUSH_BODY = "torso_link"
FRAME_S = 1.0 / 60.0
MAX_DT = 0.1  # a stalled frame (window drag, breakpoint) must not fast-forward the sim
TEXT_PERIOD_S = 0.1
STATUS_FIELDS = ("phase", "support", "fallen", "harness_engaged", "com_margin_m")

KEY_SPACE = 32
KEYS: dict[int, str] = {
    **{ord(str(i)): f"stance:{i - 1}" for i in range(1, 10)},
    ord("M"): "mirror",
    ord("P"): "push",
    ord("B"): "big_push",
    ord("R"): "reset",
    ord("N"): "tour",
    KEY_SPACE: "pause",
}


def action_for_key(keycode: int) -> str | None:
    """Viewer action for a GLFW key code (letters arrive as upper-case ASCII)."""
    if 97 <= keycode <= 122:  # be lenient with lower-case codes from other callers
        keycode -= 32
    return KEYS.get(keycode)


# --------------------------------------------------------------------------- the sim the viewer drives


class Sim(Protocol):
    """What the viewer needs from the physics (PhysicsTeacher or PassiveSim)."""

    model: mujoco.MjModel
    data: mujoco.MjData

    def stance_ids(self) -> list[str]: ...
    def set_target(
        self, stance_id: str, *, mirrored: bool = False, duration_s: float = 2.0
    ) -> None: ...
    def update(self, dt: float) -> None: ...
    def push(
        self, force_n: np.ndarray, *, body: str = PUSH_BODY, duration_s: float = 0.15
    ) -> None: ...
    def reset(self, stance_id: str = "ready") -> None: ...


@dataclass
class _Push:
    body: int
    force: np.ndarray
    until: float  # sim time


class PassiveSim:
    """No controller: ctrl holds the "stand" keyframe's joint targets on the position servos."""

    STANCE = "stand"
    FALL_PELVIS_Z = 0.5

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = build_model(settings)
        self.data = mujoco.MjData(self.model)
        self._key = self.model.key(self.STANCE).id
        self._pelvis = self.model.body("pelvis").id
        self._pushes: list[_Push] = []
        self._carry = 0.0
        self.reset()

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    def stance_ids(self) -> list[str]:
        return [self.STANCE]

    def set_target(
        self, stance_id: str, *, mirrored: bool = False, duration_s: float = 2.0
    ) -> None:
        """Stances need the controller; the passive robot keeps standing."""

    def reset(self, stance_id: str = "ready") -> None:
        m, d = self.model, self.data
        # In place, so the viewer (which holds this MjData) sees the reset.
        mujoco.mj_resetDataKeyframe(m, d, self._key)
        d.ctrl[:] = m.key_ctrl[self._key]
        self._pushes.clear()
        self._carry = 0.0
        mujoco.mj_forward(m, d)

    def push(self, force_n: np.ndarray, *, body: str = PUSH_BODY, duration_s: float = 0.15) -> None:
        f = np.zeros(6)
        f[:3] = np.asarray(force_n, float)[:3]
        self._pushes.append(_Push(self.model.body(body).id, f, self.sim_time + duration_s))

    def update(self, dt: float) -> None:
        m, d = self.model, self.data
        self._carry += max(float(dt), 0.0)
        n = min(int(self._carry / m.opt.timestep), self.settings.sim.max_substeps_per_frame)
        self._carry = (
            0.0
            if n == self.settings.sim.max_substeps_per_frame
            else self._carry - n * m.opt.timestep
        )
        for _ in range(n):
            live = [p for p in self._pushes if d.time < p.until]
            # Added and removed around the step, so the viewer's own mouse force
            # in xfrc_applied is left alone.
            for p in live:
                d.xfrc_applied[p.body] += p.force
            mujoco.mj_step(m, d)
            for p in live:
                d.xfrc_applied[p.body] -= p.force
            self._pushes = [p for p in self._pushes if d.time < p.until]

    @property
    def fallen(self) -> bool:
        return float(self.data.xpos[self._pelvis, 2]) < self.FALL_PELVIS_Z

    @property
    def status(self) -> dict[str, Any]:
        return {"phase": "passive (no controller)", "fallen": self.fallen}

    def close(self) -> None:
        pass


def make_sim(settings: Settings, *, passive: bool) -> tuple[Any, str]:
    """(sim, mode). Falls back to PassiveSim when the controller is missing or fails to start."""
    if passive:
        return PassiveSim(settings), "passive"
    try:
        from rth.sim.teacher import PhysicsTeacher
        from rth.stances import load_stances

        return PhysicsTeacher(settings, load_stances(settings.stances_path)), "physics"
    except Exception as e:  # noqa: BLE001 - any controller trouble: still show the dojo
        print(f"[sim] physics controller unavailable ({type(e).__name__}: {e}); passive mode")
        return PassiveSim(settings), "passive"


# --------------------------------------------------------------------------- key handling


@dataclass
class ViewerState:
    """What the keys control. ``handle`` returns commands; ``apply`` runs them on a sim."""

    stances: list[str]
    stance_index: int = 0
    mirrored: bool = False
    paused: bool = False
    tour: bool = False
    tour_s: float = 6.0
    next_tour_at: float = 0.0
    message: str = ""
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @property
    def stance(self) -> str | None:
        return self.stances[self.stance_index] if self.stances else None

    def handle(self, action: str | None, now: float) -> list[tuple]:
        """Apply one action. Commands: ("stance", id, mirrored), ("push", big), ("reset",)."""
        if action is None:
            return []
        if action.startswith("stance:"):
            i = int(action.split(":")[1])
            if i >= len(self.stances):
                self.message = f"no stance {i + 1} (have {len(self.stances)})"
                return []
            self.stance_index = i
            self.next_tour_at = now + self.tour_s
            return [("stance", self.stance, self.mirrored)]
        if action == "mirror":
            self.mirrored = not self.mirrored
            return [("stance", self.stance, self.mirrored)] if self.stance else []
        if action in ("push", "big_push"):
            return [("push", action == "big_push")]
        if action == "reset":
            self.stance_index = self.stances.index("ready") if "ready" in self.stances else 0
            self.mirrored = False
            self.next_tour_at = now + self.tour_s
            return [("reset",)]
        if action == "pause":
            self.paused = not self.paused
            return []
        if action == "tour":
            self.tour = not self.tour
            self.next_tour_at = now  # start straight away
            return []
        return []

    def tick(self, now: float) -> list[tuple]:
        """Tour mode: the next stance once its time is up."""
        if not self.tour or self.paused or not self.stances or now < self.next_tour_at:
            return []
        self.stance_index = (self.stance_index + 1) % len(self.stances)
        self.next_tour_at = now + self.tour_s
        return [("stance", self.stance, self.mirrored)]

    def push_force(self, big: bool, settings: Settings) -> np.ndarray:
        """A horizontal push from a random direction (newtons, world frame)."""
        s = settings.sim
        mag = s.viewer_big_push_n if big else s.viewer_push_n
        a = self.rng.uniform(0.0, 2.0 * math.pi)
        return np.array([mag * math.cos(a), mag * math.sin(a), 0.0])


def apply(
    commands: list[tuple], sim: Sim, state: ViewerState, settings: Settings, mode: str
) -> None:
    for cmd in commands:
        if cmd[0] == "stance":
            if mode == "passive":
                state.message = "passive mode: no controller, the robot only stands"
                continue
            sim.set_target(
                cmd[1], mirrored=cmd[2], duration_s=settings.session.default_transition_s
            )
            state.message = f"-> {cmd[1]}{' (mirrored)' if cmd[2] else ''}"
        elif cmd[0] == "push":
            f = state.push_force(cmd[1], settings)
            sim.push(f, body=PUSH_BODY, duration_s=settings.sim.viewer_push_s)
            state.message = f"{'big ' if cmd[1] else ''}push {np.linalg.norm(f):.0f} N"
        elif cmd[0] == "reset":
            sim.reset(stance_id=state.stance or "ready")
            state.message = "reset"


# --------------------------------------------------------------------------- overlay text


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:+.3f}" if abs(v) < 10 else f"{v:.1f}"
    if v is None:
        return "-"
    return str(getattr(v, "value", v))


def status_items(status: Any) -> list[tuple[str, str]]:
    """(name, value) pairs from a dataclass, mapping or plain object status."""
    if status is None:
        return []
    if dataclasses.is_dataclass(status) and not isinstance(status, type):
        items = dataclasses.asdict(status).items()
    elif isinstance(status, Mapping):
        items = status.items()
    else:
        items = ((k, getattr(status, k)) for k in STATUS_FIELDS if hasattr(status, k))
    return [(str(k), _fmt(v)) for k, v in items]


def overlay_text(state: ViewerState, sim: Any, mode: str) -> tuple[str, str]:
    """Left and right columns of the status box."""
    rows = [
        ("mode", mode),
        ("time", f"{getattr(sim, 'sim_time', sim.data.time):.1f} s"),
        ("stance", f"{state.stance}{' (mirrored)' if state.mirrored else ''}"),
    ]
    if state.paused:
        rows.append(("", "PAUSED"))
    if state.tour:
        rows.append(("tour", f"next in {max(state.next_tour_at - time.monotonic(), 0):.0f} s"))
    try:
        rows += status_items(getattr(sim, "status", None))
    except Exception as e:  # noqa: BLE001 - a broken status must not kill the viewer
        rows.append(("status", type(e).__name__))
    if state.message:
        rows.append(("", state.message))
    return "\n".join(k for k, _ in rows), "\n".join(v for _, v in rows)


KEY_HINT = (
    "1-7  M\nP  B\nR  N\nSpace\nmouse",
    "stance, mirror\npush, big push\nreset, tour\npause\ndbl-click a body, Ctrl+right-drag",
)


# --------------------------------------------------------------------------- mjpython re-exec


def under_mjpython() -> bool:
    """True when the mjpython launcher started this process (what launch_passive checks)."""
    import mujoco.viewer

    return getattr(mujoco.viewer, "_MJPYTHON", None) is not None


def mjpython_path() -> Path:
    """The venv's mjpython launcher (next to this interpreter)."""
    return Path(sys.executable).with_name("mjpython")


def reexec_command(
    argv: list[str], *, platform: str, in_mjpython: bool, env: Mapping[str, str]
) -> list[str] | None:
    """The command to re-run under mjpython, or None if this process can open the viewer."""
    if platform != "darwin" or in_mjpython:
        return None
    if env.get(REEXEC_ENV):
        raise RuntimeError(
            "re-ran under mjpython but the viewer still can't start; "
            f"try `{mjpython_path()} -m rth.sim.viewer` directly"
        )
    return [str(mjpython_path()), "-m", "rth.sim.viewer", *argv]


def reexec_env(env: Mapping[str, str], executable: str) -> dict[str, str]:
    """Environment for the mjpython re-exec.

    mjpython dlopens the venv's ``python3``, a symlink into uv's Python, whose
    libpython is found relative to the real binary, not the symlink; add that
    lib directory to the dyld fallback path so the load succeeds.
    """
    out = dict(env)
    out[REEXEC_ENV] = "1"
    lib = str(Path(os.path.realpath(executable)).parent.parent / "lib")
    old = out.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if lib not in old.split(":"):
        # Keep dyld's defaults: setting the variable at all replaces them.
        out["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            p for p in (lib, old or "/usr/local/lib:/usr/lib") if p
        )
    return out


# --------------------------------------------------------------------------- main


class _KeepSimForces:
    """Stop ``viewer.sync()`` from wiping forces the sim keeps in ``xfrc_applied``.

    sync() zeroes xfrc_applied and writes the mouse perturbation into it. The
    part the sim added since the last sync (a push in progress) is the
    difference from what sync left there, so it is added back afterwards.
    """

    def __init__(self, data: mujoco.MjData) -> None:
        self.data = data
        self.from_viewer = np.zeros_like(data.xfrc_applied)
        self.from_sim = np.zeros_like(data.xfrc_applied)

    def before_sync(self) -> None:
        self.from_sim = self.data.xfrc_applied - self.from_viewer

    def after_sync(self) -> None:
        self.from_viewer = self.data.xfrc_applied.copy()
        self.data.xfrc_applied += self.from_sim


class _ViewOptions:
    """Undo MuJoCo's own shortcuts for our keys (1-5 toggle geom groups, M/B/N/P vis flags).

    Each loop iteration remembers the viewer options; when one of our keys
    arrives, any option that changed since is put back.
    """

    def __init__(self, opt: mujoco.MjvOption) -> None:
        self.opt = opt
        self.snapshot()

    def snapshot(self) -> None:
        self.geomgroup = np.array(self.opt.geomgroup).copy()
        self.flags = np.array(self.opt.flags).copy()

    def restore(self) -> None:
        self.opt.geomgroup[:] = self.geomgroup
        self.opt.flags[:] = self.flags


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="rth sim",
        description=HELP.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP,
    )
    p.add_argument("--stance", help="start in this stance (default: ready)")
    p.add_argument("--scene", choices=SCENES, help="dojo or plain (default: settings.sim.scene)")
    p.add_argument(
        "--passive", action="store_true", help="no controller: the robot only holds 'stand'"
    )
    p.add_argument(
        "--tour", action="store_true", help="start in tour mode (next stance on a timer)"
    )
    p.add_argument("--seconds", type=float, default=0.0, help="close after this long (0 = never)")
    return p.parse_args(argv)


def _initial_camera(cam: mujoco.MjvCamera) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, 0.8)
    cam.distance, cam.azimuth, cam.elevation = 4.2, 205.0, -14.0


def run(args: argparse.Namespace) -> int:
    import mujoco.viewer

    settings = load_settings()
    if args.scene:
        settings.sim.scene = args.scene
    sim, mode = make_sim(settings, passive=args.passive)
    stances = list(sim.stance_ids()) if mode == "physics" else ["stand"]
    state = ViewerState(stances, tour=args.tour, tour_s=settings.sim.viewer_tour_s)
    if args.stance:
        if args.stance in stances:
            state.stance_index = stances.index(args.stance)
            sim.reset(stance_id=args.stance)
        else:
            print(f"[sim] unknown stance {args.stance!r}; have {stances}")
    elif "ready" in stances:
        state.stance_index = stances.index("ready")
    print(HELP, flush=True)
    print(
        f"[sim] {mode} mode, {settings.sim.scene} scene; stances: "
        + ", ".join(f"{i + 1}={s}" for i, s in enumerate(stances[:9])),
        flush=True,
    )

    keys: queue.SimpleQueue[int] = queue.SimpleQueue()
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys.put) as v:
        _initial_camera(v.cam)
        view = _ViewOptions(v.opt)
        forces = _KeepSimForces(sim.data)
        said = ""
        start = last = time.monotonic()
        last_text = 0.0
        while v.is_running():
            now = time.monotonic()
            dt, last = min(now - last, MAX_DT), now
            with v.lock():
                pending = []
                while not keys.empty():
                    pending.append(action_for_key(keys.get()))
                if any(pending):
                    view.restore()
                cmds = [c for a in pending for c in state.handle(a, now)] + state.tick(now)
                apply(cmds, sim, state, settings, mode)
                if not state.paused:
                    sim.update(dt * settings.sim.realtime_factor)
                view.snapshot()
            if now - last_text > TEXT_PERIOD_S:
                left, right = overlay_text(state, sim, mode)
                v.set_texts(
                    [
                        (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, left, right),
                        (None, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT, *KEY_HINT),
                    ]
                )
                last_text = now
            with v.lock():
                forces.before_sync()
            v.sync()
            with v.lock():
                forces.after_sync()
            if state.message != said:
                print(f"[sim] {state.message}", flush=True)
                said = state.message
            if args.seconds and now - start > args.seconds:
                break
            time.sleep(max(0.0, FRAME_S - (time.monotonic() - now)))
    if hasattr(sim, "close"):
        sim.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    cmd = reexec_command(argv, platform=sys.platform, in_mjpython=under_mjpython(), env=os.environ)
    if cmd is not None:
        if not Path(cmd[0]).exists():
            print(f"[sim] {cmd[0]} not found; the MuJoCo viewer needs mjpython on macOS")
            return 2
        sys.stdout.flush()
        os.execve(cmd[0], cmd, reexec_env(os.environ, sys.executable))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
