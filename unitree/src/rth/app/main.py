"""`rth demo`: the booth. Wires camera, tracker, robot, coach and screen into one frame loop.

Per frame: camera frame (unmirrored) -> tracker -> limb directions -> smoother ->
compare with the robot's limb directions -> session -> screen. Coach calls run in the
background (``CoachRunner``); the session decides when to make them.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rth.app.coach_runner import PICK, SENSEI, CoachRunner, Delivery
from rth.app.session import (
    Command,
    FrameInput,
    GoNeutral,
    Phase,
    RequestPick,
    RequestSensei,
    RequestVisionTip,
    ResetTracking,
    Say,
    Session,
    SetTarget,
)
from rth.app.ui import DESIGN_H, DESIGN_W, BoothUI, ScreenState, make_layout
from rth.config import Settings, load_settings, project_path
from rth.contracts import Joint, Limb, PersonPose
from rth.limbs import compare, hint_for, limb_directions, mirror_directions, overall
from rth.stances import StanceLibrary, load_stances

log = logging.getLogger("rth.demo")

WINDOW = "Robot Teaches Human"
BODY_JOINTS = (Joint.LEFT_HIP, Joint.RIGHT_HIP, Joint.LEFT_KNEE, Joint.RIGHT_KNEE,
               Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)
MAX_TYPED = 60
NOTICE_S = 2.0


class SetupError(RuntimeError):
    """Something the operator can fix; printed as a message, not a traceback."""


class SimClock:
    """Fixed-step clock for headless runs, so a run of N frames covers N/fps seconds."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# --------------------------------------------------------------------------- wiring


def force_offline(settings: Settings) -> None:
    """Never call Nebius: disable every role and hide the key from the factories."""
    n = settings.nebius
    n.enable_picker = n.enable_sensei = n.enable_vision_tip = False
    n.api_key_env = "RTH_OFFLINE_NO_KEY"


def _optional(what: str, make: Callable[[], Any], fallback: Callable[[], Any]) -> Any:
    try:
        return make()
    except ImportError as e:
        log.warning("%s not available yet (%s); using the built-in stand-in", what, e)
    except Exception:
        log.exception("%s failed to start; using the built-in stand-in", what)
    return fallback()


@dataclass
class Coach:
    picker: Any
    offline_picker: Any
    sensei: Any
    rule_sensei: Any
    vision_coach: Any
    voice: Any


def make_coach(settings: Settings, stances: StanceLibrary, *, silent: bool) -> Coach:
    from rth.app import fakes

    def offline_picker():
        from rth.coach.picker import OfflineMovePicker

        return OfflineMovePicker(stances)

    def rule_sensei():
        from rth.coach.sensei import RuleSensei

        return RuleSensei(settings, stances)

    def picker():
        from rth.coach.picker import make_picker

        return make_picker(settings, stances)

    def sensei():
        from rth.coach.sensei import make_sensei

        return make_sensei(settings, stances)

    def vision_coach():
        from rth.coach.vision_tip import make_vision_coach

        return make_vision_coach(settings, stances)

    def voice():
        from rth.coach.voice import make_voice

        return make_voice(settings)

    s = settings.session
    fb_picker = _optional("offline move picker", offline_picker,
                          lambda: fakes.KeywordPicker(stances))
    fb_sensei = _optional("rule sensei", rule_sensei, lambda: fakes.SimpleSensei(
        stances, s.default_hold_s, s.default_transition_s))
    return Coach(
        picker=_optional("move picker", picker, lambda: fb_picker),
        offline_picker=fb_picker,
        sensei=_optional("sensei", sensei, lambda: fb_sensei),
        rule_sensei=fb_sensei,
        vision_coach=_optional("vision tip", vision_coach, lambda: None),
        voice=fakes.NullVoice() if silent else _optional("voice", voice, fakes.NullVoice),
    )


def make_robot(settings: Settings, stances: StanceLibrary, *, fake: bool,
               mode: str | None = None) -> Any:
    if fake:
        from rth.app.fakes import FakeRobot

        return FakeRobot(stances)
    model = project_path(settings.robot.model_path)
    if not model.exists():
        raise SetupError(
            f"The G1 robot model is missing ({model}).\n"
            "  Fix:  uv run rth fetch-assets\n"
            "  Or run with a stand-in robot:  uv run rth demo --fake-robot"
        )
    try:
        from rth.robot.factory import make_teacher
    except ImportError as e:
        raise SetupError(
            f"The robot module isn't available ({e}).\n"
            "  Run with a stand-in robot:  uv run rth demo --fake-robot"
        ) from e
    try:
        return make_teacher(settings, stances, mode=mode)
    except ImportError as e:
        raise SetupError(
            f"The {mode or settings.robot.mode} robot isn't available ({e}).\n"
            "  Try:  uv run rth demo --robot kinematic"
        ) from e
    except (FileNotFoundError, ValueError, OSError) as e:
        raise SetupError(
            f"The G1 robot couldn't load: {e}\n  Fix:  uv run rth fetch-assets"
        ) from e


def make_smoother(alpha: float) -> Any:
    from rth.app.fakes import EmaSmoother

    def real():
        from rth.vision.smoothing import DirectionSmoother

        return DirectionSmoother(alpha)

    return _optional("direction smoother", real, lambda: EmaSmoother(alpha))


def make_recognizer(settings: Settings) -> Any:
    if not settings.recognizer.enabled:
        return None

    def real():
        from rth.ml.recognizer import load_recognizer

        return load_recognizer(settings)

    return _optional("stance recogniser", real, lambda: None)


class LiveSensor:
    """Webcam + MediaPipe."""

    def __init__(self, settings: Settings) -> None:
        try:
            from rth.vision.camera import Camera
            from rth.vision.pose import MediaPipePoseTracker
        except ImportError as e:
            raise SetupError(
                f"The vision module isn't available ({e}).\n"
                "  Try the camera-free demo:  uv run rth demo --fake-person"
            ) from e
        model = project_path(settings.vision.model_path)
        if not model.exists():
            raise SetupError(f"The pose model is missing ({model}).\n"
                             "  Fix:  uv run rth fetch-assets")
        from rth.vision.camera import CameraError

        self.tracker = MediaPipePoseTracker(settings.vision)
        try:
            self.camera = Camera(settings.camera)
        except CameraError as e:
            self.tracker.close()
            hint = "  Or try the camera-free demo:  uv run rth demo --fake-person"
            raise SetupError(f"{e}\n{hint}") from e
        except Exception:
            self.tracker.close()
            raise
        self._last_ts = -1

    def sense(self, dt: float, ts_ms: int, target: Any) -> tuple[np.ndarray | None, Any]:
        frame = self.camera.read()
        if frame is None:
            return None, None
        ts_ms = max(ts_ms, self._last_ts + 1)  # MediaPipe needs strictly increasing stamps
        self._last_ts = ts_ms
        return frame, self.tracker.process(frame, ts_ms)

    def close(self) -> None:
        for part in (self.camera, self.tracker):
            try:
                part.close()
            except Exception:
                log.exception("closing %s", type(part).__name__)


class FakeSensor:
    """A synthetic visitor instead of the webcam."""

    def __init__(self, settings: Settings, seed: int) -> None:
        from rth.app.fakes import FakeVisitor

        self.visitor = FakeVisitor(settings.camera.width, settings.camera.height, seed=seed)

    def sense(self, dt: float, ts_ms: int, target: Any) -> tuple[np.ndarray, Any]:
        self.visitor.update(dt, target)
        return self.visitor.frame_and_pose(ts_ms)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- app


class BoothApp:
    def __init__(self, settings: Settings, stances: StanceLibrary, *, robot: Any, sensor: Any,
                 coach: Coach, clock: Callable[[], float] = time.monotonic,
                 size: tuple[int, int] = (DESIGN_W, DESIGN_H)) -> None:
        self.settings = settings
        self.stances = stances
        self.robot = robot
        self.sensor = sensor
        self.coach = coach
        self.clock = clock
        self.size = size
        sc, ne = settings.scoring, settings.nebius
        self.mirror_view = sc.mirror_view
        self.smoother = make_smoother(sc.smoothing_alpha)
        self.recognizer = make_recognizer(settings)
        self.session = Session(settings.session, stances, coach.rule_sensei,
                               tolerance_deg=sc.tolerance_deg, min_limbs_seen=sc.min_limbs_seen,
                               think_timeout_s=ne.timeout_s + 2.0, clock=clock)
        self.runner = CoachRunner(
            picker=coach.picker, sensei=coach.sensei, fallback_picker=coach.offline_picker,
            fallback_sensei=coach.rule_sensei, vision_coach=coach.vision_coach,
            deadline_s=ne.timeout_s + 1.0, clock=clock,
        )
        self.runner.sources.update(self._expected_sources())
        robot_ids = set(robot.stance_ids() or stances.ids())
        self.stance_keys = tuple(i for i in stances.ids() if i in robot_ids)[:9]
        self.ui = BoothUI({s.id: s.name for s in stances.stances}, tolerance_deg=sc.tolerance_deg)
        self.typing: str | None = None
        self._notice: tuple[str, float] | None = None
        self._last = clock()
        self._last_wall = time.perf_counter()
        self.fps: float | None = None
        self._frame: np.ndarray | None = None
        self.last_state: ScreenState | None = None
        self._robot_cap: tuple[int, int] | None = None
        self._robot_img: tuple[tuple[int, int], np.ndarray] | None = None
        self._robot_was_still = False

    def _expected_sources(self) -> dict[str, str]:
        ne = self.settings.nebius
        online = ne.api_key is not None
        out = {
            "picker": "nebius" if online and ne.enable_picker else "offline",
            "sensei": "nebius" if online and ne.enable_sensei else "offline",
        }
        if self.coach.vision_coach is not None:
            out["vision"] = "nebius"
        return out

    # ----------------------------------------------------------------- commands

    def run_commands(self, cmds: list[Command]) -> None:
        for c in cmds:
            if isinstance(c, SetTarget):
                self.robot.set_target(c.stance_id, mirrored=c.mirrored, duration_s=c.duration_s)
            elif isinstance(c, GoNeutral):
                self.robot.go_neutral(c.duration_s)
            elif isinstance(c, Say):
                log.info("say: %s", c.text)
                try:
                    self.coach.voice.say(c.text)
                except Exception:
                    log.exception("voice failed")
            elif isinstance(c, RequestPick):
                self.runner.submit_pick(c.token, c.text)
            elif isinstance(c, RequestSensei):
                self.runner.submit_sensei(c.token, c.history, c.reps_left)
            elif isinstance(c, RequestVisionTip):
                self.runner.submit_tip(c.token, self._frame, c.rep)
            elif isinstance(c, ResetTracking):
                self.smoother.reset()
                if self.recognizer is not None:
                    self.recognizer.reset()

    def deliver(self, d: Delivery) -> None:
        log.info("%s answered from %s in %.2fs%s", d.role, d.source, d.latency_s,
                 " (deadline)" if d.timed_out else "")
        if d.role == PICK:
            self.run_commands(self.session.deliver_pick(d.token, d.value))
        elif d.role == SENSEI:
            self.run_commands(self.session.deliver_decision(d.token, d.value))
        else:
            self.run_commands(self.session.deliver_tip(d.token, d.value))

    def notice(self, text: str) -> None:
        self._notice = (text, self.clock() + NOTICE_S)

    # ----------------------------------------------------------------- frame

    def step(self) -> np.ndarray:
        """Advance one frame and return the composed screen."""
        now = self.clock()
        dt = min(max(now - self._last, 0.0), 0.1)
        self._last = now
        wall = time.perf_counter()
        if wall > self._last_wall:
            inst = 1.0 / (wall - self._last_wall)
            self.fps = inst if self.fps is None else 0.9 * self.fps + 0.1 * inst
        self._last_wall = wall
        ts_ms = int(now * 1000)
        sc = self.settings.scoring

        self.robot.update(dt)
        robot_dirs = limb_directions(self.robot.keypoints())
        target = mirror_directions(robot_dirs) if self.mirror_view else robot_dirs
        frame, pose = self.sensor.sense(dt, ts_ms, target)
        self._frame = frame
        if pose is not None:
            person = self.smoother.update(
                limb_directions(pose.keypoints, pose.visibility, sc.min_visibility))
        else:
            self.smoother.reset()
            person = dict.fromkeys(Limb)
        scores = compare(person, robot_dirs, sc.tolerance_deg, mirror_view=self.mirror_view)
        all_ok = overall(scores, sc.min_limbs_seen)[0]
        hint = hint_for(scores, person, robot_dirs, mirror_view=self.mirror_view) if pose else None
        recognised = None
        if self.recognizer is not None:
            label, conf = self.recognizer.update(pose.landmarks_world if pose else None, ts_ms)
            recognised = (label, float(conf))
        full_body = _full_body(pose, sc.min_visibility)

        for d in self.runner.poll():
            self.deliver(d)
        self.run_commands(self.session.update(FrameInput(
            scores=scores, person_present=pose is not None, full_body=full_body,
            robot_at_target=self.robot.at_target(), robot_recovering=self.robot_recovering(),
            recognised=recognised[0] if recognised else None,
            recognised_conf=recognised[1] if recognised else 0.0,
        )))

        lay = make_layout(*self.size)
        notice = None
        if self._notice is not None:
            notice = self._notice[0] if now < self._notice[1] else None
        state = ScreenState(
            view=self.session.view(), camera_bgr=frame, pose=pose, scores=scores,
            robot_bgr=self._render_robot(lay.robot.w, lay.robot.h), hint=hint,
            person_present=pose is not None, full_body=full_body, all_ok=all_ok,
            mirror_display=self.settings.camera.mirror_display, mirror_view=self.mirror_view,
            sources=dict(self.runner.sources), recognised=recognised, fps=self.fps,
            typing=self.typing, notice=notice, stance_keys=self.stance_keys,
        )
        self.last_state = state
        return self.ui.render(state, self.size)

    def robot_recovering(self) -> bool:
        status = getattr(self.robot, "status", None)
        if callable(status):
            status = status()
        return isinstance(status, dict) and status.get("phase") == "recovering"

    def _render_robot(self, w: int, h: int) -> np.ndarray | None:
        # A kinematic robot holding still looks the same every frame: skip the GPU round trip.
        # A physics robot keeps wobbling, so it is always rendered (it reports a status).
        at_target = bool(self.robot.at_target()) and not hasattr(self.robot, "status")
        cached = self._robot_img
        if at_target and self._robot_was_still and cached is not None and cached[0] == (w, h):
            return cached[1]
        img = self._render_robot_now(w, h)
        self._robot_img = ((w, h), img.copy())  # the robot may reuse its buffer
        self._robot_was_still = at_target
        return img

    def _render_robot_now(self, w: int, h: int) -> np.ndarray:
        # The G1's offscreen buffer has a maximum size; past it, render smaller at the same
        # aspect and let the screen scale it up rather than lose the robot on a big window.
        if self._robot_cap is not None:
            scale = min(1.0, self._robot_cap[0] / w, self._robot_cap[1] / h)
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
        try:
            return self.robot.render(w, h)
        except ValueError:
            if self._robot_cap is not None:
                raise
            rs = self.settings.robot
            self._robot_cap = (rs.render_width, rs.render_height)
            log.info("robot render capped at %dx%d", *self._robot_cap)
            return self._render_robot_now(w, h)

    # ----------------------------------------------------------------- keys

    def handle_key(self, key: int) -> bool:
        """Apply one key press. Returns False to quit."""
        if key < 0:
            return True
        k = key & 0xFF
        if self.typing is not None:
            if k == 27:
                self.typing = None
            elif k in (10, 13):
                text, self.typing = self.typing, None
                self.run_commands(self.session.request(text))
            elif k in (8, 127):
                self.typing = self.typing[:-1]
            elif 32 <= k < 127 and len(self.typing) < MAX_TYPED:
                self.typing += chr(k)
            return True
        ch = chr(k).lower() if 32 <= k < 127 else ""
        if k == 27 or ch == "q":
            return False
        if ch == "t" or k in (10, 13):
            self.typing = ""
        elif ch.isdigit() and ch != "0":
            i = int(ch) - 1
            if i < len(self.stance_keys):
                self.run_commands(self.session.choose(self.stance_keys[i]))
        elif ch == " ":
            self.run_commands(self.session.skip())
        elif ch == "m":
            self.mirror_view = not self.mirror_view
            self.notice("Copy like a mirror" if self.mirror_view else "Copy the same side")
        elif ch == "r":
            self.run_commands(self.session.restart())
            self.notice("Restarted")
        return True

    def close(self) -> None:
        for name, fn in (("coach", self.runner.close), ("sensor", self.sensor.close),
                         ("robot", self.robot.close)):
            try:
                fn()
            except Exception:
                log.exception("closing %s", name)
        closer = getattr(self.coach.voice, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                log.exception("closing voice")


def _full_body(pose: PersonPose | None, min_visibility: float) -> bool:
    if pose is None:
        return False
    return all(pose.visibility.get(j, 0.0) >= min_visibility for j in BODY_JOINTS)


# --------------------------------------------------------------------------- loops


def run_window(app: BoothApp) -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, *app.size)
    fullscreen = False
    try:
        while True:
            try:
                _, _, w, h = cv2.getWindowImageRect(WINDOW)
                if w >= 640 and h >= 360:
                    app.size = (w, h)  # render at window size so text stays crisp
            except cv2.error:
                pass
            cv2.imshow(WINDOW, app.step())
            key = cv2.waitKeyEx(1)
            if app.typing is None and key >= 0 and chr(key & 0xFF).lower() == "f":
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                continue
            if not app.handle_key(key):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the mouse
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def run_headless(app: BoothApp, clock: SimClock, frames: int, out: Path, every: int,
                 fps: float) -> dict[str, int]:
    """Run ``frames`` steps on a fixed clock; save every k-th frame and each phase change."""
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    last_phase: Phase | None = None
    for i in range(frames):
        clock.advance(1.0 / fps)
        canvas = app.step()
        phase = app.session.phase
        counts[phase.value] = counts.get(phase.value, 0) + 1
        if every > 0 and i % every == 0:
            cv2.imwrite(str(out / f"frame_{i:05d}.png"), canvas)
        if phase is not last_phase:
            cv2.imwrite(str(out / f"phase_{i:05d}_{phase.value}.png"), canvas)
            last_phase = phase
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rth demo", description="Run the Robot Teaches Human booth.")
    p.add_argument("--offline", action="store_true", help="never call Nebius")
    p.add_argument("--fake-person", action="store_true",
                   help="no webcam: a synthetic visitor copies the robot")
    p.add_argument("--fake-robot", action="store_true",
                   help="a stick-figure robot instead of the MuJoCo G1")
    p.add_argument("--robot", choices=("physics", "kinematic"),
                   help="robot simulation (default: robot.mode in settings)")
    p.add_argument("--headless", type=int, metavar="N", help="run N frames without a window")
    p.add_argument("--out", type=Path, help="folder for headless frames (PNG)")
    p.add_argument("--every", type=int, default=30, metavar="K",
                   help="headless: save every K-th frame (default 30)")
    p.add_argument("--fps", type=float, default=30.0, help="headless: simulated frame rate")
    p.add_argument("--camera", type=int, metavar="INDEX", help="webcam index")
    p.add_argument("--request", metavar="TEXT", help="ask for a stance at start, as if typed")
    p.add_argument("--size", default=f"{DESIGN_W}x{DESIGN_H}", help="window size, e.g. 1600x900")
    p.add_argument("--seed", type=int, default=7, help="fake visitor random seed")
    p.add_argument("--settings", default="config/settings.yaml", help=argparse.SUPPRESS)
    return p


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError as e:
        raise SetupError(f"--size must look like 1600x900, not {text!r}") from e
    return max(640, w), max(360, h)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    if args.headless is not None and args.out is None:
        print("--headless needs --out DIR for the saved frames", file=sys.stderr)
        return 2
    settings = load_settings(args.settings)
    stances = load_stances(settings.stances_path)
    if args.offline:
        force_offline(settings)
    if args.camera is not None:
        settings.camera.index = args.camera

    headless = args.headless is not None
    clock: Callable[[], float] = SimClock() if headless else time.monotonic
    robot = sensor = app = None
    try:
        size = _parse_size(args.size)
        robot = make_robot(settings, stances, fake=args.fake_robot, mode=args.robot)
        sensor = FakeSensor(settings, args.seed) if args.fake_person else LiveSensor(settings)
        coach = make_coach(settings, stances, silent=headless)
        app = BoothApp(settings, stances, robot=robot, sensor=sensor, coach=coach, clock=clock,
                       size=size)
        if args.request:
            app.run_commands(app.session.request(args.request))
        if headless:
            assert isinstance(clock, SimClock)
            counts = run_headless(app, clock, args.headless, args.out, args.every, args.fps)
            print(f"headless: {args.headless} frames -> {args.out}  phases: {counts}")
        else:
            run_window(app)
    except SetupError as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if app is not None:
            app.close()
        else:
            for part in (sensor, robot):
                if part is not None:
                    try:
                        part.close()
                    except Exception:  # keep closing the rest
                        log.exception("closing %s", type(part).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
