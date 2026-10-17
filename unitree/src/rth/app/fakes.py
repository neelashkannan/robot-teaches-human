"""Stand-ins so the booth runs with no webcam, robot assets or coach module.

* ``FakeRobot``: a stick-figure teacher with a hand-written pose per stance.
* ``FakeVisitor``: a synthetic person whose limbs follow the robot with lag and noise,
  and who sometimes gets a limb wrong, so the whole loop can run headless.
* ``KeywordPicker`` / ``SimpleSensei``: offline coach logic used only when ``rth.coach``
  is not importable yet.
* ``EmaSmoother`` / ``NullVoice``: fallbacks for the vision smoother and voice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from rth.contracts import Joint, Limb, PersonPose, PickResult, RepResult, SenseiDecision
from rth.limbs import mirror_directions
from rth.stances import StanceLibrary

Dirs = dict[Limb, np.ndarray | None]

L = Limb


def _u(*v: float) -> np.ndarray:
    a = np.asarray(v, float)
    return a / np.linalg.norm(a)


def _sym(arm_up, arm_fore, thigh, shin, r_arm_up=None, r_arm_fore=None, r_thigh=None,
         r_shin=None) -> Dirs:
    """Pose from left-side limb directions; right side mirrors left unless given."""
    def flip(v):
        return np.array([v[0], -v[1], v[2]])

    return {
        L.LEFT_UPPER_ARM: _u(*arm_up),
        L.LEFT_FOREARM: _u(*arm_fore),
        L.LEFT_THIGH: _u(*thigh),
        L.LEFT_SHIN: _u(*shin),
        L.RIGHT_UPPER_ARM: _u(*r_arm_up) if r_arm_up else flip(_u(*arm_up)),
        L.RIGHT_FOREARM: _u(*r_arm_fore) if r_arm_fore else flip(_u(*arm_fore)),
        L.RIGHT_THIGH: _u(*r_thigh) if r_thigh else flip(_u(*thigh)),
        L.RIGHT_SHIN: _u(*r_shin) if r_shin else flip(_u(*shin)),
    }


# Limb directions in the torso frame (forward, left, up), roughly matching each description.
FAKE_POSES: dict[str, Dirs] = {
    "ready": _sym((0, 0.12, -1), (0.1, 0.05, -1), (0, 0.06, -1), (0, 0.02, -1)),
    "horse": _sym((-0.5, 0.15, -0.85), (1, 0, 0.25), (0.65, 0.65, -0.4), (0.05, 0.2, -1)),
    "bow": _sym((1, 0, 0.05), (1, 0, 0.05), (0.8, 0.1, -0.6), (-0.05, 0.02, -1),
                r_arm_up=(-0.5, -0.15, -0.85), r_arm_fore=(1, 0, 0.25),
                r_thigh=(-0.55, -0.12, -0.8), r_shin=(-0.55, -0.1, -0.8)),
    "guard": _sym((0.45, 0.15, -0.9), (0.35, -0.25, 0.9), (0.25, 0.12, -1), (-0.15, 0.04, -1),
                  r_thigh=(-0.12, -0.12, -1), r_shin=(-0.05, -0.04, -1)),
    "crane": _sym((0, 1, 0.45), (0, 1, 0.8), (1, 0.1, 0.12), (0.05, 0, -1),
                  r_thigh=(0, -0.05, -1), r_shin=(0, -0.02, -1)),
    "white_crane": _sym((0.15, 0.35, -1), (0.7, 0.25, -0.3), (0.4, 0.1, -1), (0.1, 0.02, -1),
                        r_arm_up=(0.2, -0.45, 0.9), r_arm_fore=(0.2, 0.25, 1),
                        r_thigh=(0.1, -0.12, -1), r_shin=(-0.1, -0.02, -1)),
    "archer": _sym((0, 1, 0.02), (0, 1, 0.02), (0, 0.5, -1), (0, 0.3, -1),
                   r_arm_up=(0.45, 0.65, 0.05), r_arm_fore=(0.1, -1, 0.02)),
}

SEGMENT_M = {"upper_arm": 0.29, "forearm": 0.27, "thigh": 0.43, "shin": 0.42}


def _length(limb: Limb) -> float:
    return SEGMENT_M[limb.value.split("_", 1)[1]]


def keypoints_from_dirs(dirs: Mapping[Limb, np.ndarray | None], axes: np.ndarray,
                        shoulder_w: float = 0.36, hip_w: float = 0.22,
                        torso: float = 0.5) -> dict[Joint, np.ndarray]:
    """Chain limb directions into 12 joint positions.

    ``axes`` is a 3x3 matrix whose rows are the body's forward, left and up axes expressed
    in the output coordinates, so the same pose can be built in MuJoCo or MediaPipe frames.
    """
    fwd, left, up = axes
    kp = {
        Joint.LEFT_HIP: left * hip_w / 2,
        Joint.RIGHT_HIP: -left * hip_w / 2,
        Joint.LEFT_SHOULDER: up * torso + left * shoulder_w / 2,
        Joint.RIGHT_SHOULDER: up * torso - left * shoulder_w / 2,
    }
    for limb in Limb:  # proximal limbs come first in the enum, so parents exist
        a, b = limb.joints
        d = dirs.get(limb)
        d = np.array([0.0, 0.0, -1.0]) if d is None else d
        kp[b] = kp[a] + (d[0] * fwd + d[1] * left + d[2] * up) * _length(limb)
    return kp


def _blend(a: Dirs, b: Dirs, w: float) -> Dirs:
    out: Dirs = {}
    for limb in Limb:
        va, vb = a.get(limb), b.get(limb)
        if va is None or vb is None:
            out[limb] = vb if va is None else va
            continue
        v = (1 - w) * va + w * vb
        n = np.linalg.norm(v)
        out[limb] = vb if n < 1e-6 else v / n
    return out


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# --------------------------------------------------------------------------- robot


class FakeRobot:
    """A ``RobotTeacher`` drawn as a stick figure; for running without the G1 assets."""

    # MuJoCo-style world: x forward (towards the viewer), y left, z up.
    AXES = np.eye(3)

    def __init__(self, stances: StanceLibrary | None = None) -> None:
        self._ids = stances.ids() if stances else list(FAKE_POSES)
        self._from: Dirs = dict(FAKE_POSES["ready"])
        self._to: Dirs = dict(FAKE_POSES["ready"])
        self._t, self._dur = 1.0, 1.0

    def stance_ids(self) -> list[str]:
        return list(self._ids)

    def _current(self) -> Dirs:
        return _blend(self._from, self._to, _smoothstep(self._t / self._dur))

    def set_target(self, stance_id: str, *, mirrored: bool = False, duration_s: float = 2.0) -> None:
        pose = FAKE_POSES.get(stance_id, FAKE_POSES["ready"])
        self._from = self._current()
        self._to = mirror_directions(pose) if mirrored else dict(pose)
        self._t, self._dur = 0.0, max(duration_s, 1e-3)

    def go_neutral(self, duration_s: float = 1.5) -> None:
        self.set_target("ready", duration_s=duration_s)

    def update(self, dt: float) -> None:
        self._t = min(self._dur, self._t + dt)

    def at_target(self) -> bool:
        return self._t >= self._dur

    def keypoints(self) -> dict[Joint, np.ndarray]:
        kp = keypoints_from_dirs(self._current(), self.AXES)
        return {j: p + np.array([0.0, 0.0, 0.95]) for j, p in kp.items()}

    def render(self, width: int, height: int) -> np.ndarray:
        img = np.empty((height, width, 3), np.uint8)
        # A soft vertical gradient reads as a stage rather than a flat box.
        top, bottom = np.array([44, 36, 30], float), np.array([30, 25, 21], float)
        ramp = np.linspace(0, 1, height)[:, None]
        img[:] = (top * (1 - ramp) + bottom * ramp)[:, None, :].astype(np.uint8)
        kp = self.keypoints()
        s = height / 2.15
        floor_y = int(height * 0.9)
        low = min(p[2] for p in kp.values())

        def px(p: np.ndarray) -> tuple[int, int]:
            # Three-quarter view so forward/back limbs stay visible; viewer faces the robot.
            u = width / 2 + (p[1] + 0.32 * p[0]) * s
            v = floor_y - (p[2] - low + 0.12 * p[0]) * s
            return round(u * 4), round(v * 4)  # 2 fractional bits for AA

        cv2.ellipse(img, (width // 2, floor_y + 6), (int(0.22 * s), int(0.035 * s)), 0, 0, 360,
                    (22, 18, 15), -1, cv2.LINE_AA)
        body, edge, joint = (205, 200, 196), (120, 112, 106), (70, 160, 235)
        thick = max(3, int(0.045 * s))
        ls, rs = px(kp[Joint.LEFT_SHOULDER]), px(kp[Joint.RIGHT_SHOULDER])
        lh, rh = px(kp[Joint.LEFT_HIP]), px(kp[Joint.RIGHT_HIP])
        torso = np.array([ls, rs, rh, lh], np.int32)
        cv2.fillConvexPoly(img, torso, body, cv2.LINE_AA, 2)
        cv2.polylines(img, [torso], True, edge, 2, cv2.LINE_AA, 2)
        neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        head = (neck[0], neck[1] - int(0.16 * s * 4))
        cv2.circle(img, head, int(0.1 * s * 4), body, -1, cv2.LINE_AA, 2)
        cv2.circle(img, head, int(0.1 * s * 4), edge, 2, cv2.LINE_AA, 2)
        # Far side first so the near limbs overlap it.
        order = [lb for lb in Limb if lb.side == "right"] + [lb for lb in Limb if lb.side == "left"]
        for limb in order:
            a, b = (px(kp[j]) for j in limb.joints)
            cv2.line(img, a, b, edge, thick + 4, cv2.LINE_AA, 2)
            cv2.line(img, a, b, body, thick, cv2.LINE_AA, 2)
        for j in Joint:
            cv2.circle(img, px(kp[j]), int(thick * 0.45 * 4), joint, -1, cv2.LINE_AA, 2)
        return img

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- visitor

# MediaPipe pose landmark indices for the 12 joints, so the fake fills the 33-point arrays.
MP_INDEX = {
    Joint.LEFT_SHOULDER: 11, Joint.RIGHT_SHOULDER: 12, Joint.LEFT_ELBOW: 13,
    Joint.RIGHT_ELBOW: 14, Joint.LEFT_WRIST: 15, Joint.RIGHT_WRIST: 16, Joint.LEFT_HIP: 23,
    Joint.RIGHT_HIP: 24, Joint.LEFT_KNEE: 25, Joint.RIGHT_KNEE: 26, Joint.LEFT_ANKLE: 27,
    Joint.RIGHT_ANKLE: 28,
}
LEG_JOINTS = (Joint.LEFT_KNEE, Joint.RIGHT_KNEE, Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)

# MediaPipe world landmarks: x to the image right, y down, z away from the camera. A person
# facing the camera has their left on the image right, so forward is -z.
MP_AXES = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class FakeVisitor:
    """A synthetic visitor who copies the robot imperfectly.

    Limbs ease towards the target with lag and jitter. After each new stance there is a
    good chance one limb is wrong for a few seconds, and a small chance it stays wrong
    (the rep is not held). They arrive a second in, standing too close (legs out of view).
    """

    def __init__(self, width: int = 1280, height: int = 720, seed: int = 7,
                 arrive_s: float = 1.0, legs_hidden_s: float = 1.6, lag_s: float = 0.45,
                 noise_deg: float = 4.0) -> None:
        self.width, self.height = width, height
        self.rng = np.random.default_rng(seed)
        self.arrive_s, self.legs_hidden_s = arrive_s, legs_hidden_s
        self.lag_s, self.noise_deg = lag_s, noise_deg
        self.t = 0.0
        self.dirs: Dirs = dict(FAKE_POSES["ready"])
        self._target: Dirs | None = None
        self._since_change = 0.0
        self._wrong: tuple[Limb, np.ndarray, float] | None = None  # limb, offset, until
        self._background = self._make_background()

    def present(self) -> bool:
        return self.t >= self.arrive_s

    def _make_background(self) -> np.ndarray:
        h, w = self.height, self.width
        img = np.empty((h, w, 3), np.uint8)
        ramp = np.linspace(0, 1, h)[:, None]
        top, bottom = np.array([70, 64, 58], float), np.array([40, 37, 34], float)
        img[:] = (top * (1 - ramp) + bottom * ramp)[:, None, :].astype(np.uint8)
        floor = int(h * 0.86)
        img[floor:] = (img[floor:] * 0.75).astype(np.uint8)
        cv2.line(img, (0, floor), (w, floor), (85, 80, 75), 2, cv2.LINE_AA)
        cv2.ellipse(img, (w // 2, int(h * 0.93)), (130, 22), 0, 0, 360, (60, 140, 220), 3,
                    cv2.LINE_AA)  # the floor mark
        return img

    def _new_episode(self) -> None:
        """Maybe get one limb wrong after a stance change."""
        roll = self.rng.random()
        self._wrong = None
        if roll < 0.7:
            limb = list(Limb)[int(self.rng.integers(len(Limb)))]
            offset = self.rng.normal(size=3)
            offset /= np.linalg.norm(offset)
            stubborn = roll < 0.12
            self._wrong = (limb, offset * 1.1, math.inf if stubborn else
                           float(self.rng.uniform(2.0, 4.5)))

    def update(self, dt: float, target: Mapping[Limb, np.ndarray | None]) -> None:
        """Advance by ``dt`` towards ``target`` (directions the visitor should match)."""
        self.t += dt
        self._since_change += dt
        if self._target is not None:
            moved = max(
                (float(np.degrees(np.arccos(np.clip(np.dot(target[lb], self._target[lb]), -1, 1))))
                 for lb in Limb if target.get(lb) is not None and self._target.get(lb) is not None),
                default=0.0,
            )
            if moved > 8.0:
                self._since_change = 0.0
                self._new_episode()
        self._target = {lb: (None if v is None else np.array(v)) for lb, v in target.items()}
        goal = dict(self._target)
        if self._wrong is not None:
            limb, offset, until = self._wrong
            if self._since_change < until + 2.0 and goal.get(limb) is not None:
                v = goal[limb] + offset
                goal[limb] = v / np.linalg.norm(v)
        w = 1.0 - math.exp(-dt / self.lag_s)
        self.dirs = _blend(self.dirs, goal, w)
        for limb, v in self.dirs.items():
            if v is not None:
                j = self.rng.normal(scale=math.radians(self.noise_deg), size=3)
                n = v + j
                self.dirs[limb] = n / np.linalg.norm(n)

    def frame_and_pose(self, timestamp_ms: int) -> tuple[np.ndarray, PersonPose | None]:
        """An unmirrored camera frame and the pose a tracker would report for it."""
        img = self._background.copy()
        if not self.present():
            return img, None
        kp = keypoints_from_dirs(self.dirs, MP_AXES)
        hip_y = 0.0
        legs_seen = self.t >= self.arrive_s + self.legs_hidden_s
        # Normalised image coordinates: hips a bit below centre, ~1.7 m tall in 80% of height.
        sv = 0.78 / 1.75
        su = sv * self.height / self.width
        sway = 0.01 * math.sin(self.t * 0.9)
        image_points = {
            j: (0.5 + sway + float(p[0]) * su, 0.47 + float(p[1] - hip_y) * sv) for j, p in kp.items()
        }
        if not legs_seen:
            # Standing too close: zoom in so the knees and ankles fall off the bottom.
            image_points = {j: (0.5 + (x - 0.5) * 1.6, 0.15 + (y - 0.25) * 1.6)
                            for j, (x, y) in image_points.items()}
        visibility = {j: 0.95 for j in Joint}
        if not legs_seen:
            for j in LEG_JOINTS:
                visibility[j] = 0.1
        self._draw_person(img, image_points)
        lm_world = np.zeros((33, 3))
        lm_image = np.zeros((33, 3))
        lm_vis = np.full(33, 0.9)
        for j, i in MP_INDEX.items():
            lm_world[i] = kp[j]
            lm_image[i, :2] = image_points[j]
            lm_vis[i] = visibility[j]
        pose = PersonPose(
            keypoints=kp,
            visibility=visibility,
            image_points=image_points,
            landmarks_world=lm_world,
            landmarks_image=lm_image,
            landmarks_visibility=lm_vis,
            timestamp_ms=timestamp_ms,
        )
        return img, pose

    def _draw_person(self, img: np.ndarray, pts: Mapping[Joint, tuple[float, float]]) -> None:
        h, w = img.shape[:2]

        def px(j: Joint) -> tuple[int, int]:
            x, y = pts[j]
            return int(x * w * 4), int(y * h * 4)

        cloth, skin = (120, 96, 78), (150, 170, 200)
        width = int(h * 0.05)
        ls, rs, lh, rh = (px(j) for j in (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER,
                                          Joint.LEFT_HIP, Joint.RIGHT_HIP))
        cv2.fillConvexPoly(img, np.array([ls, rs, rh, lh], np.int32), cloth, cv2.LINE_AA, 2)
        neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        head_r = int(h * 0.055)
        cv2.circle(img, (neck[0], neck[1] - int(head_r * 1.5 * 4)), head_r * 4, skin, -1,
                   cv2.LINE_AA, 2)
        for limb in Limb:
            a, b = (px(j) for j in limb.joints)
            colour = skin if "arm" in limb.value and "fore" in limb.value else cloth
            cv2.line(img, a, b, colour, width, cv2.LINE_AA, 2)


# --------------------------------------------------------------------------- coach fallbacks


class KeywordPicker:
    """Offline move picker by word overlap with stance names, tags and descriptions."""

    def __init__(self, stances: StanceLibrary) -> None:
        self.stances = stances

    def pick(self, request: str) -> PickResult:
        words = {w.strip(".,!?").lower() for w in request.split() if len(w) > 2}
        scored = []
        for s in self.stances.stances:
            text = " ".join([s.id, s.name, s.description, *s.tags]).lower()
            hits = sum(1 for w in words if w in text)
            scored.append((s.id, hits / max(1, len(words))))
        scored.sort(key=lambda x: -x[1])
        best, sim = scored[0]
        return PickResult(best, float(sim), scored[1:4], source="offline")


class SimpleSensei:
    """Offline rules: repeat a missed stance once, more slowly; otherwise move on."""

    def __init__(self, stances: StanceLibrary, default_hold_s: float = 3.0,
                 default_transition_s: float = 2.0) -> None:
        self.stances = stances
        self.hold_s, self.transition_s = default_hold_s, default_transition_s

    def decide(self, history: Sequence[RepResult], *, reps_left: int) -> SenseiDecision:
        if reps_left <= 0:
            return SenseiDecision("ready", False, self.transition_s, self.hold_s,
                                  "That's the session. Well trained!", finished=True)
        last = history[-1] if history else None
        ids = self.stances.ids()
        if last is not None and not last.held and not (
            len(history) >= 2 and history[-2].stance_id == last.stance_id
        ):
            limb = last.worst_limb.label.lower() if last.worst_limb else "stance"
            return SenseiDecision(last.stance_id, last.mirrored, self.transition_s * 1.6,
                                  last.hold_s, f"Once more, slower. Watch your {limb}.")
        done = {r.stance_id for r in history}
        nxt = next((i for i in ids if i not in done), ids[len(history) % len(ids)])
        name = self.stances.get(nxt).name
        say = f"Good. Now the {name}." if last and last.held else f"Let's move on to the {name}."
        return SenseiDecision(nxt, False, self.transition_s, self.stances.get(nxt).hold_s, say)


# --------------------------------------------------------------------------- vision/voice


class EmaSmoother:
    """Exponential smoothing of limb directions; a limb that drops out starts fresh."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._state: Dirs = {}

    def update(self, dirs: Mapping[Limb, np.ndarray | None]) -> Dirs:
        out: Dirs = {}
        for limb in Limb:
            new, old = dirs.get(limb), self._state.get(limb)
            if new is None:
                out[limb] = None
            elif old is None:
                out[limb] = np.asarray(new, float)
            else:
                v = self.alpha * np.asarray(new, float) + (1 - self.alpha) * old
                n = np.linalg.norm(v)
                out[limb] = np.asarray(new, float) if n < 1e-6 else v / n
        self._state = out
        return out

    def reset(self) -> None:
        self._state = {}


class NullVoice:
    """Silent voice for headless runs; keeps the last line for inspection."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)
