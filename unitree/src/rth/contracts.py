"""Types and interfaces shared across modules.

This file is the contract between the robot, vision, ML, coach and app modules.
Change it only by agreement: every module codes against these names.

Coordinate conventions
----------------------
* ``Keypoints`` are 3D positions in metres in any right-handed frame. The
  robot gives MuJoCo world coordinates; the person gives MediaPipe world
  landmarks from the *unmirrored* camera frame. Scoring never compares raw
  positions: ``rth.limbs`` converts both into each body's own torso frame.
* Left and right are always the body's own anatomical sides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

Vec3 = np.ndarray  # shape (3,), float


class Joint(str, Enum):
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


class Limb(str, Enum):
    """The 8 limb segments both bodies are reduced to (the shared language)."""

    LEFT_UPPER_ARM = "left_upper_arm"
    LEFT_FOREARM = "left_forearm"
    RIGHT_UPPER_ARM = "right_upper_arm"
    RIGHT_FOREARM = "right_forearm"
    LEFT_THIGH = "left_thigh"
    LEFT_SHIN = "left_shin"
    RIGHT_THIGH = "right_thigh"
    RIGHT_SHIN = "right_shin"

    @property
    def joints(self) -> tuple[Joint, Joint]:
        """(proximal, distal) joints; the limb direction points proximal -> distal."""
        return LIMB_JOINTS[self]

    @property
    def side(self) -> str:
        return "left" if self.value.startswith("left") else "right"

    @property
    def mirror(self) -> Limb:
        """The same limb on the other side of the body."""
        other = "right" if self.side == "left" else "left"
        return Limb(other + self.value[len(self.side):])

    @property
    def label(self) -> str:
        """Human-readable name, e.g. 'Left forearm'."""
        return self.value.replace("_", " ").capitalize()


LIMB_JOINTS: dict[Limb, tuple[Joint, Joint]] = {
    Limb.LEFT_UPPER_ARM: (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    Limb.LEFT_FOREARM: (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    Limb.RIGHT_UPPER_ARM: (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    Limb.RIGHT_FOREARM: (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    Limb.LEFT_THIGH: (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    Limb.LEFT_SHIN: (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    Limb.RIGHT_THIGH: (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    Limb.RIGHT_SHIN: (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
}

Keypoints = Mapping[Joint, Vec3]
LimbDirections = Mapping[Limb, "Vec3 | None"]  # unit vectors in the torso frame; None = not seen


@dataclass(frozen=True)
class LimbScore:
    limb: Limb
    error_deg: float | None  # angle between person and robot directions; None = not seen
    ok: bool  # within tolerance (False when not seen)


Scores = Mapping[Limb, LimbScore]


# --------------------------------------------------------------------------- vision


@dataclass
class PersonPose:
    """One frame of body tracking for the visitor."""

    keypoints: dict[Joint, Vec3]  # world landmarks, metres, hip-centred
    visibility: dict[Joint, float]  # 0..1 per joint
    image_points: dict[Joint, tuple[float, float]]  # normalised (x, y) in the unmirrored frame
    landmarks_world: np.ndarray  # (33, 3) all MediaPipe world landmarks, for the recogniser
    landmarks_image: np.ndarray  # (33, 3) normalised image landmarks (x, y, z)
    landmarks_visibility: np.ndarray  # (33,)
    timestamp_ms: int


class PoseTracker(Protocol):
    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PersonPose | None:
        """Track the visitor in an *unmirrored* BGR frame. None when nobody is found."""

    def close(self) -> None: ...


# --------------------------------------------------------------------------- robot


class RobotTeacher(Protocol):
    def stance_ids(self) -> list[str]:
        """Stances the robot can show (ids from config/stances.yaml)."""

    def set_target(self, stance_id: str, *, mirrored: bool = False, duration_s: float = 2.0) -> None:
        """Start moving smoothly from the current pose into a stance over ``duration_s``.

        ``mirrored`` shows the other-side version of the stance (left and right swapped).
        """

    def go_neutral(self, duration_s: float = 1.5) -> None: ...

    def update(self, dt: float) -> None:
        """Advance the motion by ``dt`` seconds of wall-clock time."""

    def at_target(self) -> bool: ...

    def keypoints(self) -> Keypoints:
        """Current world positions of the 12 joints (shoulders, elbows, wrists, hips, knees, ankles)."""

    def render(self, width: int, height: int) -> np.ndarray:
        """BGR uint8 image of the robot, facing the viewer."""

    def close(self) -> None: ...


# --------------------------------------------------------------------------- ML


class StanceRecognizer(Protocol):
    def update(self, landmarks_world: np.ndarray | None, timestamp_ms: int) -> tuple[str | None, float]:
        """Feed one frame; return (smoothed stance id or None, confidence 0..1)."""

    def reset(self) -> None: ...


# --------------------------------------------------------------------------- coach


@dataclass
class PickResult:
    stance_id: str
    similarity: float  # 0..1, higher is better
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    source: str = "offline"  # "nebius" or "offline"


@dataclass
class RepResult:
    """What happened in one rep (one attempt at a stance), sent to the sensei."""

    rep_index: int
    stance_id: str
    mirrored: bool
    held: bool  # True if the visitor held the stance for the full hold time
    hold_s: float  # required hold time for this rep
    attempt_s: float  # seconds from the robot reaching the stance to the rep ending
    limb_errors_deg: dict[Limb, float | None]  # mean error per limb over the best window
    worst_limb: Limb | None
    recognised_as: str | None = None  # stance recogniser's label, if running


@dataclass
class SenseiDecision:
    next_stance_id: str
    mirrored: bool
    transition_s: float  # how long the robot takes to move into the stance (slower = bigger)
    hold_s: float
    say: str  # one short sentence for the visitor
    finished: bool = False  # True ends the session and shows the scorecard
    source: str = "offline"  # "nebius" or "offline"


@dataclass
class VisionTip:
    tip: str
    source: str = "nebius"


class MovePicker(Protocol):
    def pick(self, request: str) -> PickResult: ...


class Sensei(Protocol):
    def decide(self, history: Sequence[RepResult], *, reps_left: int) -> SenseiDecision: ...


class VisionCoach(Protocol):
    def tip(self, frame_bgr: np.ndarray, rep: RepResult) -> VisionTip | None: ...


class Voice(Protocol):
    def say(self, text: str) -> None:
        """Speak without blocking; a new sentence interrupts the previous one."""
