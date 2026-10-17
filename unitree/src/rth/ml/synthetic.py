"""Synthetic 33-landmark poses for testing the recogniser without a camera.

Each stance is a set of limb directions in the body frame (forward, left, up).
A synthetic "person" has their own size, limb proportions, habitual way of
doing each stance (a small consistent bias) and camera angle, so leave-one-
person-out evaluation on this data behaves like the real thing in miniature.
Output is in MediaPipe world-landmark axes (x right in the image, y down,
z away from the camera) for a person facing the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_D = dict[str, tuple[float, float, float]]

_DOWN = (0.0, 0.0, -1.0)

# Limb directions per stance, body frame (forward, left, up). Right-side vectors are given
# explicitly, so asymmetric stances (bow, crane, ...) are "left side forward" versions.
STANCES: dict[str, _D] = {
    "ready": {
        "left_upper_arm": (0.0, 0.15, -1), "left_forearm": (0.1, 0.1, -1),
        "right_upper_arm": (0.0, -0.15, -1), "right_forearm": (0.1, -0.1, -1),
        "left_thigh": _DOWN, "left_shin": _DOWN, "right_thigh": _DOWN, "right_shin": _DOWN,
    },
    "horse": {
        "left_upper_arm": (-0.6, 0.2, -0.8), "left_forearm": (1, 0, 0.1),
        "right_upper_arm": (-0.6, -0.2, -0.8), "right_forearm": (1, 0, 0.1),
        "left_thigh": (0.7, 0.7, -0.35), "left_shin": (0, 0.2, -1),
        "right_thigh": (0.7, -0.7, -0.35), "right_shin": (0, -0.2, -1),
    },
    "bow": {
        "left_upper_arm": (-0.6, 0.2, -0.8), "left_forearm": (1, 0, 0.1),
        "right_upper_arm": (1, 0.1, 0), "right_forearm": (1, 0.05, 0),
        "left_thigh": (0.9, 0.1, -0.45), "left_shin": (-0.1, 0, -1),
        "right_thigh": (-0.6, -0.1, -0.8), "right_shin": (-0.6, -0.05, -0.8),
    },
    "guard": {
        "left_upper_arm": (0.5, 0.3, -0.8), "left_forearm": (0.3, -0.25, 1),
        "right_upper_arm": (0.4, -0.3, -0.8), "right_forearm": (0.25, 0.25, 1),
        "left_thigh": (0.25, 0.1, -1), "left_shin": (0, 0, -1),
        "right_thigh": (-0.15, -0.1, -1), "right_shin": (0.1, 0, -1),
    },
    "crane": {
        "left_upper_arm": (0, 1, 0.5), "left_forearm": (0, 1, 0.6),
        "right_upper_arm": (0, -1, 0.5), "right_forearm": (0, -1, 0.6),
        "left_thigh": (1, 0.05, 0.1), "left_shin": (0, 0, -1),
        "right_thigh": _DOWN, "right_shin": _DOWN,
    },
    "white_crane": {
        "left_upper_arm": (0.15, 0.35, -1), "left_forearm": (0.7, 0, -0.4),
        "right_upper_arm": (0.2, -0.3, 1), "right_forearm": (0, 0.35, 1),
        "left_thigh": (0.45, 0.1, -1), "left_shin": (0.2, 0, -1),
        "right_thigh": (0.25, -0.15, -1), "right_shin": (-0.25, 0, -1),
    },
    "archer": {
        "left_upper_arm": (0, 1, 0), "left_forearm": (0, 1, 0.05),
        "right_upper_arm": (0.7, 0.35, 0), "right_forearm": (-0.1, 1, 0),
        "left_thigh": (0, 0.55, -1), "left_shin": (0, 0.4, -1),
        "right_thigh": (0, -0.55, -1), "right_shin": (0, -0.4, -1),
    },
}

_SEGMENTS = {"upper_arm": 0.29, "forearm": 0.26, "thigh": 0.43, "shin": 0.42}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rot(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Rotation about up (z), then left (y), then forward (x) of the body frame."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


# Body frame (forward, left, up) -> MediaPipe world (x right, y down, z away from camera)
# for a person facing the camera: forward = -z, left = +x (unmirrored image), up = -y.
_BODY_TO_MP = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])


@dataclass
class SyntheticPerson:
    """A body shape, a personal style and a camera set-up."""

    seed: int
    scale: float = 1.0
    segments: dict[str, float] = field(default_factory=dict)
    bias: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # stance -> limb -> offset
    yaw: float = 0.0
    pitch: float = 0.0

    @classmethod
    def random(cls, seed: int, style_deg: float = 10.0) -> SyntheticPerson:
        rng = np.random.default_rng(seed)
        segs = {k: v * rng.uniform(0.9, 1.1) for k, v in _SEGMENTS.items()}
        bias = {
            s: {limb: rng.normal(0, np.radians(style_deg), 3) for limb in dirs}
            for s, dirs in STANCES.items()
        }
        return cls(
            seed=seed,
            scale=float(rng.uniform(0.8, 1.2)),
            segments=segs,
            bias=bias,
            yaw=float(rng.uniform(-0.5, 0.5)),
            pitch=float(rng.uniform(-0.1, 0.1)),
        )


def pose_from_directions(
    dirs: dict[str, np.ndarray],
    segments: dict[str, float] | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """(33, 3) landmarks in the body frame, mid-hip at the origin."""
    seg = segments or _SEGMENTS
    lm = np.zeros((33, 3))
    pts = {
        11: (0, 0.19, 0.5), 12: (0, -0.19, 0.5), 23: (0, 0.1, 0), 24: (0, -0.1, 0),
        0: (0.1, 0, 0.72), 1: (0.09, 0.015, 0.75), 2: (0.09, 0.03, 0.75), 3: (0.085, 0.045, 0.75),
        4: (0.09, -0.015, 0.75), 5: (0.09, -0.03, 0.75), 6: (0.085, -0.045, 0.75),
        7: (0.0, 0.07, 0.73), 8: (0.0, -0.07, 0.73), 9: (0.09, 0.025, 0.69), 10: (0.09, -0.025, 0.69),
    }
    for i, p in pts.items():
        lm[i] = p
    chains = {
        "left": [(11, 13, "upper_arm"), (13, 15, "forearm"), (23, 25, "thigh"), (25, 27, "shin")],
        "right": [(12, 14, "upper_arm"), (14, 16, "forearm"), (24, 26, "thigh"), (26, 28, "shin")],
    }
    for side, links in chains.items():
        for a, b, name in links:
            lm[b] = lm[a] + seg[name] * _unit(np.asarray(dirs[f"{side}_{name}"], float))
        s = 1.0 if side == "left" else -1.0
        wrist, elbow = (15, 13) if side == "left" else (16, 14)
        fdir = _unit(lm[wrist] - lm[elbow])
        pinky, index, thumb = (17, 19, 21) if side == "left" else (18, 20, 22)
        lm[pinky] = lm[wrist] + 0.07 * fdir + (0, 0.02 * s, 0)
        lm[index] = lm[wrist] + 0.08 * fdir
        lm[thumb] = lm[wrist] + 0.05 * fdir + (0.02, 0, 0)
        ankle, heel, toe = (27, 29, 31) if side == "left" else (28, 30, 32)
        lm[heel] = lm[ankle] + (-0.05, 0, -0.05)
        lm[toe] = lm[ankle] + (0.15, 0.02 * s, -0.06)
    return lm * scale


def mirror_directions(dirs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Other-side version: swap left/right limbs and flip the left axis."""
    out = {}
    for limb, v in dirs.items():
        other = limb.replace("left", "@").replace("right", "left").replace("@", "right")
        out[other] = np.asarray(v, float) * (1, -1, 1)
    return out


def to_mediapipe(lm_body: np.ndarray, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                 offset: np.ndarray | None = None) -> np.ndarray:
    """Rotate a body-frame pose by a camera angle and express it in MediaPipe world axes."""
    out = lm_body @ _rot(yaw, pitch, roll).T @ _BODY_TO_MP.T
    return out if offset is None else out + offset


def synth_frames(
    stance_id: str,
    person: SyntheticPerson,
    n_frames: int,
    rng: np.random.Generator,
    *,
    mirrored: bool = False,
    jitter_deg: float = 6.0,
    noise_m: float = 0.01,
) -> np.ndarray:
    """(n_frames, 33, 3) of one person holding a stance: personal bias, slow drift, noise."""
    base = {
        limb: _unit(np.asarray(v, float)) + person.bias[stance_id][limb]
        for limb, v in STANCES[stance_id].items()
    }
    drift = {limb: np.zeros(3) for limb in base}
    frames = []
    for _ in range(n_frames):
        dirs = {}
        for limb, v in base.items():
            drift[limb] = 0.95 * drift[limb] + rng.normal(0, np.radians(jitter_deg) * 0.3, 3)
            dirs[limb] = v + drift[limb] + rng.normal(0, np.radians(jitter_deg) * 0.3, 3)
        if mirrored:
            dirs = mirror_directions(dirs)
        lm = pose_from_directions(dirs, person.segments, person.scale)
        lm = to_mediapipe(lm, person.yaw + rng.normal(0, 0.03), person.pitch, rng.normal(0, 0.02))
        frames.append(lm + rng.normal(0, noise_m, lm.shape))
    return np.stack(frames)


def random_poses(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 33, 3) poses with every limb pointing in a uniformly random direction."""
    out = []
    for _ in range(n):
        dirs = {limb: rng.normal(size=3) for limb in STANCES["ready"]}
        out.append(to_mediapipe(pose_from_directions(dirs), rng.uniform(-0.5, 0.5)))
    return np.stack(out)


def write_recordings(
    root: str | Path,
    n_people: int = 4,
    n_frames: int = 60,
    stances: list[str] | None = None,
    seed: int = 0,
    fps: float = 30.0,
) -> list[Path]:
    """Write a synthetic recordings tree (same layout as real ones); returns the files."""
    from rth.ml.dataset import save_recording

    rng = np.random.default_rng(seed)
    paths = []
    for p in range(n_people):
        person = SyntheticPerson.random(seed * 100 + p)
        for i, stance in enumerate(stances or list(STANCES)):
            lm = synth_frames(stance, person, n_frames, rng, mirrored=(p + i) % 3 == 0)
            ts = (np.arange(n_frames) * 1000 / fps).astype(np.int64)
            vis = np.full((n_frames, 33), 0.95)
            paths.append(save_recording(root, f"person{p}", stance, lm, vis, ts, source="synthetic"))
    return paths
