"""Turn MediaPipe 21 hand landmarks into the 8-number HandVec.

Convention
----------
- Input ``lm`` is 21 rows of ``(x, y, z)`` in MediaPipe order.
- ``handedness`` is ``"Left"`` or ``"Right"``.
- This file assumes the webcam image is mirrored (selfie view). For a Left
  hand in a mirrored image we flip the x axis before measuring, so a mirrored
  left hand produces the same HandVec as the equivalent right hand. Spread
  signs therefore already match the right-hand convention.
- Bad input: wrong shape or an all-zero array raises ``ValueError``. NaN
  values produce the neutral pose (all zeros) instead of crashing.

The 8-vector order is defined in ``hand.vec``: for thumb, index, middle, ring,
curl then spread.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import vec


# MediaPipe joint indices for the four robot fingers (little finger ignored).
FINGER_IDX = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
}

# Metacarpophalangeal joint used as the finger base for spread measurement.
_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13}


@dataclass(frozen=True)
class Calibration:
    """Per-person calibration from a short open-hand + fist warm-up."""

    open_curl: np.ndarray  # per-finger raw summed curl angle for open hand
    fist_curl: np.ndarray  # per-finger raw summed curl angle for a fist
    neutral_spread: np.ndarray  # per-finger raw spread angle for open hand
    max_spread: np.ndarray  # per-finger raw angle that maps to spread == 1


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return a unit vector; return zero if the input is too short."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle (radians) between two vectors, robust to zero vectors."""
    ua = _normalize(a)
    ub = _normalize(b)
    if np.allclose(ua, 0) or np.allclose(ub, 0):
        return 0.0
    cos = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    return float(np.arccos(cos))


def _mirror_left(lm: np.ndarray, handedness: str) -> np.ndarray:
    """Flip x for a left hand so mirrored-webcam left == right."""
    if handedness not in ("Left", "Right"):
        raise ValueError(
            f"handedness must be 'Left' or 'Right', got {handedness!r}"
        )
    if handedness == "Right":
        return lm
    out = lm.copy()
    out[:, 0] *= -1.0
    return out


def _raw_curl(lm: np.ndarray, finger: str) -> float:
    """Sum the angles between consecutive bones of one finger."""
    idx = FINGER_IDX[finger]
    total = 0.0
    for i in range(len(idx) - 2):
        a = lm[idx[i + 1]] - lm[idx[i]]
        b = lm[idx[i + 2]] - lm[idx[i + 1]]
        total += _angle_between(a, b)
    return total


def _raw_curl_all(lm: np.ndarray) -> np.ndarray:
    """Per-finger raw curl for all four fingers."""
    return np.array([_raw_curl(lm, f) for f in vec.FINGERS], dtype=float)


def _raw_spread(lm: np.ndarray, finger: str) -> float:
    """Signed in-palm angle between the finger's first bone (MCP -> next joint) and the palm axis.

    The palm axis is wrist -> middle MCP. Measuring the bone direction, not the knuckle position,
    is what makes spread follow the finger: the knuckles sit still when you fan your fingers out.
    The bone is projected onto the palm plane first so bending the knuckle does not read as spread.
    Positive means toward the thumb side of the axis; negative means the pinky side.
    """
    wrist = lm[0]
    mcp = lm[_MCP[finger]]
    axis = lm[_MCP["middle"]] - wrist
    thumb_vec = lm[_MCP["thumb"]] - wrist
    bone = lm[_MCP[finger] + 1] - mcp

    # The thumb defines the palm plane and which side of the axis is the thumb side.
    normal = _normalize(np.cross(axis, thumb_vec))
    if not normal.any():
        return 0.0
    in_plane = bone - float(np.dot(bone, normal)) * normal
    if float(np.linalg.norm(in_plane)) < 1e-12:
        return 0.0
    return float(np.arctan2(np.dot(np.cross(axis, in_plane), normal), np.dot(axis, in_plane)))


def _raw_spread_all(lm: np.ndarray) -> np.ndarray:
    """Per-finger raw spread for all four fingers."""
    return np.array([_raw_spread(lm, f) for f in vec.FINGERS], dtype=float)


def _canonical_open_hand() -> np.ndarray:
    """Generic open right hand used to seed the default calibration."""
    lm = np.zeros((21, 3), dtype=float)
    lm[0] = (0.0, 0.0, 0.0)
    # thumb (positive y is the thumb side of the hand)
    lm[1] = (0.4, 0.22, 0.0)
    lm[2] = (0.65, 0.35, 0.0)
    lm[3] = (0.9, 0.48, 0.0)
    lm[4] = (1.15, 0.61, 0.0)
    # index
    lm[5] = (1.0, 0.0, 0.0)
    lm[6] = (2.0, 0.0, 0.0)
    lm[7] = (3.0, 0.0, 0.0)
    lm[8] = (4.0, 0.0, 0.0)
    # middle
    lm[9] = (1.0, 0.12, 0.0)
    lm[10] = (2.0, 0.12, 0.0)
    lm[11] = (3.0, 0.12, 0.0)
    lm[12] = (4.0, 0.12, 0.0)
    # ring
    lm[13] = (0.95, 0.24, 0.0)
    lm[14] = (1.9, 0.24, 0.0)
    lm[15] = (2.85, 0.24, 0.0)
    lm[16] = (3.8, 0.24, 0.0)
    # little (ignored by this module, but included for completeness)
    lm[17] = (0.85, 0.38, 0.0)
    lm[18] = (1.7, 0.38, 0.0)
    lm[19] = (2.55, 0.38, 0.0)
    lm[20] = (3.4, 0.38, 0.0)
    return lm


def _canonical_fist_hand() -> np.ndarray:
    """Generic fist used to seed the default calibration."""
    lm = np.zeros((21, 3), dtype=float)
    lm[0] = (0.0, 0.0, 0.0)
    # thumb folded across palm (positive y is the thumb side)
    lm[1] = (0.35, 0.15, 0.0)
    lm[2] = (0.6, 0.45, 0.0)
    lm[3] = (0.75, 0.75, 0.0)
    lm[4] = (0.85, 1.05, 0.0)
    # index curled (~60 deg between segments -> total ~120 deg)
    lm[5] = (1.0, 0.0, 0.0)
    lm[6] = (2.0, 0.0, 0.0)
    lm[7] = (2.5, 0.87, 0.0)
    lm[8] = (2.0, 1.73, 0.0)
    # middle curled
    lm[9] = (1.0, 0.12, 0.0)
    lm[10] = (2.0, 0.12, 0.0)
    lm[11] = (2.5, 0.99, 0.0)
    lm[12] = (2.0, 1.85, 0.0)
    # ring curled
    lm[13] = (0.95, 0.24, 0.0)
    lm[14] = (1.9, 0.24, 0.0)
    lm[15] = (2.4, 1.11, 0.0)
    lm[16] = (1.9, 1.97, 0.0)
    # little
    lm[17] = (0.85, 0.38, 0.0)
    lm[18] = (1.7, 0.38, 0.0)
    lm[19] = (2.2, 1.25, 0.0)
    lm[20] = (1.7, 2.11, 0.0)
    return lm


def _default_calibration() -> Calibration:
    """Fixed defaults used when no calibration is supplied."""
    open_lm = _canonical_open_hand()
    fist_lm = _canonical_fist_hand()
    open_curl = _raw_curl_all(open_lm)
    fist_curl = _raw_curl_all(fist_lm)
    neutral_spread = _raw_spread_all(open_lm)
    max_spread = np.deg2rad(np.array([40.0, 25.0, 25.0, 25.0]))
    return Calibration(
        open_curl=open_curl,
        fist_curl=fist_curl,
        neutral_spread=neutral_spread,
        max_spread=max_spread,
    )


def fit_calibration(
    open_frames: Sequence[np.ndarray],
    fist_frames: Sequence[np.ndarray],
) -> Calibration:
    """Create a Calibration from open-hand and fist landmark frames.

    Uses the median raw curl/spread angle across frames to ignore noise.
    Falls back to the default calibration if too few frames are supplied or
    the measured range is not monotonic.
    """
    if not open_frames or not fist_frames:
        return _default_calibration()

    open_curls = np.array([_raw_curl_all(f) for f in open_frames])
    fist_curls = np.array([_raw_curl_all(f) for f in fist_frames])
    neutral_spreads = np.array([_raw_spread_all(f) for f in open_frames])

    open_curl = np.nanmedian(open_curls, axis=0)
    fist_curl = np.nanmedian(fist_curls, axis=0)
    neutral_spread = np.nanmedian(neutral_spreads, axis=0)

    default = _default_calibration()
    max_spread = default.max_spread.copy()

    # If the user did not actually open/close, keep the default range.
    for i in range(len(vec.FINGERS)):
        if not (fist_curl[i] > open_curl[i] + 0.05):
            open_curl[i] = default.open_curl[i]
            fist_curl[i] = default.fist_curl[i]
            neutral_spread[i] = default.neutral_spread[i]

    return Calibration(
        open_curl=open_curl,
        fist_curl=fist_curl,
        neutral_spread=neutral_spread,
        max_spread=max_spread,
    )


def landmarks_to_vec(
    lm: np.ndarray,
    handedness: str,
    calib: Calibration | None = None,
) -> np.ndarray:
    """Convert 21 MediaPipe landmarks to an 8-number HandVec.

    Parameters
    ----------
    lm : np.ndarray
        21 rows of ``(x, y, z)`` in MediaPipe order.
    handedness : str
        ``"Left"`` or ``"Right"``. Left hands are x-flipped assuming a
        mirrored webcam.
    calib : Calibration, optional
        Per-person calibration. If ``None``, sensible fixed defaults are used.

    Returns
    -------
    np.ndarray
        Shape ``(8,)``, values in range and finite.

    Raises
    ------
    ValueError
        If ``lm`` does not have shape ``(21, 3)`` or is all zeros.
    """
    arr = np.asarray(lm, dtype=float)
    if arr.shape != (21, 3):
        raise ValueError(f"expected lm shape (21, 3), got {arr.shape}")
    if np.allclose(arr, 0):
        raise ValueError("landmarks are all zero")

    if calib is None:
        calib = _default_calibration()

    # NaN: return the neutral pose rather than propagating garbage.
    if not np.isfinite(arr).all():
        return vec.clamp(np.zeros(vec.N))

    lm = _mirror_left(arr, handedness)

    curls = _raw_curl_all(lm)
    spreads = _raw_spread_all(lm)

    out = np.zeros(vec.N, dtype=float)
    for i, _ in enumerate(vec.FINGERS):
        open_c = calib.open_curl[i]
        fist_c = calib.fist_curl[i]
        curl_range = fist_c - open_c
        if curl_range > 1e-6:
            c = (curls[i] - open_c) / curl_range
        else:
            c = 0.0 if curls[i] <= open_c + 1e-6 else 1.0
        out[i * 2] = float(np.clip(c, 0.0, 1.0))

        s = (spreads[i] - calib.neutral_spread[i]) / calib.max_spread[i]
        out[i * 2 + 1] = float(np.clip(s, -1.0, 1.0))

    return vec.clamp(out)
