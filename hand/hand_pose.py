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
    neutral_spread: np.ndarray  # per-finger raw spread angle that reads as spread 0 (not fitted)
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


# Defaults measured on a laptop webcam with MediaPipe world landmarks (app --debug readout, 17 Oct).
# The depth estimate is noisy, so a straight finger never reads 0: straight fingers read 26-65 degrees
# of bend and curled ones 93-143 (the ring finger cannot curl fully while the middle finger is up, as
# in TWO); a relaxed thumb reads 12-28 and a folded one 45-84. Open and fist sit so that the gap
# between those groups falls in the middle of the range.
DEFAULT_OPEN_CURL_DEG = (15.0, 40.0, 40.0, 40.0)
DEFAULT_FIST_CURL_DEG = (75.0, 120.0, 120.0, 120.0)
# Spread 0 is a finger parallel to the palm axis (wrist -> middle knuckle); a relaxed thumb sits about
# 20 degrees off that axis.
DEFAULT_NEUTRAL_SPREAD_DEG = (20.0, 0.0, 0.0, 0.0)
DEFAULT_MAX_SPREAD_DEG = (40.0, 25.0, 25.0, 25.0)


def _default_calibration() -> Calibration:
    """Fixed defaults used when no calibration is supplied."""
    return Calibration(
        open_curl=np.deg2rad(np.array(DEFAULT_OPEN_CURL_DEG)),
        fist_curl=np.deg2rad(np.array(DEFAULT_FIST_CURL_DEG)),
        neutral_spread=np.deg2rad(np.array(DEFAULT_NEUTRAL_SPREAD_DEG)),
        max_spread=np.deg2rad(np.array(DEFAULT_MAX_SPREAD_DEG)),
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

    open_curl = np.nanmedian(open_curls, axis=0)
    fist_curl = np.nanmedian(fist_curls, axis=0)

    default = _default_calibration()

    # If the user did not actually open/close, keep the default range.
    for i in range(len(vec.FINGERS)):
        if not (fist_curl[i] > open_curl[i] + 0.05):
            open_curl[i] = default.open_curl[i]
            fist_curl[i] = default.fist_curl[i]

    # Spread keeps the default neutral (fingers parallel to the palm axis). An open hand held up for
    # calibration is splayed, and the sign table counts that splay as spread (FIVE fans the index
    # and ring out), so taking the neutral from it would cancel exactly what FIVE asks for.
    return Calibration(
        open_curl=open_curl,
        fist_curl=fist_curl,
        neutral_spread=default.neutral_spread.copy(),
        max_spread=default.max_spread.copy(),
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


def hand_quality(lm_img: np.ndarray, margin: float = 0.02) -> tuple[bool, str]:
    """Can this view be marked fairly? Takes the 21 normalised IMAGE landmarks (x, y in 0..1).

    A webcam reads fingers badly when the hand is cut off, seen edge-on, or pointing at the camera
    (fingers foreshortened, thumb out of plane). Return (True, "") when the view is usable, else
    (False, one short instruction for the person). Marking should be paused, not failed, when False.
    """
    arr = np.asarray(lm_img, dtype=float)
    if arr.shape[0] != 21 or arr.shape[1] < 2 or not np.isfinite(arr).all():
        return False, "No hand detected."
    xy = arr[:, :2]
    if (xy < margin).any() or (xy > 1.0 - margin).any():
        return False, "Show the whole hand inside the picture."

    wrist, index_mcp, middle_mcp, little_mcp = xy[0], xy[5], xy[9], xy[17]
    a, b = index_mcp - wrist, little_mcp - wrist
    cross = abs(a[0] * b[1] - a[1] * b[0])
    if cross / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9) < 0.45:
        return False, "Turn your palm toward the camera."

    axis = middle_mcp - wrist
    length = float(np.linalg.norm(axis))
    if length < 0.04:
        return False, "Hold your hand closer to the camera."
    # Fingers should point up the picture (image y grows downward).
    if -axis[1] / length < np.cos(np.deg2rad(55.0)):
        return False, "Point your fingers up."
    return True, ""
