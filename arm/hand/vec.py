"""The shared language: a person's hand and the robot hand are both 8 numbers.

HandVec order: [thumb_curl, thumb_spread, index_curl, index_spread,
                middle_curl, middle_spread, ring_curl, ring_spread]
  curl   in [0, 1]   0 = finger straight, 1 = fully curled
  spread in [-1, 1]  0 = neutral, +1 = fingertip fully toward the thumb side, -1 = fully away

The Amazing Hand has 4 fingers (finger1..finger4 in the MJCF). Which of them plays the thumb is
settled in hand/teacher.py (FINGER_TO_MJCF) once checked against the model.
"""
from __future__ import annotations

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring")
NAMES = tuple(f"{f}_{k}" for f in FINGERS for k in ("curl", "spread"))
N = len(NAMES)  # 8

# Used only to turn normalised errors into words for the learner ("curl 20 degrees more").
CURL_FULL_DEG = 90.0
SPREAD_FULL_DEG = 20.0

LOW = np.array([0.0, -1.0] * 4)
HIGH = np.array([1.0, 1.0] * 4)


def clamp(v) -> np.ndarray:
    """Return v as a float array of shape (8,), clipped to range. NaN becomes 0."""
    a = np.nan_to_num(np.asarray(v, dtype=float).reshape(N), nan=0.0)
    return np.clip(a, LOW, HIGH)


def curl(v, finger: str) -> float:
    return float(v[FINGERS.index(finger) * 2])


def spread(v, finger: str) -> float:
    return float(v[FINGERS.index(finger) * 2 + 1])
