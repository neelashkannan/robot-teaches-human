"""Mark a user's HandVec against a target and say what to fix.

HandVec order comes from hand.vec: curl then spread for thumb, index, middle, ring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hand.vec import CURL_FULL_DEG, FINGERS, N, SPREAD_FULL_DEG, clamp


@dataclass(frozen=True)
class FingerResult:
    ok: bool
    curl_err_deg: float
    spread_err_deg: float


@dataclass(frozen=True)
class Result:
    per_finger: Tuple[FingerResult, ...]
    worst: str | None
    message: str
    all_ok: bool


def _round5(deg: float) -> int:
    return int(round(abs(deg) / 5.0) * 5)


def _message(worst: str, c_err_norm: float, s_err_norm: float) -> str:
    if abs(c_err_norm) >= abs(s_err_norm):
        deg = _round5(c_err_norm * CURL_FULL_DEG)
        if c_err_norm > 0:
            return f"{worst.capitalize()} finger: curl {deg} degrees more."
        return f"{worst.capitalize()} finger: straighten by {deg} degrees."
    deg = _round5(s_err_norm * SPREAD_FULL_DEG)
    if s_err_norm > 0:
        return f"{worst.capitalize()} finger: spread {deg} degrees toward the thumb."
    return f"{worst.capitalize()} finger: spread {deg} degrees away from the thumb."


def _valid_vec(v) -> bool:
    """Check a vector is a finite 8-D array without NaNs/Inf."""
    try:
        a = np.asarray(v, dtype=float)
    except Exception:
        return False
    return a.shape == (N,) and np.isfinite(a).all()


# Webcam landmarks wobble more sideways than along the finger, and a fully curled finger cannot
# spread at all (the teacher hand ignores it too), so spread is marked loosely and only on fingers
# the sign holds out.
SPREAD_TOL = 0.3
SPREAD_IGNORED_ABOVE_CURL = 0.8


def compare(user, target, tol: float = 0.15, spread_tol: float = SPREAD_TOL) -> Result:
    """Return per-finger marking, worst finger, and a one-line fix message.

    Signed errors are target - user in degrees: positive means the user needs
    more curl / more spread toward the thumb.
    """
    if not (_valid_vec(user) and _valid_vec(target)):
        not_ok = tuple(
            FingerResult(ok=False, curl_err_deg=0.0, spread_err_deg=0.0) for _ in FINGERS
        )
        return Result(per_finger=not_ok, worst=None, message="No hand detected.", all_ok=False)

    u = clamp(user)
    t = clamp(target)

    curl_err_norm = t[0::2] - u[0::2]
    spread_err_norm = np.where(t[0::2] <= SPREAD_IGNORED_ABOVE_CURL, t[1::2] - u[1::2], 0.0)

    per_finger = []
    worst_idx = None
    worst_norm = -1.0
    for i, finger in enumerate(FINGERS):
        c = curl_err_norm[i]
        s = spread_err_norm[i]
        ok = abs(c) <= tol and abs(s) <= spread_tol
        per_finger.append(
            FingerResult(
                ok=ok,
                curl_err_deg=float(c * CURL_FULL_DEG),
                spread_err_deg=float(s * SPREAD_FULL_DEG),
            )
        )
        if not ok:
            err = max(abs(c), abs(s) * tol / spread_tol)
            if err > worst_norm:
                worst_norm = err
                worst_idx = i

    all_ok = worst_idx is None
    if all_ok:
        return Result(
            per_finger=tuple(per_finger),
            worst=None,
            message="Handshape matches.",
            all_ok=True,
        )

    worst = FINGERS[worst_idx]
    msg = _message(worst, curl_err_norm[worst_idx], spread_err_norm[worst_idx])
    return Result(
        per_finger=tuple(per_finger),
        worst=worst,
        message=msg,
        all_ok=False,
    )


class Hold:
    """Accumulate seconds of `all_ok` true, tolerating a brief flicker."""

    def __init__(self, seconds: float = 3.0, flicker: float = 0.3):
        self.seconds = max(float(seconds), 0.0)
        self.flicker = max(float(flicker), 0.0)
        self._start: float | None = None
        self._last_ok: float = 0.0

    def update(self, all_ok: bool, now: float) -> tuple[float, bool]:
        if all_ok:
            self._last_ok = float(now)
            if self._start is None:
                self._start = float(now)
            progress = min((now - self._start) / self.seconds, 1.0) if self.seconds > 0 else 1.0
            return progress, progress >= 1.0

        if self._start is not None and (now - self._last_ok) > self.flicker:
            self._start = None

        if self._start is None:
            return 0.0, False

        progress = min((self._last_ok - self._start) / self.seconds, 1.0)
        return progress, progress >= 1.0
