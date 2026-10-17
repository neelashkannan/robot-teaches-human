"""Per-limb smoothing of unit direction vectors between frames."""

from __future__ import annotations

import numpy as np

from rth.contracts import Limb, LimbDirections

_EPS = 1e-6


class DirectionSmoother:
    """EMA on each limb's unit vector, renormalised after every step.

    ``alpha`` is the weight of the newest frame (1 = no smoothing). A limb that
    drops out keeps its last value for ``hold_frames`` frames, so a single missed
    detection doesn't flicker the score, then becomes None. A jump bigger than
    ``snap_deg`` replaces the old value outright: a real, fast move should not
    sweep slowly through the poses in between.
    """

    def __init__(self, alpha: float, *, hold_frames: int = 5, snap_deg: float = 60.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.hold_frames = int(hold_frames)
        self._snap_cos = float(np.cos(np.radians(snap_deg)))
        self._state: dict[Limb, np.ndarray] = {}
        self._missing: dict[Limb, int] = {}

    def update(self, dirs: LimbDirections) -> dict[Limb, np.ndarray | None]:
        """Feed one frame of limb directions; returns the smoothed directions."""
        out: dict[Limb, np.ndarray | None] = {}
        for limb in Limb:
            new = dirs.get(limb)
            prev = self._state.get(limb)
            if new is None:
                out[limb] = self._hold(limb)
                continue
            new = np.asarray(new, dtype=float)
            n = float(np.linalg.norm(new))
            if n < _EPS:
                out[limb] = self._hold(limb)
                continue
            new = new / n
            self._missing[limb] = 0
            if prev is None or float(np.dot(prev, new)) < self._snap_cos:
                cur = new
            else:
                mixed = self.alpha * new + (1.0 - self.alpha) * prev
                m = float(np.linalg.norm(mixed))
                cur = new if m < _EPS else mixed / m
            self._state[limb] = cur
            out[limb] = cur.copy()
        return out

    def _hold(self, limb: Limb) -> np.ndarray | None:
        prev = self._state.get(limb)
        if prev is None:
            return None
        missing = self._missing.get(limb, 0) + 1
        self._missing[limb] = missing
        if missing > self.hold_frames:
            del self._state[limb]
            return None
        return prev.copy()

    def reset(self) -> None:
        self._state.clear()
        self._missing.clear()
