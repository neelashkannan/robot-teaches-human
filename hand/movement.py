"""Check a sign's movement template against the wrist path seen by the camera.

Only `circle` is supported (the BSL sign SORRY). The path is judged in the image plane, scaled by
hand size so it does not matter how far you sit from the camera. Direction does not matter, and
where on the screen (or your chest) you draw the circle is not checked.
"""
from __future__ import annotations

from collections import deque

import numpy as np

MIN_RADIUS = 0.3   # circle radius / hand size (wrist -> middle knuckle) must be at least this ...
MAX_RADIUS = 3.0   # ... and at most this
MAX_RADIUS_SPREAD = 0.6  # std/mean of the distance to the centre; rejects lines and zig-zags
CYCLE_FRACTION = 0.85    # a turn counts as a full cycle at 85% of a full revolution


class CircleCheck:
    """Feed it the wrist position each frame; `progress` goes 0 -> 1 as the cycles are drawn."""

    def __init__(self, cycles: float = 2.0, duration_s: float = 1.5):
        self.cycles = max(float(cycles), 0.5)
        self.window_s = max(float(duration_s), 0.5) * 2.0  # allow twice the template's time
        self._pts: deque[tuple[float, float, float, float]] = deque()  # (t, x, y, hand_size)
        self.progress = 0.0

    def reset(self) -> None:
        self._pts.clear()
        self.progress = 0.0

    def update(self, now: float, wrist, hand_size: float) -> float:
        """wrist is (x, y) in image space; hand_size in the same units. Returns progress 0..1."""
        if wrist is None or not hand_size or hand_size <= 0 or not np.isfinite(hand_size):
            self.reset()
            return 0.0
        self._pts.append((float(now), float(wrist[0]), float(wrist[1]), float(hand_size)))
        while self._pts and now - self._pts[0][0] > self.window_s:
            self._pts.popleft()
        self.progress = self._measure()
        return self.progress

    @property
    def done(self) -> bool:
        return self.progress >= 1.0

    def _measure(self) -> float:
        if len(self._pts) < 8:
            return 0.0
        a = np.array(self._pts)
        size = float(np.median(a[:, 3]))
        xy = a[:, 1:3] / size
        d = xy - xy.mean(axis=0)
        r = np.linalg.norm(d, axis=1)
        mean_r = float(r.mean())
        if not (MIN_RADIUS <= mean_r <= MAX_RADIUS) or float(r.std()) / mean_r > MAX_RADIUS_SPREAD:
            return 0.0
        ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        turns = abs(float(ang[-1] - ang[0])) / (2 * np.pi)
        return float(min(turns / (self.cycles * CYCLE_FRACTION), 1.0))


def make_check(movement: dict | None):
    """CircleCheck for a sign's movement template, or None if the sign is static / unsupported."""
    if movement and movement.get("type") == "circle":
        return CircleCheck(movement.get("cycles", 2), movement.get("duration_s", 1.5))
    return None
