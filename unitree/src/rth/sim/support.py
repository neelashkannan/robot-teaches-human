"""Support polygon geometry: convex hull of the sole contacts and the COM's margin inside it.

All functions work on 2D points (x, y) on the floor. A positive margin means the
COM projection is inside the polygon by that many metres; negative means outside.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise hull vertices (monotone chain). Fewer than 3 points come back as is."""
    pts = np.unique(np.round(np.asarray(points, float).reshape(-1, 2), 9), axis=0)
    if len(pts) < 3:
        return pts
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= _EPS:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= _EPS:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _closest_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    e = b - a
    t = float(np.clip(np.dot(p - a, e) / max(float(np.dot(e, e)), _EPS), 0.0, 1.0))
    return a + t * e


def margin(hull: np.ndarray, p: np.ndarray) -> float:
    """Signed distance of ``p`` to the hull boundary (positive inside)."""
    n = len(hull)
    if n == 0:
        return -np.inf
    if n == 1:
        return -float(np.linalg.norm(p - hull[0]))
    dist = min(
        float(np.linalg.norm(p - _closest_on_segment(p, hull[i], hull[(i + 1) % n]))) for i in range(n)
    )
    if n < 3:
        return -dist  # a line has no inside
    inside = all(_cross(hull[i], hull[(i + 1) % n], p) >= 0.0 for i in range(n))
    return dist if inside else -dist


def shrink(hull: np.ndarray, by: float) -> np.ndarray:
    """The hull with every edge moved inward by ``by`` (falls back to the centroid if it vanishes)."""
    n = len(hull)
    if n < 3:
        return hull.copy()
    lines = []  # (point on the shifted edge, edge direction)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = (b - a) / max(float(np.linalg.norm(b - a)), _EPS)
        lines.append((a + by * np.array([-e[1], e[0]]), e))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < _EPS:
            out.append(p2)
            continue
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        out.append(p1 + t * d1)
    small = np.array(out)
    # Shrinking by more than the inradius turns the polygon inside out.
    if any(_cross(small[i], small[(i + 1) % n], small[(i + 2) % n]) < -_EPS for i in range(n)):
        return hull.mean(axis=0, keepdims=True)
    return small


def nearest_inside(hull: np.ndarray, p: np.ndarray, want_margin: float) -> np.ndarray:
    """Closest point to ``p`` that is at least ``want_margin`` inside the hull."""
    inner = shrink(hull, want_margin)
    if len(inner) < 3:
        return inner.mean(axis=0)
    if margin(inner, p) >= 0.0:
        return np.asarray(p, float).copy()
    n = len(inner)
    cands = [_closest_on_segment(p, inner[i], inner[(i + 1) % n]) for i in range(n)]
    return min(cands, key=lambda c: float(np.linalg.norm(c - p)))
