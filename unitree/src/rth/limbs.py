"""The shared language: both bodies become 8 limb directions in their own torso frame.

Directions ignore limb length, so the G1's proportions don't matter, and the
torso frame means it doesn't matter where the visitor stands or which way the
camera looks. Torso-frame axes are (forward, left, up).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from rth.contracts import Joint, Keypoints, Limb, LimbScore, Scores

_EPS = 1e-6

FORWARD, LEFT, UP = 0, 1, 2


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < _EPS else v / n


def torso_frame(kp: Keypoints) -> np.ndarray | None:
    """3x3 matrix whose rows are the body's forward, left and up axes in input coordinates.

    Up runs from mid-hip to mid-shoulder. Left is the hip and shoulder lines, made
    orthogonal to up. Forward = left x up, which assumes right-handed coordinates.
    Returns None if the torso is degenerate (missing joints or zero-length axes).
    """
    try:
        ls, rs = kp[Joint.LEFT_SHOULDER], kp[Joint.RIGHT_SHOULDER]
        lh, rh = kp[Joint.LEFT_HIP], kp[Joint.RIGHT_HIP]
    except KeyError:
        return None
    up = _unit((ls + rs) / 2 - (lh + rh) / 2)
    if up is None:
        return None
    across = (ls - rs) + (lh - rh)
    left = _unit(across - np.dot(across, up) * up)
    if left is None:
        return None
    forward = np.cross(left, up)
    return np.stack([forward, left, up])


def limb_directions(
    kp: Keypoints,
    visibility: Mapping[Joint, float] | None = None,
    min_visibility: float = 0.5,
) -> dict[Limb, np.ndarray | None]:
    """Unit direction of each limb (proximal -> distal) in the body's torso frame.

    A limb is None when either joint is missing or below ``min_visibility``,
    or when the torso itself can't be found (then every limb is None).
    """
    frame = torso_frame(kp)
    out: dict[Limb, np.ndarray | None] = {}
    for limb in Limb:
        a, b = limb.joints
        seen = a in kp and b in kp
        if seen and visibility is not None:
            seen = visibility.get(a, 0.0) >= min_visibility and visibility.get(b, 0.0) >= min_visibility
        if frame is None or not seen:
            out[limb] = None
            continue
        d = _unit(np.asarray(kp[b], float) - np.asarray(kp[a], float))
        out[limb] = None if d is None else frame @ d
    return out


def mirror_directions(dirs: Mapping[Limb, np.ndarray | None]) -> dict[Limb, np.ndarray | None]:
    """The mirror image of a pose: swap left and right limbs and flip the left axis.

    Used when the visitor copies the robot like a mirror: their left arm
    matches the robot's right arm, reflected.
    """
    flip = np.array([1.0, -1.0, 1.0])
    return {limb: (None if dirs[limb.mirror] is None else dirs[limb.mirror] * flip) for limb in Limb}


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


# How far a limb must reach across the camera's view (its length in the left/up plane, as a
# fraction of 1) before its up/down-left/right angle is trusted over the full 3D angle.
FRONTAL_MIN, FRONTAL_FULL = 0.2, 0.5


def _reach(d: np.ndarray) -> float:
    return float(np.hypot(d[LEFT], d[UP]))


def frontal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions seen face-on: their left/up parts only."""
    a2, b2 = a[[LEFT, UP]], b[[LEFT, UP]]
    na, nb = float(np.linalg.norm(a2)), float(np.linalg.norm(b2))
    if na < _EPS or nb < _EPS:
        return angle_deg(a, b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a2 / na, b2 / nb), -1.0, 1.0))))


def limb_error(person_dir: np.ndarray, target_dir: np.ndarray) -> float:
    """Degrees between a visitor's limb and the robot's, robust to webcam depth errors.

    Depth is a single webcam's weakest axis: on real visitors MediaPipe put hanging hands
    20-30 degrees in front of the body and pulled-back elbows in front of the shoulders. So
    when both limbs lie across the camera's view, only their face-on angle counts. When either
    points at or away from the camera, depth is the only signal and the full 3D angle is used.
    In between the two are blended.
    """
    full = angle_deg(person_dir, target_dir)
    reach = min(_reach(person_dir), _reach(target_dir))
    trust = float(np.clip((reach - FRONTAL_MIN) / (FRONTAL_FULL - FRONTAL_MIN), 0.0, 1.0))
    if trust == 0.0:
        return full
    return trust * frontal_angle_deg(person_dir, target_dir) + (1.0 - trust) * full


def compare(
    person: Mapping[Limb, np.ndarray | None],
    robot: Mapping[Limb, np.ndarray | None],
    tolerance_deg: float,
    *,
    mirror_view: bool = True,
) -> dict[Limb, LimbScore]:
    """Score each of the visitor's limbs against the robot's.

    With ``mirror_view`` the visitor is expected to copy the robot as a mirror
    image (they face it, like a class facing a teacher).
    """
    target = mirror_directions(robot) if mirror_view else robot
    scores: dict[Limb, LimbScore] = {}
    for limb in Limb:
        p, r = person.get(limb), target.get(limb)
        if p is None or r is None:
            scores[limb] = LimbScore(limb, None, False)
        else:
            err = limb_error(p, r)
            scores[limb] = LimbScore(limb, err, err <= tolerance_deg)
    return scores


def overall(scores: Scores, min_seen: int = 6) -> tuple[bool, float | None]:
    """(all seen limbs ok and enough seen, mean error of seen limbs)."""
    seen = [s for s in scores.values() if s.error_deg is not None]
    if not seen:
        return False, None
    mean = float(np.mean([s.error_deg for s in seen]))
    return len(seen) >= min_seen and all(s.ok for s in seen), mean


def worst_limb(scores: Scores) -> Limb | None:
    seen = [s for s in scores.values() if s.error_deg is not None and not s.ok]
    return max(seen, key=lambda s: s.error_deg).limb if seen else None


_ARM_WORDS = {
    (UP, +1): "raise it",
    (UP, -1): "lower it",
    (FORWARD, +1): "bring it forward",
    (FORWARD, -1): "bring it back",
}
_LEG_WORDS = {
    (UP, +1): "lift it higher",
    (UP, -1): "point it more down",
    (FORWARD, +1): "bring it forward",
    (FORWARD, -1): "bring it back",
}


def correction_hint(
    limb: Limb,
    person_dir: np.ndarray,
    target_dir: np.ndarray,
) -> str:
    """Short coaching line for one limb, e.g. 'Left forearm: raise it'.

    ``target_dir`` must already be in the frame the visitor should match
    (i.e. mirrored if ``mirror_view``). Picks the axis with the largest change.
    """
    delta = target_dir - person_dir
    # Forward/back only when the limb points at or away from the camera: elsewhere webcam
    # depth is too unreliable to coach on. On near-ties prefer up/down.
    forward_w = 0.0 if min(_reach(person_dir), _reach(target_dir)) >= FRONTAL_FULL else 1.02
    axis = int(np.argmax(np.abs(delta) * np.array([forward_w, 1.0, 1.05])))
    sign = 1 if delta[axis] > 0 else -1
    if axis == LEFT:
        # Towards the body's left is "out" for a left limb and "in" for a right limb.
        outward = sign > 0 if limb.side == "left" else sign < 0
        words = "move it out to the side" if outward else "bring it in"
    else:
        table = _ARM_WORDS if "arm" in limb.value else _LEG_WORDS
        words = table[(axis, sign)]
    return f"{limb.label}: {words}"


def hint_for(
    scores: Scores,
    person: Mapping[Limb, np.ndarray | None],
    robot: Mapping[Limb, np.ndarray | None],
    *,
    mirror_view: bool = True,
) -> str | None:
    """Coaching line for the worst limb, or None if every seen limb is ok."""
    limb = worst_limb(scores)
    if limb is None:
        return None
    target = mirror_directions(robot) if mirror_view else robot
    p, t = person.get(limb), target.get(limb)
    if p is None or t is None:
        return None
    return correction_hint(limb, p, t)
