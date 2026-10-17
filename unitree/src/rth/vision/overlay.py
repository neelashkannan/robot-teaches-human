"""Draw the visitor's skeleton on the display image, limbs coloured by score."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from rth.contracts import Joint, Limb, LimbScore, PersonPose

OK_BGR = (80, 200, 60)  # green: limb matches the robot
BAD_BGR = (40, 40, 230)  # red: limb is off
UNSEEN_BGR = (150, 150, 150)  # grey: limb not seen this frame
NEUTRAL_BGR = (230, 200, 40)  # no score yet (e.g. vision-check)
TORSO_BGR = (235, 235, 235)
OUTLINE_BGR = (20, 20, 20)

MIN_VISIBILITY = 0.5

_TORSO_LINES = (
    (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
    (Joint.LEFT_HIP, Joint.RIGHT_HIP),
    (Joint.LEFT_SHOULDER, Joint.LEFT_HIP),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_HIP),
)


def to_pixel(
    xy: tuple[float, float], width: int, height: int, *, mirrored: bool
) -> tuple[int, int]:
    """Normalised point in the unmirrored frame -> pixel in the display image."""
    x, y = xy
    if mirrored:
        x = 1.0 - x
    return round(x * width), round(y * height)


def limb_color(limb: Limb, scores: Mapping[Limb, LimbScore] | None, seen: bool) -> tuple:
    if not seen:
        return UNSEEN_BGR
    if scores is None or limb not in scores:
        return NEUTRAL_BGR
    s = scores[limb]
    if s.error_deg is None:
        return UNSEEN_BGR
    return OK_BGR if s.ok else BAD_BGR


def _line(img: np.ndarray, a, b, color, thick: int) -> None:
    # A dark outline under every stroke keeps it readable on any background.
    cv2.line(img, a, b, OUTLINE_BGR, thick + 4, cv2.LINE_AA)
    cv2.line(img, a, b, color, thick, cv2.LINE_AA)


def draw_skeleton(
    img: np.ndarray,
    pose: PersonPose | None,
    scores: Mapping[Limb, LimbScore] | None = None,
    *,
    mirrored: bool,
) -> None:
    """Draw in place on the display image.

    ``mirrored`` says the display image is the horizontally flipped camera frame,
    so x becomes 1 - x. Image points always come from the unmirrored frame.
    """
    if pose is None:
        return
    h, w = img.shape[:2]
    thick = max(3, h // 80)
    px = {j: to_pixel(p, w, h, mirrored=mirrored) for j, p in pose.image_points.items()}

    def seen(j: Joint) -> bool:
        return j in px and pose.visibility.get(j, 0.0) >= MIN_VISIBILITY

    for a, b in _TORSO_LINES:
        if seen(a) and seen(b):
            _line(img, px[a], px[b], TORSO_BGR, max(2, thick // 2))

    for limb in Limb:
        a, b = limb.joints
        if a not in px or b not in px:
            continue
        ok = seen(a) and seen(b)
        _line(img, px[a], px[b], limb_color(limb, scores, ok), thick if ok else max(2, thick // 2))

    r = max(3, int(thick * 0.8))
    for j, p in px.items():
        if seen(j):
            cv2.circle(img, p, r + 2, OUTLINE_BGR, -1, cv2.LINE_AA)
            cv2.circle(img, p, r, TORSO_BGR, -1, cv2.LINE_AA)
