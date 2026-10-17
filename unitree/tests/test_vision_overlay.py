import numpy as np

from rth.contracts import Joint, Limb, LimbScore, PersonPose
from rth.vision.overlay import (
    BAD_BGR,
    NEUTRAL_BGR,
    OK_BGR,
    UNSEEN_BGR,
    draw_skeleton,
    limb_color,
    to_pixel,
)

# Normalised image points of a person standing on the left third of the unmirrored frame.
_POINTS = {
    Joint.LEFT_SHOULDER: (0.30, 0.30),
    Joint.RIGHT_SHOULDER: (0.20, 0.30),
    Joint.LEFT_ELBOW: (0.33, 0.42),
    Joint.RIGHT_ELBOW: (0.17, 0.42),
    Joint.LEFT_WRIST: (0.34, 0.54),
    Joint.RIGHT_WRIST: (0.16, 0.54),
    Joint.LEFT_HIP: (0.28, 0.58),
    Joint.RIGHT_HIP: (0.22, 0.58),
    Joint.LEFT_KNEE: (0.28, 0.74),
    Joint.RIGHT_KNEE: (0.22, 0.74),
    Joint.LEFT_ANKLE: (0.28, 0.90),
    Joint.RIGHT_ANKLE: (0.22, 0.90),
}


def _pose(vis: float = 1.0) -> PersonPose:
    return PersonPose(
        keypoints={j: np.zeros(3) for j in Joint},
        visibility={j: vis for j in Joint},
        image_points=dict(_POINTS),
        landmarks_world=np.zeros((33, 3)),
        landmarks_image=np.zeros((33, 3)),
        landmarks_visibility=np.full(33, vis),
        timestamp_ms=0,
    )


def _ink_columns(img):
    cols = np.where(img.any(axis=(0, 2)))[0]
    return cols.min(), cols.max()


def test_to_pixel_mirrors_x_only():
    assert to_pixel((0.25, 0.5), 400, 200, mirrored=False) == (100, 100)
    assert to_pixel((0.25, 0.5), 400, 200, mirrored=True) == (300, 100)


def test_unmirrored_draws_where_the_person_is():
    img = np.zeros((360, 640, 3), np.uint8)
    draw_skeleton(img, _pose(), mirrored=False)
    _lo, hi = _ink_columns(img)
    assert hi < 640 * 0.5


def test_mirrored_draws_on_the_flipped_side():
    img = np.zeros((360, 640, 3), np.uint8)
    draw_skeleton(img, _pose(), mirrored=True)
    lo, _hi = _ink_columns(img)
    assert lo > 640 * 0.5
    # Drawing on the mirrored frame equals mirroring the unmirrored drawing.
    plain = np.zeros_like(img)
    draw_skeleton(plain, _pose(), mirrored=False)
    flipped = plain[:, ::-1]
    overlap = (img.any(axis=2) & flipped.any(axis=2)).sum() / img.any(axis=2).sum()
    assert overlap > 0.9


def _pixel_on_limb(img, limb, mirrored=False):
    a, b = (
        np.array(to_pixel(_POINTS[j], img.shape[1], img.shape[0], mirrored=mirrored))
        for j in limb.joints
    )
    x, y = ((a + b) / 2).astype(int)
    return tuple(int(c) for c in img[y, x])


def test_limbs_coloured_by_score():
    img = np.zeros((720, 1280, 3), np.uint8)
    scores = {limb: LimbScore(limb, 5.0, True) for limb in Limb}
    scores[Limb.RIGHT_FOREARM] = LimbScore(Limb.RIGHT_FOREARM, 70.0, False)
    scores[Limb.LEFT_SHIN] = LimbScore(Limb.LEFT_SHIN, None, False)
    draw_skeleton(img, _pose(), scores, mirrored=True)
    assert _pixel_on_limb(img, Limb.LEFT_THIGH, mirrored=True) == OK_BGR
    assert _pixel_on_limb(img, Limb.RIGHT_FOREARM, mirrored=True) == BAD_BGR
    assert _pixel_on_limb(img, Limb.LEFT_SHIN, mirrored=True) == UNSEEN_BGR


def test_limb_color_rules():
    ok = {Limb.LEFT_THIGH: LimbScore(Limb.LEFT_THIGH, 3.0, True)}
    assert limb_color(Limb.LEFT_THIGH, ok, True) == OK_BGR
    assert limb_color(Limb.LEFT_THIGH, ok, False) == UNSEEN_BGR
    assert limb_color(Limb.LEFT_THIGH, None, True) == NEUTRAL_BGR
    assert limb_color(Limb.RIGHT_THIGH, ok, True) == NEUTRAL_BGR


def test_low_visibility_limbs_are_grey():
    img = np.zeros((720, 1280, 3), np.uint8)
    scores = {limb: LimbScore(limb, 5.0, True) for limb in Limb}
    draw_skeleton(img, _pose(vis=0.1), scores, mirrored=False)
    assert _pixel_on_limb(img, Limb.LEFT_THIGH) == UNSEEN_BGR


def test_no_pose_draws_nothing():
    img = np.zeros((100, 100, 3), np.uint8)
    draw_skeleton(img, None, mirrored=True)
    assert not img.any()
