import numpy as np
import pytest

from rth.contracts import Joint, Limb
from rth.limbs import (
    angle_deg,
    compare,
    correction_hint,
    limb_directions,
    limb_error,
    mirror_directions,
    overall,
    torso_frame,
    worst_limb,
)


def standing(arms_out: bool = False) -> dict[Joint, np.ndarray]:
    """A body in MuJoCo-style coordinates: faces +x, left is +y, up is +z."""
    kp = {
        Joint.LEFT_SHOULDER: [0, 0.2, 1.4],
        Joint.RIGHT_SHOULDER: [0, -0.2, 1.4],
        Joint.LEFT_HIP: [0, 0.1, 0.9],
        Joint.RIGHT_HIP: [0, -0.1, 0.9],
        Joint.LEFT_KNEE: [0, 0.1, 0.5],
        Joint.RIGHT_KNEE: [0, -0.1, 0.5],
        Joint.LEFT_ANKLE: [0, 0.1, 0.1],
        Joint.RIGHT_ANKLE: [0, -0.1, 0.1],
    }
    if arms_out:
        kp |= {
            Joint.LEFT_ELBOW: [0, 0.5, 1.4],
            Joint.LEFT_WRIST: [0, 0.8, 1.4],
            Joint.RIGHT_ELBOW: [0, -0.5, 1.4],
            Joint.RIGHT_WRIST: [0, -0.8, 1.4],
        }
    else:
        kp |= {
            Joint.LEFT_ELBOW: [0, 0.2, 1.1],
            Joint.LEFT_WRIST: [0, 0.2, 0.8],
            Joint.RIGHT_ELBOW: [0, -0.2, 1.1],
            Joint.RIGHT_WRIST: [0, -0.2, 0.8],
        }
    return {j: np.asarray(v, float) for j, v in kp.items()}


def transform(kp, rot, shift):
    return {j: rot @ p + shift for j, p in kp.items()}


def rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def test_torso_frame_axes():
    f = torso_frame(standing())
    np.testing.assert_allclose(f, np.eye(3), atol=1e-9)


def test_directions_hanging_and_out():
    d = limb_directions(standing())
    np.testing.assert_allclose(d[Limb.LEFT_UPPER_ARM], [0, 0, -1], atol=1e-9)
    d = limb_directions(standing(arms_out=True))
    np.testing.assert_allclose(d[Limb.LEFT_UPPER_ARM], [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(d[Limb.RIGHT_FOREARM], [0, -1, 0], atol=1e-9)


def test_invariant_to_position_and_heading():
    base = limb_directions(standing(arms_out=True))
    moved = limb_directions(transform(standing(arms_out=True), rot_z(73), np.array([3.0, -2.0, 0.5])))
    for limb in Limb:
        np.testing.assert_allclose(base[limb], moved[limb], atol=1e-9)


def test_mediapipe_style_coordinates_match():
    """Person facing the camera in MediaPipe world coords (x right in image, y down, z away)."""
    # Body axes in camera coords: forward = -z, left = +x, up = -y.
    to_cam = np.stack([[0, 0, -1], [1, 0, 0], [0, -1, 0]], axis=1)
    person = transform(standing(arms_out=True), to_cam, np.zeros(3))
    a, b = limb_directions(person), limb_directions(standing(arms_out=True))
    for limb in Limb:
        np.testing.assert_allclose(a[limb], b[limb], atol=1e-9)


def test_visibility_hides_limbs():
    kp = standing()
    vis = {j: 1.0 for j in kp} | {Joint.LEFT_WRIST: 0.1}
    d = limb_directions(kp, vis)
    assert d[Limb.LEFT_FOREARM] is None
    assert d[Limb.LEFT_UPPER_ARM] is not None


def test_mirror_swaps_sides():
    kp = standing()
    kp[Joint.LEFT_ELBOW] = np.array([0, 0.5, 1.4])  # only left upper arm out
    d = mirror_directions(limb_directions(kp))
    np.testing.assert_allclose(d[Limb.RIGHT_UPPER_ARM], [0, -1, 0], atol=1e-9)
    np.testing.assert_allclose(d[Limb.LEFT_UPPER_ARM], [0, 0, -1], atol=1e-9)


def test_compare_mirror_view():
    robot = standing()
    robot[Joint.LEFT_ELBOW] = np.array([0, 0.5, 1.4])
    robot[Joint.LEFT_WRIST] = np.array([0, 0.8, 1.4])
    person = standing()
    person[Joint.RIGHT_ELBOW] = np.array([0, -0.5, 1.4])
    person[Joint.RIGHT_WRIST] = np.array([0, -0.8, 1.4])
    rd, pd = limb_directions(robot), limb_directions(person)
    mirrored = compare(pd, rd, 20, mirror_view=True)
    assert all(s.ok for s in mirrored.values())
    direct = compare(pd, rd, 20, mirror_view=False)
    assert not direct[Limb.LEFT_UPPER_ARM].ok
    assert direct[Limb.LEFT_UPPER_ARM].error_deg == pytest.approx(90)
    assert worst_limb(direct) in (Limb.LEFT_UPPER_ARM, Limb.LEFT_FOREARM, Limb.RIGHT_UPPER_ARM, Limb.RIGHT_FOREARM)
    assert overall(mirrored) == (True, pytest.approx(0, abs=1e-6))
    assert overall(direct)[0] is False


def test_overall_needs_enough_limbs():
    scores = compare({l: None for l in Limb}, limb_directions(standing()), 20)
    assert overall(scores) == (False, None)


def test_hint_words():
    up, down, fwd = np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), np.array([1.0, 0, 0])
    assert correction_hint(Limb.LEFT_UPPER_ARM, down, up) == "Left upper arm: raise it"
    assert correction_hint(Limb.RIGHT_THIGH, down, fwd).startswith("Right thigh:")
    # Mostly sideways corrections, with a little vertical change.
    near_l, far_l = np.array([0, 0.17, -0.98]), np.array([0, 0.94, -0.34])
    assert correction_hint(Limb.LEFT_UPPER_ARM, near_l, far_l).endswith("out to the side")
    assert correction_hint(Limb.LEFT_UPPER_ARM, far_l, near_l).endswith("bring it in")
    near_r, far_r = near_l * [1, -1, 1], far_l * [1, -1, 1]
    assert correction_hint(Limb.RIGHT_UPPER_ARM, near_r, far_r).endswith("out to the side")
    assert correction_hint(Limb.RIGHT_UPPER_ARM, far_r, near_r).endswith("bring it in")
    # Exact ties prefer the vertical word.
    assert correction_hint(Limb.LEFT_UPPER_ARM, down, np.array([0, 1.0, 0])).endswith("raise it")


def _u(*v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def test_limb_error_forgives_webcam_depth_bias():
    # Real visitor, arms hanging: MediaPipe put the hands well in front of the elbows.
    person, robot = _u(0.45, -0.10, -0.88), _u(0.07, -0.15, -0.99)
    assert angle_deg(person, robot) > 20
    assert limb_error(person, robot) < 10


def test_limb_error_still_catches_pointing_at_the_camera():
    # Arm straight out towards the camera while the robot's hangs down: a real mistake.
    assert limb_error(_u(1, 0, 0), _u(0, 0, -1)) == pytest.approx(90, abs=1)
    # Both pointing at the camera (horse-stance fists): only the 3D angle can tell them apart.
    p, r = _u(0.91, -0.28, -0.11), _u(1.0, 0.0, -0.05)
    assert limb_error(p, r) == pytest.approx(angle_deg(p, r))


def test_limb_error_face_on_difference_counts_fully():
    assert limb_error(_u(0, 0, -1), _u(0, 1, 0)) == pytest.approx(90)


def test_hint_ignores_depth_for_limbs_across_the_view():
    hint = correction_hint(Limb.LEFT_FOREARM, _u(0.45, 0.6, -0.6), _u(-0.3, 0.2, -0.95))
    assert "forward" not in hint and "back" not in hint
