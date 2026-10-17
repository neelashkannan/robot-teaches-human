import numpy as np
import pytest

from hand.teacher import FINGER_TO_MJCF, Teacher, vec_to_ctrl
from hand.vec import FINGERS


@pytest.fixture(scope="module")
def teacher():
    t = Teacher()
    yield t
    t.close()


def settle(t, vec):
    t.set_pose(vec)
    t.step(600)
    return t.tip_positions()


def palm_point(t):
    # a point on the palm, below the knuckles (world frame, hand base is fixed)
    return np.array([0.04, -0.01, 0.03])


def test_finger_map_is_a_permutation():
    assert sorted(FINGER_TO_MJCF) == sorted(FINGERS)
    assert sorted(FINGER_TO_MJCF.values()) == [1, 2, 3, 4]


def test_ctrl_within_actuator_range():
    for v in (np.zeros(8), np.array([1, 1] * 4, float), np.array([1, -1] * 4, float)):
        c = vec_to_ctrl(v)
        assert c.shape == (8,) and np.all(np.abs(c) <= 1.5708)


def test_open_is_straight_and_fist_is_closer_to_palm(teacher):
    op = settle(teacher, np.zeros(8))
    fist = settle(teacher, np.array([1.0, 0.0] * 4))
    p = palm_point(teacher)
    for f in FINGERS:
        assert np.linalg.norm(fist[f] - p) < np.linalg.norm(op[f] - p), f
    # open hand: the three fingertips are well above the palm
    for f in ("index", "middle", "ring"):
        assert op[f][2] > 0.13


def test_each_curl_moves_only_that_finger(teacher):
    base = settle(teacher, np.zeros(8))
    for i, f in enumerate(FINGERS):
        v = np.zeros(8)
        v[2 * i] = 1.0
        tips = settle(teacher, v)
        assert np.linalg.norm(tips[f] - base[f]) > 0.03, f
        for g in FINGERS:
            if g != f:
                assert np.linalg.norm(tips[g] - base[g]) < 1e-3, (f, g)


def test_spread_moves_tip_sideways(teacher):
    base = settle(teacher, np.zeros(8))
    for i, f in enumerate(FINGERS):
        v = np.zeros(8)
        v[2 * i + 1] = 1.0
        tips = settle(teacher, v)
        assert abs(tips[f][1] - base[f][1]) > 0.01, f


def test_render_shape_and_not_blank(teacher):
    img = teacher.render(320, 240)
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8
    assert img.std() > 5
