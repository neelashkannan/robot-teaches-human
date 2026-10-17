"""Tests for hand.compare."""
import math

import numpy as np
import pytest

from hand.compare import Hold, compare


def vec(*pairs):
    """Build an 8-vector from ((curl, spread), ...) in finger order."""
    return np.array([v for pair in pairs for v in pair], dtype=float)


def test_identical_passes():
    v = vec((0.2, 0.1), (0.3, -0.1), (0.5, 0.0), (0.0, 0.4))
    r = compare(v, v)
    assert r.all_ok
    assert r.worst is None
    assert r.message == "Handshape matches."
    assert all(f.ok for f in r.per_finger)
    assert all(f.curl_err_deg == 0.0 and f.spread_err_deg == 0.0 for f in r.per_finger)


def test_one_finger_error_flags_only_that_finger():
    target = vec((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    user = vec((0.0, 0.0), (0.5, 0.0), (0.0, 0.0), (0.0, 0.0))
    r = compare(user, target)
    assert not r.all_ok
    assert r.worst == "index"
    assert not r.per_finger[1].ok
    assert r.per_finger[1].curl_err_deg == pytest.approx(-0.5 * 90.0)
    assert all(r.per_finger[i].ok for i in (0, 2, 3))


def test_worst_finger_choice():
    # index curl off by 0.40, middle curl off by 0.60 -> worst is middle.
    target = vec((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    user = vec((0.0, 0.0), (0.4, 0.0), (0.6, 0.0), (0.0, 0.0))
    r = compare(user, target)
    assert r.worst == "middle"


def test_spread_direction_message():
    # user needs to spread index more toward thumb side.
    target = vec((0.0, 0.0), (0.0, 0.5), (0.0, 0.0), (0.0, 0.0))
    user = vec((0.0, 0.0), (0.0, -0.5), (0.0, 0.0), (0.0, 0.0))
    r = compare(user, target)
    assert r.worst == "index"
    assert "spread" in r.message.lower()
    assert "toward the thumb" in r.message


def test_curl_too_much_message():
    target = vec((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    user = vec((0.0, 0.0), (0.5, 0.0), (0.0, 0.0), (0.0, 0.0))
    r = compare(user, target)
    assert r.worst == "index"
    assert "straighten" in r.message


def test_curl_too_little_message():
    target = vec((0.0, 0.0), (0.5, 0.0), (0.0, 0.0), (0.0, 0.0))
    user = vec((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    r = compare(user, target)
    assert r.worst == "index"
    assert "curl" in r.message
    assert "more" in r.message


def test_tolerance_edge():
    tol = 0.15
    # curl error exactly tol -> ok.
    target = vec((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    user = vec((0.0, 0.0), (tol, 0.0), (0.0, 0.0), (0.0, 0.0))
    assert compare(user, target, tol=tol).all_ok
    user = vec((0.0, 0.0), (tol + 0.01, 0.0), (0.0, 0.0), (0.0, 0.0))
    assert not compare(user, target, tol=tol).all_ok


def test_degrees_rounded_to_nearest_five():
    # 0.46 * 90 = 41.4 degrees -> rounds to 40
    target = vec((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    user = vec((0.0, 0.0), (0.46, 0.0), (0.0, 0.0), (0.0, 0.0))
    r = compare(user, target)
    assert "40" in r.message


def test_bad_input_returns_no_hand_detected():
    r = compare(np.array([1, 2, 3]), np.zeros(8))
    assert r.message == "No hand detected."
    assert not r.all_ok
    assert r.worst is None

    r = compare(np.zeros(8), np.array([np.nan] * 8))
    assert r.message == "No hand detected."


def test_hold_completes():
    h = Hold(seconds=3.0)
    p, done = h.update(True, 0.0)
    assert p == 0.0 and not done
    p, done = h.update(True, 1.5)
    assert p == 0.5 and not done
    p, done = h.update(True, 3.0)
    assert p == 1.0 and done
    p, done = h.update(True, 4.0)
    assert p == 1.0 and done


def test_hold_resets_after_long_gap():
    h = Hold(seconds=3.0, flicker=0.3)
    h.update(True, 0.0)
    h.update(True, 2.0)
    p, done = h.update(False, 2.5)
    assert not done
    assert p < 1.0
    # gap of 0.5 s exceeded flicker (0.3 s), so this True starts a new hold
    p, done = h.update(True, 2.7)
    assert not done
    assert p == 0.0
    # confirming it really started fresh
    p, done = h.update(True, 5.7)
    assert p == 1.0 and done


def test_hold_tolerates_flicker():
    h = Hold(seconds=3.0, flicker=0.3)
    h.update(True, 0.0)
    h.update(True, 1.0)
    h.update(False, 1.1)
    h.update(False, 1.2)
    h.update(True, 1.3)
    p, done = h.update(True, 3.0)
    assert done


def test_hold_not_done_with_no_ok():
    h = Hold(seconds=3.0)
    p, done = h.update(False, 0.0)
    assert p == 0.0 and not done
    p, done = h.update(False, 10.0)
    assert p == 0.0 and not done


def test_folded_finger_may_curl_further_and_its_spread_is_not_marked():
    # ONE from data/signs.json: thumb folded (0.8), index up, middle and ring curled.
    target = vec((0.8, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0))
    user = vec((1.0, -1.0), (0.05, 0.1), (1.0, 0.7), (1.0, 0.5))
    assert compare(user, target).all_ok


def test_folded_finger_not_curled_enough_still_fails():
    target = vec((0.8, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0))
    user = vec((0.3, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0))
    r = compare(user, target)
    assert r.worst == "thumb"
    assert "curl" in r.message


def test_thumb_out_is_judged_on_curl_not_spread():
    target = vec((0.0, 0.6), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0))  # SEVEN
    assert compare(vec((0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)), target).all_ok
    assert compare(vec((0.0, -1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)), target).all_ok  # spread unmarked
    r = compare(vec((1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)), target)  # thumb tucked in
    assert r.worst == "thumb" and "straighten" in r.message
