"""Tests for hand.hand_pose.

All tests use synthetic landmarks so they run without a camera or MediaPipe.
"""
from __future__ import annotations

import numpy as np
import pytest

from hand.hand_pose import fit_calibration, landmarks_to_vec
from hand.vec import FINGERS


# MediaPipe indices for the four fingers we model.
_FINGER_IDX = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
}


_DEFAULT_BASE = {
    "thumb": np.deg2rad(50.0),  # thumb side of the middle-finger axis
    "index": 0.0,
    "middle": 0.0,
    "ring": 0.0,
}


def _make_hand(
    curl: dict[str, float] | None = None,
    spread: dict[str, float] | None = None,
) -> np.ndarray:
    """Build synthetic 21-point landmarks.

    ``curl[f]`` is the total raw curl angle (two equal bends along the
    finger). ``spread[f]`` is an extra rotation on top of the finger's
    default base direction; positive angles move toward the thumb side.
    """
    lm = np.zeros((21, 3), dtype=float)
    lm[0] = (0.0, 0.0, 0.0)
    curl = curl or {}
    spread = spread or {}

    for finger, idx in _FINGER_IDX.items():
        theta = _DEFAULT_BASE[finger] + spread.get(finger, 0.0)
        alpha = curl.get(finger, 0.0)

        d1 = np.array([np.cos(theta), np.sin(theta), 0.0])
        d2 = np.array([np.cos(theta + alpha / 2), np.sin(theta + alpha / 2), 0.0])
        d3 = np.array([np.cos(theta + alpha), np.sin(theta + alpha), 0.0])

        mcp = d1
        lm[idx[0]] = mcp
        lm[idx[1]] = mcp + d2
        lm[idx[2]] = mcp + d2 + d3
        lm[idx[3]] = mcp + d2 + d3 + d3

    return lm


def _calib(open_extra=None, fist_extra=None):
    """Helper calibration with a straight open hand and a curled fist."""
    open_lm = _make_hand()
    fist_lm = _make_hand(curl={f: 2.0 for f in FINGERS})
    if open_extra:
        open_lm = open_extra
    if fist_extra:
        fist_lm = fist_extra
    return fit_calibration([open_lm], [fist_lm])


@pytest.mark.parametrize("finger", FINGERS)
def test_straight_finger_has_low_curl(finger):
    calib = _calib()
    lm = _make_hand()
    v = landmarks_to_vec(lm, "Right", calib)
    assert v[FINGERS.index(finger) * 2] == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("finger", FINGERS)
def test_curled_finger_maps_to_one(finger):
    calib = _calib()
    lm = _make_hand(curl={f: 2.0 for f in FINGERS})
    v = landmarks_to_vec(lm, "Right", calib)
    assert v[FINGERS.index(finger) * 2] == pytest.approx(1.0, abs=0.05)


def test_partial_curl_between_zero_and_one():
    calib = _calib()
    lm = _make_hand(curl={"index": 1.0})
    v = landmarks_to_vec(lm, "Right", calib)
    assert v[2] == pytest.approx(0.5, abs=0.05)


def test_spread_index_toward_thumb_positive():
    calib = _calib()
    angle = np.deg2rad(15.0)
    lm = _make_hand(spread={"index": angle})
    v = landmarks_to_vec(lm, "Right", calib)
    expected = angle / calib.max_spread[FINGERS.index("index")]
    assert v[3] == pytest.approx(expected, abs=0.02)
    assert v[3] > 0.0


def test_spread_ring_away_negative():
    calib = _calib()
    angle = np.deg2rad(-12.0)
    lm = _make_hand(spread={"ring": angle})
    v = landmarks_to_vec(lm, "Right", calib)
    expected = angle / calib.max_spread[FINGERS.index("ring")]
    assert v[7] == pytest.approx(expected, abs=0.02)
    assert v[7] < 0.0


def test_scale_invariant():
    calib = _calib()
    lm = _make_hand(curl={"index": 1.0}, spread={"index": np.deg2rad(10.0)})
    v1 = landmarks_to_vec(lm, "Right", calib)
    v2 = landmarks_to_vec(lm * 10.0, "Right", calib)
    np.testing.assert_allclose(v1, v2, atol=1e-6)


def test_mirrored_left_equals_right():
    calib = _calib()
    lm_right = _make_hand(curl={"index": 1.0}, spread={"index": np.deg2rad(10.0)})
    v_right = landmarks_to_vec(lm_right, "Right", calib)

    lm_left = lm_right.copy()
    lm_left[:, 0] *= -1.0
    v_left = landmarks_to_vec(lm_left, "Left", calib)

    np.testing.assert_allclose(v_right, v_left, atol=1e-6)


def test_default_calibration_finite():
    lm = _make_hand(curl={"index": 0.5}, spread={"index": np.deg2rad(5.0)})
    v = landmarks_to_vec(lm, "Right")
    assert v.shape == (8,)
    assert np.isfinite(v).all()
    assert (v[0::2] >= 0.0).all() and (v[0::2] <= 1.0).all()
    assert (v[1::2] >= -1.0).all() and (v[1::2] <= 1.0).all()


def test_calibration_uses_median():
    """Noisy frames should still produce a stable open/fist range."""
    base_open = _make_hand()
    base_fist = _make_hand(curl={f: 2.0 for f in FINGERS})
    noise = np.random.default_rng(42).normal(0.0, 0.01, size=(10, 21, 3))
    open_frames = [base_open + n for n in noise[:5]]
    fist_frames = [base_fist + n for n in noise[5:]]
    calib = fit_calibration(open_frames, fist_frames)

    v_open = landmarks_to_vec(base_open, "Right", calib)
    v_fist = landmarks_to_vec(base_fist, "Right", calib)
    assert np.allclose(v_open[0::2], 0.0, atol=0.05)
    assert np.allclose(v_fist[0::2], 1.0, atol=0.05)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        landmarks_to_vec(np.zeros((10,)), "Right")
    with pytest.raises(ValueError):
        landmarks_to_vec(np.zeros((21, 2)), "Right")


def test_all_zero_raises():
    with pytest.raises(ValueError):
        landmarks_to_vec(np.zeros((21, 3)), "Right")


def test_nan_returns_neutral():
    lm = _make_hand()
    lm[5, 0] = np.nan
    v = landmarks_to_vec(lm, "Right")
    np.testing.assert_allclose(v, np.zeros(8), atol=1e-6)


def test_invalid_handedness_raises():
    with pytest.raises(ValueError):
        landmarks_to_vec(_make_hand(), "left")
