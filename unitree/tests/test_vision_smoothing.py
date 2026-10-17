import numpy as np
import pytest

from rth.contracts import Limb
from rth.vision.smoothing import DirectionSmoother


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _all(v):
    return {limb: (None if v is None else _unit(v)) for limb in Limb}


def _deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def test_first_frame_passes_through_and_outputs_are_unit():
    s = DirectionSmoother(0.5)
    out = s.update(_all([1, 1, 0]))
    for limb in Limb:
        np.testing.assert_allclose(out[limb], _unit([1, 1, 0]))


def test_small_moves_are_smoothed_and_stay_unit_length():
    s = DirectionSmoother(0.5)
    a, b = _unit([1, 0, 0]), _unit([np.cos(np.radians(30)), np.sin(np.radians(30)), 0])
    s.update({Limb.LEFT_FOREARM: a})
    out = s.update({Limb.LEFT_FOREARM: b})[Limb.LEFT_FOREARM]
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert 10 < _deg(a, out) < 20  # halfway-ish, not all the way
    for _ in range(30):
        out = s.update({Limb.LEFT_FOREARM: b})[Limb.LEFT_FOREARM]
    assert _deg(out, b) < 0.5


def test_big_jump_snaps():
    s = DirectionSmoother(0.2)
    s.update({Limb.RIGHT_THIGH: _unit([0, 0, -1])})
    out = s.update({Limb.RIGHT_THIGH: _unit([1, 0, 0])})[Limb.RIGHT_THIGH]
    np.testing.assert_allclose(out, [1, 0, 0])


def test_missing_limb_holds_then_goes_none():
    s = DirectionSmoother(0.5, hold_frames=3)
    v = _unit([0, 1, 0])
    s.update({Limb.LEFT_SHIN: v})
    for _ in range(3):
        np.testing.assert_allclose(s.update({Limb.LEFT_SHIN: None})[Limb.LEFT_SHIN], v)
    assert s.update({Limb.LEFT_SHIN: None})[Limb.LEFT_SHIN] is None
    # After expiring, the next sighting starts fresh instead of blending with stale data.
    w = _unit([0, 0.9, 0.44])
    np.testing.assert_allclose(s.update({Limb.LEFT_SHIN: w})[Limb.LEFT_SHIN], w)


def test_reappearing_within_hold_resets_the_count():
    s = DirectionSmoother(1.0, hold_frames=2)
    v = _unit([0, 0, 1])
    s.update({Limb.LEFT_THIGH: v})
    s.update({})
    s.update({})
    s.update({Limb.LEFT_THIGH: v})
    s.update({})
    s.update({})
    assert s.update({Limb.LEFT_THIGH: v})[Limb.LEFT_THIGH] is not None


def test_never_seen_is_none_and_zero_vectors_count_as_missing():
    s = DirectionSmoother(0.5)
    out = s.update({Limb.LEFT_THIGH: np.zeros(3)})
    assert all(out[limb] is None for limb in Limb)


def test_reset_forgets_everything():
    s = DirectionSmoother(0.1)
    s.update(_all([1, 0, 0]))
    s.reset()
    assert all(v is None for v in s.update(_all(None)).values())
    out = s.update(_all([0.8, 0.6, 0]))
    np.testing.assert_allclose(out[Limb.LEFT_FOREARM], [0.8, 0.6, 0])


def test_outputs_are_copies():
    s = DirectionSmoother(0.5)
    out = s.update(_all([1, 0, 0]))
    out[Limb.LEFT_FOREARM][:] = 0
    np.testing.assert_allclose(s.update(_all(None))[Limb.LEFT_FOREARM], [1, 0, 0])


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_bad_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        DirectionSmoother(alpha)
