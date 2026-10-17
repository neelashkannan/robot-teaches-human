import numpy as np

from hand.compare import compare
from hand.hand_pose import hand_quality


def upright_hand():
    """Palm to the camera, fingers up, well inside the frame (normalised image coords)."""
    lm = np.zeros((21, 2))
    lm[0] = (0.50, 0.80)  # wrist
    lm[5] = (0.58, 0.55)  # index MCP
    lm[9] = (0.50, 0.52)  # middle MCP
    lm[13] = (0.43, 0.54)
    lm[17] = (0.38, 0.58)  # little MCP
    for i in (1, 2, 3, 4):
        lm[i] = (0.62 + 0.01 * i, 0.72 - 0.05 * i)
    for base, x in ((5, 0.58), (9, 0.50), (13, 0.43), (17, 0.38)):
        for k in range(1, 4):
            lm[base + k] = (x, lm[base][1] - 0.07 * k)
    return lm


def test_good_view_is_usable():
    ok, why = hand_quality(upright_hand())
    assert ok and why == ""


def test_cut_off_hand():
    lm = upright_hand()
    lm[8] = (0.5, 0.0)
    ok, why = hand_quality(lm)
    assert not ok and "whole hand" in why


def test_edge_on_palm():
    lm = upright_hand()
    lm[17] = lm[5] + (0.001, 0.001)  # knuckles line up: palm seen edge-on
    ok, why = hand_quality(lm)
    assert not ok and "palm" in why


def test_fingers_pointing_sideways_or_down():
    lm = upright_hand()
    flipped = lm.copy()
    flipped[:, 1] = 1.0 - flipped[:, 1] + 0.0  # upside down
    ok, why = hand_quality(flipped)
    assert not ok and "fingers up" in why


def test_bad_input():
    assert hand_quality(np.zeros((5, 2))) == (False, "No hand detected.")
    nan = upright_hand()
    nan[3, 0] = np.nan
    assert hand_quality(nan) == (False, "No hand detected.")


def test_thumb_is_marked_loosely_and_without_spread():
    target = np.array([0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    near = target.copy()
    near[0] = 0.8 - 0.35  # thumb off by 0.35 curl: passes now (was a fail at 0.15)
    near[1] = 0.9  # thumb spread way off: not marked
    assert compare(near, target).per_finger[0].ok
    far = target.copy()
    far[0] = 0.8 - 0.6
    assert not compare(far, target).per_finger[0].ok


def test_other_fingers_keep_the_strict_tolerance():
    target = np.array([0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    off = target.copy()
    off[2] = 0.4  # index off by 0.4: still a fail (finger tolerance 0.35, measured on webcam; thumb 0.4)
    assert not compare(off, target).per_finger[1].ok


def test_side_on_signs_skip_only_the_orientation_checks():
    """GOOD's thumbs-up and ME pointing at the chest are made side-on: framing still matters."""
    edge_on = upright_hand()
    edge_on[17] = edge_on[5] + (0.001, 0.001)
    assert hand_quality(edge_on, orientation=False) == (True, "")
    sideways = upright_hand()
    sideways[:, [0, 1]] = sideways[:, [1, 0]]  # fingers point across the picture
    assert hand_quality(sideways, orientation=False) == (True, "")
    cut = edge_on.copy()
    cut[8] = (0.5, 0.0)
    assert "whole hand" in hand_quality(cut, orientation=False)[1]
    tiny = 0.5 + (upright_hand() - 0.5) * 0.05
    assert "closer" in hand_quality(tiny, orientation=False)[1]


def test_a_few_points_off_the_edge_can_be_allowed():
    lm = upright_hand()
    lm[8] = (0.58, -0.01)  # index and middle tips just above the picture
    lm[12] = (0.50, -0.02)
    assert not hand_quality(lm)[0]  # strict by default
    assert hand_quality(lm, max_off=4) == (True, "")
    lm[[6, 7, 10, 11, 16]] = (0.5, -0.05)  # most of the fingers gone
    assert "whole hand" in hand_quality(lm, max_off=4)[1]
