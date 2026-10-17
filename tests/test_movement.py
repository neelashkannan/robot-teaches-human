import numpy as np

from hand.compare import compare
from hand.movement import CircleCheck
from hand.signs import Session, load_signs


def run(check, points, fps=30.0, size=0.15):
    for k, (x, y) in enumerate(points):
        check.update(k / fps, (x, y), size)
    return check


def circle(cycles, seconds, r=0.2, fps=30.0):
    n = int(seconds * fps)
    t = np.linspace(0, 2 * np.pi * cycles, n)
    return list(zip(0.5 + r * np.cos(t), 0.5 + r * np.sin(t)))


def test_two_circles_in_time_pass_either_direction():
    for pts in (circle(2, 1.5), circle(2, 1.5)[::-1]):
        assert run(CircleCheck(2, 1.5), pts).done


def test_one_circle_is_not_enough():
    assert not run(CircleCheck(2, 1.5), circle(1, 1.5)).done


def test_still_hand_and_straight_line_fail():
    assert not run(CircleCheck(2, 1.5), [(0.5, 0.5)] * 60).done
    assert not run(CircleCheck(2, 1.5), [(0.2 + 0.01 * k, 0.5) for k in range(60)]).done


def test_tiny_jitter_fails():
    rng = np.random.default_rng(0)
    pts = [(0.5 + rng.normal(0, 0.004), 0.5 + rng.normal(0, 0.004)) for _ in range(90)]
    assert not run(CircleCheck(2, 1.5), pts).done


def test_scale_follows_hand_size():
    far = [(0.5 + 0.05 * np.cos(t), 0.5 + 0.05 * np.sin(t)) for t in np.linspace(0, 4 * np.pi, 45)]
    assert run(CircleCheck(2, 1.5), far, size=0.04).done


def test_sorry_needs_fist_and_circles():
    signs = load_signs()
    k = [s.id for s in signs].index("sorry")
    s = Session(signs)
    for _ in range(k):
        s.skip()
    sign = s.sign
    ok = compare(sign.target, sign.target)
    # fist but not moving: no pass
    for i in range(120):
        _, done = s.update(ok, i / 30.0, (0.5, 0.5), 0.15)
        assert not done
    # fist and circling: passes
    s = Session(signs)
    for _ in range(k):
        s.skip()
    passed = False
    for i, p in enumerate(circle(4, 3.0)):
        _, done = s.update(ok, i / 30.0, p, 0.15)
        passed = passed or done
    assert passed and s.result["sorry"] == "passed"
