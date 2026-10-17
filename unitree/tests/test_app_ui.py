"""The booth screen renders every phase without error, at any window size."""

from __future__ import annotations

import numpy as np
import pytest

from rth.app.fakes import FakeRobot, FakeVisitor
from rth.app.session import (
    Exercise,
    Phase,
    SessionView,
    build_scorecard,
)
from rth.app.ui import BoothUI, Rect, ScreenState, TextRenderer, contain, cover, make_layout
from rth.contracts import Limb, LimbScore, RepResult
from rth.limbs import compare, limb_directions, mirror_directions

NAMES = {"horse": "Horse stance", "crane": "Crane stance", "ready": "Ready stance"}


def rep(i, stance="horse", held=True, err=10.0):
    return RepResult(
        i,
        stance,
        False,
        held,
        3.0,
        4.2,
        dict.fromkeys(Limb, err),
        Limb.LEFT_THIGH if not held else None,
    )


def view(phase: Phase, **kw) -> SessionView:
    history = kw.pop("history", (rep(0), rep(1, "crane", held=False, err=40.0)))
    base = {
        "phase": phase,
        "now": 12.3,
        "exercise": Exercise("horse", True, 2.0, 3.0),
        "hold_progress": 0.4,
        "holding": True,
        "copy_elapsed_s": 2.0,
        "reps_done": len(history),
        "reps_total": 5,
        "history": history,
        "last_rep": history[-1] if history else None,
        "say": "Once more, slower. Watch your left thigh.",
        "say_source": "nebius",
        "pending_request": None,
        "tip": None,
        "scorecard": None,
        "scorecard_left_s": 0.0,
        "final_rep": False,
        "best_hold_s": 1.2,
    }
    base.update(kw)
    return SessionView(**base)


@pytest.fixture(scope="module")
def scene():
    robot = FakeRobot()
    robot.set_target("horse", duration_s=0.01)
    robot.update(1.0)
    visitor = FakeVisitor(seed=1, arrive_s=0.0, legs_hidden_s=0.0)
    robot_dirs = limb_directions(robot.keypoints())
    for _ in range(30):
        visitor.update(1 / 30, mirror_directions(robot_dirs))
    frame, pose = visitor.frame_and_pose(1000)
    person = limb_directions(pose.keypoints, pose.visibility)
    scores = compare(person, robot_dirs, 25.0)
    return robot, frame, pose, scores


def state(scene, phase, size=(1600, 900), **kw):
    robot, frame, pose, scores = scene
    lay = make_layout(*size)
    v = kw.pop("view", None) or view(phase)
    defaults = {
        "view": v,
        "camera_bgr": frame,
        "pose": pose,
        "scores": scores,
        "robot_bgr": robot.render(lay.robot.w, lay.robot.h),
        "hint": "Left thigh: lift it higher",
        "person_present": True,
        "full_body": True,
        "all_ok": False,
        "sources": {"picker": "nebius", "sensei": "offline"},
        "recognised": ("horse", 0.82),
        "fps": 29.7,
        "stance_keys": ("ready", "horse", "crane"),
    }
    defaults.update(kw)
    return ScreenState(**defaults)


@pytest.fixture(scope="module")
def ui():
    return BoothUI(NAMES)


SIZES = [(1600, 900), (1280, 720), (1920, 1200), (1000, 900)]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    "phase", [Phase.WAITING, Phase.DEMONSTRATING, Phase.COPYING, Phase.THINKING]
)
def test_every_phase_renders(ui, scene, phase, size):
    img = ui.render(state(scene, phase, size), size)
    assert img.shape == (size[1], size[0], 3) and img.dtype == np.uint8
    assert img.std() > 5  # something was drawn


@pytest.mark.parametrize("size", SIZES)
def test_scorecard_renders(ui, scene, size):
    history = tuple(rep(i, held=i % 2 == 0, err=40 - 5 * i) for i in range(5))
    v = view(
        Phase.SCORECARD, history=history, scorecard=build_scorecard(history), scorecard_left_s=9.2
    )
    img = ui.render(state(scene, Phase.SCORECARD, size, view=v), size)
    assert img.shape == (size[1], size[0], 3)


def test_edge_cases_render(ui, scene):
    cases = [
        state(
            scene,
            Phase.WAITING,
            person_present=False,
            pose=None,
            camera_bgr=None,
            robot_bgr=None,
            recognised=None,
            fps=None,
            sources={},
        ),
        state(scene, Phase.COPYING, full_body=False, all_ok=True, hint=None),
        state(scene, Phase.COPYING, person_present=False, pose=None),
        state(scene, Phase.COPYING, typing="something for bal"),
        state(scene, Phase.WAITING, typing=""),
        state(scene, Phase.THINKING, view=view(Phase.THINKING, final_rep=True, tip="Hips lower")),
        state(scene, Phase.THINKING, view=view(Phase.THINKING, history=(), last_rep=None)),
        state(
            scene,
            Phase.DEMONSTRATING,
            notice="Copy the same side",
            mirror_view=False,
            view=view(Phase.DEMONSTRATING, pending_request="x" * 200, say=None),
        ),
        state(
            scene,
            Phase.COPYING,
            recognised=(None, 0.0),
            hint="A" * 300,
            robot_bgr=np.zeros((300, 200, 3), np.uint8),
        ),
    ]
    for st in cases:
        assert ui.render(st).shape == (900, 1600, 3)


def test_hershey_fallback_renders(scene):
    ui = BoothUI(NAMES, use_truetype=False)
    assert ui.render(state(scene, Phase.COPYING)).shape == (900, 1600, 3)


def test_custom_skeleton_is_used(scene):
    calls = []

    def skeleton(img, pose, scores=None, *, mirrored):
        calls.append(mirrored)  # returns None: drawing in place is fine too

    ui = BoothUI(NAMES, skeleton=skeleton)
    ui.render(state(scene, Phase.COPYING, mirror_display=True))
    assert calls == [True]


def test_text_helpers():
    t = TextRenderer()
    assert t.width("Hello", 30) > t.width("Hi", 30) > 0
    assert t.fit(
        "one two three four five", 30, "bold", t.width("one two", 30, "bold") + 40
    ).endswith("…")
    lines = t.wrap("a b c d e f g h i j k l m n o p", 30, "regular", 120, max_lines=2)
    assert len(lines) <= 2
    canvas = np.zeros((50, 50, 3), np.uint8)
    t.draw(canvas, "Clipped text far outside", 40, 40, 30, (255, 255, 255))
    t.draw(canvas, "Off canvas", 500, 500, 30, (255, 255, 255))


def test_cover_and_contain_sizes():
    img = np.zeros((720, 1280, 3), np.uint8)
    assert cover(img, 700, 600).shape == (600, 700, 3)
    assert contain(img, 700, 600).shape == (600, 700, 3)
    assert make_layout(1600, 900).cam == Rect(28, 112, 762, 612)


def test_scores_with_unseen_limbs_render(ui, scene):
    unseen = {limb: LimbScore(limb, None, False) for limb in Limb}
    assert ui.render(state(scene, Phase.COPYING, scores=unseen)).shape == (900, 1600, 3)
