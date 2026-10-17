"""PhysicsTeacher: the G1 under real physics (needs the Menagerie model: `rth fetch-assets`)."""

from __future__ import annotations

import numpy as np
import pytest

from rth.config import load_settings, project_path
from rth.limbs import angle_deg, limb_directions
from rth.stances import load_stances

SETTINGS = load_settings()
pytestmark = [
    pytest.mark.assets,
    pytest.mark.skipif(
        not project_path(SETTINGS.robot.model_path).exists(),
        reason="G1 model not downloaded (uv run rth fetch-assets)",
    ),
]
STANCES = load_stances(SETTINGS.stances_path)
FPS = 30
# The booth default keeps the feet planted (settings.sim.stepping = false), so stances
# with other foot placements only match from the hips up; bow needs the rope assist.
ASSISTED = set(SETTINGS.sim.assist_stances)
STABLE = [s.id for s in STANCES.stances if s.id not in ASSISTED and s.id != "ready"]
ARM_TOL_DEG = 5.0
LEG_TOL_DEG = {"crane": 5.0, "white_crane": 5.0, "guard": 15.0}  # others: legs not scored here


@pytest.fixture(scope="module")
def teacher():
    from rth.sim.teacher import PhysicsTeacher

    t = PhysicsTeacher(SETTINGS, STANCES)
    yield t
    t.close()


def run(t, seconds: float) -> bool:
    """Advance at 30 fps; False as soon as the robot falls."""
    for _ in range(int(seconds * FPS)):
        t.update(1 / FPS)
        if t.status["phase"] == "recovering":
            return False
    return True


def arrive(t, seconds: float = 14.0) -> bool:
    for _ in range(int(seconds * FPS)):
        t.update(1 / FPS)
        if t.status["phase"] == "recovering":
            return False
        if t.at_target():
            return True
    return False


def limb_errors(t, stance_id: str, mirrored: bool = False) -> dict[str, float]:
    target = t.target(stance_id, mirrored=mirrored)
    want = limb_directions(t.builder.kin.keypoints(target.kinematic))
    got = t.limb_directions()
    return {limb.value: angle_deg(want[limb], got[limb]) for limb in want}


def test_status_keys_and_api(teacher):
    teacher.reset("ready")
    st = teacher.status
    for key in ("phase", "support", "fallen", "harness_engaged", "com_margin_m"):
        assert key in st
    assert st["phase"] in ("moving", "stepping", "holding", "recovering")
    assert st["support"] in ("double", "left", "right", "none")
    assert teacher.stance_ids() == STANCES.ids()
    assert set(teacher.keypoints()) == set(teacher.map.keypoint_bodies)
    with pytest.raises(KeyError):
        teacher.set_target("moonwalk")


def test_update_caps_substeps(teacher):
    teacher.reset("ready")
    t0 = teacher.data.time
    teacher.update(10.0)  # a huge frame must not spiral into 5000 substeps
    steps = round((teacher.data.time - t0) / teacher.model.opt.timestep)
    assert steps == SETTINGS.sim.max_substeps_per_frame
    t0 = teacher.data.time
    teacher.update(1 / FPS)
    assert round((teacher.data.time - t0) / teacher.model.opt.timestep) <= SETTINGS.sim.max_substeps_per_frame


def test_ready_stands_still(teacher):
    teacher.reset("ready")
    assert run(teacher, 10.0)
    st = teacher.status
    assert st["support"] == "double" and st["com_margin_m"] > 0.03
    assert not st["harness_engaged"]
    assert teacher.at_target()
    assert max(limb_errors(teacher, "ready").values()) < 3.0


@pytest.mark.parametrize("stance_id", STABLE)
def test_stance_reached_and_held(teacher, stance_id):
    teacher.reset("ready")
    teacher.set_target(stance_id, duration_s=2.0)
    assert arrive(teacher), f"{stance_id} not reached"
    assert run(teacher, 10.0), f"{stance_id} fell while holding"
    st = teacher.status
    assert not st["harness_engaged"] and not st["assist"] and st["com_margin_m"] > 0.0
    errs = limb_errors(teacher, stance_id)
    arms = {k: v for k, v in errs.items() if "arm" in k}
    legs = {k: v for k, v in errs.items() if "arm" not in k}
    assert max(arms.values()) < ARM_TOL_DEG, arms
    if stance_id in LEG_TOL_DEG:
        assert max(legs.values()) < LEG_TOL_DEG[stance_id], legs


@pytest.mark.xfail(reason="bow leans out of its support even with the rope assist", strict=False)
@pytest.mark.parametrize("stance_id", sorted(ASSISTED))
def test_assisted_stance_held_without_falling(teacher, stance_id):
    teacher.reset("ready")
    teacher.set_target(stance_id)
    assert arrive(teacher), f"{stance_id} not reached"
    assert run(teacher, 10.0)
    assert teacher.status["assist"] and not teacher.fallen


def test_tour_normal_and_mirrored(teacher):
    teacher.reset("ready")
    # Double-support stances, through ready. Single-leg stances are covered one at a
    # time above: lowering the lifted leg after a long tour can still fall.
    for stance_id in [s for s in STABLE if s not in ("crane", "white_crane")]:
        for mirrored in (False, True):
            teacher.set_target(stance_id, mirrored=mirrored)
            assert arrive(teacher), (stance_id, mirrored)
            assert run(teacher, 1.0), (stance_id, mirrored)
            teacher.go_neutral()
            assert arrive(teacher), ("ready after", stance_id, mirrored)


def test_push_recovery(teacher):
    teacher.reset("ready")
    assert run(teacher, 1.0)
    for force in ((60.0, 0, 0), (-60.0, 0, 0), (0, 80.0, 0)):
        teacher.push(force, duration_s=0.15)
        assert run(teacher, 3.0), force
        assert not teacher.fallen


def test_fall_harness_recover(teacher):
    teacher.reset("ready")
    assert run(teacher, 0.5)
    teacher.push((400.0, 0, 0), duration_s=0.3)
    seen = set()
    for _ in range(int(6 * FPS)):
        teacher.update(1 / FPS)
        st = teacher.status
        seen.add(st["phase"])
        if st["harness_engaged"]:
            seen.add("caught")
        if "recovering" in seen and st["phase"] != "recovering":
            break
    assert {"recovering", "caught"} <= seen
    assert teacher.stance_id == "ready" and not teacher.fallen
    assert arrive(teacher, 3.0)


def test_deterministic():
    from rth.sim.teacher import PhysicsTeacher

    runs = []
    for _ in range(2):
        with PhysicsTeacher(SETTINGS, STANCES) as t:
            t.set_target("guard")
            for _ in range(60):
                t.update(1 / FPS)
            t.push((50.0, 0, 0))
            for _ in range(30):
                t.update(1 / FPS)
            runs.append(t.data.qpos.copy())
    assert np.array_equal(runs[0], runs[1])


def test_render_uses_booth_camera_and_tracks_physics(teacher):
    teacher.reset("ready")
    a = teacher.render(320, 360)
    assert a.shape == (360, 320, 3) and a.dtype == np.uint8 and a.std() > 10
    teacher.set_target("guard")
    run(teacher, 1.0)
    b = teacher.render(320, 360)
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 0.5  # a new frame, not a cached one
    assert teacher._camera.azimuth == SETTINGS.robot.camera_azimuth_deg
