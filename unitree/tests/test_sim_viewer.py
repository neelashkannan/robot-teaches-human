"""`rth sim` logic without a window: keys, state machine, re-exec decision, passive fallback."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import mujoco
import numpy as np
import pytest

from rth.config import load_settings, project_path
from rth.sim import viewer as V

SETTINGS = load_settings()
STANCES = ["ready", "horse", "bow", "guard", "crane", "white_crane", "archer"]
needs_model = pytest.mark.skipif(
    not project_path(SETTINGS.robot.model_path).exists(),
    reason="G1 model not downloaded (uv run rth fetch-assets)",
)


# --------------------------------------------------------------------------- keys


def test_key_mapping():
    for i in range(1, 8):
        assert V.action_for_key(ord(str(i))) == f"stance:{i - 1}"
    assert V.action_for_key(ord("M")) == "mirror"
    assert V.action_for_key(ord("P")) == "push"
    assert V.action_for_key(ord("B")) == "big_push"
    assert V.action_for_key(ord("R")) == "reset"
    assert V.action_for_key(ord("N")) == "tour"
    assert V.action_for_key(32) == "pause"
    assert V.action_for_key(ord("p")) == "push"  # lower case is accepted too
    assert V.action_for_key(ord("Q")) is None
    assert V.action_for_key(256) is None  # GLFW escape


def test_stance_keys_and_mirror():
    s = V.ViewerState(list(STANCES))
    assert s.handle("stance:1", 0.0) == [("stance", "horse", False)]
    assert s.handle("mirror", 0.0) == [("stance", "horse", True)]
    assert s.handle("stance:6", 0.0) == [("stance", "archer", True)]  # mirroring sticks
    assert s.handle("stance:8", 0.0) == [] and "no stance 9" in s.message
    assert s.handle(None, 0.0) == []


def test_push_reset_pause():
    s = V.ViewerState(list(STANCES), stance_index=3, mirrored=True)
    assert s.handle("push", 0.0) == [("push", False)]
    assert s.handle("big_push", 0.0) == [("push", True)]
    assert s.handle("reset", 0.0) == [("reset",)]
    assert s.stance == "ready" and not s.mirrored
    assert s.handle("pause", 0.0) == [] and s.paused
    s.handle("pause", 0.0)
    assert not s.paused


def test_push_force_is_horizontal_with_the_set_size():
    s = V.ViewerState(list(STANCES), rng=np.random.default_rng(0))
    small = s.push_force(False, SETTINGS)
    big = s.push_force(True, SETTINGS)
    assert small[2] == 0.0 and big[2] == 0.0
    assert np.linalg.norm(small) == pytest.approx(SETTINGS.sim.viewer_push_n)
    assert np.linalg.norm(big) == pytest.approx(SETTINGS.sim.viewer_big_push_n)


def test_tour_steps_on_a_timer_and_pauses():
    s = V.ViewerState(list(STANCES), tour_s=5.0)
    assert s.tick(0.0) == []  # tour off
    s.handle("tour", 10.0)
    assert s.tick(10.0) == [("stance", "horse", False)]
    assert s.tick(14.0) == []
    assert s.tick(15.1) == [("stance", "bow", False)]
    s.handle("pause", 15.2)
    assert s.tick(30.0) == []
    s.handle("pause", 30.0)
    s.handle("stance:0", 30.0)  # picking a stance restarts the timer
    assert s.tick(34.0) == [] and s.tick(35.1) == [("stance", "horse", False)]
    s.handle("tour", 36.0)
    assert not s.tour and s.tick(99.0) == []


# --------------------------------------------------------------------------- applying commands


class FakeSim:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_target(self, stance_id, *, mirrored=False, duration_s=2.0):
        self.calls.append(("set_target", stance_id, mirrored, duration_s))

    def push(self, force_n, *, body="torso_link", duration_s=0.15):
        self.calls.append(("push", tuple(np.round(force_n, 3)), body, duration_s))

    def reset(self, stance_id="ready"):
        self.calls.append(("reset", stance_id))


def test_apply_drives_the_physics_teacher_api():
    sim, s = FakeSim(), V.ViewerState(list(STANCES))
    V.apply([("stance", "bow", True), ("push", True), ("reset",)], sim, s, SETTINGS, "physics")
    kinds = [c[0] for c in sim.calls]
    assert kinds == ["set_target", "push", "reset"]
    assert sim.calls[0][1:3] == ("bow", True)
    assert sim.calls[1][2] == "torso_link" and sim.calls[1][3] == SETTINGS.sim.viewer_push_s
    assert sim.calls[2] == ("reset", "ready")


def test_apply_in_passive_mode_skips_stances():
    sim, s = FakeSim(), V.ViewerState(["stand"])
    V.apply([("stance", "stand", False), ("push", False)], sim, s, SETTINGS, "passive")
    assert [c[0] for c in sim.calls] == ["push"]


@dataclass
class _Status:
    phase: str
    fallen: bool
    com_margin_m: float | None


def test_status_items_from_dataclass_dict_and_object():
    assert V.status_items(_Status("hold", False, 0.0421)) == [
        ("phase", "hold"),
        ("fallen", "no"),
        ("com_margin_m", "+0.042"),
    ]
    assert V.status_items({"harness_engaged": True}) == [("harness_engaged", "yes")]
    obj = types.SimpleNamespace(phase="step", support="left", other=1)
    assert V.status_items(obj) == [("phase", "step"), ("support", "left")]
    assert V.status_items(None) == []


# --------------------------------------------------------------------------- mjpython re-exec


def test_reexec_only_on_macos_outside_mjpython():
    argv = ["--scene", "plain"]
    assert V.reexec_command(argv, platform="linux", in_mjpython=False, env={}) is None
    assert V.reexec_command(argv, platform="darwin", in_mjpython=True, env={}) is None
    cmd = V.reexec_command(argv, platform="darwin", in_mjpython=False, env={})
    assert cmd[0].endswith("mjpython") and cmd[1:] == ["-m", "rth.sim.viewer", *argv]
    with pytest.raises(RuntimeError):  # already re-executed once: don't loop
        V.reexec_command(argv, platform="darwin", in_mjpython=False, env={V.REEXEC_ENV: "1"})


def test_reexec_env_finds_libpython():
    env = V.reexec_env({"DYLD_FALLBACK_LIBRARY_PATH": "/opt/x"}, sys.executable)
    paths = env["DYLD_FALLBACK_LIBRARY_PATH"].split(":")
    assert env[V.REEXEC_ENV] == "1" and paths[-1] == "/opt/x"
    assert paths[0].endswith("/lib")
    fresh = V.reexec_env({}, sys.executable)["DYLD_FALLBACK_LIBRARY_PATH"].split(":")
    assert fresh[1:] == ["/usr/local/lib", "/usr/lib"]  # dyld's defaults are kept


def test_not_under_mjpython_in_tests():
    assert V.under_mjpython() is False


# --------------------------------------------------------------------------- viewer-side helpers


def test_keep_sim_forces_survives_a_sync_that_zeroes_them():
    d = types.SimpleNamespace(xfrc_applied=np.zeros((3, 6)))
    keep = V._KeepSimForces(d)
    d.xfrc_applied[1, 0] = 50.0  # the sim's push
    keep.before_sync()
    d.xfrc_applied[:] = 0.0  # what viewer.sync() does ...
    d.xfrc_applied[2, 1] = 7.0  # ... before writing the mouse force
    keep.after_sync()
    assert d.xfrc_applied[1, 0] == 50.0 and d.xfrc_applied[2, 1] == 7.0
    d.xfrc_applied[1, 0] = 0.0  # push over
    keep.before_sync()
    d.xfrc_applied[:] = 0.0
    keep.after_sync()
    assert not d.xfrc_applied.any()  # the old mouse force isn't kept either


def test_view_options_restore():
    opt = mujoco.MjvOption()
    view = V._ViewOptions(opt)
    opt.geomgroup[2] = 0  # MuJoCo's own '2' shortcut hides the robot's meshes
    opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1  # and 'M' shows centres of mass
    view.restore()
    assert opt.geomgroup[2] == 1 and opt.flags[mujoco.mjtVisFlag.mjVIS_COM] == 0


def test_args():
    a = V.parse_args(["--stance", "horse", "--scene", "plain", "--passive", "--tour"])
    assert (a.stance, a.scene, a.passive, a.tour) == ("horse", "plain", True, True)
    with pytest.raises(SystemExit):
        V.parse_args(["--scene", "moon"])


# --------------------------------------------------------------------------- passive fallback


@pytest.mark.assets
@needs_model
def test_falls_back_to_passive_when_the_controller_fails(monkeypatch):
    broken = types.ModuleType("rth.sim.teacher")

    class PhysicsTeacher:
        def __init__(self, *a, **k):
            raise RuntimeError("controller not ready")

    broken.PhysicsTeacher = PhysicsTeacher
    monkeypatch.setitem(sys.modules, "rth.sim.teacher", broken)
    sim, mode = V.make_sim(SETTINGS, passive=False)
    assert mode == "passive" and isinstance(sim, V.PassiveSim)
    sim2, mode2 = V.make_sim(SETTINGS, passive=True)
    assert mode2 == "passive" and isinstance(sim2, V.PassiveSim)


@pytest.mark.assets
@needs_model
def test_passive_sim_stands_falls_when_pushed_and_resets():
    sim = V.PassiveSim(SETTINGS)
    pelvis = sim.model.body("pelvis").id
    start = sim.data.xpos[pelvis].copy()
    for _ in range(60):
        sim.update(1 / 60)
    assert sim.sim_time == pytest.approx(1.0, abs=0.01)
    assert np.linalg.norm(sim.data.xpos[pelvis] - start) < 0.01 and not sim.fallen
    sim.push(np.array([0.0, 400.0, 0.0]), duration_s=0.3)
    for _ in range(180):
        sim.update(1 / 60)
    assert not sim.data.xfrc_applied.any()  # the push is gone once it's over
    assert np.linalg.norm(sim.data.xpos[pelvis, :2] - start[:2]) > 0.1  # it moved
    data = sim.data
    sim.reset()
    assert sim.data is data  # the viewer holds this MjData: reset in place
    assert sim.sim_time == 0.0 and np.linalg.norm(sim.data.xpos[pelvis] - start) < 1e-6
    assert sim.status["phase"].startswith("passive")


@pytest.mark.assets
@needs_model
def test_passive_sim_caps_substeps():
    sim = V.PassiveSim(SETTINGS)
    sim.update(10.0)  # a huge stall
    cap = SETTINGS.sim.max_substeps_per_frame
    assert sim.sim_time == pytest.approx(cap * sim.model.opt.timestep)
