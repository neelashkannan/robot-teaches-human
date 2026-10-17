"""The booth wiring, keys and headless mode, with the fake robot and fake visitor."""

from __future__ import annotations

import pytest

from rth.app import main as app_main
from rth.app.fakes import FakeRobot, NullVoice
from rth.app.main import BoothApp, FakeSensor, SetupError, SimClock, force_offline, make_coach
from rth.app.session import Phase
from rth.config import load_settings
from rth.stances import load_stances


@pytest.fixture
def settings():
    s = load_settings()
    force_offline(s)
    return s


@pytest.fixture
def stances():
    return load_stances()


@pytest.fixture
def app(settings, stances):
    clock = SimClock()
    a = BoothApp(settings, stances, robot=FakeRobot(stances), sensor=FakeSensor(settings, 1),
                 coach=make_coach(settings, stances, silent=True), clock=clock, size=(960, 540))
    a.sim = clock
    yield a
    a.close()


def step(app, n=1):
    img = None
    for _ in range(n):
        app.sim.advance(1 / 30)
        img = app.step()
    return img


def test_offline_never_has_a_key(settings):
    assert settings.nebius.api_key is None
    assert not settings.nebius.enable_sensei


def test_step_renders_at_requested_size(app):
    assert step(app).shape == (540, 960, 3)
    assert isinstance(app.coach.voice, NullVoice)
    assert app.runner.sources == {"picker": "offline", "sensei": "offline"}


def test_visitor_arrives_and_session_starts(app):
    step(app, 30 * 5)
    assert app.session.phase in (Phase.DEMONSTRATING, Phase.COPYING)
    assert app.coach.voice.spoken


def test_keys_typing_and_digits(app):
    assert app.handle_key(ord("t"))
    assert app.typing == ""
    for ch in "balanx":
        app.handle_key(ord(ch))
    app.handle_key(127)  # backspace
    for ch in "ce":
        app.handle_key(ord(ch))
    assert app.typing == "balance"
    app.handle_key(ord("q"))  # typed, not quit
    app.handle_key(27)
    assert app.typing is None
    app.handle_key(13)  # Enter opens typing too
    for ch in "one leg":
        app.handle_key(ord(ch))
    app.handle_key(13)
    assert app.typing is None and app.session.pending_request == "one leg"
    for _ in range(60):
        step(app)
        if app.session.phase is Phase.DEMONSTRATING:
            break
    assert app.session.phase is Phase.DEMONSTRATING
    app.handle_key(ord("2"))
    assert app.session.exercise.stance_id == app.stance_keys[1]
    mirror = app.mirror_view
    app.handle_key(ord("m"))
    assert app.mirror_view is not mirror
    app.handle_key(ord(" "))
    assert app.session.phase is Phase.THINKING
    app.handle_key(ord("r"))
    assert app.session.phase is Phase.WAITING
    assert app.handle_key(-1)
    assert not app.handle_key(ord("q"))
    assert not app.handle_key(27)


def test_missing_robot_assets_explain_the_fix(settings, stances):
    settings.robot.model_path = "assets/does_not_exist/scene.xml"
    with pytest.raises(SetupError, match="fetch-assets"):
        app_main.make_robot(settings, stances, fake=False)


def test_headless_main_writes_frames(tmp_path):
    out = tmp_path / "frames"
    code = app_main.main(["--offline", "--fake-person", "--fake-robot", "--headless", "75",
                          "--out", str(out), "--every", "25", "--size", "960x540"])
    assert code == 0
    frames = sorted(p.name for p in out.glob("frame_*.png"))
    assert frames == ["frame_00000.png", "frame_00025.png", "frame_00050.png"]
    assert any(out.glob("phase_*_waiting.png"))


def test_headless_needs_out(capsys):
    assert app_main.main(["--headless", "5"]) == 2


class FallingRobot(FakeRobot):
    """Looks like a physics robot: reports a status and keeps moving."""

    def __init__(self, stances):
        super().__init__(stances)
        self.status = {"phase": "standing", "fallen": False}
        self.renders = 0

    def render(self, width, height):
        self.renders += 1
        return super().render(width, height)


def test_physics_status_pauses_and_always_renders(settings, stances):
    clock = SimClock()
    robot = FallingRobot(stances)
    a = BoothApp(settings, stances, robot=robot, sensor=FakeSensor(settings, 1),
                 coach=make_coach(settings, stances, silent=True), clock=clock, size=(960, 540))
    try:
        a.run_commands(a.session.choose("horse"))
        robot.status = {"phase": "recovering", "fallen": True}
        for _ in range(30 * 8):
            clock.advance(1 / 30)
            a.step()
        assert a.session.phase is Phase.DEMONSTRATING
        assert a.last_state.view.robot_recovering
        assert robot.renders == 30 * 8  # no caching for a robot that wobbles
    finally:
        a.close()


def test_unknown_physics_module_explains(settings, stances, monkeypatch):
    from rth.robot import factory

    def boom(*a, **kw):
        raise ImportError("No module named 'rth.sim.teacher'")

    monkeypatch.setattr(factory, "make_teacher", boom)
    with pytest.raises(SetupError, match="--robot kinematic"):
        app_main.make_robot(settings, stances, fake=False, mode="physics")
