"""vision-check helpers and the camera wrapper (the webcam itself is never opened here)."""

import numpy as np
import pytest

from rth.config import CameraSettings
from rth.contracts import Limb
from rth.vision import camera as camera_mod
from rth.vision.check import describe, direction_words


def test_direction_words():
    assert direction_words(Limb.LEFT_FOREARM, np.array([0.0, 0.0, 1.0])) == "up"
    assert direction_words(Limb.LEFT_FOREARM, np.array([0.0, 0.7, 0.7])) in ("up/out", "out/up")
    # "Out" means towards the limb's own side for both arms.
    assert direction_words(Limb.RIGHT_UPPER_ARM, np.array([0.0, -1.0, 0.0])) == "out"
    assert direction_words(Limb.LEFT_UPPER_ARM, np.array([0.0, -1.0, 0.0])) == "in"
    assert direction_words(Limb.RIGHT_THIGH, np.array([0.8, 0.0, -0.6])) == "fwd/down"
    assert direction_words(Limb.RIGHT_SHIN, None) == "not seen"


def test_describe_has_one_line_per_limb():
    lines = describe({limb: None for limb in Limb})
    assert len(lines) == 8
    assert lines[1] == "L forearm: not seen"


class _FakeCapture:
    opened = True

    def __init__(self, index, api):
        self.index, self.api, self.props, self.released = index, api, {}, False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        return True, np.zeros((4, 6, 3), np.uint8)

    def release(self):
        self.released = True


def test_camera_requests_size_and_reads_frames(monkeypatch):
    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", _FakeCapture)
    with camera_mod.Camera(CameraSettings(index=2, width=640, height=480)) as cam:
        assert cam._cap.index == 2
        assert cam._cap.api == camera_mod.backend()
        assert cam.size == (640, 480)
        frame = cam.read()
        assert frame.shape == (4, 6, 3)
    assert cam._cap.released


def test_camera_failure_explains_macos_permission(monkeypatch):
    class Closed(_FakeCapture):
        opened = False

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", Closed)
    with pytest.raises(camera_mod.CameraError, match="Privacy & Security"):
        camera_mod.Camera(CameraSettings(permission_wait_s=0))


def test_camera_waits_for_permission_then_opens(monkeypatch):
    attempts = []

    class AllowedOnThirdTry(_FakeCapture):
        def __init__(self, *args):
            super().__init__(*args)
            attempts.append(1)
            self.opened = len(attempts) >= 3

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", AllowedOnThirdTry)
    monkeypatch.setattr(camera_mod.time, "sleep", lambda s: None)
    with camera_mod.Camera(CameraSettings(permission_wait_s=30)) as cam:
        assert cam._cap.opened
    assert len(attempts) == 3


def test_camera_read_returns_none_on_failure(monkeypatch):
    class Empty(_FakeCapture):
        def read(self):
            return False, None

    monkeypatch.setattr(camera_mod.cv2, "VideoCapture", Empty)
    with camera_mod.Camera(CameraSettings()) as cam:
        assert cam.read() is None


def test_macos_uses_avfoundation(monkeypatch):
    monkeypatch.setattr(camera_mod.sys, "platform", "darwin")
    assert camera_mod.backend() == camera_mod.cv2.CAP_AVFOUNDATION
