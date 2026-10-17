"""Webcam capture. Frames come back unmirrored; flip only for display."""

from __future__ import annotations

import sys
import time
from typing import Self

import cv2
import numpy as np

from rth.config import CameraSettings

PERMISSION_HELP = (
    "On macOS, allow camera access for the app you run this from (Terminal, iTerm, VS Code...): "
    "System Settings -> Privacy & Security -> Camera, switch it on, then quit and reopen that app. "
    "Also check no other app is using the camera and that camera.index in config/settings.yaml "
    "is right (0 is usually the built-in camera)."
)


class CameraError(RuntimeError):
    pass


def backend() -> int:
    """OpenCV capture backend: AVFoundation on macOS, OpenCV's choice elsewhere."""
    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


class Camera:
    """Open webcam ``cfg.index`` at the requested size; use as a context manager."""

    def __init__(self, cfg: CameraSettings) -> None:
        self.cfg = cfg
        self._cap = self._open(cfg)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    @staticmethod
    def _open(cfg: CameraSettings) -> cv2.VideoCapture:
        """Open the camera, retrying for up to ``cfg.permission_wait_s`` seconds.

        The first open on macOS shows the permission dialog and fails straight away;
        once Allow is clicked a fresh open succeeds, so retry rather than quit.
        """
        deadline = time.monotonic() + max(0.0, cfg.permission_wait_s)
        told = False
        while True:
            cap = cv2.VideoCapture(cfg.index, backend())
            if cap.isOpened():
                return cap
            cap.release()
            if time.monotonic() >= deadline:
                raise CameraError(f"could not open camera {cfg.index}. {PERMISSION_HELP}")
            if not told:
                print(
                    f"Waiting up to {cfg.permission_wait_s:.0f} s for camera {cfg.index}: "
                    "if macOS asks for camera access, click Allow.",
                    flush=True,
                )
                told = True
            time.sleep(1.0)

    @property
    def size(self) -> tuple[int, int]:
        """Actual (width, height); the camera may not honour the request."""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> np.ndarray | None:
        """Next unmirrored BGR frame, or None if the camera gave nothing."""
        ok, frame = self._cap.read()
        return frame if ok and frame is not None and frame.size else None

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
