"""Webcam frame -> MediaPipe hand landmarks (Tasks API, VIDEO mode, one hand).

Frames are used in memory only. The model file is hashed and loaded from the same bytes, so a file
swapped between the check and the load cannot be used. No network calls here.

Mirror convention (see hand/hand_pose.py): pass a MIRRORED (selfie view) frame. MediaPipe's
handedness label assumes a selfie image, so on a mirrored frame "Right" is the person's right hand.
`process` returns the image landmarks (for drawing) and sets `.world` to the 3D world landmarks
(metres, wrist-centred) that `landmarks_to_vec` measures angles on.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "models" / "hand_landmarker.task"
DEFAULT_HASH = REPO / "models" / "hand_landmarker.sha256"


class ModelIntegrityError(RuntimeError):
    pass


def verify_model(model_path=DEFAULT_MODEL, hash_path=DEFAULT_HASH) -> bytes:
    """Return the model bytes if their sha256 matches the pinned hash, else raise."""
    data = Path(model_path).read_bytes()
    pinned = Path(hash_path).read_text().split()
    if not pinned:
        raise ModelIntegrityError(f"{hash_path} is empty")
    actual = hashlib.sha256(data).hexdigest()
    if actual != pinned[0].lower():
        raise ModelIntegrityError(
            f"{Path(model_path).name}: sha256 {actual} does not match pinned {pinned[0]}; refusing to load"
        )
    return data


class HandTracker:
    def __init__(self, model_path=DEFAULT_MODEL, hash_path=DEFAULT_HASH):
        data = verify_model(model_path, hash_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=data),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._lm = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1
        self.world: np.ndarray | None = None

    def process(self, bgr_frame: np.ndarray, t_ms: int):
        """-> (21x3 normalised image landmarks | None, "Left"|"Right" | None)."""
        self.world = None
        t_ms = int(t_ms)
        if t_ms <= self._last_ms:  # detect_for_video needs strictly increasing timestamps
            t_ms = self._last_ms + 1
        self._last_ms = t_ms
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(image, t_ms)
        if not res.hand_landmarks:
            return None, None
        lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=float)
        self.world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]], dtype=float)
        return lm, res.handedness[0][0].category_name

    def close(self) -> None:
        self._lm.close()
