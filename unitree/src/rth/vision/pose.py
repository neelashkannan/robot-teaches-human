"""Visitor body tracking with the MediaPipe Pose Landmarker (Tasks API, VIDEO mode).

Coordinate notes, checked on real photos (tests/test_vision_pose_photos.py):

* MediaPipe's LEFT_* / RIGHT_* landmarks are the person's anatomical sides on an
  *unmirrored* image: someone facing the camera has their left shoulder on the
  image's right.
* World landmarks are metres, hip-centred, x towards image right, y down, z away
  from the camera. That is a right-handed frame, so ``rth.limbs.torso_frame``
  works on them unchanged and its forward axis points at the camera for someone
  facing it.
* Tracking must run on the raw camera frame. On a horizontally flipped frame the
  model still labels sides as if the image were real, so a mirrored person's
  right arm comes back as LEFT_*: left/right swap and every score would be
  compared against the wrong limb. Mirror only for display.
"""

from __future__ import annotations

from typing import Self

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from rth.config import VisionSettings, project_path
from rth.contracts import Joint, PersonPose

N_LANDMARKS = 33

# MediaPipe Pose landmark index for each contract joint.
JOINT_LANDMARK: dict[Joint, int] = {
    Joint.LEFT_SHOULDER: 11,
    Joint.RIGHT_SHOULDER: 12,
    Joint.LEFT_ELBOW: 13,
    Joint.RIGHT_ELBOW: 14,
    Joint.LEFT_WRIST: 15,
    Joint.RIGHT_WRIST: 16,
    Joint.LEFT_HIP: 23,
    Joint.RIGHT_HIP: 24,
    Joint.LEFT_KNEE: 25,
    Joint.RIGHT_KNEE: 26,
    Joint.LEFT_ANKLE: 27,
    Joint.RIGHT_ANKLE: 28,
}


def landmarks_to_arrays(landmarks) -> tuple[np.ndarray, np.ndarray]:
    """(33, 3) xyz and (33,) visibility from a MediaPipe landmark list."""
    xyz = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)
    # Visibility alone stays high for limbs cut off by the frame edge; presence catches those.
    vis = np.array(
        [
            min(p.visibility or 0.0, p.presence if p.presence is not None else 1.0)
            for p in landmarks
        ],
        dtype=float,
    )
    return xyz, vis


def build_pose(
    world: np.ndarray, image: np.ndarray, vis: np.ndarray, timestamp_ms: int
) -> PersonPose:
    """Map the 33 MediaPipe landmarks onto the contract's 12 joints."""
    return PersonPose(
        keypoints={j: world[i].copy() for j, i in JOINT_LANDMARK.items()},
        visibility={j: float(vis[i]) for j, i in JOINT_LANDMARK.items()},
        image_points={
            j: (float(image[i, 0]), float(image[i, 1])) for j, i in JOINT_LANDMARK.items()
        },
        landmarks_world=world,
        landmarks_image=image,
        landmarks_visibility=vis,
        timestamp_ms=int(timestamp_ms),
    )


class TimestampGuard:
    """Make timestamps strictly increasing; VIDEO mode raises on repeats or going back."""

    def __init__(self) -> None:
        self._last: int | None = None

    def next(self, timestamp_ms: int) -> int:
        t = int(timestamp_ms)
        if self._last is not None and t <= self._last:
            t = self._last + 1
        self._last = t
        return t


class MediaPipePoseTracker:
    """``PoseTracker`` for one visitor; feed it unmirrored BGR frames."""

    def __init__(self, cfg: VisionSettings) -> None:
        path = project_path(cfg.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"pose model not found at {path}; run `uv run rth fetch-assets` first"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=cfg.min_detection_confidence,
            min_pose_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._clock = TimestampGuard()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PersonPose | None:
        ts = self._clock.next(timestamp_ms)
        # Sliced views (e.g. frame[:, ::-1]) have negative strides that OpenCV rejects.
        rgb = cv2.cvtColor(np.ascontiguousarray(frame_bgr), cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
        )
        if not result.pose_world_landmarks or not result.pose_landmarks:
            return None
        world, vis = landmarks_to_arrays(result.pose_world_landmarks[0])
        image, _ = landmarks_to_arrays(result.pose_landmarks[0])
        return build_pose(world, image, vis, ts)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
