"""Landmark -> Joint mapping, timestamp guarding and coordinate handedness (no model needed)."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from mediapipe.tasks.python.vision import PoseLandmark

from rth.config import VisionSettings
from rth.contracts import Joint, Limb
from rth.limbs import FORWARD, LEFT, UP, limb_directions, torso_frame
from rth.vision.pose import (
    JOINT_LANDMARK,
    N_LANDMARKS,
    MediaPipePoseTracker,
    TimestampGuard,
    build_pose,
    landmarks_to_arrays,
)


def test_every_joint_maps_to_the_same_named_mediapipe_landmark():
    assert set(JOINT_LANDMARK) == set(Joint)
    for joint, idx in JOINT_LANDMARK.items():
        assert PoseLandmark(idx).name == joint.name


def _fake_landmarks(xyz, vis=0.9, presence=0.95):
    return [SimpleNamespace(x=p[0], y=p[1], z=p[2], visibility=vis, presence=presence) for p in xyz]


def test_landmarks_to_arrays_takes_the_lower_of_visibility_and_presence():
    xyz = np.arange(N_LANDMARKS * 3, dtype=float).reshape(N_LANDMARKS, 3)
    arr, vis = landmarks_to_arrays(_fake_landmarks(xyz, vis=0.9, presence=0.3))
    np.testing.assert_allclose(arr, xyz)
    assert vis.shape == (N_LANDMARKS,)
    np.testing.assert_allclose(vis, 0.3)


def test_landmarks_to_arrays_handles_missing_visibility():
    xyz = np.zeros((N_LANDMARKS, 3))
    _, vis = landmarks_to_arrays(_fake_landmarks(xyz, vis=None, presence=None))
    np.testing.assert_allclose(vis, 0.0)


def test_build_pose_picks_the_right_rows():
    world = np.arange(N_LANDMARKS * 3, dtype=float).reshape(N_LANDMARKS, 3)
    image = world / 100.0
    vis = np.linspace(0, 1, N_LANDMARKS)
    pose = build_pose(world, image, vis, 1234)
    assert pose.timestamp_ms == 1234
    np.testing.assert_allclose(pose.keypoints[Joint.LEFT_WRIST], world[15])
    np.testing.assert_allclose(pose.keypoints[Joint.RIGHT_ANKLE], world[28])
    assert pose.visibility[Joint.RIGHT_HIP] == pytest.approx(vis[24])
    assert pose.image_points[Joint.LEFT_SHOULDER] == pytest.approx((image[11, 0], image[11, 1]))
    assert pose.landmarks_world.shape == (33, 3)
    assert pose.landmarks_image.shape == (33, 3)
    assert pose.landmarks_visibility.shape == (33,)
    # Keypoints are copies, so later edits to the big arrays don't leak into them.
    world[15] = -1
    assert pose.keypoints[Joint.LEFT_WRIST][0] != -1


def test_timestamp_guard_makes_timestamps_strictly_increasing():
    g = TimestampGuard()
    out = [g.next(t) for t in (0, 33, 33, 20, 100, 100)]
    assert out == [0, 33, 34, 35, 100, 101]
    assert all(b > a for a, b in itertools.pairwise(out))


def _mediapipe_style_person_facing_camera(right_arm_up: bool = True) -> dict[Joint, np.ndarray]:
    """Synthetic world landmarks in MediaPipe's convention: x image-right, y down, z away.

    A person facing the camera has their left side on the image's right (+x).
    """
    kp = {
        Joint.LEFT_SHOULDER: [0.18, -0.5, 0.0],
        Joint.RIGHT_SHOULDER: [-0.18, -0.5, 0.0],
        Joint.LEFT_HIP: [0.1, 0.0, 0.0],
        Joint.RIGHT_HIP: [-0.1, 0.0, 0.0],
        Joint.LEFT_ELBOW: [0.2, -0.22, 0.0],
        Joint.LEFT_WRIST: [0.2, 0.0, 0.0],
        Joint.RIGHT_ELBOW: [-0.2, -0.78, 0.0] if right_arm_up else [-0.2, -0.22, 0.0],
        Joint.RIGHT_WRIST: [-0.2, -1.05, 0.0] if right_arm_up else [-0.2, 0.0, 0.0],
        Joint.LEFT_KNEE: [0.1, 0.45, 0.0],
        Joint.RIGHT_KNEE: [-0.1, 0.45, 0.0],
        Joint.LEFT_ANKLE: [0.1, 0.9, 0.0],
        Joint.RIGHT_ANKLE: [-0.1, 0.9, 0.0],
    }
    return {j: np.array(v, float) for j, v in kp.items()}


def test_mediapipe_convention_is_right_handed_and_forward_faces_the_camera():
    kp = _mediapipe_style_person_facing_camera()
    frame = torso_frame(kp)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    # Forward is towards the camera, i.e. -z (MediaPipe z grows away from the camera).
    np.testing.assert_allclose(frame[FORWARD], [0, 0, -1], atol=1e-9)
    dirs = limb_directions(kp)
    assert dirs[Limb.RIGHT_UPPER_ARM][UP] > 0.9
    assert dirs[Limb.LEFT_UPPER_ARM][UP] < -0.9
    assert dirs[Limb.LEFT_THIGH][UP] < -0.9
    assert abs(dirs[Limb.LEFT_THIGH][LEFT]) < 0.1


def test_tracker_needs_the_model_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-assets"):
        MediaPipePoseTracker(VisionSettings(model_path=str(tmp_path / "missing.task")))
