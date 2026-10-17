"""Real photos through the real model: sides, handedness and limb directions.

Needs the pose model (`uv run rth fetch-assets`). Photos and licences: tests/data/vision/SOURCES.md.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from rth.config import VisionSettings, project_path
from rth.contracts import Limb, PersonPose
from rth.limbs import FORWARD, LEFT, UP, limb_directions, torso_frame

pytestmark = pytest.mark.assets

DATA = Path(__file__).parent / "data" / "vision"
NOSE = 0
L = Limb


@pytest.fixture(scope="module")
def settings() -> VisionSettings:
    cfg = VisionSettings()
    if not project_path(cfg.model_path).exists():
        pytest.skip("pose model not downloaded; run `uv run rth fetch-assets`")
    return cfg


def _track(settings, name: str, *, flip: bool = False) -> PersonPose:
    from rth.vision.pose import MediaPipePoseTracker

    img = cv2.imread(str(DATA / name))
    assert img is not None, name
    if flip:
        img = np.ascontiguousarray(img[:, ::-1])
    with MediaPipePoseTracker(settings) as tracker:
        pose = tracker.process(img, 0)
    assert pose is not None, f"nobody found in {name}"
    return pose


def _dirs(pose: PersonPose):
    return limb_directions(pose.keypoints, pose.visibility, 0.5)


PHOTOS = [
    "ranger_arm_up.jpg",
    "ranger_standing.jpg",
    "ranger_tree.jpg",
    "taichi_wide.jpg",
    "warrior2.jpg",
]


@pytest.mark.parametrize("name", PHOTOS)
@pytest.mark.parametrize("flip", [False, True])
def test_world_landmarks_are_right_handed_and_forward_faces_camera(settings, name, flip):
    pose = _track(settings, name, flip=flip)
    frame = torso_frame(pose.keypoints)
    assert frame is not None
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1e-6)
    # Everyone in these photos faces the camera; MediaPipe z grows away from it.
    assert frame[FORWARD][2] < -0.7
    # The nose sits in front of the torso, so forward really is the way the body faces.
    hips = (pose.landmarks_world[23] + pose.landmarks_world[24]) / 2
    nose = frame @ (pose.landmarks_world[NOSE] - hips)
    assert nose[FORWARD] > 0.03
    assert nose[UP] > 0.4


@pytest.mark.parametrize("name", PHOTOS)
def test_left_side_is_on_the_image_right_for_a_person_facing_camera(settings, name):
    pose = _track(settings, name)
    assert (
        pose.image_points[L.LEFT_UPPER_ARM.joints[0]][0]
        > pose.image_points[L.RIGHT_UPPER_ARM.joints[0]][0]
    )
    assert pose.landmarks_world.shape == (33, 3)
    assert pose.landmarks_image.shape == (33, 3)
    assert pose.landmarks_visibility.shape == (33,)


def test_raised_left_arm(settings):
    # Ranger, facing the camera: left arm raised overhead, right arm out to her right,
    # standing on her left leg with the right knee lifted.
    d = _dirs(_track(settings, "ranger_arm_up.jpg"))
    assert d[L.LEFT_UPPER_ARM][UP] > 0.4 and d[L.LEFT_UPPER_ARM][LEFT] > 0.4
    assert d[L.LEFT_FOREARM][UP] > 0.75
    assert d[L.RIGHT_UPPER_ARM][LEFT] < -0.7
    assert d[L.LEFT_THIGH][UP] < -0.7
    assert d[L.LEFT_SHIN][UP] < -0.7
    assert d[L.RIGHT_THIGH][UP] > -0.5


def test_flipped_image_swaps_left_and_right(settings):
    # The same photo mirrored: the model now reports the raised arm as the RIGHT one,
    # which is why tracking must use the raw, unmirrored camera frame.
    d = _dirs(_track(settings, "ranger_arm_up.jpg", flip=True))
    assert d[L.RIGHT_FOREARM][UP] > 0.75
    assert d[L.LEFT_FOREARM][UP] < 0.6
    assert d[L.RIGHT_THIGH][UP] < -0.7  # the standing leg is now labelled right


def test_standing_relaxed(settings):
    d = _dirs(_track(settings, "ranger_standing.jpg"))
    for limb in (
        L.LEFT_UPPER_ARM,
        L.RIGHT_UPPER_ARM,
        L.LEFT_THIGH,
        L.RIGHT_THIGH,
        L.LEFT_SHIN,
        L.RIGHT_SHIN,
    ):
        assert d[limb][UP] < -0.8, limb
    for limb in (L.LEFT_FOREARM, L.RIGHT_FOREARM):
        assert d[limb][UP] < -0.5, limb


def test_warrior_two(settings):
    # Arms straight out to the sides, right knee bent out to his right, left leg straight.
    d = _dirs(_track(settings, "warrior2.jpg"))
    assert d[L.LEFT_UPPER_ARM][LEFT] > 0.85 and d[L.LEFT_FOREARM][LEFT] > 0.85
    assert d[L.RIGHT_UPPER_ARM][LEFT] < -0.85 and d[L.RIGHT_FOREARM][LEFT] < -0.6
    assert d[L.RIGHT_THIGH][LEFT] < -0.6 and d[L.RIGHT_THIGH][UP] > -0.5
    assert d[L.RIGHT_SHIN][UP] < -0.85
    assert d[L.LEFT_THIGH][LEFT] > 0.5 and d[L.LEFT_THIGH][UP] < -0.3


def test_tree_pose(settings):
    # Standing on the left leg, right knee turned out to her right.
    d = _dirs(_track(settings, "ranger_tree.jpg"))
    assert d[L.LEFT_THIGH][UP] < -0.9
    assert d[L.LEFT_SHIN][UP] < -0.8
    assert d[L.RIGHT_THIGH][LEFT] < -0.5


def test_wide_stance(settings):
    d = _dirs(_track(settings, "taichi_wide.jpg"))
    for limb in (L.LEFT_THIGH, L.RIGHT_THIGH, L.LEFT_SHIN, L.RIGHT_SHIN):
        assert d[limb][UP] < -0.5, limb
    assert d[L.LEFT_THIGH][LEFT] > d[L.RIGHT_THIGH][LEFT] + 0.3


def test_video_mode_tolerates_repeated_and_backwards_timestamps(settings):
    from rth.vision.pose import MediaPipePoseTracker

    img = cv2.imread(str(DATA / "warrior2.jpg"))
    with MediaPipePoseTracker(settings) as tracker:
        stamps = [tracker.process(img, t).timestamp_ms for t in (100, 100, 50, 200)]
        assert stamps == [100, 101, 102, 200]
        assert tracker.process(np.zeros((360, 640, 3), np.uint8), 300) is None


def test_vision_check_writes_annotated_png(settings, tmp_path, capsys):
    from rth.vision.check import main

    out = tmp_path / "out.png"
    assert main(["--image", str(DATA / "ranger_arm_up.jpg"), "--out", str(out)]) == 0
    img = cv2.imread(str(out))
    assert img is not None and img.shape[:2] == (623, 800)
    assert "L forearm: up" in capsys.readouterr().out
