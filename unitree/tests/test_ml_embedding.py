import numpy as np
import pytest

from rth.contracts import Joint
from rth.limbs import torso_frame
from rth.ml import synthetic as syn
from rth.ml.embedding import (
    EMBEDDING_DIM,
    MIRROR_INDEX,
    embed,
    embed_batch,
    mirror_embedding,
    mirror_landmarks,
    torso_frames,
)

RNG = np.random.default_rng(0)


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _poses(n: int = 5) -> np.ndarray:
    person = syn.SyntheticPerson.random(3)
    return np.concatenate([syn.synth_frames(s, person, n, RNG) for s in ("bow", "crane", "archer")])


def test_shapes_and_dim():
    lm = _poses()
    e = embed_batch(lm)
    assert e.shape == (len(lm), EMBEDDING_DIM)
    assert embed(lm[0]).shape == (EMBEDDING_DIM,)
    with pytest.raises(ValueError):
        embed_batch(np.zeros((2, 17, 3)))


def test_invariant_to_translation_scale_and_rotation():
    lm = _poses()
    e = embed_batch(lm)
    gravity_yaw = _rotation(np.array([0, 1.0, 0]), 0.8)  # MediaPipe y is vertical
    any_rot = _rotation(np.array([0.3, -0.5, 0.8]), 2.1)
    for rot in (gravity_yaw, any_rot):
        moved = 1.7 * lm @ rot.T + np.array([0.4, -1.2, 3.0])
        np.testing.assert_allclose(embed_batch(moved), e, atol=1e-9)


def test_matches_limbs_torso_frame():
    lm = _poses(1)[0]
    kp = {Joint.LEFT_SHOULDER: lm[11], Joint.RIGHT_SHOULDER: lm[12],
          Joint.LEFT_HIP: lm[23], Joint.RIGHT_HIP: lm[24]}
    frames, _, valid = torso_frames(lm[None])
    assert valid[0]
    np.testing.assert_allclose(frames[0], torso_frame(kp), atol=1e-12)


def test_degenerate_pose_is_none():
    assert embed(np.zeros((33, 3))) is None
    nan = _poses(1)[0].copy()
    nan[11] = np.nan
    assert embed(nan) is None
    assert np.isnan(embed_batch(np.stack([nan, _poses(1)[0]]))[0]).all()


def test_mirror_index_is_an_involution():
    np.testing.assert_array_equal(MIRROR_INDEX[MIRROR_INDEX], np.arange(33))
    assert MIRROR_INDEX[11] == 12 and MIRROR_INDEX[0] == 0 and MIRROR_INDEX[31] == 32


def test_mirror_twice_is_identity_and_matches_embedding_mirror():
    lm = _poses()
    np.testing.assert_allclose(mirror_landmarks(mirror_landmarks(lm)), lm, atol=1e-9)
    np.testing.assert_allclose(mirror_embedding(embed_batch(lm)), embed_batch(mirror_landmarks(lm)),
                               atol=1e-9)
    np.testing.assert_allclose(mirror_landmarks(lm[0]), mirror_landmarks(lm)[0])


def test_mirror_does_not_depend_on_camera_axes():
    lm = _poses(2)
    rot = _rotation(np.array([0.2, 1.0, -0.4]), 1.3)
    np.testing.assert_allclose(mirror_landmarks(lm @ rot.T), mirror_landmarks(lm) @ rot.T, atol=1e-9)


def test_mirrored_stance_equals_other_side_synthetic():
    # A left-foot-forward bow, mirrored, is the right-foot-forward bow.
    dirs = {k: np.asarray(v, float) for k, v in syn.STANCES["bow"].items()}
    left = syn.to_mediapipe(syn.pose_from_directions(dirs), yaw=0.4)
    right = syn.to_mediapipe(syn.pose_from_directions(syn.mirror_directions(dirs)), yaw=-0.4)
    np.testing.assert_allclose(embed(mirror_landmarks(left)), embed(right), atol=1e-9)
    assert np.abs(embed(left) - embed(right)).max() > 0.3  # the two sides really differ


@pytest.mark.assets
@pytest.mark.parametrize("name", ["warrior2", "ranger_tree", "taichi_wide"])
def test_mirror_matches_mediapipe_on_flipped_photo(name):
    # MediaPipe on a flipped photo sees the mirror-image person; our mirror should agree.
    import cv2

    from rth.config import ROOT, load_settings

    settings = load_settings()
    if not (ROOT / settings.vision.model_path).exists():
        pytest.skip("pose model not downloaded")
    from rth.vision.pose import MediaPipePoseTracker

    img = cv2.imread(str(ROOT / f"tests/data/vision/{name}.jpg"))
    poses = []
    for frame in (img, cv2.flip(img, 1)):
        with MediaPipePoseTracker(settings.vision) as tracker:
            poses.append(tracker.process(frame, 0).landmarks_world)
    orig, flipped = (embed(p) for p in poses)
    mirrored = embed(mirror_landmarks(poses[0]))
    assert np.abs(flipped - mirrored).mean() < 0.5 * np.abs(flipped - orig).mean()
