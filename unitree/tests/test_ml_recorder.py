import numpy as np
import pytest

from rth.contracts import PersonPose
from rth.ml import recorder
from rth.ml import synthetic as syn


def _pose(vis: float = 0.9) -> PersonPose:
    lm = syn.synth_frames("horse", syn.SyntheticPerson.random(1), 1, np.random.default_rng(0))[0]
    image = np.column_stack([np.linspace(0.3, 0.7, 33), np.linspace(0.1, 0.9, 33), np.zeros(33)])
    return PersonPose(keypoints={}, visibility={}, image_points={}, landmarks_world=lm,
                      landmarks_image=image, landmarks_visibility=np.full(33, vis), timestamp_ms=5)


def test_full_body_ok():
    assert recorder.full_body_ok(_pose(), 0.6)
    assert not recorder.full_body_ok(_pose(0.3), 0.6)
    assert not recorder.full_body_ok(None, 0.6)
    p = _pose()
    p.landmarks_visibility[27] = 0.1  # an ankle out of frame
    assert not recorder.full_body_ok(p, 0.6)


def test_take_saves_only_with_enough_frames(tmp_path):
    take = recorder.Take()
    for _ in range(recorder.MIN_FRAMES - 1):
        take.add(_pose())
    assert take.save(tmp_path, "ana", "horse", "test") is None
    take.add(_pose())
    path = take.save(tmp_path, "ana", "horse", "test")
    assert path is not None and path.parent.name == "horse"


def test_draw_screen_without_camera():
    frame = np.zeros((360, 640, 3), np.uint8)
    ref = np.full((400, 320, 3), 128, np.uint8)
    img = recorder.draw_screen(frame, _pose(), title="Horse", description="Feet wide " * 20,
                               status="REC", status_color=(0, 0, 255), big="3", reference=ref,
                               mirrored=True)
    assert img.shape == frame.shape and img.any()


def test_cli_rejects_unknown_stance_and_bad_combos():
    with pytest.raises(SystemExit):
        recorder.main(["--person", "ana", "--stance", "not_a_stance"])
    with pytest.raises(SystemExit):
        recorder.main(["--person", "ana", "--all", "--from-video", "x.mp4"])
    with pytest.raises(SystemExit):
        recorder.main(["--person", "!!!", "--stance", "horse"])


def test_missing_video_fails_cleanly(tmp_path, capsys):
    assert recorder.main(["--person", "ana", "--stance", "horse",
                          "--from-video", str(tmp_path / "missing.mp4")]) == 1
    assert "could not open video" in capsys.readouterr().out


@pytest.mark.assets
def test_from_video_end_to_end(tmp_path):
    import cv2

    from rth.config import ROOT, load_settings
    from rth.ml.dataset import load_dataset

    if not (ROOT / load_settings().vision.model_path).exists():
        pytest.skip("pose model not downloaded")
    img = cv2.imread(str(ROOT / "tests/data/vision/ranger_standing.jpg"))
    h, w = img.shape[:2]
    video = tmp_path / "clip.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for _ in range(30):
        out.write(img)
    out.release()
    rec = tmp_path / "recordings"
    code = recorder.main(["--person", "Ana", "--stance", "ready", "--from-video", str(video),
                          "--recordings", str(rec)])
    assert code == 0
    ds = load_dataset(rec)
    assert ds.people == ["ana"] and ds.stances == ["ready"] and len(ds) >= recorder.MIN_FRAMES
