import numpy as np
import pytest

from rth.config import RecognizerSettings, Settings
from rth.ml import synthetic as syn
from rth.ml.knn import KNNStanceClassifier
from rth.ml.recognizer import SmoothedRecognizer, load_recognizer

RNG = np.random.default_rng(0)


def _frames(stance: str, n: int, seed: int = 50) -> np.ndarray:
    return syn.synth_frames(stance, syn.SyntheticPerson.random(seed), n, RNG)


@pytest.fixture(scope="module")
def clf() -> KNNStanceClassifier:
    x, y, g = [], [], []
    for p in range(3):
        person = syn.SyntheticPerson.random(p)
        for s in ("ready", "horse", "guard"):
            x.append(syn.synth_frames(s, person, 20, RNG))
            y += [s] * 20
            g += [p] * 20
    rand = syn.random_poses(60, RNG)  # explicit 'none' class
    return KNNStanceClassifier().fit(
        np.concatenate([*x, rand]), y + ["none"] * 60, g + [i % 3 for i in range(60)]
    )


def test_warm_up_then_recognised(clf):
    rec = SmoothedRecognizer(clf, alpha=0.2, min_confidence=0.6)
    out = [rec.update(lm, i * 33) for i, lm in enumerate(_frames("horse", 15))]
    assert out[0] == (None, pytest.approx(0.2, abs=0.05))  # one frame is not enough
    label, conf = out[-1]
    assert label == "horse" and conf > 0.9


def test_none_frames_decay_and_reset(clf):
    rec = SmoothedRecognizer(clf, alpha=0.3, min_confidence=0.6)
    for i, lm in enumerate(_frames("guard", 15)):
        rec.update(lm, i)
    assert rec.update(None, 100)[0] == "guard"  # one dropped frame doesn't lose it
    for i in range(10):
        label, conf = rec.update(None, 101 + i)
    assert label is None and conf < 0.1
    for lm in _frames("guard", 15):
        rec.update(lm, 0)
    rec.reset()
    assert all(v == 0 for v in rec.probabilities().values())


def test_switching_stance(clf):
    rec = SmoothedRecognizer(clf, alpha=0.2)
    for lm in _frames("ready", 20):
        rec.update(lm, 0)
    labels = [rec.update(lm, 0)[0] for lm in _frames("horse", 25)]
    assert labels[0] == "ready" and labels[-1] == "horse"


def test_random_movement_is_not_a_stance(clf):
    rec = SmoothedRecognizer(clf, alpha=0.2)
    labels = [rec.update(lm, 0)[0] for lm in syn.random_poses(40, np.random.default_rng(5))]
    assert labels.count(None) >= 38


def test_bad_input_counts_as_no_pose(clf):
    rec = SmoothedRecognizer(clf)
    assert rec.update(np.full((33, 3), np.nan), 0) == (None, 0.0)
    assert rec.update(np.zeros((5, 3)), 0) == (None, 0.0)


def test_load_recognizer_missing_disabled_and_corrupt(tmp_path, clf):
    missing = RecognizerSettings(model_path=str(tmp_path / "nope.npz"))
    assert load_recognizer(missing) is None
    assert load_recognizer(Settings(recognizer=missing)) is None
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a zip")
    assert load_recognizer(RecognizerSettings(model_path=str(bad))) is None
    good = clf.save(tmp_path / "good.npz")
    assert load_recognizer(RecognizerSettings(model_path=str(good), enabled=False)) is None
    rec = load_recognizer(Settings(recognizer=RecognizerSettings(model_path=str(good))))
    assert isinstance(rec, SmoothedRecognizer)
    for lm in _frames("ready", 10):
        label, _ = rec.update(lm, 0)
    assert label == "ready"
