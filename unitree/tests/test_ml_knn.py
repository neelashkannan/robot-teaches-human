import numpy as np
import pytest

from rth.ml import synthetic as syn
from rth.ml.embedding import embed_batch
from rth.ml.knn import KNNStanceClassifier

STANCES = list(syn.STANCES)


def _data(n_people: int = 4, frames: int = 30, seed: int = 0, mirrored: bool = False):
    rng = np.random.default_rng(seed)
    x, y, g = [], [], []
    for p in range(n_people):
        person = syn.SyntheticPerson.random(seed * 10 + p)
        for s in STANCES:
            x.append(syn.synth_frames(s, person, frames, rng, mirrored=mirrored))
            y += [s] * frames
            g += [p] * frames
    return np.concatenate(x), np.array(y), np.array(g)


@pytest.fixture(scope="module")
def fitted():
    x, y, g = _data()
    return KNNStanceClassifier().fit(x, y, g)


def test_unseen_people_are_recognised(fitted):
    x, y, _ = _data(n_people=2, seed=7)  # different people from training
    pred = np.array(fitted.predict_batch(x), dtype=object)
    assert np.mean(pred == y) > 0.9


def test_leave_one_person_out():
    x, y, g = _data()
    for held in range(4):
        clf = KNNStanceClassifier().fit(x[g != held], y[g != held], g[g != held])
        pred = np.array(clf.predict_batch(x[g == held]), dtype=object)
        assert np.mean(pred == y[g == held]) > 0.85


def test_mirrored_stance_counts_as_same_id(fitted):
    x, y, _ = _data(n_people=2, seed=7, mirrored=True)
    pred = np.array(fitted.predict_batch(x), dtype=object)
    assert np.mean(pred == y) > 0.9


def test_without_mirror_augmentation_other_side_is_missed():
    x, y, g = _data()
    clf = KNNStanceClassifier(augment_mirror=False).fit(x, y, g)
    xm, ym, _ = _data(n_people=2, seed=7, mirrored=True)
    asym = np.isin(ym, ["bow", "white_crane", "crane", "archer"])
    pred = np.array(clf.predict_batch(xm[asym]), dtype=object)
    assert np.mean(pred == ym[asym]) < 0.5  # shows the augmentation is what does the job


def test_random_poses_are_rejected(fitted):
    rand = syn.random_poses(300, np.random.default_rng(1))
    probs = fitted.predict_proba_batch(rand)
    assert np.mean([p is None for p in fitted.predict_batch(rand)]) > 0.95
    assert probs.sum(axis=1).mean() < 0.05
    assert np.isfinite(fitted.threshold_)


def test_predict_proba_dict(fitted):
    x, y, _ = _data(n_people=1, frames=1, seed=7)
    p = fitted.predict_proba(x[0])
    assert set(p) == set(STANCES)
    assert 0.99 <= sum(p.values()) <= 1.0 + 1e-9
    assert max(p, key=p.get) == y[0]


def test_fit_on_embeddings_equals_fit_on_landmarks(fitted):
    x, y, g = _data()
    clf = KNNStanceClassifier().fit(embed_batch(x), y, g)
    xt, _, _ = _data(n_people=1, seed=7)
    np.testing.assert_allclose(clf.predict_proba_batch(xt), fitted.predict_proba_batch(xt))


def test_invalid_query_gets_no_votes(fitted):
    probs = fitted.predict_proba_batch(np.zeros((2, 33, 3)))
    assert probs.sum() == 0


def test_save_load_roundtrip(fitted, tmp_path):
    path = fitted.save(tmp_path / "m.npz", meta={"people": 4})
    loaded = KNNStanceClassifier.load(path)
    xt, _, _ = _data(n_people=1, seed=9)
    np.testing.assert_allclose(loaded.predict_proba_batch(xt), fitted.predict_proba_batch(xt))
    assert loaded.classes_ == fitted.classes_
    assert loaded.threshold_ == pytest.approx(fitted.threshold_)
    assert loaded.meta == {"people": 4}
    with np.load(path, allow_pickle=False) as d:  # no pickled objects inside
        assert d["labels"].dtype.kind == "U"


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        KNNStanceClassifier().predict_proba_batch(np.zeros((1, 33, 3)))
