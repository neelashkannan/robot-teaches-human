import json

import numpy as np
import pytest

from rth.config import RecognizerSettings
from rth.ml import synthetic as syn
from rth.ml import train
from rth.ml.dataset import load_dataset, save_recording
from rth.ml.recognizer import load_recognizer


def _run(tmp_path, recordings) -> int:
    return train.main([
        "--recordings", str(recordings),
        "--model", str(tmp_path / "model.npz"),
        "--report-dir", str(tmp_path / "reports"),
    ])


def test_lopo_end_to_end(tmp_path, capsys):
    rec = tmp_path / "recordings"
    syn.write_recordings(rec, n_people=4, n_frames=40)
    assert _run(tmp_path, rec) == 0
    out = capsys.readouterr().out
    assert "leave-one-person-out" in out

    reports = tmp_path / "reports"
    res = json.loads((reports / "recognizer.json").read_text())
    assert res["protocol"] == "leave-one-person-out"
    assert len(res["folds"]) == 4 and set(res["per_person"]) == {f"person{i}" for i in range(4)}
    assert res["overall"]["frame_accuracy"] > 0.85
    assert res["overall"]["recordings"] == 4 * len(syn.STANCES)
    counts = np.array(res["confusion"]["counts"])
    assert counts.sum() == res["frames_tested"] == 4 * len(syn.STANCES) * 40
    assert res["confusion"]["cols"][-1] == train.NO_STANCE

    md = (reports / "recognizer.md").read_text()
    assert "leave-one-person-out" in md and "Confusion matrix" in md
    assert "SYNTHETIC" in md and res["sources"] == {"synthetic": 4 * len(syn.STANCES)}
    png = (reports / "recognizer_confusion.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000

    rec_model = load_recognizer(RecognizerSettings(model_path=str(tmp_path / "model.npz")))
    assert rec_model is not None
    lm = syn.synth_frames("crane", syn.SyntheticPerson.random(99), 15, np.random.default_rng(3))
    for frame in lm:
        label, _ = rec_model.update(frame, 0)
    assert label == "crane"


def test_single_person_time_split(tmp_path, capsys):
    rec = tmp_path / "recordings"
    syn.write_recordings(rec, n_people=1, n_frames=40, stances=["ready", "horse", "archer"])
    assert _run(tmp_path, rec) == 0
    res = json.loads((tmp_path / "reports" / "recognizer.json").read_text())
    assert res["protocol"].startswith("time-split")
    assert "NOT a test on unseen people" in (tmp_path / "reports" / "recognizer.md").read_text()
    assert res["frames_tested"] == 3 * 12  # last 30% of each recording


def test_none_class_counts_rejection_as_correct(tmp_path):
    rec = tmp_path / "recordings"
    syn.write_recordings(rec, n_people=2, n_frames=30, stances=["ready", "horse"])
    rng = np.random.default_rng(0)
    for p in range(2):
        lm = syn.random_poses(30, rng)
        save_recording(rec, f"person{p}", "none", lm, np.ones((30, 33)), np.arange(30) * 33)
    assert _run(tmp_path, rec) == 0
    res = json.loads((tmp_path / "reports" / "recognizer.json").read_text())
    assert "none" in res["confusion"]["rows"] and "none" not in res["confusion"]["cols"]
    assert res["per_stance"]["none"]["frame_accuracy"] > 0.9


def test_cap_and_time_split_helpers(tmp_path):
    syn.write_recordings(tmp_path, n_people=1, n_frames=50, stances=["ready", "horse"])
    ds = load_dataset(tmp_path)
    assert train.cap_per_recording(ds, 10).sum() == 20
    tr, te = train.time_split(ds)
    assert tr.sum() == 70 and te.sum() == 30
    for r in np.unique(ds.recording):
        m = ds.recording == r
        assert ds.timestamps_ms[tr & m].max() < ds.timestamps_ms[te & m].min()
    groups = train.segment_groups(ds)
    assert len(set(groups.tolist())) == 4  # 2 recordings x 2 chunks for 1.6 s each


def test_no_recordings_fails_cleanly(tmp_path, capsys):
    assert _run(tmp_path, tmp_path / "empty") == 1
    assert "no usable recordings" in capsys.readouterr().out


@pytest.mark.parametrize("stances", [["ready"]])
def test_needs_two_stances(tmp_path, stances):
    syn.write_recordings(tmp_path / "r", n_people=2, n_frames=15, stances=stances)
    assert _run(tmp_path, tmp_path / "r") == 1
