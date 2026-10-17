from datetime import UTC, datetime

import numpy as np
import pytest

from rth.ml import synthetic as syn
from rth.ml.dataset import (
    list_recordings,
    load_dataset,
    load_recording,
    recording_path,
    safe_name,
    save_recording,
    summary_table,
)


def test_safe_name():
    assert safe_name("  Ana María ") == "ana_mar_a"
    assert safe_name("bob-2") == "bob-2"
    with pytest.raises(ValueError):
        safe_name("!!!")


def test_recording_path_never_collides(tmp_path):
    when = datetime(2026, 10, 17, 12, 0, 0, tzinfo=UTC)
    p1 = recording_path(tmp_path, "Ana", "horse", when)
    assert p1.relative_to(tmp_path).as_posix() == "ana/horse/20261017T120000.000Z.npz"
    p1.parent.mkdir(parents=True)
    p1.touch()
    assert recording_path(tmp_path, "Ana", "horse", when) != p1


def test_round_trip(tmp_path):
    lm = np.random.default_rng(0).normal(size=(12, 33, 3))
    vis = np.full((12, 33), 0.9)
    ts = np.arange(12) * 33
    path = save_recording(tmp_path, "Ana", "horse", lm, vis, ts)
    r = load_recording(path)
    assert (r.person, r.stance_id) == ("ana", "horse")
    np.testing.assert_allclose(r.landmarks_world, lm, atol=1e-6)
    np.testing.assert_array_equal(r.timestamps_ms, ts)
    with pytest.raises(ValueError):
        save_recording(tmp_path, "Ana", "horse", lm, vis[:3], ts)


def test_load_dataset_and_summary(tmp_path):
    syn.write_recordings(tmp_path, n_people=3, n_frames=10, stances=["ready", "horse"])
    assert len(list_recordings(tmp_path)) == 6
    ds = load_dataset(tmp_path)
    assert len(ds) == 60 and ds.people == ["person0", "person1", "person2"]
    assert ds.stances == ["horse", "ready"]
    assert len(set(ds.recording.tolist())) == 6
    only = load_dataset(tmp_path, stance_ids=["ready"])
    assert only.stances == ["ready"]
    table = summary_table(ds)
    assert "| person0 | 10 (1) | 10 (1) | 20 (2) |" in table
    assert "60 (6)" in table


def test_empty_dataset(tmp_path):
    ds = load_dataset(tmp_path / "missing")
    assert len(ds) == 0 and ds.people == []
