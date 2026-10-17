import json
from pathlib import Path

import pytest

from coach_fakes import FakeOpenAI, connection_error, fake_client, make_settings, stances
from rth.coach.picker import EmbeddingMovePicker, OfflineMovePicker, make_picker


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("something for balance", {"crane", "white_crane"}),
        ("teach me to block", {"guard"}),
        ("I want strong legs", {"horse"}),
        ("how do I punch", {"bow"}),
        ("tai chi please", {"white_crane"}),
        ("like shooting a bow and arrow", {"archer"}),
        ("Horse stance", {"horse"}),
        ("defend myself", {"guard"}),
    ],
)
def test_offline_picker_requests(request_text, expected):
    r = OfflineMovePicker(stances()).pick(request_text)
    assert r.stance_id in expected
    assert r.source == "offline"
    assert 0.0 < r.similarity <= 1.0


@pytest.mark.parametrize("request_text", ["", "   ", "surprise me", "zzz qqq"])
def test_offline_picker_no_match_gives_valid_stance(request_text):
    lib = stances()
    r = OfflineMovePicker(lib).pick(request_text)
    assert r.stance_id in lib and r.stance_id != "ready"
    assert r.similarity == 0.0


def test_make_picker_offline_without_key(tmp_path):
    assert isinstance(make_picker(make_settings(tmp_path), stances()), OfflineMovePicker)


def test_embedding_picker_and_cache(tmp_path):
    s = make_settings(tmp_path)
    lib = stances()
    fake = FakeOpenAI()
    r = EmbeddingMovePicker(fake_client(s, fake), s, lib).pick("something for balance")
    assert r.source == "nebius"
    assert r.stance_id in {"crane", "white_crane"}
    # First pick: one batch for all stances, one for the query.
    assert [len(c["input"]) for c in fake.embed_calls] == [len(lib.ids()), 1]
    cache = json.loads(Path(s.nebius.embedding_cache).read_text())
    assert len(cache) == len(lib.ids())

    # A fresh picker reuses the disk cache: only the query is embedded.
    fake2 = FakeOpenAI()
    r2 = EmbeddingMovePicker(fake_client(s, fake2), s, lib).pick("teach me to block")
    assert r2.stance_id == "guard"
    assert [len(c["input"]) for c in fake2.embed_calls] == [1]


def test_embedding_cache_keyed_by_model(tmp_path):
    lib = stances()
    s = make_settings(tmp_path)
    EmbeddingMovePicker(fake_client(s, FakeOpenAI()), s, lib).pick("balance")
    s2 = make_settings(tmp_path, embedding_model="other/model")
    fake = FakeOpenAI()
    EmbeddingMovePicker(fake_client(s2, fake), s2, lib).pick("balance")
    assert len(fake.embed_calls[0]["input"]) == len(lib.ids())  # cache miss for a new model


def test_embedding_failure_falls_back_offline(tmp_path):
    s = make_settings(tmp_path)
    fake = FakeOpenAI(embed=[connection_error(), connection_error()])
    r = EmbeddingMovePicker(fake_client(s, fake), s, stances()).pick("I want strong legs")
    assert r.source == "offline" and r.stance_id == "horse"


def test_corrupt_cache_is_ignored(tmp_path):
    s = make_settings(tmp_path)
    Path(s.nebius.embedding_cache).write_text("{not json")
    r = EmbeddingMovePicker(fake_client(s, FakeOpenAI()), s, stances()).pick("how do I punch")
    assert r.source == "nebius" and r.stance_id == "bow"
