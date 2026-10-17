import json
import os

import pytest

from coach_fakes import FakeOpenAI, fake_client, make_settings, stances
from rth.coach import check
from rth.coach.check import closest_ids, run


def test_missing_key_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check, "load_settings", lambda: make_settings(tmp_path))
    assert check.main([]) == 2
    out = capsys.readouterr().out
    assert "RTH_TEST_NEBIUS_KEY_UNSET" in out and ".env.example" in out


def test_closest_ids():
    avail = ["openbmb/MiniCPM-V-4_5", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-Embedding-8B"]
    assert closest_ids("openbmb/MiniCPM-V-4.5", avail)[0] == "openbmb/MiniCPM-V-4_5"
    assert "Qwen/Qwen3-Embedding-8B" in closest_ids("Qwen3-Embedding-8B", avail)


def _sensei_reply(**kw):
    return json.dumps({"next_stance_id": "horse", "mirrored": False, "transition_s": 3,
                       "hold_s": 2.5, "say": "Sink lower into your left thigh."})


def test_run_all_ok(tmp_path):
    s = make_settings(tmp_path, enable_vision_tip=False)
    n = s.nebius
    fake = FakeOpenAI(chat=[_sensei_reply(), "Make fists with both hands."],
                      models=[n.embedding_model, n.sensei_model, n.vision_model])
    lines: list[str] = []
    assert run(s, stances(), fake_client(s, fake), vision=True, out=lines.append) == 0
    text = "\n".join(lines)
    assert "All good." in text and "FAIL" not in text
    assert "Make fists" in text and "output mode json_schema" in text


def test_run_reports_missing_model(tmp_path):
    s = make_settings(tmp_path)
    fake = FakeOpenAI(chat=[_sensei_reply()],
                      models=[s.nebius.embedding_model, "Qwen/Qwen3-30B-A3B-Instruct-2508"])
    lines: list[str] = []
    assert run(s, stances(), fake_client(s, fake), out=lines.append) == 1
    text = "\n".join(lines)
    assert "Qwen/Qwen3-30B-A3B-Instruct-2508" in text  # suggested as closest
    assert "skip  vision" in text


def test_run_bad_sensei_reply_fails(tmp_path):
    s = make_settings(tmp_path)
    n = s.nebius
    fake = FakeOpenAI(chat=["no json here"], models=[n.embedding_model, n.sensei_model])
    lines: list[str] = []
    assert run(s, stances(), fake_client(s, fake), out=lines.append) == 1
    assert any("FAIL  sensei" in x for x in lines)


@pytest.mark.nebius
@pytest.mark.skipif(not os.environ.get("NEBIUS_API_KEY"), reason="NEBIUS_API_KEY not set")
def test_live_check_nebius():
    """Live: one embedding batch, one sensei call. Run with `uv run pytest -m nebius`."""
    from rth.coach.client import NebiusClient
    from rth.config import load_settings

    s = load_settings()
    lines: list[str] = []
    code = run(s, stances(), NebiusClient(s.nebius), out=lines.append)
    print("\n".join(lines))
    assert code == 0, "\n".join(lines)
