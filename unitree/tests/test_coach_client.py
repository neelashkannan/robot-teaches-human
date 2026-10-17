import json
from pathlib import Path

import pytest

from coach_fakes import (
    FAKE_KEY,
    FakeOpenAI,
    connection_error,
    fake_client,
    make_settings,
    status_error,
    timeout_error,
)
from rth.coach.client import NebiusClient, NebiusError


def _log_lines(settings):
    return [json.loads(x) for x in open(settings.nebius.usage_log).read().splitlines()]


def test_missing_key_not_available(tmp_path):
    s = make_settings(tmp_path)
    c = NebiusClient(s.nebius)
    assert not c.available
    with pytest.raises(NebiusError, match="no API key"):
        c.chat("sensei", [], model="m", max_tokens=5)


def test_chat_success_logs_usage_without_key(tmp_path):
    s = make_settings(tmp_path)
    fake = FakeOpenAI(chat=["hello"])
    c = fake_client(s, fake)
    reply = c.chat("sensei", [{"role": "user", "content": "hi"}], model="m", max_tokens=5)
    assert reply.text == "hello"
    assert reply.prompt_tokens == 50 and reply.completion_tokens == 20
    (line,) = _log_lines(s)
    assert line["role"] == "sensei" and line["model"] == "m" and line["ok"] is True
    assert line["prompt_tokens"] == 50 and isinstance(line["latency_ms"], int)
    assert FAKE_KEY not in Path(s.nebius.usage_log).read_text()


def test_timeout_is_not_retried(tmp_path):
    s = make_settings(tmp_path)
    fake = FakeOpenAI(chat=[timeout_error(), "late"])
    with pytest.raises(NebiusError, match="timeout"):
        fake_client(s, fake).chat("sensei", [], model="m", max_tokens=5)
    assert len(fake.chat_calls) == 1
    assert _log_lines(s)[-1]["ok"] is False


def test_connection_error_retried_once(tmp_path):
    s = make_settings(tmp_path)
    fake = FakeOpenAI(chat=[connection_error(), "second time lucky"])
    assert fake_client(s, fake).chat("sensei", [], model="m", max_tokens=5).text.startswith("sec")
    assert len(fake.chat_calls) == 2


def test_connection_error_twice_fails(tmp_path):
    s = make_settings(tmp_path)
    fake = FakeOpenAI(chat=[connection_error(), connection_error(), "never"])
    with pytest.raises(NebiusError, match="connection"):
        fake_client(s, fake).chat("sensei", [], model="m", max_tokens=5)
    assert len(fake.chat_calls) == 2


def test_http_error_not_retried_and_key_scrubbed(tmp_path):
    s = make_settings(tmp_path)
    fake = FakeOpenAI(chat=[status_error(401, f"invalid key {FAKE_KEY}")])
    with pytest.raises(NebiusError) as ei:
        fake_client(s, fake).chat("sensei", [], model="m", max_tokens=5)
    assert "HTTP 401" in str(ei.value)
    assert FAKE_KEY not in str(ei.value)
    assert len(fake.chat_calls) == 1
    assert FAKE_KEY not in Path(s.nebius.usage_log).read_text()


def test_embed_and_list_models(tmp_path):
    s = make_settings(tmp_path)
    c = fake_client(s, FakeOpenAI(models=["b/x", "a/y"]))
    vecs = c.embed("picker", ["balance", "punch"], model="e")
    assert len(vecs) == 2 and vecs[0] != vecs[1]
    assert c.list_models() == ["a/y", "b/x"]
    assert [x["role"] for x in _log_lines(s)] == ["picker", "models"]


def test_malformed_chat_reply(tmp_path):
    s = make_settings(tmp_path)
    fake = FakeOpenAI(chat=[lambda **kw: object()])
    with pytest.raises(NebiusError, match="malformed"):
        fake_client(s, fake).chat("sensei", [], model="m", max_tokens=5)
