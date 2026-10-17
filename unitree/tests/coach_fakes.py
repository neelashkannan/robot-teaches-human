"""Fake OpenAI-like client and fixtures shared by the tests/test_coach_*.py files."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import openai

from rth.coach.client import NebiusClient
from rth.config import NebiusSettings, Settings
from rth.contracts import Limb, RepResult
from rth.stances import load_stances

FAKE_KEY = "sk-test-SECRET-should-never-appear"
_REQ = httpx2.Request("POST", "https://api.tokenfactory.nebius.com/v1/chat/completions")


def timeout_error() -> Exception:
    return openai.APITimeoutError(request=_REQ)


def connection_error() -> Exception:
    return openai.APIConnectionError(request=_REQ)


def status_error(code: int, msg: str = "bad request") -> Exception:
    resp = httpx2.Response(code, request=_REQ)
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError}.get(code,
                                                                          openai.APIStatusError)
    return cls(msg, response=resp, body=None)


def chat_response(text: str, prompt_tokens: int = 50, completion_tokens: int = 20) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def embed_vector(text: str, dim: int = 8) -> list[float]:
    """Deterministic bag-of-keywords vector so cosine similarity is meaningful."""
    keys = ["balance", "leg", "block", "punch", "tai chi", "bow", "arm", "calm"]
    t = text.lower()
    v = [1.0 if k in t else 0.0 for k in keys[:dim]]
    return v if any(v) else [0.01] * dim


class FakeOpenAI:
    """Each queue item is a reply text, an exception to raise, or a callable(**kwargs)."""

    def __init__(
        self,
        chat: list[Any] | None = None,
        embed: list[Any] | None = None,
        models: list[str] | None = None,
    ) -> None:
        self.chat_queue = list(chat or [])
        self.embed_queue = list(embed or [])
        self.model_ids = models or []
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)
        self.models = SimpleNamespace(list=self._models)

    def _chat(self, **kw: Any) -> Any:
        self.chat_calls.append(kw)
        item = self.chat_queue.pop(0) if self.chat_queue else "{}"
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(**kw)
        return chat_response(item)

    def _embed(self, **kw: Any) -> Any:
        self.embed_calls.append(kw)
        item = self.embed_queue.pop(0) if self.embed_queue else None
        if isinstance(item, Exception):
            raise item
        data = [SimpleNamespace(index=i, embedding=embed_vector(t)) for i, t in enumerate(kw["input"])]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=7 * len(data)))

    def _models(self) -> Any:
        return [SimpleNamespace(id=m) for m in self.model_ids]


def make_settings(tmp_path: Path, **nebius: Any) -> Settings:
    """Default settings with logs and caches redirected into ``tmp_path``."""
    n = NebiusSettings(
        api_key_env="RTH_TEST_NEBIUS_KEY_UNSET",
        usage_log=str(tmp_path / "usage.jsonl"),
        embedding_cache=str(tmp_path / "emb.json"),
        **nebius,
    )
    return Settings(nebius=n)


def fake_client(settings: Settings, fake: FakeOpenAI) -> NebiusClient:
    return NebiusClient(settings.nebius, client=fake, api_key=FAKE_KEY)


def stances():
    return load_stances()


def rep(
    i: int,
    stance_id: str,
    held: bool,
    *,
    err: float = 10.0,
    worst: Limb | None = None,
    mirrored: bool = False,
    hold_s: float = 3.0,
) -> RepResult:
    errs: dict[Limb, float | None] = {limb: err for limb in Limb}
    if worst is not None:
        errs[worst] = err + 25.0
    return RepResult(rep_index=i, stance_id=stance_id, mirrored=mirrored, held=held,
                     hold_s=hold_s, attempt_s=6.0, limb_errors_deg=errs, worst_limb=worst)
