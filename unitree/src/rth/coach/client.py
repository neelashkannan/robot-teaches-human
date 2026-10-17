"""Thin wrapper around the OpenAI SDK pointed at Nebius Token Factory.

Every call has a hard timeout, at most one retry (connection errors only, never
timeouts, so a slow API can't double the wait) and one JSON line in the usage log.
Callers catch ``NebiusError`` and fall back to offline behaviour.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai

from rth.config import NebiusSettings, project_path


class NebiusError(RuntimeError):
    """Any failure talking to Nebius (no key, timeout, HTTP error, bad reply)."""


@dataclass
class ChatReply:
    text: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class CallRecord:
    """What the last call did; also the shape of one usage-log line."""

    role: str
    model: str
    latency_ms: int
    ok: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _usage(resp: Any) -> tuple[int | None, int | None]:
    u = getattr(resp, "usage", None)
    if u is None:
        return None, None
    return getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)


class NebiusClient:
    """OpenAI-compatible client for Nebius. Pass ``client`` to inject a fake for tests."""

    def __init__(
        self,
        settings: NebiusSettings,
        *,
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings
        self._key = api_key if api_key is not None else settings.api_key
        self._client = client
        self._lock = threading.Lock()
        self.last_call: CallRecord | None = None

    @property
    def available(self) -> bool:
        """True when a key (or an injected client) is present."""
        return self._client is not None or bool(self._key)

    def _sdk(self) -> Any:
        if self._client is None:
            if not self._key:
                raise NebiusError(f"no API key: set {self.settings.api_key_env} in .env")
            # SDK retries are off: our own retry policy is stricter and logged.
            self._client = openai.OpenAI(
                base_url=self.settings.base_url,
                api_key=self._key,
                timeout=self.settings.timeout_s,
                max_retries=0,
            )
        return self._client

    # ------------------------------------------------------------------ calls

    def chat(
        self,
        role: str,
        messages: Sequence[dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.6,
        response_format: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatReply:
        """One chat completion; returns the reply text or raises ``NebiusError``."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        t0 = time.perf_counter()
        resp = self._call(role, model, t0, lambda sdk: sdk.chat.completions.create(**kwargs),
                          timeout_s)
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as e:
            raise self._fail(role, model, t0, NebiusError(f"malformed reply: {e}")) from e
        p, c = _usage(resp)
        reply = ChatReply(text, model, self._ms(t0), p, c)
        self._record(CallRecord(role, model, reply.latency_ms, True, p, c))
        return reply

    def embed(
        self,
        role: str,
        texts: Sequence[str],
        *,
        model: str,
        timeout_s: float | None = None,
    ) -> list[list[float]]:
        """Embed several texts in one call; vectors come back in input order."""
        t0 = time.perf_counter()
        resp = self._call(
            role, model, t0,
            lambda sdk: sdk.embeddings.create(model=model, input=list(texts)),
            timeout_s,
        )
        try:
            items = sorted(resp.data, key=lambda d: d.index)
            vecs = [list(map(float, d.embedding)) for d in items]
        except (AttributeError, TypeError, ValueError) as e:
            raise self._fail(role, model, t0, NebiusError(f"malformed embeddings: {e}")) from e
        if len(vecs) != len(texts):
            raise self._fail(role, model, t0, NebiusError(
                f"asked for {len(texts)} embeddings, got {len(vecs)}"))
        p, _ = _usage(resp)
        self._record(CallRecord(role, model, self._ms(t0), True, p, None,
                                extra={"n": len(texts), "dim": len(vecs[0]) if vecs else 0}))
        return vecs

    def list_models(self, timeout_s: float | None = None) -> list[str]:
        """IDs of the models this key can call."""
        t0 = time.perf_counter()
        resp = self._call("models", "-", t0, lambda sdk: sdk.models.list(), timeout_s)
        try:
            ids = sorted(m.id for m in resp)
        except (AttributeError, TypeError) as e:
            raise self._fail("models", "-", t0, NebiusError(f"malformed model list: {e}")) from e
        self._record(CallRecord("models", "-", self._ms(t0), True, extra={"n": len(ids)}))
        return ids

    # ------------------------------------------------------------------ plumbing

    def _call(self, role: str, model: str, t0: float, fn: Any, timeout_s: float | None) -> Any:
        try:
            sdk = self._sdk()
        except NebiusError as e:
            raise self._fail(role, model, t0, e) from None
        timeout = timeout_s if timeout_s is not None else self.settings.timeout_s
        try:
            target = sdk.with_options(timeout=timeout) if hasattr(sdk, "with_options") else sdk
        except Exception:  # noqa: BLE001 - fakes: use the client default
            target = sdk
        for attempt in (0, 1):
            try:
                return fn(target)
            except openai.APITimeoutError as e:
                raise self._fail(role, model, t0, NebiusError(f"timeout after {timeout:g}s")) from e
            except openai.APIConnectionError as e:
                if attempt == 0:
                    continue
                raise self._fail(role, model, t0, NebiusError(f"connection error: {e}")) from e
            except openai.APIStatusError as e:
                msg = f"HTTP {e.status_code}: {self._scrub(str(e))[:200]}"
                raise self._fail(role, model, t0, NebiusError(msg)) from e
            except Exception as e:
                msg = f"{type(e).__name__}: {self._scrub(str(e))[:200]}"
                raise self._fail(role, model, t0, NebiusError(msg)) from e
        raise AssertionError("unreachable")

    def _fail(self, role: str, model: str, t0: float, err: Exception) -> NebiusError:
        msg = self._scrub(str(err))
        self._record(CallRecord(role, model, self._ms(t0), False, error=msg))
        return err if isinstance(err, NebiusError) else NebiusError(msg)

    def _scrub(self, text: str) -> str:
        """Never let the key reach a log line or an error message."""
        return text.replace(self._key, "***") if self._key else text

    @staticmethod
    def _ms(t0: float) -> int:
        return round((time.perf_counter() - t0) * 1000)

    def _record(self, rec: CallRecord) -> None:
        self.last_call = rec
        line = {
            "ts": round(time.time(), 3),
            "role": rec.role,
            "model": rec.model,
            "latency_ms": rec.latency_ms,
            "prompt_tokens": rec.prompt_tokens,
            "completion_tokens": rec.completion_tokens,
            "ok": rec.ok,
            "error": rec.error,
            **rec.extra,
        }
        path: Path = project_path(self.settings.usage_log)
        try:
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(self._scrub(json.dumps(line)) + "\n")
        except OSError:
            pass  # logging must never break the demo


def make_client(settings: NebiusSettings) -> NebiusClient | None:
    """A client when a key is configured, else None."""
    client = NebiusClient(settings)
    return client if client.available else None
