"""Move picker: turn a visitor's request ("something for balance") into a stance.

``EmbeddingMovePicker`` embeds the request on Nebius and compares it with the
stance texts (embedded once, cached on disk). ``OfflineMovePicker`` needs no
network and is the fallback for every failure.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from rth.coach.client import NebiusClient
from rth.config import Settings, project_path
from rth.contracts import MovePicker, PickResult
from rth.stances import Stance, StanceLibrary

# Phrase -> stance ids. Checked before word overlap, and a matched phrase is removed from
# the request so its words can't also vote (e.g. "bow and arrow" must not mean bow stance).
SYNONYMS: dict[str, tuple[str, ...]] = {
    "bow and arrow": ("archer",),
    "drawing a bow": ("archer",),
    "archery": ("archer",),
    "archer": ("archer",),
    "arrow": ("archer",),
    "shoot": ("archer",),
    "shooting": ("archer",),
    "stretch": ("archer",),
    "shoulders": ("archer",),
    "tai chi": ("white_crane",),
    "taichi": ("white_crane",),
    "gentle": ("white_crane",),
    "slow": ("white_crane",),
    "relaxing": ("white_crane",),
    "karate kid": ("crane",),
    "one leg": ("crane",),
    "balance": ("crane", "white_crane"),
    "balancing": ("crane", "white_crane"),
    "wobbly": ("crane", "white_crane"),
    "block": ("guard",),
    "blocking": ("guard",),
    "defend": ("guard",),
    "defence": ("guard",),
    "defense": ("guard",),
    "protect": ("guard",),
    "boxing": ("guard",),
    "boxer": ("guard",),
    "fight": ("guard",),
    "fighting": ("guard",),
    "self defence": ("guard",),
    "strong legs": ("horse",),
    "leg strength": ("horse",),
    "legs": ("horse",),
    "thighs": ("horse",),
    "squat": ("horse",),
    "strength": ("horse",),
    "strong": ("horse",),
    "horse": ("horse",),
    "punch": ("bow",),
    "punching": ("bow",),
    "strike": ("bow",),
    "attack": ("bow",),
    "lunge": ("bow",),
    "hit": ("bow",),
    "warm up": ("ready",),
    "warmup": ("ready",),
    "easy": ("ready",),
    "beginner": ("ready",),
}

_STOPWORDS = frozenset({
    "a", "an", "and", "the", "to", "of", "for", "in", "on", "at", "me", "my", "i", "you",
    "your", "it", "is", "be", "do", "how", "can", "want", "like", "some", "something",
    "teach", "show", "give", "let", "lets", "please", "with", "what", "stance", "pose",
    "move", "one", "other", "would", "could", "should", "get", "make", "try",
})

_W_SYNONYM, _W_NAME, _W_TAG, _W_DESC = 3.0, 2.0, 1.5, 1.0
_FULL_SCORE = 4.0  # an offline score this high counts as similarity 1.0

# Qwen3-Embedding is trained with an instruction on the query side only.
QUERY_INSTRUCTION = (
    "Instruct: Given a visitor's request, retrieve the martial arts stance that best matches it\n"
    "Query: "
)


def as_stance_list(stances: StanceLibrary | Iterable[Stance]) -> list[Stance]:
    return list(stances.stances if isinstance(stances, StanceLibrary) else stances)


def _stem(word: str) -> str:
    # Crude, but enough to make "legs"/"leg" and "punches"/"punch" meet.
    for suffix in ("ing", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS}


def _phrase_hits(text: str) -> tuple[list[str], str]:
    """Synonym phrases found in ``text`` (longest first) and the text with them removed."""
    t = " " + re.sub(r"[^a-z]+", " ", text.lower()) + " "
    hits: list[str] = []
    for phrase in sorted(SYNONYMS, key=len, reverse=True):
        needle = f" {phrase} "
        if needle in t:
            hits.append(phrase)
            t = t.replace(needle, " ")
    return hits, t


def stance_text(s: Stance) -> str:
    """The text embedded for a stance."""
    tags = ", ".join(s.tags)
    return f"{s.name}. {s.description} Good for: {tags}." if tags else f"{s.name}. {s.description}"


class OfflineMovePicker:
    """Keyword and synonym overlap over each stance's name, description and tags."""

    def __init__(self, stances: StanceLibrary | Iterable[Stance]) -> None:
        self.stances = as_stance_list(stances)
        self._ids = {s.id for s in self.stances}
        self._name = {s.id: tokens(s.name) for s in self.stances}
        self._tags = {s.id: tokens(" ".join(s.tags)) for s in self.stances}
        self._desc = {s.id: tokens(s.description) for s in self.stances}

    def scores(self, request: str) -> dict[str, float]:
        hits, rest = _phrase_hits(request)
        out = {s.id: 0.0 for s in self.stances}
        for phrase in hits:
            for sid in SYNONYMS[phrase]:
                if sid in self._ids:
                    out[sid] += _W_SYNONYM
        words = tokens(rest)
        for sid in out:
            out[sid] += _W_NAME * len(words & self._name[sid])
            out[sid] += _W_TAG * len(words & self._tags[sid])
            out[sid] += _W_DESC * len(words & self._desc[sid])
        return out

    def default_stance(self) -> Stance:
        """Easiest stance that isn't the neutral starting pose; used when nothing matches."""
        active = [s for s in self.stances if not {"neutral", "start"} & set(s.tags)]
        pool = active or self.stances
        return min(pool, key=lambda s: s.difficulty)

    def pick(self, request: str) -> PickResult:
        scores = self.scores(request or "")
        # Stable order: score, then library order, so ties are deterministic.
        order = sorted(self.stances, key=lambda s: -scores[s.id])
        best = order[0]
        if scores[best.id] <= 0:
            fallback = self.default_stance()
            alts = [(s.id, 0.0) for s in self.stances if s.id != fallback.id][:3]
            return PickResult(fallback.id, 0.0, alts, source="offline")

        def sim(sid: str) -> float:
            return round(min(1.0, scores[sid] / _FULL_SCORE), 3)

        alts = [(s.id, sim(s.id)) for s in order[1:4] if scores[s.id] > 0]
        return PickResult(best.id, sim(best.id), alts, source="offline")


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()[:24]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


class EmbeddingMovePicker:
    """Nebius embeddings + cosine similarity; any failure gives the offline answer."""

    def __init__(
        self,
        client: NebiusClient,
        settings: Settings,
        stances: StanceLibrary | Iterable[Stance],
    ) -> None:
        self.client = client
        self.settings = settings
        self.stances = as_stance_list(stances)
        self.offline = OfflineMovePicker(self.stances)
        self.cache_path: Path = project_path(settings.nebius.embedding_cache)
        self._vectors: dict[str, np.ndarray] | None = None

    @property
    def model(self) -> str:
        return self.settings.nebius.embedding_model

    def _load_cache(self) -> dict[str, list[float]]:
        try:
            data = json.loads(self.cache_path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_cache(self, cache: dict[str, list[float]]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            tmp.replace(self.cache_path)
        except OSError:
            pass  # a missing cache only costs one extra embedding call next time

    def stance_vectors(self) -> dict[str, np.ndarray]:
        """Stance id -> embedding; embeds only texts missing from the disk cache."""
        if self._vectors is not None:
            return self._vectors
        cache = self._load_cache()
        keys = {s.id: _cache_key(self.model, stance_text(s)) for s in self.stances}
        missing = [s for s in self.stances if keys[s.id] not in cache]
        if missing:
            vecs = self.client.embed("picker", [stance_text(s) for s in missing], model=self.model)
            for s, v in zip(missing, vecs):
                cache[keys[s.id]] = v
            self._save_cache(cache)
        self._vectors = {s.id: np.asarray(cache[keys[s.id]], float) for s in self.stances}
        return self._vectors

    def pick_remote(self, request: str) -> PickResult:
        """Embedding pick; raises on any failure (``check-nebius`` wants to see why)."""
        stance_vecs = self.stance_vectors()
        (q,) = self.client.embed("picker", [QUERY_INSTRUCTION + request], model=self.model)
        qv = np.asarray(q, float)
        sims = {sid: _cosine(qv, v) for sid, v in stance_vecs.items()}
        order = sorted(self.stances, key=lambda s: -sims[s.id])
        best = order[0].id

        def sim01(sid: str) -> float:
            return round(max(0.0, min(1.0, sims[sid])), 3)

        alts = [(s.id, sim01(s.id)) for s in order[1:4]]
        return PickResult(best, sim01(best), alts, source="nebius")

    def pick(self, request: str) -> PickResult:
        if not request or not request.strip():
            return self.offline.pick(request)
        try:
            return self.pick_remote(request.strip())
        except Exception:  # noqa: BLE001 - any failure means offline
            return self.offline.pick(request)


def make_picker(settings: Settings, stances: StanceLibrary | Sequence[Stance]) -> MovePicker:
    """Nebius picker when a key is present and the picker is enabled, else offline."""
    if settings.nebius.enable_picker:
        client = NebiusClient(settings.nebius)
        if client.available:
            return EmbeddingMovePicker(client, settings, stances)
    return OfflineMovePicker(stances)
