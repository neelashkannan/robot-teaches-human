"""k-nearest-neighbour stance classifier, after MediaPipe's PoseClassifier.

Two-stage search as in MediaPipe: shortlist the ``top_n_by_max`` training poses
with the smallest *maximum* per-dimension difference (this throws out poses
that match well overall but have one limb badly off), then vote among the ``k``
of those with the smallest *mean* difference.

Rejection: a neighbour only votes if its mean distance is under a threshold
learned from the training data itself: how far a genuine example of a stance
sits from the same stance performed by *other* people (or other recordings).
Random poses have no neighbour that close, so their probabilities sum to ~0
instead of lighting up whichever stance happens to be nearest.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rth.ml.embedding import EMBEDDING_DIM, N_LANDMARKS, embed_batch, mirror_embedding

_CHUNK_BYTES = 64 * 2**20  # bound the (queries, train, dims) difference tensor
_THRESHOLD_SAMPLES = 1500


def to_embeddings(x: np.ndarray) -> np.ndarray:
    """Accept landmarks (33, 3) / (N, 33, 3) or embeddings (D,) / (N, D); return (N, D)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-2:] == (N_LANDMARKS, 3):
        return embed_batch(x.reshape(-1, N_LANDMARKS, 3))
    if x.shape[-1] == EMBEDDING_DIM:
        return x.reshape(-1, EMBEDDING_DIM)
    raise ValueError(f"expected landmarks (..., 33, 3) or embeddings (..., {EMBEDDING_DIM})")


class KNNStanceClassifier:
    """Stance votes from the nearest training poses, with distance-based rejection."""

    def __init__(
        self,
        k: int = 10,
        top_n_by_max: int = 30,
        forward_weight: float = 0.5,
        reject_quantile: float = 0.95,
        reject_margin: float = 1.1,
        augment_mirror: bool = True,
        none_label: str = "none",
    ) -> None:
        self.k = k
        self.top_n_by_max = max(top_n_by_max, k)
        self.forward_weight = forward_weight
        self.reject_quantile = reject_quantile
        self.reject_margin = reject_margin
        self.augment_mirror = augment_mirror
        self.none_label = none_label
        self.classes_: list[str] = []
        self.threshold_: float = np.inf
        self.meta: dict = {}
        self._emb: np.ndarray = np.zeros((0, EMBEDDING_DIM), np.float32)
        self._train: np.ndarray = self._emb  # weighted copy, what distances are measured on
        self._label_idx: np.ndarray = np.zeros(0, np.int64)

    # ------------------------------------------------------------------ fitting

    @property
    def weights(self) -> np.ndarray:
        # Per-dimension weights: each pair contributes (forward, left, up).
        return np.tile([self.forward_weight, 1.0, 1.0], EMBEDDING_DIM // 3)

    @property
    def n_train(self) -> int:
        return len(self._train)

    def fit(
        self,
        x: np.ndarray,
        labels: Sequence[str],
        groups: Sequence[object] | None = None,
    ) -> KNNStanceClassifier:
        """Fit on landmarks or embeddings with one label each.

        ``groups`` (person or recording per sample) is used to learn the rejection
        threshold from cross-group distances; frames within one recording are
        near-duplicates, so within-group distances would make it far too strict.
        With mirror augmentation the other-side version of every pose gets the same label.
        """
        emb = to_embeddings(x)
        labels = np.asarray(labels, dtype=str)
        groups = np.arange(len(emb)) if groups is None else np.asarray(groups).astype(str)
        if len(labels) != len(emb) or len(groups) != len(emb):
            raise ValueError("x, labels and groups must have the same length")
        keep = ~np.isnan(emb).any(axis=1)
        emb, labels, groups = emb[keep], labels[keep], groups[keep]
        if len(emb) == 0:
            raise ValueError("no valid poses to fit on")
        if self.augment_mirror:
            emb = np.concatenate([emb, mirror_embedding(emb)])
            labels = np.concatenate([labels, labels])
            groups = np.concatenate([groups, groups])
        self.classes_ = sorted(set(labels.tolist()))
        self._label_idx = np.searchsorted(self.classes_, labels)
        self._set_train(emb)
        self.threshold_ = self._learn_threshold(groups)
        return self

    def _set_train(self, emb: np.ndarray) -> None:
        self._emb = emb.astype(np.float32)
        self._train = (emb * self.weights).astype(np.float32)

    def _learn_threshold(self, groups: np.ndarray) -> float:
        """Quantile of the k-th same-stance, other-group neighbour distance, times a margin.

        The 'none' class is left out: its examples are deliberately spread out, and
        including them would loosen the threshold for every real stance.
        """
        rng = np.random.default_rng(0)
        stance_rows = np.flatnonzero(np.asarray(self.classes_)[self._label_idx] != self.none_label)
        size = min(len(stance_rows), _THRESHOLD_SAMPLES)
        sample = rng.choice(stance_rows, size=size, replace=False)
        dists: list[float] = []
        for i in sample:
            mask = (self._label_idx == self._label_idx[i]) & (groups != groups[i])
            if not mask.any():
                continue
            cand = self._train[mask]
            diff = np.abs(cand - self._train[i])
            mean = self._two_stage(diff.max(axis=1), diff.mean(axis=1))
            dists.append(float(mean[-1]))
        if not dists:
            return np.inf  # a single group: no honest way to learn a threshold
        return float(np.quantile(dists, self.reject_quantile) * self.reject_margin)

    def _two_stage(self, max_d: np.ndarray, mean_d: np.ndarray) -> np.ndarray:
        """Sorted mean distances of the k neighbours chosen MediaPipe's way (1-D inputs)."""
        n = min(self.top_n_by_max, len(max_d))
        short = np.argpartition(max_d, n - 1)[:n] if n < len(max_d) else np.arange(len(max_d))
        return np.sort(mean_d[short])[: self.k]

    # --------------------------------------------------------------- prediction

    def kneighbors(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(indices (M, k), mean distances (M, k)) of each query's voting neighbours, nearest first.

        Invalid queries get index -1 and distance inf.
        """
        if self.n_train == 0:
            raise RuntimeError("classifier is not fitted")
        q = to_embeddings(x) * self.weights
        m = len(q)
        k = min(self.k, self.n_train)
        n_short = min(self.top_n_by_max, self.n_train)
        idx = np.full((m, k), -1, np.int64)
        dist = np.full((m, k), np.inf)
        valid = ~np.isnan(q).any(axis=1)
        rows = np.flatnonzero(valid)
        chunk = max(1, _CHUNK_BYTES // (4 * self.n_train * EMBEDDING_DIM))
        for start in range(0, len(rows), chunk):
            r = rows[start : start + chunk]
            diff = np.abs(self._train[None] - q[r, None].astype(np.float32))  # (c, N, D)
            max_d = diff.max(axis=2)
            if n_short < self.n_train:
                short = np.argpartition(max_d, n_short - 1, axis=1)[:, :n_short]
            else:
                short = np.broadcast_to(np.arange(self.n_train), (len(r), self.n_train))
            mean_d = np.take_along_axis(diff.mean(axis=2), short, axis=1)
            order = np.argsort(mean_d, axis=1)[:, :k]
            idx[r] = np.take_along_axis(short, order, axis=1)
            dist[r] = np.take_along_axis(mean_d, order, axis=1)
        return idx, dist

    def predict_proba_batch(self, x: np.ndarray) -> np.ndarray:
        """(M, n_classes) vote shares. Rows sum to 1 for familiar poses and less (down to 0)
        when neighbours are beyond the rejection threshold."""
        idx, dist = self.kneighbors(x)
        votes = (dist <= self.threshold_) & (idx >= 0)
        out = np.zeros((len(idx), len(self.classes_)))
        rows = np.repeat(np.arange(len(idx)), idx.shape[1])
        np.add.at(out, (rows[votes.ravel()], self._label_idx[idx.ravel()][votes.ravel()]), 1.0)
        return out / max(idx.shape[1], 1)

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        """Vote share per stance for one pose (landmarks (33, 3) or an embedding)."""
        p = self.predict_proba_batch(np.asarray(x)[None])[0]
        return dict(zip(self.classes_, p.tolist()))

    def predict_batch(self, x: np.ndarray, min_share: float = 0.5) -> list[str | None]:
        """Most-voted stance per pose, or None when it has less than ``min_share`` of the votes."""
        p = self.predict_proba_batch(x)
        best = p.argmax(axis=1)
        return [self.classes_[b] if p[i, b] >= min_share else None for i, b in enumerate(best)]

    def predict(self, x: np.ndarray, min_share: float = 0.5) -> str | None:
        return self.predict_batch(np.asarray(x)[None], min_share)[0]

    # ---------------------------------------------------------------- save/load

    def _params(self) -> dict:
        return {
            "k": self.k,
            "top_n_by_max": self.top_n_by_max,
            "forward_weight": self.forward_weight,
            "reject_quantile": self.reject_quantile,
            "reject_margin": self.reject_margin,
            "augment_mirror": self.augment_mirror,
            "none_label": self.none_label,
        }

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        """Write an .npz (no pickles). Embeddings are stored unweighted and already augmented."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self._emb,
            labels=np.asarray(self.classes_, dtype=str)[self._label_idx],
            classes=np.asarray(self.classes_, dtype=str),
            threshold=np.float64(self.threshold_),
            params=np.asarray(json.dumps(self._params())),
            meta=np.asarray(json.dumps(meta or {})),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> KNNStanceClassifier:
        with np.load(Path(path), allow_pickle=False) as data:
            clf = cls(**json.loads(str(data["params"])))
            clf.classes_ = [str(c) for c in data["classes"]]
            labels = [str(s) for s in data["labels"]]
            clf._label_idx = np.searchsorted(clf.classes_, labels)
            clf._set_train(data["embeddings"].astype(np.float64))
            clf.threshold_ = float(data["threshold"])
            clf.meta = json.loads(str(data["meta"]))
        return clf
