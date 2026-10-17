"""Per-frame stance recognition for the booth: k-NN votes smoothed over time.

``load_recognizer(settings)`` is the entry point the app uses. It returns None
(never raises) when the recogniser is disabled or not trained yet, so the demo
runs without it.
"""

from __future__ import annotations

import logging
import zipfile
from typing import Any

import numpy as np

from rth.config import RecognizerSettings, project_path
from rth.ml.dataset import NONE_LABEL
from rth.ml.knn import KNNStanceClassifier

log = logging.getLogger(__name__)


class SmoothedRecognizer:
    """Implements ``StanceRecognizer``: EMA over k-NN class probabilities.

    Frames with no pose (None or a degenerate torso) push every probability
    towards 0, so the label fades out instead of sticking. The returned label is
    None when the best stance is below ``min_confidence`` or is the 'none' class;
    the confidence is still that of the best real stance, for an optional "maybe" display.
    """

    def __init__(
        self,
        classifier: KNNStanceClassifier,
        alpha: float = 0.2,
        min_confidence: float = 0.6,
        none_label: str = NONE_LABEL,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.classifier = classifier
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.classes = list(classifier.classes_)
        self._stance_cols = np.array([c != none_label for c in self.classes])
        self._ema = np.zeros(len(self.classes))

    def reset(self) -> None:
        self._ema[:] = 0.0

    def probabilities(self) -> dict[str, float]:
        """Current smoothed probability per class."""
        return dict(zip(self.classes, self._ema.tolist()))

    def update(self, landmarks_world: np.ndarray | None, timestamp_ms: int) -> tuple[str | None, float]:
        p = np.zeros(len(self.classes))
        if landmarks_world is not None:
            lm = np.asarray(landmarks_world, dtype=np.float64)
            if lm.shape == (33, 3) and np.isfinite(lm).all():
                p = self.classifier.predict_proba_batch(lm[None])[0]
        self._ema = (1 - self.alpha) * self._ema + self.alpha * p
        if not self._stance_cols.any():
            return None, 0.0
        stance_p = np.where(self._stance_cols, self._ema, -1.0)
        best = int(stance_p.argmax())
        conf = float(self._ema[best])
        winner = int(self._ema.argmax())
        if conf < self.min_confidence or not self._stance_cols[winner]:
            return None, conf
        return self.classes[best], conf


def _section(settings: Any) -> RecognizerSettings:
    return settings.recognizer if hasattr(settings, "recognizer") else settings


def load_recognizer(settings: Any) -> SmoothedRecognizer | None:
    """The trained recogniser, or None if disabled, untrained or unreadable.

    Accepts the full ``Settings`` or just its ``RecognizerSettings`` section.
    """
    rs = _section(settings)
    if not rs.enabled:
        return None
    path = project_path(rs.model_path)
    if not path.is_file():
        log.info("no stance recogniser at %s (run `rth train-recognizer`)", path)
        return None
    try:
        clf = KNNStanceClassifier.load(path)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        # A corrupt or outdated model file must not take the demo down.
        log.warning("could not load stance recogniser %s: %s", path, exc)
        return None
    # k and the rejection threshold stay as trained: the threshold was learned for that k.
    return SmoothedRecognizer(clf, rs.smoothing_alpha, rs.min_confidence)
