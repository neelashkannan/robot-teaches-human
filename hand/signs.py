"""BSL signs for the app: loading data/signs.json and the pass / skip / scorecard bookkeeping.

Only signs with status `exact` or `approximated` are loaded. `unverified` and `not_representable`
are never shown. A sign's `movement` template is not checked (stretch); only the handshape is.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hand.compare import Hold, Result
from hand.vec import FINGERS, N, clamp

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SIGNS = REPO / "data" / "signs.json"
SHOWN_STATUSES = ("exact", "approximated")

CAPTION = "Checks handshape only, not location, movement or facial expression."
STATUS_LABEL = {"exact": "exact", "approximated": "approximated: closest this hand can do"}


@dataclass(frozen=True)
class Sign:
    id: str
    gloss: str
    source_url: str
    status: str
    instruction: str
    target: np.ndarray
    hold_seconds: float
    movement: dict | None

    @property
    def status_label(self) -> str:
        label = STATUS_LABEL[self.status]
        if self.movement:
            label += "; handshape only, movement not checked"
        return label

    @property
    def source_host(self) -> str:
        return self.source_url.split("//")[-1].split("/")[0]


def load_signs(path=DEFAULT_SIGNS) -> list[Sign]:
    raw = json.loads(Path(path).read_text())
    signs = []
    for entry in raw:
        if entry.get("status") not in SHOWN_STATUSES:
            continue
        target = np.asarray(entry["target"], dtype=float)
        if target.shape != (N,) or not np.isfinite(target).all():
            raise ValueError(f"sign {entry.get('id')!r}: target must be {N} finite numbers")
        signs.append(Sign(
            id=entry["id"],
            gloss=entry["gloss"],
            source_url=entry["source_url"],
            status=entry["status"],
            instruction=entry["instruction"],
            target=clamp(target),
            hold_seconds=float(entry.get("hold_seconds", 3.0)),
            movement=entry.get("movement"),
        ))
    return signs


class Session:
    """One run through the sign set: the current sign, the hold timer, results, off-finger counts."""

    def __init__(self, signs: list[Sign]):
        if not signs:
            raise ValueError("no signs to teach")
        self.signs = signs
        self.i = 0
        self.finished = False
        self.result: dict[str, str | None] = {s.id: None for s in signs}  # passed / skipped / None
        self.off = {s.id: Counter() for s in signs}  # finger -> frames it was marked wrong
        self.hold = Hold(signs[0].hold_seconds)

    @property
    def sign(self) -> Sign:
        return self.signs[self.i]

    def _go(self, i: int) -> None:
        self.i = i
        self.finished = False
        self.hold = Hold(self.sign.hold_seconds)

    def update(self, result: Result | None, now: float) -> tuple[float, bool]:
        """Feed one frame's marking (None = no hand). Returns (hold progress, passed this frame)."""
        if self.finished:
            return 0.0, False
        if result is not None and not result.all_ok:
            for finger, fr in zip(FINGERS, result.per_finger):
                if not fr.ok:
                    self.off[self.sign.id][finger] += 1
        progress, done = self.hold.update(bool(result is not None and result.all_ok), now)
        if done:
            self.result[self.sign.id] = "passed"
            self.advance()
        return progress, done

    def advance(self) -> None:
        if self.i + 1 >= len(self.signs):
            self.finished = True
        else:
            self._go(self.i + 1)

    def skip(self) -> None:
        if self.result[self.sign.id] is None:
            self.result[self.sign.id] = "skipped"
        self.advance()

    def back(self) -> None:
        self._go(max(self.i - 1, 0) if not self.finished else len(self.signs) - 1)

    def repeat(self) -> None:
        self._go(self.i)

    def restart(self) -> None:
        for s in self.signs:
            self.result[s.id] = None
            self.off[s.id].clear()
        self._go(0)

    def scorecard(self) -> list[str]:
        passed = sum(1 for r in self.result.values() if r == "passed")
        lines = [f"Scorecard: {passed} of {len(self.signs)} signs passed", ""]
        for s in self.signs:
            res = self.result[s.id] or "not tried"
            worst = self.off[s.id].most_common(1)
            off = f"most often off: {worst[0][0]}" if worst else "no finger marked wrong"
            lines.append(f"{s.gloss:<16} {res:<9} {off}")
        return lines
