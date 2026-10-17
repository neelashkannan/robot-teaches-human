"""BSL signs for the app: loading data/signs.json and the pass / skip / scorecard bookkeeping.

Only signs with status `exact` or `approximated` are loaded. `unverified` and `not_representable`
are never shown. A sign with a `movement` template (circle) is also checked on the wrist path: see hand.movement.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hand.compare import Hold, Result
from hand.movement import make_check
from hand.vec import FINGERS, N, clamp

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SIGNS = REPO / "data" / "signs.json"
SHOWN_STATUSES = ("exact", "approximated")
MOVE_HOLD_S = 0.5

CAPTION = "Checks handshape, and wrist circles where a sign has them. Not location, palm direction or facial expression."
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
            label += "; handshape + wrist circle checked"
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

    def __init__(self, signs: list[Sign], browse: bool = False):
        """browse=True: move freely (arrows, jump), wrap at the ends, a pass is recorded but never
        advances or finishes the run. browse=False is the practice flow: a pass advances, the end
        shows the scorecard."""
        if not signs:
            raise ValueError("no signs to teach")
        self.signs = signs
        self.browse = browse
        self.i = 0
        self.finished = False
        self.result: dict[str, str | None] = {s.id: None for s in signs}  # passed / skipped / None
        self.off = {s.id: Counter() for s in signs}  # finger -> frames it was marked wrong
        self._go(0)

    @property
    def sign(self) -> Sign:
        return self.signs[self.i]

    def _go(self, i: int) -> None:
        self.i = i
        self.finished = False
        self.move = make_check(self.sign.movement)
        # a moving sign is held only briefly once the movement is complete
        self.hold = Hold(MOVE_HOLD_S if self.move else self.sign.hold_seconds)

    def update(self, result: Result | None, now: float, wrist=None, hand_size=None) -> tuple[float, bool]:
        """Feed one frame's marking (None = no hand) and, for moving signs, the wrist path point.

        Returns (progress, passed this frame). For a moving sign progress is the movement progress
        until the handshape holds, then the short hold.
        """
        if self.finished:
            return 0.0, False
        moved = True
        if self.move is not None:
            self.move.update(now, wrist if result is not None else None, hand_size)
            moved = self.move.done
        if result is not None and not result.all_ok:
            for finger, fr in zip(FINGERS, result.per_finger):
                if not fr.ok:
                    self.off[self.sign.id][finger] += 1
        progress, done = self.hold.update(bool(result is not None and result.all_ok and moved), now)
        if self.move is not None and not done and progress == 0.0:
            progress = self.move.progress
        if done and self.browse:
            first_time = self.result[self.sign.id] != "passed"
            self.result[self.sign.id] = "passed"
            self._go(self.i)  # restart the hold; stay on this sign
            return progress, first_time
        if done:
            self.result[self.sign.id] = "passed"
            self.advance()
        return progress, done

    def next(self) -> None:
        """Free movement (wraps), never marks a result."""
        self._go((self.i + 1) % len(self.signs))

    def prev(self) -> None:
        self._go((self.i - 1) % len(self.signs))

    def jump(self, position: int) -> bool:
        """Go to a sign by its 1-based position; False (and no move) if there is no such sign."""
        if not 1 <= position <= len(self.signs):
            return False
        self._go(position - 1)
        return True

    def toggle_mode(self) -> None:
        self.browse = not self.browse
        self._go(self.i)

    def advance(self) -> None:
        if self.browse:
            self._go((self.i + 1) % len(self.signs))
        elif self.i + 1 >= len(self.signs):
            self.finished = True
        else:
            self._go(self.i + 1)

    def force_pass(self) -> None:
        """Mark the current sign passed without matching it (to click through every sign)."""
        self.result[self.sign.id] = "passed"
        self.advance()

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
