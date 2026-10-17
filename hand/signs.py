"""BSL signs for the app: loading data/signs.json and the pass / skip / scorecard bookkeeping.

Only signs with status `exact` or `approximated` are loaded. `unverified` and `not_representable`
are never shown. A sign with a `movement` template (circle) is also checked on the wrist path: see hand.movement.

A statement (data/statements.json, or typed by the learner) is a short sentence made of signs. It is
taught in two rounds: learn each sign on its own, then watch the robot sign the whole statement and
sign it back.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from hand.compare import Hold, Result
from hand.movement import make_check
from hand.vec import FINGERS, N, clamp

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SIGNS = REPO / "data" / "signs.json"
DEFAULT_STATEMENTS = REPO / "data" / "statements.json"
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
    view: str = "palm"  # "palm": marked only palm-forward, fingers up; "any": made side-on or on the body

    @property
    def status_label(self) -> str:
        label = STATUS_LABEL[self.status]
        if self.movement:
            label += "; handshape + wrist circle checked"
        return label

    @property
    def source_host(self) -> str:
        return self.source_url.split("//")[-1].split("/")[0]


def _parse_sign(entry: dict) -> Sign | None:
    """One JSON sign entry -> Sign, or None if its status is never shown."""
    if entry.get("status") not in SHOWN_STATUSES:
        return None
    target = np.asarray(entry["target"], dtype=float)
    if target.shape != (N,) or not np.isfinite(target).all():
        raise ValueError(f"sign {entry.get('id')!r}: target must be {N} finite numbers")
    return Sign(
        id=entry["id"],
        gloss=entry["gloss"],
        source_url=entry["source_url"],
        status=entry["status"],
        instruction=entry["instruction"],
        target=clamp(target),
        hold_seconds=float(entry.get("hold_seconds", 3.0)),
        movement=entry.get("movement"),
        view=entry.get("view", "palm"),
    )


def load_signs(path=DEFAULT_SIGNS) -> list[Sign]:
    raw = json.loads(Path(path).read_text())
    return [sign for sign in map(_parse_sign, raw) if sign is not None]


@dataclass(frozen=True)
class Statement:
    id: str
    english: str
    gloss: str
    signs: tuple[Sign, ...]
    unknown: tuple[str, ...] = ()  # typed words with no sign in the phrasebook (skipped)


def _vocabulary(path=DEFAULT_STATEMENTS, signs_path=DEFAULT_SIGNS) -> tuple[dict, dict[str, Sign]]:
    """statements.json as loaded, and every shown sign by id (statement-only signs override signs.json)."""
    raw = json.loads(Path(path).read_text())
    known = {s.id: s for s in load_signs(signs_path)}
    known.update({s.id: s for s in map(_parse_sign, raw.get("signs", [])) if s is not None})
    return raw, known


def load_statement(statement_id: str = "im-sorry", path=DEFAULT_STATEMENTS, signs_path=DEFAULT_SIGNS) -> Statement:
    """A statement with its signs in order. Its own signs (statements.json) come first, then signs.json."""
    raw, known = _vocabulary(path, signs_path)
    for st in raw["statements"]:
        if st["id"] == statement_id:
            missing = [sid for sid in st["signs"] if sid not in known]
            if missing or not st["signs"]:
                raise ValueError(f"statement {statement_id!r}: no shown sign for {missing or 'any id'}")
            return Statement(st["id"], st["english"], st["gloss"], tuple(known[sid] for sid in st["signs"]))
    raise ValueError(f"no statement {statement_id!r} in {path}")


CONTRACTIONS = {"i'm": "i am", "you're": "you are", "i've": "i have", "it's": "it is", "that's": "that is"}
MAX_TYPED_SIGNS = 8


def phrasebook_words(path=DEFAULT_STATEMENTS) -> list[str]:
    """The English words and phrases that have a sign, for showing the learner what they can type."""
    return sorted(w for w in json.loads(Path(path).read_text()).get("words", {}) if not w.isdigit())


def statement_from_text(text: str, path=DEFAULT_STATEMENTS, signs_path=DEFAULT_SIGNS) -> Statement | None:
    """Typed English -> a Statement of the phrasebook's signs in the typed order (None if no word has a sign).

    Two-word phrases ("thank you") win over single words. Grammar words BSL does not sign (skip_words) are
    dropped; any other word without a sign is listed in `unknown` rather than guessed. The order is the
    typed English order, not BSL grammar.
    """
    raw, known = _vocabulary(path, signs_path)
    words = {w: sid for w, sid in raw.get("words", {}).items() if sid in known}
    skip = set(raw.get("skip_words", []))
    reasons = raw.get("cannot_sign", {})
    tokens = []
    for tok in re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'")):
        tokens += CONTRACTIONS.get(tok, tok).split()
    signs: list[Sign] = []
    unknown: list[str] = []
    k = 0
    while k < len(tokens):
        pair = " ".join(tokens[k:k + 2])
        if k + 1 < len(tokens) and pair in words:
            signs.append(known[words[pair]])
            k += 2
            continue
        tok = tokens[k].strip("'")
        if tok in words:
            signs.append(known[words[tok]])
        elif tok and tok not in skip:
            shown = f"{tok} ({reasons[tok]})" if tok in reasons else tok
            if shown not in unknown:
                unknown.append(shown)
        k += 1
    if not signs:
        return None
    signs = signs[:MAX_TYPED_SIGNS]
    english = " ".join(text.split())
    return Statement("typed", english, " ".join(s.id.upper() for s in signs), tuple(signs), tuple(unknown))


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

    @property
    def heading(self) -> str:
        return f"Sign {self.i + 1} of {len(self.signs)}: {self.sign.gloss}"

    @property
    def note(self) -> str:
        """Extra line for the learner (nothing, for a plain sign set)."""
        return ""

    def demo_sign(self, now: float) -> Sign | None:
        """The sign the robot is demonstrating while marking is paused (never, for a plain sign set)."""
        return None

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


LEARN_HOLD_S = 1.5  # per sign when learning a statement's signs one by one
TOGETHER_HOLD_S = 0.8  # per sign when signing the whole statement
DEMO_STEP_S = 1.2  # robot shows each sign of the statement this long before the learner's turn


class StatementLesson:
    """Teach one statement: learn each sign (normal hold), watch the robot sign it all, sign it all back.

    Same interface as Session (sign, i, signs, finished, heading, note, result, move, hold, browse,
    update, skip, back, next, prev, jump, force_pass, toggle_mode, repeat, restart, scorecard,
    demo_sign), so the app runs either one. A statement is always practice: there is no browse mode.
    """

    browse = False

    def __init__(self, statement: Statement):
        self.statement = statement
        # A statement is marked on handshapes only: the movement is in each instruction but not
        # checked, so a sign like SORRY (circle on the chest) does not hold the whole sentence up.
        self.learn = Session([replace(s, hold_seconds=LEARN_HOLD_S, movement=None) for s in statement.signs])
        self.together = Session([replace(s, hold_seconds=TOGETHER_HOLD_S, movement=None) for s in statement.signs])
        self.round = self.learn
        self.demo_start: float | None = None

    @property
    def signs(self) -> list[Sign]:
        return self.round.signs

    @property
    def i(self) -> int:
        return self.round.i

    @property
    def sign(self) -> Sign:
        return self.round.sign

    @property
    def finished(self) -> bool:
        return self.round is self.together and self.together.finished

    @property
    def result(self) -> dict[str, str | None]:
        return self.round.result

    @property
    def move(self):
        return self.round.move

    @property
    def hold(self):
        return self.round.hold

    @property
    def heading(self) -> str:
        quote = f'"{self.statement.english}"'
        if self.round is self.learn:
            return f"{quote}  Learn sign {self.i + 1} of {len(self.signs)}: {self.sign.gloss}"
        words = " ".join(f"[{s.id.upper()}]" if k == self.i else s.id.upper() for k, s in enumerate(self.signs))
        return f"{quote}  Now sign it all: {words}"

    @property
    def note(self) -> str:
        """Typed words the statement leaves out, kept on screen for the whole lesson."""
        unknown = self.statement.unknown
        return f"Not taught (no sign yet): {', '.join(unknown)}" if unknown else ""

    def demo_sign(self, now: float) -> Sign | None:
        if self.round is not self.together or self.demo_start is None:
            return None
        k = int((now - self.demo_start) / DEMO_STEP_S)
        return self.signs[k] if 0 <= k < len(self.signs) else None

    def _next_round(self) -> None:
        if self.round is self.learn and self.learn.finished:
            self.round = self.together
            self.together.restart()
            self.demo_start = None

    def update(self, result: Result | None, now: float, wrist=None, hand_size=None) -> tuple[float, bool]:
        if self.round is self.together and self.demo_start is None:
            self.demo_start = now
        if self.demo_sign(now) is not None:
            return 0.0, False
        progress, done = self.round.update(result, now, wrist, hand_size)
        self._next_round()
        return progress, done

    def skip(self) -> None:
        self.round.skip()
        self._next_round()

    def next(self) -> None:
        self.skip()

    def prev(self) -> None:
        self.back()

    def jump(self, position: int) -> bool:
        return self.round.jump(position)

    def force_pass(self) -> None:
        self.round.force_pass()
        self._next_round()

    def toggle_mode(self) -> None:
        """No browse mode for a statement."""

    def back(self) -> None:
        if self.round is self.together and self.i == 0 and not self.together.finished:
            self.round = self.learn
            self.learn.back()
        else:
            self.round.back()

    def repeat(self) -> None:
        self.round.repeat()
        if self.round is self.together:
            self.demo_start = None

    def restart(self) -> None:
        self.learn.restart()
        self.together.restart()
        self.round = self.learn
        self.demo_start = None

    def scorecard(self) -> list[str]:
        st = self.statement
        whole = all(r == "passed" for r in self.together.result.values())
        lines = [f'Statement: "{st.english}"   Signs: {st.gloss}',
                 f"No sign yet for: {', '.join(st.unknown)} (skipped)" if st.unknown else "",
                 "You signed the whole statement." if whole else "Whole statement: not finished.", ""]
        for s in st.signs:
            res = self.learn.result[s.id] or "not tried"
            worst = self.learn.off[s.id].most_common(1)
            off = f"most often off: {worst[0][0]}" if worst else "no finger marked wrong"
            lines.append(f"{s.gloss:<16} {res:<9} {off}")
        return lines
