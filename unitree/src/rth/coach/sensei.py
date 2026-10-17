"""The sensei: after each rep, decide the next stance, its timing and one sentence to say.

``RuleSensei`` is deterministic and good enough on its own. ``NebiusSensei`` asks an
LLM, validates the reply against the stance list, and falls back to the rule
decision on any failure, so the booth never waits on the API.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from rth.coach.client import NebiusClient, NebiusError
from rth.coach.picker import as_stance_list
from rth.config import Settings
from rth.contracts import Limb, RepResult, Sensei, SenseiDecision
from rth.stances import Stance, StanceLibrary

TRANSITION_RANGE = (1.0, 5.0)
HOLD_RANGE = (1.5, 6.0)
MAX_SAY_WORDS = 20

_ONE_SIDED = re.compile(r"\b(one (leg|arm|hand|fist|foot|knee)|the other|front|back leg)\b")


def is_one_sided(stance: Stance) -> bool:
    """True if the stance has a left and a right version worth mirroring.

    The stance file has no explicit flag, so this reads the description.
    """
    return bool(_ONE_SIDED.search(stance.description.lower()))


def clamp(x: float, lo: float, hi: float, default: float) -> float:
    if x is None or not isinstance(x, (int, float)) or math.isnan(x):
        return default
    return float(min(hi, max(lo, x)))


def mean_error(rep: RepResult) -> float | None:
    errs = [e for e in rep.limb_errors_deg.values() if e is not None]
    return sum(errs) / len(errs) if errs else None


def limb_words(limb: Limb | str | None) -> str | None:
    """'left forearm' from Limb.LEFT_FOREARM (or its value)."""
    if limb is None:
        return None
    try:
        return Limb(limb).value.replace("_", " ")
    except ValueError:
        return None


def trailing_misses(history: Sequence[RepResult]) -> int:
    """Consecutive not-held reps of the same stance at the end of the history."""
    if not history:
        return 0
    last = history[-1]
    n = 0
    for rep in reversed(history):
        if rep.held or rep.stance_id != last.stance_id or rep.mirrored != last.mirrored:
            break
        n += 1
    return n


def _first_sentence(text: str) -> str:
    """The first sentence, plus the next one when the first is only a short cheer.

    "Great hold! Now the crane: knee high." keeps its cue; a full first sentence stands alone.
    """
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    if not m:
        return text
    first = m.group(1)
    if len(first.split()) <= 3:
        rest = re.match(r"(.+?[.!?])(\s|$)", text[m.end():].lstrip())
        if rest:
            return f"{first} {rest.group(1)}"
    return first


def clean_say(text: str, max_words: int = MAX_SAY_WORDS) -> str:
    """One short sentence: strip quotes and markup, cap the word count."""
    t = re.sub(r"[\"`*\u201c\u201d]", "", str(text))
    t = re.sub(r"^\s*(say|sensei|tip)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip().strip("'").strip()
    t = _first_sentence(t)
    words = t.split()
    if len(words) > max_words:
        t = " ".join(words[:max_words]).rstrip(",;:-") + "."
    return t


# Coaching only, never health advice: anything like this is replaced by the rule line.
_MEDICAL = re.compile(
    r"\b(injur\w*|pain\w*|hurt\w*|medical\w*|doctor\w*|cure\w*|heal\w*|therap\w*|diagnos\w*|"
    r"arthritis|disease\w*|rehab\w*|physio\w*|symptom\w*|treat\w*)\b",
    re.IGNORECASE,
)


def is_safe_say(text: str) -> bool:
    return bool(text) and not _MEDICAL.search(text)


# --------------------------------------------------------------------------- phrasing

_START = [
    "Welcome! Copy me: {name}.",
    "Let's begin with the {name}. Mirror me.",
    "Watch me, then copy: {name}.",
]
_PRAISE = [
    "Excellent! Now the {name}.",
    "Well held! Next up: {name}.",
    "Strong work. Let's try the {name}.",
    "Nice and steady. Now the {name}.",
]
_PRAISE_MIRROR = [
    "Great! Now the same on the other side.",
    "Lovely. Let's mirror it: other side now.",
]
_RETRY = [
    "Close! Watch your {limb}. Once more, a little slower.",
    "Almost there. Check your {limb} and try again.",
    "Good effort. Focus on your {limb} this time.",
]
_RETRY_NO_LIMB = [
    "Nearly! Let's try that one more time, slower.",
    "Keep going, you'll get it. Once more.",
]
_EASIER = [
    "Let's try something easier: the {name}.",
    "No problem. Let's switch to the {name}.",
]
_FINISH = [
    "Wonderful training! You held {held} of {total} stances.",
    "Session complete. {held} of {total} stances held. Bow to your sensei!",
]


def _pick(options: list[str], index: int, **kw: Any) -> str:
    return options[index % len(options)].format(**kw)


class RuleSensei:
    """Deterministic coach: repeat slower on a miss, step down after two, else progress."""

    def __init__(
        self,
        settings: Settings,
        stances: StanceLibrary | Iterable[Stance],
        *,
        seed: int = 0,
    ) -> None:
        self.settings = settings
        self.stances = as_stance_list(stances)
        self._by_id = {s.id: s for s in self.stances}
        self._order = {s.id: i for i, s in enumerate(self.stances)}
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ helpers

    def _stance(self, stance_id: str) -> Stance | None:
        return self._by_id.get(stance_id)

    def _hold(self, stance: Stance | None) -> float:
        base = stance.hold_s if stance else self.settings.session.default_hold_s
        return clamp(base, *HOLD_RANGE, default=self.settings.session.default_hold_s)

    def _transition(self, scale: float = 1.0) -> float:
        base = self.settings.session.default_transition_s
        return round(clamp(base * scale, *TRANSITION_RANGE, default=base), 2)

    def start_stance(self) -> Stance:
        sid = self.settings.session.start_stance
        return self._by_id.get(sid) or min(self.stances, key=lambda s: s.difficulty)

    def _rank(self, s: Stance) -> tuple[int, int]:
        return s.difficulty, self._order[s.id]

    def _next_new(self, history: Sequence[RepResult], current: str) -> Stance | None:
        """Easiest stance not yet held, skipping the current one."""
        held = {r.stance_id for r in history if r.held}
        tried = {r.stance_id for r in history}
        pool = [s for s in self.stances if s.id not in held and s.id != current]
        if not pool:
            return None
        # Prefer stances never tried, then ones tried but not held; easy to hard within each.
        return min(pool, key=lambda s: (s.id in tried, *self._rank(s)))

    def _easier(self, current: Stance | None, history: Sequence[RepResult]) -> Stance:
        cur_d = current.difficulty if current else 99
        cur_id = current.id if current else ""
        held = {r.stance_id for r in history if r.held}
        pool = [s for s in self.stances if s.id != cur_id and s.difficulty <= cur_d]
        if not pool:
            pool = [s for s in self.stances if s.id != cur_id] or self.stances
        return min(pool, key=lambda s: (s.difficulty, s.id in held, self._order[s.id]))

    def _revisit(self, history: Sequence[RepResult]) -> tuple[Stance, bool]:
        """Everything held already: revisit the stance with the worst mean error."""
        worst: dict[str, float] = {}
        for r in history:
            e = mean_error(r)
            worst[r.stance_id] = max(worst.get(r.stance_id, 0.0), e if e is not None else 90.0)
        last = history[-1].stance_id
        candidates = [s for s in self.stances if s.id != last] or self.stances
        pick = max(candidates, key=lambda s: (worst.get(s.id, 0.0), -self._order[s.id]))
        mirrored = is_one_sided(pick) and self._rng.random() < 0.5
        return pick, mirrored

    # ------------------------------------------------------------------ decision

    def decide(self, history: Sequence[RepResult], *, reps_left: int) -> SenseiDecision:
        # Vary phrasing by position so repeated situations don't sound canned.
        n = len(history) + (history[-1].rep_index if history else 0)
        if reps_left <= 0:
            last = history[-1].stance_id if history else self.start_stance().id
            held = sum(r.held for r in history)
            say = _pick(_FINISH, n, held=held, total=len(history))
            return SenseiDecision(last, False, self._transition(), self._hold(None), say,
                                  finished=True)

        if not history:
            s = self.start_stance()
            return SenseiDecision(s.id, False, self._transition(), self._hold(s),
                                  _pick(_START, 0, name=s.name.lower()))

        last = history[-1]
        cur = self._stance(last.stance_id)
        if not last.held:
            misses = trailing_misses(history)
            if misses >= 2:
                s = self._easier(cur, history)
                return SenseiDecision(s.id, False, self._transition(1.25),
                                      clamp(self._hold(s) - 0.5, *HOLD_RANGE, default=2.0),
                                      _pick(_EASIER, n, name=s.name.lower()))
            words = limb_words(last.worst_limb)
            say = (_pick(_RETRY, n, limb=words) if words else _pick(_RETRY_NO_LIMB, n))
            hold = clamp(last.hold_s - 0.5, *HOLD_RANGE, default=self._hold(cur))
            return SenseiDecision(last.stance_id, last.mirrored, self._transition(1.5), hold, say)

        err = mean_error(last)
        good = err is not None and err <= 0.6 * self.settings.scoring.tolerance_deg
        done = {(r.stance_id, r.mirrored) for r in history if r.held}
        if (
            good and cur is not None and is_one_sided(cur)
            and (cur.id, not last.mirrored) not in done
            and self._rng.random() < 0.5
        ):
            hold = clamp(last.hold_s + 0.5, *HOLD_RANGE, default=self._hold(cur))
            return SenseiDecision(cur.id, not last.mirrored, self._transition(0.75), hold,
                                  _pick(_PRAISE_MIRROR, n))

        nxt = self._next_new(history, last.stance_id)
        mirrored = False
        if nxt is None:
            nxt, mirrored = self._revisit(history)
        # Smooth holds are rewarded with a slightly quicker robot.
        transition = self._transition(0.85 if good else 1.0)
        return SenseiDecision(nxt.id, mirrored, transition, self._hold(nxt),
                              _pick(_PRAISE, n, name=nxt.name.lower()))


# --------------------------------------------------------------------------- Nebius


class SenseiReply(BaseModel):
    """What the model must return; numbers are clamped after validation."""

    next_stance_id: str
    mirrored: bool = False
    transition_s: float = 2.0
    hold_s: float = 3.0
    say: str


SENSEI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "next_stance_id": {"type": "string"},
        "mirrored": {"type": "boolean"},
        "transition_s": {"type": "number"},
        "hold_s": {"type": "number"},
        "say": {"type": "string"},
    },
    "required": ["next_stance_id", "mirrored", "transition_s", "hold_s", "say"],
    "additionalProperties": False,
}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """First JSON object in a reply, tolerating code fences, prose and <think> blocks."""
    t = _THINK.sub("", text or "").strip()
    candidates = [t] + [m.group(1).strip() for m in _FENCE.finditer(t)]
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(t, i)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("no JSON object in reply")


def _response_format(mode: str) -> dict[str, Any] | None:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "sensei_decision", "schema": SENSEI_SCHEMA, "strict": True},
        }
    if mode == "json_object":
        return {"type": "json_object"}
    return None


def _fmt_rep(rep: RepResult) -> str:
    err = mean_error(rep)
    worst = limb_words(rep.worst_limb) or "none"
    worst_err = rep.limb_errors_deg.get(rep.worst_limb) if rep.worst_limb else None
    worst_s = f"{worst} ({worst_err:.0f} deg)" if worst_err is not None else worst
    return (
        f"rep {rep.rep_index}: {rep.stance_id}{' mirrored' if rep.mirrored else ''}, "
        f"{'HELD' if rep.held else 'not held'}, hold {rep.hold_s:.1f}s, took {rep.attempt_s:.1f}s, "
        f"mean error {'n/a' if err is None else f'{err:.0f} deg'}, worst limb {worst_s}"
    )


class NebiusSensei:
    """LLM sensei with strict validation; any failure returns the rule decision."""

    def __init__(
        self,
        client: NebiusClient,
        settings: Settings,
        stances: StanceLibrary | Iterable[Stance],
        *,
        max_history: int = 8,
    ) -> None:
        self.client = client
        self.settings = settings
        self.stances = as_stance_list(stances)
        self._by_id = {s.id: s for s in self.stances}
        self.rules = RuleSensei(settings, self.stances)
        self.max_history = max_history
        self._format = settings.nebius.structured_output

    def system_prompt(self) -> str:
        lines = [
            f"- {s.id}: {s.name} (difficulty {s.difficulty}/3, "
            f"{'one-sided, can be mirrored' if is_one_sided(s) else 'symmetric'}). {s.description}"
            for s in self.stances
        ]
        tol = self.settings.scoring.tolerance_deg
        return (
            "You are a warm, encouraging martial arts sensei. A robot shows a stance and a "
            "visitor copies it in front of a camera. After each attempt you choose what comes "
            "next.\n\nStances (use these ids only):\n" + "\n".join(lines) + "\n\n"
            "Rules:\n"
            "- If the last attempt was not held, repeat it more slowly or, after two misses, "
            "choose an easier stance.\n"
            "- If it was held, usually move on to a stance not yet held, roughly easy to hard. "
            "Sometimes ask for the mirrored (other side) version of a one-sided stance.\n"
            f"- Limb errors are angles in degrees; under {tol:.0f} counts as correct.\n"
            f"- transition_s: robot move time, {TRANSITION_RANGE[0]:g} to {TRANSITION_RANGE[1]:g} "
            "seconds (slower for beginners).\n"
            f"- hold_s: seconds to hold, {HOLD_RANGE[0]:g} to {HOLD_RANGE[1]:g}.\n"
            "- say: ONE sentence, at most 18 words, spoken aloud. Encouraging and specific. "
            "If the last attempt was not held, name its worst limb in plain words and how to "
            "fix it. If it was held, praise it in two or three words, then give one body cue "
            "for the next stance, e.g. \"Great hold! Now the crane: knee high, arms wide like "
            "wings.\" No numbers or degrees, no health or medical advice.\n\n"
            'Reply with JSON only: {"next_stance_id": str, "mirrored": bool, '
            '"transition_s": number, "hold_s": number, "say": str}'
        )

    def user_prompt(
        self,
        history: Sequence[RepResult],
        reps_left: int,
        plan: SenseiDecision | None = None,
    ) -> str:
        recent = list(history)[-self.max_history:]
        held = sum(r.held for r in history)
        body = "\n".join(_fmt_rep(r) for r in recent) or "(no attempts yet: choose a first stance)"
        closing = (
            "This is the end of the session: say a short closing line for the visitor and "
            "repeat the last stance id."
            if reps_left <= 0 else f"Stances left after this one: {reps_left - 1}."
        )
        text = f"Attempts so far ({held} of {len(history)} held):\n{body}\n{closing}"
        if plan is not None and reps_left > 0:
            # The rule plan anchors the model: it only needs to improve on a sane default.
            text += (
                f"\nDefault plan (keep it unless you have a clear reason): {plan.next_stance_id}"
                f"{' mirrored' if plan.mirrored else ''}, move {plan.transition_s:g}s, "
                f"hold {plan.hold_s:g}s. Write your own say line."
            )
        return text

    def _ask(self, messages: list[dict[str, Any]]) -> str:
        n = self.settings.nebius
        kw = {"model": n.sensei_model, "max_tokens": n.sensei_max_tokens,
                  "temperature": n.sensei_temperature}
        fmt = _response_format(self._format)
        try:
            return self.client.chat("sensei", messages, response_format=fmt, **kw).text
        except NebiusError as e:
            # Some models refuse a response_format; if prompt-only JSON works, keep using it
            # for the rest of the session rather than failing every rep.
            if fmt is None or not str(e).startswith(("HTTP 400", "HTTP 422")):
                raise
            text = self.client.chat("sensei", messages, response_format=None, **kw).text
            self._format = "none"
            return text

    def parse(self, text: str, fallback: SenseiDecision, *, reps_left: int) -> SenseiDecision:
        """Validate a model reply; raises ValueError if it can't be used."""
        try:
            reply = SenseiReply.model_validate(extract_json(text))
        except ValidationError as e:
            raise ValueError(f"reply does not match schema: {e.errors()[:2]}") from e
        finished = reps_left <= 0
        sid = reply.next_stance_id.strip()
        stance = self._by_id.get(sid)
        if finished:
            # Only the closing line matters; models often put "end" or "none" as the id.
            stance = self._by_id.get(fallback.next_stance_id, stance)
        if stance is None:
            raise ValueError(f"unknown stance id {sid!r}")
        say = clean_say(reply.say)
        if not is_safe_say(say):
            say = fallback.say
        return SenseiDecision(
            next_stance_id=fallback.next_stance_id if finished else stance.id,
            mirrored=bool(reply.mirrored) and is_one_sided(stance) and not finished,
            transition_s=round(clamp(reply.transition_s, *TRANSITION_RANGE,
                                     default=fallback.transition_s), 2),
            hold_s=round(clamp(reply.hold_s, *HOLD_RANGE, default=fallback.hold_s), 2),
            say=say,
            finished=finished,
            source="nebius",
        )

    def decide_remote(self, history: Sequence[RepResult], *, reps_left: int) -> SenseiDecision:
        """Model decision; raises on any failure (``check-nebius`` wants to see why)."""
        fallback = self.rules.decide(history, reps_left=reps_left)
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.user_prompt(history, reps_left, fallback)},
        ]
        return self.parse(self._ask(messages), fallback, reps_left=reps_left)

    def decide(self, history: Sequence[RepResult], *, reps_left: int) -> SenseiDecision:
        try:
            return self.decide_remote(history, reps_left=reps_left)
        except Exception:  # noqa: BLE001 - any failure means offline
            return self.rules.decide(history, reps_left=reps_left)


def make_sensei(settings: Settings, stances: StanceLibrary | Sequence[Stance]) -> Sensei:
    """Nebius sensei when a key is present and the sensei is enabled, else rules only."""
    if settings.nebius.enable_sensei:
        client = NebiusClient(settings.nebius)
        if client.available:
            return NebiusSensei(client, settings, stances)
    return RuleSensei(settings, stances)
