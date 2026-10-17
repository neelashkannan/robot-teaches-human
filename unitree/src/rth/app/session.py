"""One visitor's run of stances, as a pure and deterministic state machine.

The frame loop feeds it per-frame scores and delivers coach answers whenever they
arrive; it answers with commands for the robot, the voice and the coach. There is no
OpenCV and no threading here, and time comes from an injected clock, so every path can
be tested with a fake clock and scripted scores.

WAITING -> DEMONSTRATING -> COPYING -> THINKING -> DEMONSTRATING ... -> SCORECARD -> WAITING
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from rth.config import SessionSettings
from rth.contracts import Limb, PickResult, RepResult, Scores, Sensei, SenseiDecision, VisionTip
from rth.limbs import overall
from rth.stances import StanceLibrary

log = logging.getLogger(__name__)

Clock = Callable[[], float]

# Sensei answers outside these bounds would make the robot snap or stall the booth.
TRANSITION_RANGE_S = (0.5, 8.0)
HOLD_RANGE_S = (1.0, 15.0)
MIN_DEMO_S = 0.5  # even if the robot is already there, give the visitor a beat to look


class Phase(str, Enum):
    WAITING = "waiting"
    DEMONSTRATING = "demonstrating"
    COPYING = "copying"
    THINKING = "thinking"
    SCORECARD = "scorecard"


# --------------------------------------------------------------------------- commands


@dataclass(frozen=True)
class SetTarget:
    stance_id: str
    mirrored: bool
    duration_s: float


@dataclass(frozen=True)
class GoNeutral:
    duration_s: float = 1.5


@dataclass(frozen=True)
class Say:
    text: str


@dataclass(frozen=True)
class RequestPick:
    token: int
    text: str


@dataclass(frozen=True)
class RequestSensei:
    token: int
    history: tuple[RepResult, ...]
    reps_left: int


@dataclass(frozen=True)
class RequestVisionTip:
    token: int
    rep: RepResult


@dataclass(frozen=True)
class ResetTracking:
    """A new visitor: forget smoothed limb directions and recogniser state."""


Command = SetTarget | GoNeutral | Say | RequestPick | RequestSensei | RequestVisionTip | ResetTracking


# --------------------------------------------------------------------------- inputs and views


@dataclass(frozen=True)
class FrameInput:
    scores: Scores | None = None
    person_present: bool = False
    full_body: bool = False  # hips, knees and ankles all visible
    robot_at_target: bool = False
    recognised: str | None = None
    recognised_conf: float = 0.0
    robot_recovering: bool = False  # the physics robot fell and is getting back up


@dataclass(frozen=True)
class Exercise:
    stance_id: str
    mirrored: bool
    transition_s: float
    hold_s: float


def accuracy_pct(error_deg: float | None) -> float | None:
    """Limb accuracy for the scorecard: 100% when parallel, 0% at a right angle or worse."""
    if error_deg is None:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - error_deg / 90.0)))


@dataclass(frozen=True)
class LimbSummary:
    limb: Limb
    first_deg: float | None
    last_deg: float | None
    best_deg: float | None
    mean_deg: float | None

    @property
    def first_pct(self) -> float | None:
        return accuracy_pct(self.first_deg)

    @property
    def last_pct(self) -> float | None:
        return accuracy_pct(self.last_deg)

    @property
    def best_pct(self) -> float | None:
        return accuracy_pct(self.best_deg)


@dataclass(frozen=True)
class Scorecard:
    limbs: tuple[LimbSummary, ...]
    reps: tuple[RepResult, ...]
    held: int
    first_mean_deg: float | None
    last_mean_deg: float | None

    @property
    def total(self) -> int:
        return len(self.reps)


def _mean(values: Sequence[float | None]) -> float | None:
    seen = [v for v in values if v is not None]
    return sum(seen) / len(seen) if seen else None


def build_scorecard(history: Sequence[RepResult]) -> Scorecard:
    """Per-limb first try vs last try (and best, mean) across a session."""
    limbs = []
    for limb in Limb:
        errs = [r.limb_errors_deg.get(limb) for r in history]
        seen = [e for e in errs if e is not None]
        limbs.append(
            LimbSummary(
                limb=limb,
                first_deg=errs[0] if errs else None,
                last_deg=errs[-1] if errs else None,
                best_deg=min(seen) if seen else None,
                mean_deg=_mean(seen),
            )
        )
    first = _mean(list(history[0].limb_errors_deg.values())) if history else None
    last = _mean(list(history[-1].limb_errors_deg.values())) if history else None
    return Scorecard(
        limbs=tuple(limbs),
        reps=tuple(history),
        held=sum(r.held for r in history),
        first_mean_deg=first,
        last_mean_deg=last,
    )


@dataclass(frozen=True)
class SessionView:
    """Everything the screen needs about the session, as one immutable snapshot."""

    phase: Phase
    now: float
    exercise: Exercise | None
    hold_progress: float  # 0..1 of the current hold
    holding: bool
    copy_elapsed_s: float
    reps_done: int
    reps_total: int
    history: tuple[RepResult, ...]
    last_rep: RepResult | None
    say: str | None
    say_source: str | None  # "nebius", "offline" or "app"
    pending_request: str | None
    tip: str | None
    scorecard: Scorecard | None
    scorecard_left_s: float
    final_rep: bool  # THINKING before the scorecard: no sensei call
    best_hold_s: float = 0.0  # longest hold in the last rep
    robot_recovering: bool = False


# --------------------------------------------------------------------------- accumulators


class _Errors:
    """Running per-limb mean error and recogniser label counts over a stretch of frames."""

    def __init__(self) -> None:
        self.sums = dict.fromkeys(Limb, 0.0)
        self.counts = dict.fromkeys(Limb, 0)
        self.red = dict.fromkeys(Limb, 0)  # frames seen but out of tolerance
        self.labels: Counter[str] = Counter()
        self.frames = 0

    def add(self, scores: Scores, label: str | None) -> None:
        self.frames += 1
        for limb, s in scores.items():
            if s.error_deg is not None:
                self.sums[limb] += s.error_deg
                self.counts[limb] += 1
                self.red[limb] += not s.ok
        if label:
            self.labels[label] += 1

    def means(self) -> dict[Limb, float | None]:
        return {
            limb: (self.sums[limb] / self.counts[limb] if self.counts[limb] else None)
            for limb in Limb
        }

    def label(self) -> str | None:
        return self.labels.most_common(1)[0][0] if self.labels else None


@dataclass
class _Hold:
    start: float
    last_ok: float
    errors: _Errors
    # Frames inside the grace period: part of the hold only if it resumes.
    flicker: list[tuple[Scores, str | None]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.last_ok - self.start


# --------------------------------------------------------------------------- session


class Session:
    """The booth flow for one visitor at a time. Every method returns commands to run."""

    def __init__(
        self,
        settings: SessionSettings,
        stances: StanceLibrary,
        fallback_sensei: Sensei,
        *,
        tolerance_deg: float = 25.0,
        min_limbs_seen: int = 6,
        think_timeout_s: float = 6.0,
        clock: Clock = time.monotonic,
    ) -> None:
        self.cfg = settings
        self.stances = stances
        self.fallback_sensei = fallback_sensei
        self.tolerance_deg = tolerance_deg
        self.min_limbs_seen = min_limbs_seen
        self.think_timeout_s = think_timeout_s
        self.clock = clock
        self._token = 0
        self._reset_state()

    # ----------------------------------------------------------------- state

    def _reset_state(self) -> None:
        self.phase = Phase.WAITING
        self.history: list[RepResult] = []
        self.exercise: Exercise | None = None
        self.say: str | None = None
        self.say_source: str | None = None
        self.tip: str | None = None
        self.pending_request: str | None = None
        self.scorecard: Scorecard | None = None
        self.best_hold_s = 0.0
        self.robot_recovering = False
        self._last_frame: float | None = None
        self._phase_start = self.clock()
        self._present_since: float | None = None
        self._absent_since: float | None = None
        self._pick_token: int | None = None
        self._sensei_token: int | None = None
        self._decision: SenseiDecision | None = None
        self._final = False
        self._tip_token: int | None = None
        self._copy_start = 0.0
        self._hold: _Hold | None = None
        self._best: _Hold | None = None
        self._attempt = _Errors()

    def _next_token(self) -> int:
        self._token += 1
        return self._token

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_start = self.clock()

    @property
    def reps_total(self) -> int:
        return self.cfg.stances_per_session

    @property
    def reps_left(self) -> int:
        return max(0, self.reps_total - len(self.history))

    @property
    def active(self) -> bool:
        return self.phase in (Phase.DEMONSTRATING, Phase.COPYING, Phase.THINKING)

    def stance_name(self, stance_id: str) -> str:
        return self.stances.get(stance_id).name if stance_id in self.stances else stance_id

    # ----------------------------------------------------------------- events

    def request(self, text: str) -> list[Command]:
        """The visitor or operator typed what they want; ask the move picker."""
        text = text.strip()
        if not text:
            return []
        self._pick_token = self._next_token()
        self.pending_request = text
        return [RequestPick(self._pick_token, text)]

    def deliver_pick(self, token: int, pick: PickResult | None) -> list[Command]:
        if token != self._pick_token:
            return []  # superseded by a newer request or a restart
        self._pick_token = None
        request, self.pending_request = self.pending_request, None
        if pick is None or pick.stance_id not in self.stances:
            self._set_say(f"Sorry, I don't have a stance for “{request}”.", "app")
            return [Say(self.say or "")]
        return self._begin(
            self._exercise(pick.stance_id),
            f"Let's try the {self.stance_name(pick.stance_id)}.",
            pick.source,
        )

    def choose(self, stance_id: str) -> list[Command]:
        """Operator picked a stance directly (number keys)."""
        if stance_id not in self.stances:
            return []
        self._pick_token, self.pending_request = None, None
        return self._begin(
            self._exercise(stance_id), f"Copy me: {self.stance_name(stance_id)}.", "app"
        )

    def deliver_decision(self, token: int, decision: SenseiDecision | None) -> list[Command]:
        if token != self._sensei_token or self.phase is not Phase.THINKING:
            return []
        self._sensei_token = None
        self._decision = decision if decision is not None else self._fallback_decision()
        return self._maybe_apply_decision(self.clock())

    def deliver_tip(self, token: int, tip: VisionTip | None) -> list[Command]:
        if tip is None or not tip.tip or not self.history:
            return []
        # Tips are tied to the rep they were asked for; accept them until the next rep ends.
        if token == self._tip_token:
            self.tip = tip.tip
        return []

    def skip(self) -> list[Command]:
        """Space: move on now, whatever the phase."""
        now = self.clock()
        if self.phase is Phase.WAITING:
            return self._begin(self._start_exercise(), self._greeting(), "app")
        if self.phase is Phase.DEMONSTRATING:
            self._copy_start = now
            return self._finish(now, held=False)
        if self.phase is Phase.COPYING:
            return self._finish(now, held=False)
        if self.phase is Phase.THINKING:
            if self._final:
                return self._show_scorecard()
            self._sensei_token = None
            decision = self._decision or self._fallback_decision()
            self._decision = None
            return self._apply_decision(decision)
        return self._go_waiting()

    def restart(self) -> list[Command]:
        """R: drop everything and wait for the next visitor."""
        return self._go_waiting()

    # ----------------------------------------------------------------- per frame

    def update(self, inp: FrameInput) -> list[Command]:
        now = self.clock()
        dt = 0.0 if self._last_frame is None else max(0.0, now - self._last_frame)
        self._last_frame = now
        self.robot_recovering = inp.robot_recovering
        if inp.robot_recovering and self.phase in (Phase.DEMONSTRATING, Phase.COPYING):
            # The robot fell: freeze every timer of the rep until it is back up.
            self._pause(dt)
            return []
        if self.phase is Phase.WAITING:
            return self._update_waiting(now, inp)
        if self.phase is Phase.SCORECARD:
            if now - self._phase_start >= self.cfg.scorecard_s:
                return self._go_waiting()
            return []
        # The visitor walked off mid-session: get ready for the next one.
        if inp.person_present:
            self._absent_since = None
        elif self._absent_since is None:
            self._absent_since = now
        elif now - self._absent_since >= self.cfg.absent_reset_s:
            return self._go_waiting()
        if self.phase is Phase.DEMONSTRATING:
            return self._update_demonstrating(now, inp)
        if self.phase is Phase.COPYING:
            return self._update_copying(now, inp)
        return self._update_thinking(now)

    def _pause(self, dt: float) -> None:
        self._phase_start += dt
        self._copy_start += dt
        if self._hold is not None:
            self._hold.start += dt
            self._hold.last_ok += dt
        if self._absent_since is not None:
            self._absent_since += dt

    def _update_waiting(self, now: float, inp: FrameInput) -> list[Command]:
        if not (inp.person_present and inp.full_body):
            self._present_since = None
            return []
        if self._present_since is None:
            self._present_since = now
        waited = now - self._present_since
        if self.cfg.auto_start_s > 0 and waited >= self.cfg.auto_start_s and not self.pending_request:
            return self._begin(self._start_exercise(), self._greeting(), "app")
        return []

    def _update_demonstrating(self, now: float, inp: FrameInput) -> list[Command]:
        assert self.exercise is not None
        elapsed = now - self._phase_start
        arrived = inp.robot_at_target and elapsed >= MIN_DEMO_S
        # Don't trust at_target alone: a robot that never reports arrival must not stall the booth.
        if arrived or elapsed >= self.exercise.transition_s + self.cfg.arrive_timeout_s:
            self._enter(Phase.COPYING)
            self._copy_start = now
            self._hold, self._best, self._attempt = None, None, _Errors()
        return []

    def _update_copying(self, now: float, inp: FrameInput) -> list[Command]:
        assert self.exercise is not None
        scores = inp.scores or {}
        ok = bool(scores) and overall(scores, self.min_limbs_seen)[0]
        self._attempt.add(scores, inp.recognised)
        if ok:
            if self._hold is None:
                self._hold = _Hold(now, now, _Errors())
            self._hold.last_ok = now
            for frame in self._hold.flicker:
                self._hold.errors.add(*frame)
            self._hold.flicker.clear()
            self._hold.errors.add(scores, inp.recognised)
        elif self._hold is not None:
            if now - self._hold.last_ok <= self.cfg.hold_grace_s:
                self._hold.flicker.append((scores, inp.recognised))  # maybe just a flicker
            else:
                self._close_hold()
        if self._hold is not None and now - self._hold.start >= self.exercise.hold_s:
            return self._finish(now, held=True)
        if now - self._copy_start >= self.cfg.give_up_s:
            return self._finish(now, held=False)
        return []

    def _update_thinking(self, now: float) -> list[Command]:
        elapsed = now - self._phase_start
        if self._final:
            return self._show_scorecard() if elapsed >= self.cfg.result_pause_s else []
        if self._decision is None and elapsed >= self.think_timeout_s:
            self._sensei_token = None
            self._decision = self._fallback_decision()
        return self._maybe_apply_decision(now)

    # ----------------------------------------------------------------- transitions

    def _close_hold(self) -> None:
        h = self._hold
        if h is not None and (self._best is None or h.duration > self._best.duration):
            self._best = h
        self._hold = None

    def _finish(self, now: float, *, held: bool) -> list[Command]:
        assert self.exercise is not None
        if held:
            window = self._hold.errors if self._hold else self._attempt
        else:
            self._close_hold()
            # No good stretch at all: describe the whole attempt instead.
            window = self._best.errors if self._best else self._attempt
        errors = window.means()
        seen = {limb: e for limb, e in errors.items() if e is not None}
        worst = max(seen, key=lambda limb: seen[limb]) if seen else None
        red = self._attempt.red
        if not held and any(red.values()):
            # The best window may be all green; what broke the hold is the limb most often red.
            worst = max(Limb, key=lambda limb: (red[limb], seen.get(limb, 0.0)))
        if held and self._hold is not None:
            self.best_hold_s = self.exercise.hold_s
        else:
            self.best_hold_s = self._best.duration if self._best else 0.0
        rep = RepResult(
            rep_index=len(self.history),
            stance_id=self.exercise.stance_id,
            mirrored=self.exercise.mirrored,
            held=held,
            hold_s=self.exercise.hold_s,
            attempt_s=max(0.0, now - self._copy_start),
            limb_errors_deg=errors,
            worst_limb=worst,
            recognised_as=window.label() or self._attempt.label(),
        )
        self.history.append(rep)
        self._hold, self._best = None, None
        self._decision = None
        self.tip = None
        self._enter(Phase.THINKING)
        # No voice line here: the sensei's sentence follows within seconds and says it better.
        cmds: list[Command] = []
        if self.reps_left == 0:
            self._final = True
            self._sensei_token = None
            self._tip_token = None
            return cmds
        self._final = False
        self._sensei_token = self._next_token()
        self._tip_token = self._sensei_token
        cmds.append(RequestSensei(self._sensei_token, tuple(self.history), self.reps_left))
        cmds.append(RequestVisionTip(self._sensei_token, rep))
        return cmds

    def _maybe_apply_decision(self, now: float) -> list[Command]:
        # Hold the answer back briefly so the visitor sees how their rep went.
        if self._decision is None or now - self._phase_start < self.cfg.result_pause_s:
            return []
        decision, self._decision = self._decision, None
        return self._apply_decision(decision)

    def _apply_decision(self, decision: SenseiDecision) -> list[Command]:
        if decision.finished:
            cmds: list[Command] = [Say(decision.say)] if decision.say else []
            return cmds + self._show_scorecard(say=not decision.say)
        if decision.next_stance_id not in self.stances:
            decision = self._fallback_decision()
        ex = Exercise(
            stance_id=decision.next_stance_id,
            mirrored=bool(decision.mirrored),
            transition_s=_clamp(decision.transition_s, TRANSITION_RANGE_S),
            hold_s=_clamp(decision.hold_s, HOLD_RANGE_S),
        )
        say = decision.say or f"Next: {self.stance_name(ex.stance_id)}."
        return self._begin(ex, say, decision.source)

    def _begin(self, ex: Exercise, say: str, source: str) -> list[Command]:
        cmds: list[Command] = []
        if not self.active:
            # A fresh session for a new visitor (or the same one going again).
            self.history, self.scorecard, self.tip = [], None, None
            self._final = False
        self._sensei_token = None
        self._decision = None
        self._final = False
        self.exercise = ex
        self._set_say(say, source)
        self._absent_since = None
        self._enter(Phase.DEMONSTRATING)
        cmds.append(SetTarget(ex.stance_id, ex.mirrored, ex.transition_s))
        cmds.append(Say(say))
        return cmds

    def _show_scorecard(self, *, say: bool = True) -> list[Command]:
        self.scorecard = build_scorecard(self.history)
        self._final = False
        self._sensei_token = None
        self._enter(Phase.SCORECARD)
        text = f"Session complete. You held {self.scorecard.held} of {self.scorecard.total} stances."
        self._set_say(text, "app")
        return [Say(text), GoNeutral()] if say else [GoNeutral()]

    def _go_waiting(self) -> list[Command]:
        self._reset_state()
        return [GoNeutral(), ResetTracking()]

    # ----------------------------------------------------------------- helpers

    def _set_say(self, text: str, source: str | None) -> None:
        self.say, self.say_source = text, source

    def _exercise(self, stance_id: str, mirrored: bool = False) -> Exercise:
        stance = self.stances.get(stance_id)
        return Exercise(stance_id, mirrored, self.cfg.default_transition_s, stance.hold_s)

    def _start_exercise(self) -> Exercise:
        sid = self.cfg.start_stance
        return self._exercise(sid if sid in self.stances else self.stances.ids()[0])

    def _greeting(self) -> str:
        return f"Welcome! Copy me: {self.stance_name(self._start_exercise().stance_id)}."

    def _fallback_decision(self) -> SenseiDecision:
        try:
            d = self.fallback_sensei.decide(tuple(self.history), reps_left=self.reps_left)
            if d.finished or d.next_stance_id in self.stances:
                d.source = "offline"
                return d
        except Exception:  # the booth must keep going whatever the sensei does
            log.exception("offline sensei failed; using the next stance in order")
        return self._default_decision()

    def _default_decision(self) -> SenseiDecision:
        """Last resort: the next stance in library order."""
        ids = self.stances.ids()
        last = self.history[-1].stance_id if self.history else ids[0]
        nxt = ids[(ids.index(last) + 1) % len(ids)] if last in ids else ids[0]
        return SenseiDecision(
            next_stance_id=nxt,
            mirrored=False,
            transition_s=self.cfg.default_transition_s,
            hold_s=self.stances.get(nxt).hold_s,
            say=f"Next: {self.stance_name(nxt)}.",
            source="offline",
        )

    # ----------------------------------------------------------------- view

    def view(self) -> SessionView:
        now = self.clock()
        hold = 0.0
        if self.phase is Phase.COPYING and self._hold is not None and self.exercise:
            hold = min(1.0, (now - self._hold.start) / max(self.exercise.hold_s, 1e-6))
        elif self.phase is Phase.THINKING and self.history and self.history[-1].held:
            hold = 1.0
        return SessionView(
            phase=self.phase,
            now=now,
            exercise=self.exercise,
            hold_progress=hold,
            holding=self.phase is Phase.COPYING and self._hold is not None,
            copy_elapsed_s=now - self._copy_start if self.phase is Phase.COPYING else 0.0,
            reps_done=len(self.history),
            reps_total=self.reps_total,
            history=tuple(self.history),
            last_rep=self.history[-1] if self.history else None,
            say=self.say,
            say_source=self.say_source,
            pending_request=self.pending_request,
            tip=self.tip,
            scorecard=self.scorecard,
            scorecard_left_s=(
                max(0.0, self.cfg.scorecard_s - (now - self._phase_start))
                if self.phase is Phase.SCORECARD
                else 0.0
            ),
            final_rep=self._final,
            best_hold_s=self.best_hold_s,
            robot_recovering=self.robot_recovering,
        )


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    return min(hi, max(lo, v))
