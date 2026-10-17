"""The booth state machine, driven by a fake clock and scripted scores."""

from __future__ import annotations

import pytest

from rth.app.fakes import SimpleSensei
from rth.app.session import (
    FrameInput,
    GoNeutral,
    Phase,
    RequestPick,
    RequestSensei,
    RequestVisionTip,
    ResetTracking,
    Say,
    Session,
    SetTarget,
    accuracy_pct,
    build_scorecard,
)
from rth.config import SessionSettings
from rth.contracts import Limb, LimbScore, PickResult, RepResult, SenseiDecision, VisionTip
from rth.stances import load_stances

FPS = 30
DT = 1.0 / FPS


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def scores(err: float = 5.0, bad: dict[Limb, float] | None = None, tol: float = 25.0):
    bad = bad or {}
    return {limb: LimbScore(limb, bad.get(limb, err), bad.get(limb, err) <= tol) for limb in Limb}


GOOD = scores(5.0)
BAD_THIGH = scores(5.0, {Limb.LEFT_THIGH: 60.0})


class ScriptedSensei:
    def __init__(self, decision: SenseiDecision | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.decision = decision

    def decide(self, history, *, reps_left):
        self.calls.append((len(history), reps_left))
        if self.decision is not None:
            return self.decision
        return SenseiDecision("guard", False, 2.0, 3.0, "Fallback says guard.")


@pytest.fixture
def stances():
    return load_stances()


def make(stances, clock=None, sensei=None, **cfg):
    defaults = {
        "stances_per_session": 3,
        "give_up_s": 10.0,
        "hold_grace_s": 0.3,
        "auto_start_s": 2.0,
        "absent_reset_s": 5.0,
        "result_pause_s": 1.0,
        "scorecard_s": 8.0,
    }
    defaults.update(cfg)
    clock = clock or FakeClock()
    s = Session(
        SessionSettings(**defaults),
        stances,
        sensei or ScriptedSensei(),
        think_timeout_s=4.0,
        clock=clock,
    )
    return s, clock


def run(session, clock, seconds, frame=None, *, until=None):
    """Feed frames for ``seconds``; return all commands. Stops early when ``until(session)``."""
    frame = frame or FrameInput(
        scores=GOOD, person_present=True, full_body=True, robot_at_target=True
    )
    cmds = []
    for _ in range(round(seconds * FPS)):
        clock.t += DT
        cmds += session.update(frame)
        if until and until(session):
            break
    return cmds


def present(sc=GOOD, at_target=True):
    return FrameInput(scores=sc, person_present=True, full_body=True, robot_at_target=at_target)


def to_copying(session, clock, stance="horse"):
    cmds = session.choose(stance)
    assert session.phase is Phase.DEMONSTRATING
    run(session, clock, 1.0, until=lambda s: s.phase is Phase.COPYING)
    assert session.phase is Phase.COPYING
    return cmds


def of(cmds, kind):
    return [c for c in cmds if isinstance(c, kind)]


# --------------------------------------------------------------------------- reps


def test_choose_starts_demonstration_and_moves_robot(stances):
    s, clock = make(stances)
    cmds = s.choose("horse")
    assert s.phase is Phase.DEMONSTRATING
    (target,) = of(cmds, SetTarget)
    assert target == SetTarget("horse", False, 2.0)
    assert of(cmds, Say)
    # Scores during the demonstration don't count, and the robot hasn't arrived yet.
    run(s, clock, 1.0, present(at_target=False))
    assert s.phase is Phase.DEMONSTRATING
    run(s, clock, 0.2, present(at_target=True))
    assert s.phase is Phase.COPYING


def test_demonstration_times_out_if_robot_never_arrives(stances):
    s, clock = make(stances, arrive_timeout_s=3.0)
    s.choose("horse")
    run(s, clock, 4.9, present(at_target=False))
    assert s.phase is Phase.DEMONSTRATING  # a slow robot gets extra time
    run(s, clock, 0.2, present(at_target=False))
    assert s.phase is Phase.COPYING


def recovering(sc=GOOD):
    return FrameInput(scores=sc, person_present=True, full_body=True, robot_at_target=False,
                      robot_recovering=True)


def test_robot_recovering_pauses_the_rep(stances):
    s, clock = make(stances, give_up_s=5.0)
    to_copying(s, clock)
    run(s, clock, 1.5, present(GOOD))
    hold = s.view().hold_progress
    run(s, clock, 10.0, recovering(BAD_THIGH))  # longer than give_up_s and the grace
    v = s.view()
    assert s.phase is Phase.COPYING and v.robot_recovering
    assert v.hold_progress == pytest.approx(hold, abs=0.02)
    run(s, clock, 1.6, present(GOOD), until=lambda s: s.phase is not Phase.COPYING)
    rep = s.history[-1]
    assert rep.held and rep.attempt_s == pytest.approx(3.0, abs=0.1)
    assert not s.view().robot_recovering


def test_robot_recovering_during_demo_waits(stances):
    s, clock = make(stances, arrive_timeout_s=1.0)
    s.choose("horse")
    run(s, clock, 10.0, recovering())
    assert s.phase is Phase.DEMONSTRATING


def test_held_rep_builds_rep_result(stances):
    s, clock = make(stances)
    to_copying(s, clock, "horse")
    cmds = run(s, clock, 3.5, until=lambda s: s.phase is not Phase.COPYING)
    assert s.phase is Phase.THINKING
    (req,) = of(cmds, RequestSensei)
    assert req.reps_left == 2
    (rep,) = req.history
    assert isinstance(rep, RepResult)
    assert rep.held and rep.stance_id == "horse" and rep.rep_index == 0
    assert rep.hold_s == stances.get("horse").hold_s
    assert rep.attempt_s == pytest.approx(3.0, abs=0.1)
    assert set(rep.limb_errors_deg) == set(Limb)
    assert all(e == pytest.approx(5.0) for e in rep.limb_errors_deg.values())
    assert of(cmds, RequestVisionTip)[0].token == req.token


def test_flicker_within_grace_keeps_the_hold(stances):
    s, clock = make(stances, hold_grace_s=0.3)
    to_copying(s, clock)
    run(s, clock, 1.5, present(GOOD))
    run(s, clock, 0.2, present(BAD_THIGH))  # a flicker shorter than the grace
    assert s.view().holding
    run(s, clock, 2.0, present(GOOD), until=lambda s: s.phase is not Phase.COPYING)
    assert s.phase is Phase.THINKING
    rep = s.history[-1]
    assert rep.held
    assert rep.attempt_s == pytest.approx(3.0, abs=0.1)


def test_dropout_longer_than_grace_restarts_the_hold(stances):
    s, clock = make(stances, hold_grace_s=0.3)
    to_copying(s, clock)
    run(s, clock, 1.5, present(GOOD))
    run(s, clock, 0.6, present(BAD_THIGH))
    assert not s.view().holding
    run(s, clock, 2.0, present(GOOD))
    assert s.phase is Phase.COPYING  # only 2 s into the new hold
    run(s, clock, 1.5, present(GOOD), until=lambda s: s.phase is not Phase.COPYING)
    assert s.history[-1].held
    assert s.history[-1].attempt_s == pytest.approx(5.1, abs=0.15)


def test_give_up_ends_rep_not_held_and_names_the_limb(stances):
    s, clock = make(stances, give_up_s=10.0)
    to_copying(s, clock)
    run(s, clock, 1.0, present(GOOD))  # a short good stretch, then the thigh goes wrong
    cmds = run(s, clock, 10.0, present(BAD_THIGH), until=lambda s: s.phase is not Phase.COPYING)
    assert s.phase is Phase.THINKING
    rep = s.history[-1]
    assert not rep.held
    assert rep.attempt_s == pytest.approx(10.0, abs=0.1)
    assert rep.worst_limb is Limb.LEFT_THIGH
    # Errors describe the best window: the good second.
    assert rep.limb_errors_deg[Limb.LEFT_THIGH] == pytest.approx(5.0)
    assert of(cmds, RequestSensei)


def test_give_up_with_no_good_window_uses_whole_attempt(stances):
    s, clock = make(stances, give_up_s=5.0)
    to_copying(s, clock)
    run(s, clock, 6.0, present(BAD_THIGH), until=lambda s: s.phase is not Phase.COPYING)
    rep = s.history[-1]
    assert not rep.held
    assert rep.limb_errors_deg[Limb.LEFT_THIGH] == pytest.approx(60.0)
    assert rep.worst_limb is Limb.LEFT_THIGH


def test_unseen_limbs_are_none(stances):
    s, clock = make(stances, give_up_s=2.0)
    to_copying(s, clock)
    sc = {limb: LimbScore(limb, None, False) for limb in Limb}
    run(s, clock, 3.0, present(sc), until=lambda s: s.phase is not Phase.COPYING)
    rep = s.history[-1]
    assert all(v is None for v in rep.limb_errors_deg.values())
    assert rep.worst_limb is None


# --------------------------------------------------------------------------- sensei


def finish_rep(s, clock, stance="horse"):
    to_copying(s, clock, stance)
    cmds = run(s, clock, 3.5, until=lambda s: s.phase is not Phase.COPYING)
    reqs = of(cmds, RequestSensei)
    return reqs[0] if reqs else None


def test_decision_waits_for_result_pause_then_demonstrates(stances):
    s, clock = make(stances, result_pause_s=1.0)
    req = finish_rep(s, clock)
    decision = SenseiDecision("bow", True, 3.0, 4.0, "Now the other side, slowly.", source="nebius")
    assert s.deliver_decision(req.token, decision) == []  # too soon after the rep
    assert s.phase is Phase.THINKING
    cmds = run(s, clock, 1.1, until=lambda s: s.phase is not Phase.THINKING)
    assert s.phase is Phase.DEMONSTRATING
    assert of(cmds, SetTarget) == [SetTarget("bow", True, 3.0)]
    assert Say("Now the other side, slowly.") in cmds
    v = s.view()
    assert v.exercise.hold_s == 4.0 and v.say_source == "nebius"
    assert len(v.history) == 1  # the session continues


def test_sensei_timeout_uses_fallback_and_ignores_late_answer(stances):
    fallback = ScriptedSensei()
    s, clock = make(stances, sensei=fallback)
    req = finish_rep(s, clock)
    run(s, clock, 3.9)
    assert s.phase is Phase.THINKING
    cmds = run(s, clock, 0.2)
    assert s.phase is Phase.DEMONSTRATING
    assert fallback.calls == [(1, 2)]
    assert of(cmds, SetTarget)[0].stance_id == "guard"
    assert s.view().say_source == "offline"
    late = SenseiDecision("crane", False, 2.0, 3.0, "Too late.")
    assert s.deliver_decision(req.token, late) == []
    assert s.exercise.stance_id == "guard"


def test_none_decision_falls_back(stances):
    s, clock = make(stances)
    req = finish_rep(s, clock)
    s.deliver_decision(req.token, None)
    run(s, clock, 1.1)
    assert s.phase is Phase.DEMONSTRATING and s.exercise.stance_id == "guard"


def test_bad_decision_is_repaired(stances):
    s, clock = make(stances)
    req = finish_rep(s, clock)
    s.deliver_decision(req.token, SenseiDecision("moonwalk", False, 2.0, 3.0, "?"))
    run(s, clock, 1.1)
    assert s.exercise.stance_id == "guard"  # the fallback's choice
    req = finish_rep(s, clock)
    s.deliver_decision(req.token, SenseiDecision("bow", False, 0.01, 999.0, "Extreme."))
    run(s, clock, 1.1)
    assert s.exercise.transition_s == 0.5 and s.exercise.hold_s == 15.0


def test_finished_decision_shows_scorecard(stances):
    s, clock = make(stances)
    req = finish_rep(s, clock)
    s.deliver_decision(req.token, SenseiDecision("ready", False, 2, 3, "Done!", finished=True))
    cmds = run(s, clock, 1.1)
    assert s.phase is Phase.SCORECARD
    assert Say("Done!") in cmds and of(cmds, GoNeutral)


def test_vision_tip_is_kept_for_its_rep(stances):
    s, clock = make(stances)
    req = finish_rep(s, clock)
    s.deliver_tip(req.token, VisionTip("Sink your hips lower."))
    assert s.view().tip == "Sink your hips lower."
    s.deliver_tip(req.token + 99, VisionTip("Stale"))
    assert s.view().tip == "Sink your hips lower."


# --------------------------------------------------------------------------- session end


def test_session_end_scorecard_then_reset(stances):
    s, clock = make(stances, stances_per_session=2, result_pause_s=1.0, scorecard_s=8.0)
    req = finish_rep(s, clock, "horse")
    s.deliver_decision(req.token, SenseiDecision("bow", False, 2.0, 3.0, "Bow next."))
    run(s, clock, 1.1)
    cmds = []
    to_copying(s, clock, "bow")
    cmds += run(s, clock, 3.5, until=lambda s: s.phase is not Phase.COPYING)
    assert s.phase is Phase.THINKING and s.view().final_rep
    assert not of(cmds, RequestSensei)  # nothing left to decide
    cmds = run(s, clock, 1.1)
    assert s.phase is Phase.SCORECARD
    sc = s.view().scorecard
    assert sc.total == 2 and sc.held == 2
    assert [r.stance_id for r in sc.reps] == ["horse", "bow"]
    assert any(isinstance(c, Say) and "2 of 2" in c.text for c in cmds)
    assert s.view().scorecard_left_s == pytest.approx(8.0 - 1.1 + 1.0, abs=0.2)
    cmds = run(s, clock, 8.0, until=lambda s: s.phase is Phase.WAITING)
    assert s.phase is Phase.WAITING
    assert of(cmds, GoNeutral) and of(cmds, ResetTracking)
    assert s.history == [] and s.view().scorecard is None


def test_build_scorecard_first_last_best():
    def rep(i, errs, held=True):
        return RepResult(i, "horse", False, held, 3.0, 3.0, errs, None)

    a = dict.fromkeys(Limb, 45.0)
    b = dict.fromkeys(Limb, 9.0)
    b[Limb.LEFT_SHIN] = None
    c = dict.fromkeys(Limb, 18.0)
    sc = build_scorecard([rep(0, a), rep(1, b, held=False), rep(2, c)])
    shin = next(ls for ls in sc.limbs if ls.limb is Limb.LEFT_SHIN)
    arm = next(ls for ls in sc.limbs if ls.limb is Limb.LEFT_UPPER_ARM)
    assert (shin.first_deg, shin.last_deg, shin.best_deg) == (45.0, 18.0, 18.0)
    assert arm.best_deg == 9.0 and arm.mean_deg == pytest.approx(24.0)
    assert sc.held == 2 and sc.total == 3
    assert sc.first_mean_deg == 45.0 and sc.last_mean_deg == 18.0
    assert arm.first_pct == pytest.approx(50.0) and arm.last_pct == pytest.approx(80.0)
    assert accuracy_pct(None) is None and accuracy_pct(120.0) == 0.0


def test_scorecard_with_no_history():
    sc = build_scorecard([])
    assert sc.total == 0 and sc.first_mean_deg is None


# --------------------------------------------------------------------------- requests


def test_typed_request_picks_then_demonstrates(stances):
    s, clock = make(stances)
    (req,) = s.request("  something for balance ")
    assert isinstance(req, RequestPick) and req.text == "something for balance"
    assert s.view().pending_request == "something for balance"
    # While the pick is outstanding the visitor's arrival doesn't auto-start anything.
    run(s, clock, 3.0, present())
    assert s.phase is Phase.WAITING
    cmds = s.deliver_pick(req.token, PickResult("crane", 0.9, source="nebius"))
    assert s.phase is Phase.DEMONSTRATING
    assert of(cmds, SetTarget)[0].stance_id == "crane"
    assert s.view().pending_request is None and s.view().say_source == "nebius"
    run(s, clock, 1.0, until=lambda s: s.phase is Phase.COPYING)
    assert s.phase is Phase.COPYING


def test_stale_and_unknown_picks(stances):
    s, _clock = make(stances)
    old = s.request("first")[0]
    new = s.request("second")[0]
    assert s.deliver_pick(old.token, PickResult("horse", 1.0)) == []
    assert s.phase is Phase.WAITING
    cmds = s.deliver_pick(new.token, PickResult("not_a_stance", 0.1))
    assert s.phase is Phase.WAITING
    assert of(cmds, Say) and "second" in of(cmds, Say)[0].text
    assert s.request("   ") == []


def test_request_mid_session_continues_the_session(stances):
    s, clock = make(stances)
    finish_rep(s, clock, "horse")
    (req,) = s.request("tai chi")
    s.deliver_pick(req.token, PickResult("white_crane", 0.8))
    assert s.phase is Phase.DEMONSTRATING and len(s.history) == 1
    assert s.deliver_decision(1, SenseiDecision("bow", False, 2, 3, "x")) == []


# --------------------------------------------------------------------------- presence and keys


def test_auto_start_needs_whole_body(stances):
    s, clock = make(stances, auto_start_s=2.0)
    legs_hidden = FrameInput(scores=GOOD, person_present=True, full_body=False)
    run(s, clock, 3.0, legs_hidden)
    assert s.phase is Phase.WAITING
    cmds = run(s, clock, 2.5, present(), until=lambda s: s.phase is not Phase.WAITING)
    assert s.phase is Phase.DEMONSTRATING
    assert of(cmds, SetTarget)[0].stance_id == "ready"


def test_auto_start_disabled(stances):
    s, clock = make(stances, auto_start_s=0.0)
    run(s, clock, 5.0, present())
    assert s.phase is Phase.WAITING
    s.skip()
    assert s.phase is Phase.DEMONSTRATING


def test_visitor_leaving_resets(stances):
    s, clock = make(stances, absent_reset_s=5.0)
    to_copying(s, clock)
    gone = FrameInput(person_present=False)
    run(s, clock, 4.0, gone)
    assert s.phase is Phase.COPYING
    run(s, clock, 0.5, present(BAD_THIGH))  # back in time
    run(s, clock, 4.0, gone)
    assert s.phase is Phase.COPYING
    cmds = run(s, clock, 1.5, gone)
    assert s.phase is Phase.WAITING and of(cmds, ResetTracking)


def test_skip_moves_on_in_every_phase(stances):
    s, clock = make(stances)
    s.skip()
    assert s.phase is Phase.DEMONSTRATING
    cmds = s.skip()
    assert s.phase is Phase.THINKING and not s.history[-1].held and of(cmds, RequestSensei)
    s.skip()  # no answer yet: use the fallback now
    assert s.phase is Phase.DEMONSTRATING
    run(s, clock, 1.0, until=lambda s: s.phase is Phase.COPYING)
    s.skip()
    assert s.phase is Phase.THINKING


def test_restart_invalidates_everything(stances):
    s, clock = make(stances)
    req = finish_rep(s, clock)
    cmds = s.restart()
    assert s.phase is Phase.WAITING and s.history == []
    assert of(cmds, GoNeutral) and of(cmds, ResetTracking)
    assert s.deliver_decision(req.token, SenseiDecision("bow", False, 2, 3, "x")) == []


def test_full_run_with_offline_sensei(stances):
    """Five reps with the built-in offline sensei, start to scorecard."""
    s, clock = make(stances, stances_per_session=5, sensei=SimpleSensei(stances))
    s.skip()
    for _ in range(5 * 12 * FPS):
        clock.t += DT
        for c in s.update(present()):
            if isinstance(c, RequestSensei):
                s.deliver_decision(
                    c.token, s.fallback_sensei.decide(c.history, reps_left=c.reps_left)
                )
        if s.phase is Phase.SCORECARD:
            break
    assert s.phase is Phase.SCORECARD
    assert s.scorecard.total == 5 and s.scorecard.held == 5
