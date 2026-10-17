"""Coach calls run in the background with a hard deadline and an offline fallback."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rth.app.coach_runner import PICK, SENSEI, TIP, CoachRunner
from rth.contracts import Limb, PickResult, RepResult, SenseiDecision, VisionTip


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class Picker:
    def __init__(self, stance="crane", source="nebius", gate=None, error=None):
        self.stance, self.source, self.gate, self.error = stance, source, gate, error

    def pick(self, request):
        if self.gate is not None:
            self.gate.wait(5)
        if self.error:
            raise self.error
        return PickResult(self.stance, 0.9, source=self.source)


class Sensei:
    def __init__(self, stance="bow", source="nebius", gate=None):
        self.stance, self.source, self.gate = stance, source, gate

    def decide(self, history, *, reps_left):
        if self.gate is not None:
            self.gate.wait(5)
        return SenseiDecision(
            self.stance, False, 2.0, 3.0, f"{len(history)} {reps_left}", source=self.source
        )


class Tipper:
    def tip(self, frame, rep):
        return VisionTip(f"tip {frame.shape[1]}")


REP = RepResult(0, "horse", False, True, 3.0, 3.0, dict.fromkeys(Limb, 5.0), None)


def wait_for(runner, n=1, timeout=2.0):
    end = time.monotonic() + timeout
    out = []
    while time.monotonic() < end:
        out += runner.poll()
        if len(out) >= n:
            return out
        time.sleep(0.005)
    return out


@pytest.fixture
def clock():
    return Clock()


def make(clock, **kw):
    args = {
        "picker": Picker(),
        "sensei": Sensei(),
        "fallback_picker": Picker("ready", "offline"),
        "fallback_sensei": Sensei("guard", "offline"),
        "deadline_s": 5.0,
        "clock": clock,
    }
    args.update(kw)
    return CoachRunner(**args)


def test_fast_answers_are_delivered_with_their_source(clock):
    r = make(clock)
    try:
        r.submit_pick(1, "balance")
        r.submit_sensei(2, [REP], 3)
        got = {d.role: d for d in wait_for(r, 2)}
        assert got[PICK].value.stance_id == "crane" and got[PICK].source == "nebius"
        assert got[PICK].token == 1 and not got[PICK].timed_out
        assert got[SENSEI].value.say == "1 3"
        assert r.sources == {PICK: "nebius", SENSEI: "nebius"}
        assert r.pending() == 0
    finally:
        r.close()


def test_deadline_uses_fallback_without_blocking(clock):
    gate = threading.Event()
    r = make(clock, sensei=Sensei(gate=gate), deadline_s=5.0)
    try:
        r.submit_sensei(7, [REP], 2)
        clock.t = 4.9
        t0 = time.monotonic()
        assert r.poll() == []
        assert time.monotonic() - t0 < 0.05  # polling never waits on the worker
        clock.t = 5.0
        (d,) = r.poll()
        assert d.timed_out and d.source == "offline" and d.token == 7
        assert d.value.next_stance_id == "guard" and d.value.source == "offline"
        assert r.sources[SENSEI] == "offline"
        gate.set()  # the abandoned call finishing later must not be delivered
        time.sleep(0.05)
        assert r.poll() == []
    finally:
        gate.set()
        r.close()


def test_errors_fall_back(clock):
    r = make(clock, picker=Picker(error=RuntimeError("boom")))
    try:
        r.submit_pick(3, "anything")
        (d,) = wait_for(r)
        assert d.value.stance_id == "ready" and d.source == "offline" and d.error == "boom"
    finally:
        r.close()


def test_failing_fallback_still_delivers_none(clock):
    r = make(
        clock,
        picker=Picker(error=RuntimeError("a")),
        fallback_picker=Picker(error=RuntimeError("b")),
    )
    try:
        r.submit_pick(3, "anything")
        (d,) = wait_for(r)
        assert d.value is None and d.source == "offline"
    finally:
        r.close()


def test_newer_request_supersedes_older(clock):
    gate = threading.Event()
    r = make(clock, picker=Picker(gate=gate))
    try:
        r.submit_pick(1, "first")
        r.submit_pick(2, "second")
        assert r.pending() == 1
        gate.set()
        (d,) = wait_for(r)
        assert d.token == 2
    finally:
        gate.set()
        r.close()


def test_vision_tip(clock):
    r = make(clock)
    try:
        assert r.submit_tip(1, np.zeros((4, 4, 3), np.uint8), REP) is False
    finally:
        r.close()
    r = make(clock, vision_coach=Tipper())
    try:
        frame = np.zeros((10, 20, 3), np.uint8)
        assert r.submit_tip(4, frame, REP) is True
        assert r.submit_tip(5, None, REP) is False
        (d,) = wait_for(r)
        assert d.role == TIP and d.value.tip == "tip 20" and d.token == 4
    finally:
        r.close()


def test_slow_tip_times_out_to_none(clock):
    gate = threading.Event()

    class Slow:
        def tip(self, frame, rep):
            gate.wait(5)
            return VisionTip("late")

    r = make(clock, vision_coach=Slow(), deadline_s=1.0)
    try:
        r.submit_tip(1, np.zeros((2, 2, 3), np.uint8), REP)
        clock.t = 1.0
        (d,) = r.poll()
        assert d.value is None and d.timed_out
    finally:
        gate.set()
        r.close()
