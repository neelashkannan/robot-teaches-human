"""Runs the move picker, sensei and vision tip off the frame loop, with a hard deadline.

Every call goes to a small thread pool so a slow network never stalls the camera. The
frame loop polls once per frame; anything still running past the deadline is abandoned
and answered by the offline fallback instead, so the booth always moves on.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rth.contracts import MovePicker, RepResult, Sensei, VisionCoach

log = logging.getLogger(__name__)

PICK, SENSEI, TIP = "picker", "sensei", "vision"


@dataclass(frozen=True)
class Delivery:
    """One finished coach call, ready to hand to the session."""

    role: str  # PICK, SENSEI or TIP
    token: int
    value: Any  # PickResult / SenseiDecision / VisionTip, or None if nothing usable came back
    source: str  # "nebius" or "offline"
    latency_s: float
    timed_out: bool = False
    error: str | None = None


@dataclass
class _Job:
    role: str
    token: int
    future: Future
    fallback: Callable[[], Any]
    started: float


@dataclass
class CoachRunner:
    picker: MovePicker
    sensei: Sensei
    fallback_picker: MovePicker
    fallback_sensei: Sensei
    vision_coach: VisionCoach | None = None
    deadline_s: float = 5.0
    clock: Callable[[], float] = time.monotonic
    max_workers: int = 3
    # Last source that answered per role, for the on-screen badges.
    sources: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(self.max_workers, thread_name_prefix="coach")
        self._jobs: list[_Job] = []

    # ----------------------------------------------------------------- submit

    def submit_pick(self, token: int, text: str) -> None:
        self._submit(PICK, token, lambda: self.picker.pick(text),
                     lambda: self.fallback_picker.pick(text))

    def submit_sensei(self, token: int, history: Sequence[RepResult], reps_left: int) -> None:
        history = tuple(history)
        self._submit(
            SENSEI,
            token,
            lambda: self.sensei.decide(history, reps_left=reps_left),
            lambda: self.fallback_sensei.decide(history, reps_left=reps_left),
        )

    def submit_tip(self, token: int, frame_bgr: np.ndarray | None, rep: RepResult) -> bool:
        """Ask for a vision tip; False when no vision coach is configured or no frame."""
        coach = self.vision_coach
        if coach is None or frame_bgr is None:
            return False
        frame = frame_bgr.copy()  # the frame loop reuses its buffers
        self._submit(TIP, token, lambda: coach.tip(frame, rep), lambda: None)
        return True

    def _submit(self, role: str, token: int, call: Callable[[], Any],
                fallback: Callable[[], Any]) -> None:
        # A newer request for the same role makes the older one pointless.
        for job in [j for j in self._jobs if j.role == role]:
            job.future.cancel()
            self._jobs.remove(job)
        self._jobs.append(_Job(role, token, self._pool.submit(call), fallback, self.clock()))

    # ----------------------------------------------------------------- poll

    def pending(self) -> int:
        return len(self._jobs)

    def poll(self) -> list[Delivery]:
        """Finished or overdue calls since the last poll. Never blocks."""
        now = self.clock()
        out: list[Delivery] = []
        for job in list(self._jobs):
            latency = now - job.started
            if job.future.done():
                self._jobs.remove(job)
                out.append(self._from_future(job, latency))
            elif latency >= self.deadline_s:
                self._jobs.remove(job)
                job.future.cancel()  # too late to matter; the thread finishes on its own
                log.warning("%s took longer than %.1fs; using offline answer", job.role,
                            self.deadline_s)
                out.append(self._from_fallback(job, latency, timed_out=True))
        for d in out:
            if d.value is not None:
                self.sources[d.role] = d.source
        return out

    def _from_future(self, job: _Job, latency: float) -> Delivery:
        try:
            value = job.future.result()
        except Exception as e:  # noqa: BLE001 - any failure falls back to offline
            log.warning("%s failed (%s); using offline answer", job.role, e)
            return self._from_fallback(job, latency, error=str(e) or type(e).__name__)
        if value is None and job.role != TIP:
            return self._from_fallback(job, latency, error="no answer")
        source = getattr(value, "source", "offline") or "offline"
        return Delivery(job.role, job.token, value, source, latency)

    def _from_fallback(self, job: _Job, latency: float, *, timed_out: bool = False,
                       error: str | None = None) -> Delivery:
        try:
            value = job.fallback()
        except Exception as e:
            log.exception("offline %s failed", job.role)
            return Delivery(job.role, job.token, None, "offline", latency, timed_out, str(e))
        if value is not None and hasattr(value, "source"):
            value.source = "offline"
        return Delivery(job.role, job.token, value, "offline", latency, timed_out, error)

    def close(self) -> None:
        for job in self._jobs:
            job.future.cancel()
        self._jobs.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
