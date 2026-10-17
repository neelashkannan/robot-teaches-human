"""One window: selfie webcam with per-finger marking | teacher hand showing the sign (| mirror hand).

Starts in BROWSE mode: move through every sign freely; marking and the hold ring run but never
block. `t` switches to PRACTICE (a pass advances, the end shows a scorecard).
Keys: Right/n next, Left/b previous, 1-9 jump to a sign, r replay the teacher's pose slowly,
      f force-pass, t browse/practice, q quit;
      s say a statement: type it, Enter teaches it with the phrasebook's signs (learn each sign,
        watch the robot sign it all, sign it back); Enter on an empty line returns to the sign set;
      c calibrate (10 s: 2 s get ready, 3 s open hand, 2 s get ready, 3 s fist),
      m toggle the mirror hand (a second sim hand copying you),
      p drop the loose objects on the bench again.
(Practice: n skips, b goes back, r repeats the sign; on the scorecard r goes again.)
Frames stay in memory: nothing is saved, encoded or sent anywhere.
Run: .venv/bin/python -m hand.app        (--say "good morning" starts on that statement,
                                          --statement starts on "I'm sorry." (ME SORRY),
                                          --debug prints the raw finger angles twice a second)
"""
from __future__ import annotations

import sys
import textwrap
import time

import cv2
import numpy as np

from hand.compare import compare
from hand.hand_pose import _raw_curl_all, _raw_spread_all, fit_calibration, hand_quality, landmarks_to_vec
from hand.signs import (CAPTION, Session, StatementLesson, load_signs, load_statement, phrasebook_words,
                        statement_from_text)
from hand.teacher import Teacher
from hand.vec import FINGERS
from hand.vision import HandTracker

H = 480
STRIP_H = 200
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16}  # MediaPipe fingertip landmarks
GREEN, RED, YELLOW, WHITE, GREY = (0, 220, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (150, 150, 150)
ORANGE = (0, 165, 255)
# (label, seconds, collect into) steps of the 10 s calibration
CAL_STEPS = [
    ("Get ready: open hand", 2.0, None),
    ("Hold your hand OPEN", 3.0, "open"),
    ("Get ready: make a fist", 2.0, None),
    ("Hold a FIST", 3.0, "fist"),
]
SMOOTH = 0.5  # weight of the new HandVec in the mirror hand's low-pass
PASSED_BANNER_S = 1.5
REPLAY_S = 2.5  # slow teacher replay: open hand -> sign
HINT_NAV = "Right/n next   Left/b previous   1-9 jump   r replay   f force-pass   t browse/practice   q quit"
HINT_OTHER = "s say a statement   c calibrate   m mirror hand   p drop objects"
MAX_TYPED_CHARS = 60
TYPED_BANNER_S = 4.0
# cv2.waitKeyEx arrow codes differ by platform (macOS Cocoa, Linux GTK/X11, Windows)
KEY_RIGHT = {63235, 65363, 2555904}
KEY_LEFT = {63234, 65361, 2424832}


class PoseView:
    """A Teacher stepped in real time; re-renders only while something is moving."""

    MAX_STEPS = 40  # cap per frame so a slow frame cannot snowball

    def __init__(self):
        self.teacher = Teacher()
        self.vec = None
        self.img = None
        self.acc = 0.0
        self.last = time.time()
        self.settle = 0  # extra frames to render after motion stops

    def drop(self) -> None:
        self.teacher.drop_objects()
        self.settle = 10

    def show(self, vec) -> np.ndarray:
        vec = np.asarray(vec, dtype=float)
        if self.vec is None or np.abs(vec - self.vec).max() > 0.01:
            self.teacher.set_pose(vec)
            self.vec = vec.copy()
            self.settle = 10
        now = time.time()
        self.acc += (now - self.last) / self.teacher.model.opt.timestep
        self.last = now
        n = min(int(self.acc), self.MAX_STEPS)
        self.acc = 0.0 if n == self.MAX_STEPS else self.acc - n
        self.teacher.step(n)
        if np.abs(self.teacher.data.qvel).max() > 5e-3:
            self.settle = 10
        if self.settle > 0 or self.img is None:
            self.img = cv2.cvtColor(self.teacher.render(H, H), cv2.COLOR_RGB2BGR)
            self.settle -= 1
        return self.img


def label(img, text, x=10, y=28, scale=0.7, color=WHITE):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def big_text(img, text, color=YELLOW) -> None:
    """Large centred message on a panel, wrapped to fit."""
    h, w = img.shape[:2]
    lines = textwrap.wrap(text, 24)
    for k, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        y = h // 2 + (k - (len(lines) - 1) / 2) * (th + 16)
        label(img, line, max(int((w - tw) / 2), 8), int(y), 1.1, color)


def draw_landmarks(img, lm) -> None:
    h, w = img.shape[:2]
    for i, (x, y, _) in enumerate(lm):
        cv2.circle(img, (int(x * w), int(y * h)), 4, GREY if i >= 17 else (0, 255, 0), -1)  # no little finger


def draw_marking(img, lm, result) -> None:
    """Green / red ring on each fingertip plus a legend; result is a compare() Result."""
    h, w = img.shape[:2]
    for k, (finger, fr) in enumerate(zip(FINGERS, result.per_finger)):
        color = GREEN if fr.ok else RED
        x, y, _ = lm[TIP[finger]]
        cv2.circle(img, (int(x * w), int(y * h)), 16, color, 3)
        label(img, f"{finger}: {'ok' if fr.ok else 'fix'}", 10, 60 + 26 * k, 0.6, color)


def draw_hold_ring(img, progress) -> None:
    h, w = img.shape[:2]
    c = (w - 50, h - 50)
    cv2.circle(img, c, 32, (60, 60, 60), 5)
    if progress > 0:
        cv2.ellipse(img, c, (32, 32), -90, 0, 360 * progress, GREEN, 6)


def strip(width, lines, names=None, current=0, results=None) -> np.ndarray:
    """Info bar under the panels: lines are (text, color, scale); names is the strip of all signs
    (current one highlighted, passed ones green), wrapped onto as many rows as it needs."""
    rows = []  # list of lists of (text, x, w)
    x, row = 10, []
    for k, name in enumerate(names or []):
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if x + tw + 14 > width and row:
            rows.append(row)
            x, row = 10, []
        row.append((k, name, x, tw))
        x += tw + 24
    if row:
        rows.append(row)
    img = np.zeros((max(STRIP_H, 26 * len(lines) + 12 + 26 * len(rows)), width, 3), np.uint8)
    for k, (text, color, scale) in enumerate(lines):
        label(img, text, 10, 28 + 26 * k, scale, color)
    y0 = 26 * len(lines) + 16
    for r, row in enumerate(rows):
        y = y0 + 26 * r + 14
        for k, name, x, tw in row:
            if k == current:
                cv2.rectangle(img, (x - 5, y - 16), (x + tw + 5, y + 6), YELLOW, -1)
                cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            else:
                cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN if results and results[k] == "passed" else GREY, 1)
    return img


def typing_lines(typed: str, words: list[str]) -> list[tuple]:
    return [
        ("Say a statement for the robot to teach, then press Enter (Esc cancels)", YELLOW, 0.7),
        (f"> {typed}_", WHITE, 0.8),
        ("Words with a sign: " + ", ".join(words), GREY, 0.5),
        ("Other words are skipped and listed, never guessed. Signs follow your English word order.", GREY, 0.5),
        ("Enter on an empty line goes back to the sign set.", GREY, 0.5),
        (CAPTION, GREY, 0.5),
    ]


def debug_line(sign_id, world, vec, result) -> str:
    """Raw angles (degrees) behind the HandVec, for tuning the default calibration and tolerances."""
    curl = np.rad2deg(_raw_curl_all(world))
    spread = np.rad2deg(_raw_spread_all(world))
    fix = [f for f, fr in zip(FINGERS, result.per_finger) if not fr.ok] if result is not None else []
    return (f"[{sign_id}] curl deg " + " ".join(f"{f[0].upper()}{c:4.0f}" for f, c in zip(FINGERS, curl))
            + " | spread deg " + " ".join(f"{f[0].upper()}{s:4.0f}" for f, s in zip(FINGERS, spread))
            + " | vec " + " ".join(f"{x:+.2f}" for x in vec)
            + (" | fix " + ",".join(fix) if fix else " | ok"))


def scorecard_image(session, width) -> np.ndarray:
    img = np.zeros((H + STRIP_H, width, 3), np.uint8)
    for k, line in enumerate(session.scorecard()):
        cv2.putText(img, line, (30, 50 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(img, "r: go again    b: back    s: say a statement    q: quit", (30, H + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    cv2.putText(img, CAPTION, (30, H + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)
    return img


def main() -> int:
    argv = sys.argv[1:]
    debug = "--debug" in argv
    debug_next = 0.0
    say = argv[argv.index("--say") + 1] if "--say" in argv[:-1] else None
    statement = statement_from_text(say) if say else None
    if say and statement is None:
        print(f"No word in {say!r} has a sign yet; starting on the default statement.", file=sys.stderr)
    if statement is None and ("--statement" in argv or say):
        statement = load_statement()
    words = phrasebook_words()
    typing = None  # the statement being typed, or None
    signs = load_signs()
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    if not ok:
        print("No camera frames. On macOS grant the terminal Camera access "
              "(System Settings -> Privacy & Security -> Camera) and restart.", file=sys.stderr)
        return 1
    tracker = HandTracker()
    teacher = PoseView()
    session = StatementLesson(statement) if statement else Session(signs, browse=True)
    replay_t0 = None  # start time of a slow teacher replay (key r)
    mirror = None
    calib = None
    cal_start = None  # time when calibration began
    cal_frames = {"open": [], "fist": []}
    smooth_vec = None
    trail = []  # recent wrist points (x, y, t) drawn for moving signs
    banner, banner_until, banner_color = "", 0.0, GREEN
    fps, t_prev, t0 = 0.0, time.time(), time.time()
    fps_samples = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # selfie view; MediaPipe handedness assumes this
        now = time.time()
        lm, hand = tracker.process(frame, int((now - t0) * 1000))
        vec = landmarks_to_vec(tracker.world, hand, calib) if lm is not None else None
        # An unreliable view (cut off, edge-on, pointing at the camera) pauses marking: no
        # red/green, no hold progress, nothing counted against the learner.
        usable, reason = hand_quality(lm) if lm is not None else (False, "")

        step_text = None
        if cal_start is not None:
            el = now - cal_start
            for name, dur, bucket in CAL_STEPS:
                if el < dur:
                    step_text = f"{name}  ({dur - el:.0f}s)"
                    if bucket and tracker.world is not None and usable:
                        cal_frames[bucket].append(tracker.world.copy())
                    break
                el -= dur
            if step_text is None:
                calib = fit_calibration(cal_frames["open"], cal_frames["fist"])
                cal_start = None
                session.repeat()
                print(f"calibrated from {len(cal_frames['open'])} open and {len(cal_frames['fist'])} fist frames")

        if session.finished:
            card = scorecard_image(session, int(frame.shape[1] * H / frame.shape[0]) + H)
            if typing is not None:
                box = strip(card.shape[1], typing_lines(typing, words))
                card = np.vstack([card[:card.shape[0] - box.shape[0]], box])
            cv2.imshow("Robot Teaches Human", card)
            progress, result = 0.0, None
        else:
            sign = session.sign
            result = compare(vec, sign.target) if vec is not None and usable else None
            if step_text is None and typing is None:
                was = sign.gloss
                wrist, size = None, None
                if lm is not None and usable:
                    aspect = frame.shape[1] / frame.shape[0]  # image space with square units
                    wrist = (lm[0][0] * aspect, lm[0][1])
                    size = float(np.hypot((lm[9][0] - lm[0][0]) * aspect, lm[9][1] - lm[0][1]))
                    trail.append((int(lm[0][0] * frame.shape[1]), int(lm[0][1] * frame.shape[0]), now))
                progress, done = session.update(result, now, wrist, size)
                if done:
                    banner, banner_until, banner_color = f"Passed: {was}", now + PASSED_BANNER_S, GREEN
                    progress = 0.0
            else:
                progress = 0.0
            if debug and vec is not None and now >= debug_next:
                print(debug_line(sign.id, tracker.world, vec, result), flush=True)
                debug_next = now + 0.5

        demo = session.demo_sign(now) if not session.finished else None

        if not session.finished:
            sign = session.sign
            left = cv2.resize(frame, (int(frame.shape[1] * H / frame.shape[0]), H))
            if lm is not None:
                draw_landmarks(left, lm)
                if result is not None and step_text is None and demo is None:
                    draw_marking(left, lm, result)
            if session.move is not None:
                trail[:] = [p for p in trail if now - p[2] < 2.0]
                for a, b in zip(trail, trail[1:]):
                    cv2.line(left, a[:2], b[:2], ORANGE, 3)
            else:
                trail.clear()
            if step_text:
                label(left, step_text, 10, 28, 0.8, YELLOW)
            elif vec is None:
                label(left, "No hand detected", 10, 28, 0.8, RED)
            elif not usable:
                big_text(left, reason)
            else:
                label(left, f"{hand} hand   {'calibrated' if calib else 'default calibration'}", 10, 28, 0.55)
            label(left, f"{fps:.0f} fps", 10, H - 12, 0.6)
            draw_hold_ring(left, progress)

            pose = demo.target if demo is not None else sign.target
            if replay_t0 is not None and demo is None:
                p = (now - replay_t0) / REPLAY_S
                if p >= 1.0:
                    replay_t0 = None
                else:
                    pose = sign.target * (p * p * (3 - 2 * p))  # smoothstep from the open hand
            right = teacher.show(pose).copy()
            label(right, f"Teacher: {demo.gloss}" if demo is not None else "Teacher")
            panels = [left, right]
            if mirror is not None:
                if vec is not None:
                    smooth_vec = vec if smooth_vec is None else SMOOTH * vec + (1 - SMOOTH) * smooth_vec
                m_img = mirror.show(smooth_vec if smooth_vec is not None else np.zeros(8)).copy()
                label(m_img, "Mirror (you)")
                panels.append(m_img)
            row = np.hstack(panels)

            if step_text:
                msg, msg_color = "Calibrating: keep your hand in view", YELLOW
            elif demo is not None:
                msg, msg_color = f"Watch the robot sign the whole statement: {demo.gloss}", YELLOW
            elif now < banner_until:
                msg, msg_color = banner, banner_color
            elif lm is not None and not usable:
                msg, msg_color = f"Marking paused: {reason}", YELLOW
            elif result is None:
                msg, msg_color = "Show your hand to the camera", RED
            elif result.all_ok and session.move is not None and not session.move.done:
                msg, msg_color = f"Handshape good. Now draw circles with your hand: {session.move.progress * 100:.0f}%", YELLOW
            elif result.all_ok:
                msg, msg_color = f"Hold it... {progress * session.hold.seconds:.1f} / {session.hold.seconds:.1f} s", GREEN
            else:
                msg, msg_color = result.message, YELLOW
            heading = session.heading if isinstance(session, StatementLesson) else (
                f"{session.heading}   [{'BROWSE' if session.browse else 'PRACTICE'}]")
            info = strip(row.shape[1], typing_lines(typing, words) if typing is not None else [
                (heading, WHITE, 0.8),
                (sign.instruction, WHITE, 0.6),
                (f"Status: {sign.status_label}   (handshape source: {sign.source_host})"
                 + (f"     {session.note}" if session.note else ""),
                 GREEN if sign.status == "exact" else ORANGE, 0.55),
                (msg, msg_color, 0.7),
                (CAPTION, GREY, 0.5),
                (HINT_NAV, YELLOW, 0.5),
                (HINT_OTHER, YELLOW, 0.5),
            ], [g.id.upper() for g in session.signs], session.i, [session.result[g.id] for g in session.signs])
            cv2.imshow("Robot Teaches Human", np.vstack([row, info]))

        dt = time.time() - t_prev
        t_prev = time.time()
        fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
        fps_samples.append(1 / dt)

        k = cv2.waitKeyEx(1)
        ch = chr(k) if 0 <= k < 128 else ""
        if typing is not None:
            if k in (13, 10):
                if not typing.strip():
                    session = Session(signs, browse=True)
                    banner, banner_until = "", 0.0
                elif (typed := statement_from_text(typing)) is None:
                    banner, banner_until, banner_color = ("None of those words has a sign yet.",
                                                          now + TYPED_BANNER_S, RED)
                else:
                    session = StatementLesson(typed)
                    banner, banner_until, banner_color = (
                        (f"No sign yet for: {', '.join(typed.unknown)} (skipped)", now + TYPED_BANNER_S, ORANGE)
                        if typed.unknown else ("", 0.0, GREEN))
                typing = None
            elif k == 27:
                typing = None
            elif k in (8, 127):
                typing = typing[:-1]
            elif 32 <= k <= 126 and len(typing) < MAX_TYPED_CHARS:
                typing += ch
        elif ch == "q":
            break
        elif ch == "s":
            typing = ""
        elif ch == "c":
            cal_start = time.time()
            cal_frames = {"open": [], "fist": []}
        elif ch == "m":
            mirror = PoseView() if mirror is None else None
        elif ch == "p":
            teacher.drop()
            if mirror is not None:
                mirror.drop()
        elif ch == "t":
            session.toggle_mode()
        elif session.finished:  # scorecard (practice mode)
            if ch == "r":
                session.restart()
            elif ch == "b" or k in KEY_LEFT:
                session.back()
        else:
            if ch == "f":
                session.force_pass()
            elif ch == "n" or k in KEY_RIGHT:
                session.next() if session.browse or k in KEY_RIGHT else session.skip()
            elif ch == "b" or k in KEY_LEFT:
                session.prev() if session.browse or k in KEY_LEFT else session.back()
            elif ch.isdigit() and ch != "0":
                session.jump(int(ch))
            elif ch == "r":
                replay_t0 = time.time()
                session.repeat()

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    if fps_samples:
        s = np.array(fps_samples[10:] or fps_samples)
        print(f"fps: mean {s.mean():.1f}, median {np.median(s):.1f}, p5 {np.percentile(s, 5):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
