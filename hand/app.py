"""One window: selfie webcam with per-finger marking | teacher hand showing the sign (| mirror hand).

Keys: q quit, n skip to next sign, b back, r repeat the sign (restart all on the scorecard),
      c calibrate (10 s: 2 s get ready, 3 s open hand, 2 s get ready, 3 s fist),
      m toggle the mirror hand (a second sim hand copying you),
      p drop the loose objects on the bench again.
Frames stay in memory: nothing is saved, encoded or sent anywhere.
Run: .venv/bin/python -m hand.app
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from hand.compare import compare
from hand.hand_pose import fit_calibration, landmarks_to_vec
from hand.signs import CAPTION, Session, load_signs
from hand.teacher import Teacher
from hand.vec import FINGERS
from hand.vision import HandTracker

H = 480
STRIP_H = 130
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


def strip(width, lines) -> np.ndarray:
    """Info bar under the panels: lines are (text, color, scale)."""
    img = np.zeros((STRIP_H, width, 3), np.uint8)
    for k, (text, color, scale) in enumerate(lines):
        label(img, text, 10, 28 + 26 * k, scale, color)
    return img


def scorecard_image(session, width) -> np.ndarray:
    img = np.zeros((H + STRIP_H, width, 3), np.uint8)
    for k, line in enumerate(session.scorecard()):
        cv2.putText(img, line, (30, 50 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(img, "r: go again    b: back    q: quit", (30, H + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    cv2.putText(img, CAPTION, (30, H + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)
    return img


def main() -> int:
    signs = load_signs()
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    if not ok:
        print("No camera frames. On macOS grant the terminal Camera access "
              "(System Settings -> Privacy & Security -> Camera) and restart.", file=sys.stderr)
        return 1
    tracker = HandTracker()
    teacher = PoseView()
    session = Session(signs)
    mirror = None
    calib = None
    cal_start = None  # time when calibration began
    cal_frames = {"open": [], "fist": []}
    smooth_vec = None
    banner, banner_until = "", 0.0
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

        step_text = None
        if cal_start is not None:
            el = now - cal_start
            for name, dur, bucket in CAL_STEPS:
                if el < dur:
                    step_text = f"{name}  ({dur - el:.0f}s)"
                    if bucket and tracker.world is not None:
                        cal_frames[bucket].append(tracker.world.copy())
                    break
                el -= dur
            if step_text is None:
                calib = fit_calibration(cal_frames["open"], cal_frames["fist"])
                cal_start = None
                session.repeat()
                print(f"calibrated from {len(cal_frames['open'])} open and {len(cal_frames['fist'])} fist frames")

        if session.finished:
            cv2.imshow("Robot Teaches Human",
                       scorecard_image(session, int(frame.shape[1] * H / frame.shape[0]) + H))
            progress, result = 0.0, None
        else:
            sign = session.sign
            result = compare(vec, sign.target) if vec is not None else None
            if step_text is None:
                was = sign.gloss
                progress, done = session.update(result, now)
                if done:
                    banner, banner_until = f"Passed: {was}", now + PASSED_BANNER_S
                    progress = 0.0
            else:
                progress = 0.0

        if not session.finished:
            sign = session.sign
            left = cv2.resize(frame, (int(frame.shape[1] * H / frame.shape[0]), H))
            if lm is not None:
                draw_landmarks(left, lm)
                if result is not None and step_text is None:
                    draw_marking(left, lm, result)
            if step_text:
                label(left, step_text, 10, 28, 0.8, YELLOW)
            elif vec is None:
                label(left, "No hand detected", 10, 28, 0.8, RED)
            else:
                label(left, f"{hand} hand   {'calibrated' if calib else 'default calibration'}", 10, 28, 0.55)
            label(left, f"{fps:.0f} fps", 10, H - 12, 0.6)
            draw_hold_ring(left, progress)

            right = teacher.show(sign.target).copy()
            label(right, "Teacher")
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
            elif now < banner_until:
                msg, msg_color = banner, GREEN
            elif result is None:
                msg, msg_color = "Show your hand to the camera", RED
            elif result.all_ok:
                msg, msg_color = f"Hold it... {progress * sign.hold_seconds:.1f} / {sign.hold_seconds:.0f} s", GREEN
            else:
                msg, msg_color = result.message, YELLOW
            info = strip(row.shape[1], [
                (f"Sign {session.i + 1} of {len(session.signs)}: {sign.gloss}", WHITE, 0.8),
                (sign.instruction, WHITE, 0.6),
                (f"Status: {sign.status_label}   (handshape source: {sign.source_host})",
                 GREEN if sign.status == "exact" else ORANGE, 0.55),
                (msg, msg_color, 0.7),
                (CAPTION + "    n skip  b back  r repeat  c calibrate  m mirror  p drop  q quit", GREY, 0.5),
            ])
            cv2.imshow("Robot Teaches Human", np.vstack([row, info]))

        dt = time.time() - t_prev
        t_prev = time.time()
        fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
        fps_samples.append(1 / dt)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            cal_start = time.time()
            cal_frames = {"open": [], "fist": []}
        elif key == ord("m"):
            mirror = PoseView() if mirror is None else None
        elif key == ord("p"):
            teacher.drop()
            if mirror is not None:
                mirror.drop()
        elif key == ord("n") and not session.finished:
            session.skip()
        elif key == ord("b"):
            session.back()
        elif key == ord("r"):
            session.restart() if session.finished else session.repeat()

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    if fps_samples:
        s = np.array(fps_samples[10:] or fps_samples)
        print(f"fps: mean {s.mean():.1f}, median {np.median(s):.1f}, p5 {np.percentile(s, 5):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
