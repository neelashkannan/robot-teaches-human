"""One window: selfie webcam with landmarks | teacher hand (| mirror hand copying you).

Keys: q quit, c calibrate (10 s: 2 s get ready, 3 s open hand, 2 s get ready, 3 s fist),
      m toggle the mirror hand, n next teacher pose.
Frames stay in memory: nothing is saved, encoded or sent anywhere.
Run: .venv/bin/python -m hand.app
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from hand.hand_pose import fit_calibration, landmarks_to_vec
from hand.teacher import Teacher
from hand.vision import HandTracker

H = 480
POSES = [
    ("open", np.zeros(8)),
    ("fist", np.array([1.0, 0.0] * 4)),
    ("index only", np.array([1.0, 0, 0, 0, 1, 0, 1, 0])),
]
# (label, seconds, collect into) steps of the 10 s calibration
CAL_STEPS = [
    ("Get ready: open hand", 2.0, None),
    ("Hold your hand OPEN", 3.0, "open"),
    ("Get ready: make a fist", 2.0, None),
    ("Hold a FIST", 3.0, "fist"),
]
SMOOTH = 0.5  # weight of the new HandVec in the mirror hand's low-pass


class PoseView:
    """A Teacher that re-renders only while its pose is changing."""

    def __init__(self):
        self.teacher = Teacher()
        self.vec = None
        self.dirty = 0
        self.img = None

    def show(self, vec) -> np.ndarray:
        vec = np.asarray(vec, dtype=float)
        if self.vec is None or np.abs(vec - self.vec).max() > 0.01:
            self.teacher.set_pose(vec)
            self.vec = vec.copy()
            self.dirty = 12  # frames for the sim to settle
        if self.dirty > 0 or self.img is None:
            self.teacher.step(30)
            self.img = cv2.cvtColor(self.teacher.render(H, H), cv2.COLOR_RGB2BGR)
            self.dirty -= 1
        return self.img


def label(img, text, y=28, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_landmarks(img, lm) -> None:
    h, w = img.shape[:2]
    for i, (x, y, _) in enumerate(lm):
        little = i >= 17  # the robot has no little finger
        cv2.circle(img, (int(x * w), int(y * h)), 4, (150, 150, 150) if little else (0, 255, 0), -1)


def main() -> int:
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    if not ok:
        print("No camera frames. On macOS grant the terminal Camera access "
              "(System Settings -> Privacy & Security -> Camera) and restart.", file=sys.stderr)
        return 1
    tracker = HandTracker()
    teacher = PoseView()
    mirror = None
    pose_i = 0
    calib = None
    cal_start = None  # time when calibration began
    cal_frames = {"open": [], "fist": []}
    smooth_vec = None
    fps, t_prev, t0 = 0.0, time.time(), time.time()
    fps_samples = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # selfie view; MediaPipe handedness assumes this
        now = time.time()
        lm, hand = tracker.process(frame, int((now - t0) * 1000))
        vec = None
        if lm is not None:
            vec = landmarks_to_vec(tracker.world, hand, calib)
            draw_landmarks(frame, lm)

        step_text = None
        if cal_start is not None:
            el, step_text = now - cal_start, None
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
                print(f"calibrated from {len(cal_frames['open'])} open and {len(cal_frames['fist'])} fist frames")

        left = cv2.resize(frame, (int(frame.shape[1] * H / frame.shape[0]), H))
        if step_text:
            label(left, step_text, 28, 0.8, (0, 255, 255))
        elif vec is None:
            label(left, "No hand detected", 28, 0.8, (0, 0, 255))
        else:
            label(left, f"{hand} hand   {'calibrated' if calib else 'default calibration'}", 28, 0.6)
        label(left, f"{fps:.0f} fps", H - 12, 0.6)

        panels = [left, teacher.show(POSES[pose_i][1]).copy()]
        label(panels[1], f"Teacher: {POSES[pose_i][0]}")
        if mirror is not None:
            if vec is not None:
                smooth_vec = vec if smooth_vec is None else SMOOTH * vec + (1 - SMOOTH) * smooth_vec
            m_img = mirror.show(smooth_vec if smooth_vec is not None else np.zeros(8)).copy()
            label(m_img, "Mirror (you)")
            panels.append(m_img)
        cv2.imshow("Robot Teaches Human", np.hstack(panels))

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
        elif key == ord("n"):
            pose_i = (pose_i + 1) % len(POSES)

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    if fps_samples:
        s = np.array(fps_samples[10:] or fps_samples)
        print(f"fps: mean {s.mean():.1f}, median {np.median(s):.1f}, p5 {np.percentile(s, 5):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
