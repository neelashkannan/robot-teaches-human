"""`rth record`: record labelled stance examples for the recogniser.

Webcam:  rth record --person NAME --all            (every stance in config/stances.yaml)
         rth record --person NAME --stance horse --seconds 20
Video:   rth record --person NAME --stance horse --from-video clip.mp4 [--start 2 --end 18]

Only frames where every body joint (shoulders to ankles) is confidently seen are
kept. Recordings go to data/recordings/<person>/<stance>/<UTC time>.npz.

The 'none' class: ``--stance none`` records "not a stance" (walking about,
shaking out, moving between stances). It is optional. The recogniser already
rejects unfamiliar poses with a learned distance threshold, which needs no extra
data; a 'none' recording adds explicit negatives for the in-between poses that
sit near a stance, and the report then measures the false-accept rate. Don't
record standing still with arms down as 'none': that is the 'ready' stance.
"""

from __future__ import annotations

import argparse
import textwrap
import time
from pathlib import Path

import numpy as np

from rth.config import Settings, load_settings, project_path
from rth.contracts import PersonPose
from rth.ml.dataset import NONE_LABEL, safe_name, save_recording

# MediaPipe indices of shoulders, elbows, wrists, hips, knees, ankles.
BODY_LANDMARKS = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
MIN_FRAMES = 10
WINDOW = "rth record"
_NONE_DESCRIPTION = "Not a stance: walk about, shake out, wave, turn. Keep your whole body in view."


def full_body_ok(pose: PersonPose | None, min_visibility: float) -> bool:
    """True when every body joint is seen confidently and the landmarks are finite."""
    if pose is None:
        return False
    vis = np.asarray(pose.landmarks_visibility)
    lm = np.asarray(pose.landmarks_world)
    return lm.shape == (33, 3) and bool(np.isfinite(lm).all()) and bool(
        (vis[BODY_LANDMARKS] >= min_visibility).all()
    )


class Take:
    """Frames kept for one recording."""

    def __init__(self) -> None:
        self.lm: list[np.ndarray] = []
        self.vis: list[np.ndarray] = []
        self.ts: list[int] = []

    def __len__(self) -> int:
        return len(self.lm)

    def add(self, pose: PersonPose) -> None:
        self.lm.append(np.asarray(pose.landmarks_world, np.float32))
        self.vis.append(np.asarray(pose.landmarks_visibility, np.float32))
        self.ts.append(int(pose.timestamp_ms))

    def save(self, root: Path, person: str, stance_id: str, source: str) -> Path | None:
        if len(self) < MIN_FRAMES:
            print(f"  only {len(self)} usable frames for {stance_id}; not saved")
            return None
        path = save_recording(root, person, stance_id, np.stack(self.lm), np.stack(self.vis),
                              np.array(self.ts), source=source)
        print(f"  saved {len(self)} frames -> {path}")
        return path


# ----------------------------------------------------------------------------- drawing


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color=(255, 255, 255),
          thick: int = 2) -> None:
    import cv2

    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _draw_pose(img: np.ndarray, pose: PersonPose | None, mirrored: bool) -> None:
    try:
        from rth.vision.overlay import draw_skeleton
    except ImportError:
        draw_skeleton = None
    if draw_skeleton is not None:
        draw_skeleton(img, pose, mirrored=mirrored)
        return
    if pose is None:  # fallback dots if the overlay module is unavailable
        return
    import cv2

    h, w = img.shape[:2]
    for x, y in pose.image_points.values():
        x = 1 - x if mirrored else x
        cv2.circle(img, (int(x * w), int(y * h)), 6, (0, 255, 255), -1)


def draw_screen(
    frame_bgr: np.ndarray,
    pose: PersonPose | None,
    *,
    title: str,
    description: str,
    status: str,
    status_color: tuple[int, int, int],
    big: str | None,
    reference: np.ndarray | None,
    mirrored: bool,
) -> np.ndarray:
    """The mirrored webcam view with skeleton, stance text, status line and robot inset."""
    import cv2

    img = cv2.flip(frame_bgr, 1) if mirrored else frame_bgr.copy()
    _draw_pose(img, pose, mirrored)
    h, w = img.shape[:2]
    s = h / 720
    cv2.rectangle(img, (0, 0), (w, int(150 * s)), (0, 0, 0), -1)
    _text(img, title, (int(20 * s), int(45 * s)), 1.2 * s)
    for i, line in enumerate(textwrap.wrap(description, 90)[:2]):
        _text(img, line, (int(20 * s), int(85 * s + i * 30 * s)), 0.7 * s, thick=1)
    _text(img, status, (int(20 * s), h - int(25 * s)), 0.9 * s, status_color)
    if big:
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 5 * s, 8)[0]
        _text(img, big, ((w - size[0]) // 2, (h + size[1]) // 2), 5 * s, (0, 220, 255), 8)
    if reference is not None:
        rh = int(h * 0.35)
        rw = int(reference.shape[1] * rh / reference.shape[0])
        inset = cv2.resize(reference, (rw, rh))
        y0, x0 = int(160 * s), w - rw - int(10 * s)
        if x0 > 0 and y0 + rh < h:
            img[y0 : y0 + rh, x0 : x0 + rw] = inset
            cv2.rectangle(img, (x0, y0), (x0 + rw, y0 + rh), (255, 255, 255), 2)
    return img


# ------------------------------------------------------------------------------ robot


class RobotReference:
    """Optional still image of the G1 in each stance; None when the robot module can't load."""

    def __init__(self, settings: Settings, stances) -> None:
        self._robot = None
        try:
            from rth.robot.g1 import G1Teacher

            self._robot = G1Teacher(settings, stances)
        except Exception as exc:  # noqa: BLE001 - optional extra; record without it
            print(f"(no robot reference image: {exc})")

    def image(self, stance_id: str) -> np.ndarray | None:
        if self._robot is None or stance_id == NONE_LABEL:
            return None
        try:
            self._robot.set_target(stance_id, duration_s=0.05)
            for _ in range(5):
                self._robot.update(0.05)
            return self._robot.render(320, 400)
        except Exception as exc:  # noqa: BLE001 - e.g. a stance without a pose yet
            print(f"(no robot image for {stance_id}: {exc})")
            return None

    def close(self) -> None:
        if self._robot is not None:
            self._robot.close()


# ----------------------------------------------------------------------------- webcam


def record_stance(cam, tracker, settings: Settings, *, stance_id: str, title: str,
                  description: str, seconds: float, countdown: float,
                  reference: np.ndarray | None, t0: float) -> Take | None:
    """One stance: wait for SPACE, count down, record. None = quit; empty Take = skipped."""
    import cv2

    mirrored = settings.camera.mirror_display
    min_vis = settings.recognizer.record_min_visibility
    take = Take()
    phase, phase_start = "wait", 0.0
    while True:
        frame = cam.read()
        if frame is None:
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                return None
            continue
        now = time.monotonic()
        pose = tracker.process(frame, int((now - t0) * 1000))
        ok = full_body_ok(pose, min_vis)
        big = None
        if phase == "wait":
            status = "SPACE: start   s: skip   q: quit"
            status += "" if ok else "   (step back: whole body must be visible)"
            color = (255, 255, 255) if ok else (0, 165, 255)
        elif phase == "countdown":
            left = countdown - (now - phase_start)
            if left <= 0:
                phase, phase_start = "record", now
                continue
            big, status, color = str(int(np.ceil(left))), "Get into the stance...", (0, 220, 255)
        else:
            elapsed = now - phase_start
            if elapsed >= seconds:
                return take
            if ok:
                take.add(pose)
            status = f"REC {elapsed:4.1f}/{seconds:.0f} s   frames kept {len(take)}"
            status += "   hold it, move a little" if ok else "   whole body not visible!"
            color = (0, 0, 255) if ok else (0, 165, 255)
        img = draw_screen(frame, pose, title=title, description=description, status=status,
                          status_color=color, big=big, reference=reference, mirrored=mirrored)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return None
        if key == ord("s"):
            return Take()
        if key == ord(" ") and phase == "wait":
            phase, phase_start = "countdown", now


def run_webcam(settings: Settings, args, stance_ids: list[str], library) -> int:
    import cv2

    from rth.vision.camera import Camera
    from rth.vision.pose import MediaPipePoseTracker

    root = project_path(settings.recognizer.recordings_dir)
    robot = RobotReference(settings, library) if library is not None else None
    saved = 0
    t0 = time.monotonic()
    try:
        with Camera(settings.camera) as cam, MediaPipePoseTracker(settings.vision) as tracker:
            for i, sid in enumerate(stance_ids):
                if sid == NONE_LABEL:
                    title, desc = "No stance (random movement)", _NONE_DESCRIPTION
                else:
                    st = library.get(sid)
                    title, desc = st.name, st.description
                title = f"[{i + 1}/{len(stance_ids)}] {title}"
                ref = robot.image(sid) if robot is not None else None
                print(f"{sid}: {title}")
                take = record_stance(cam, tracker, settings, stance_id=sid, title=title,
                                     description=desc, seconds=args.seconds,
                                     countdown=args.countdown, reference=ref, t0=t0)
                if take is None:
                    print("quit")
                    break
                saved += take.save(root, args.person, sid, "webcam") is not None
    finally:
        cv2.destroyAllWindows()
        if robot is not None:
            robot.close()
    print(f"{saved} recording(s) saved under {root}")
    return 0


# ------------------------------------------------------------------------------ video


def run_video(settings: Settings, args, stance_id: str) -> int:
    import cv2

    from rth.vision.pose import MediaPipePoseTracker

    path = Path(args.from_video).expanduser()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"could not open video {path}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps if 1 <= fps <= 240 else 30.0
    min_vis = settings.recognizer.record_min_visibility
    take, n = Take(), 0
    try:
        with MediaPipePoseTracker(settings.vision) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t_s = n / fps
                n += 1
                if t_s < args.start or (args.end is not None and t_s > args.end):
                    continue
                pose = tracker.process(frame, int(t_s * 1000))
                if full_body_ok(pose, min_vis):
                    take.add(pose)
                if args.show:
                    img = draw_screen(frame, pose, title=stance_id, description=str(path.name),
                                      status=f"{t_s:5.1f} s  kept {len(take)}",
                                      status_color=(255, 255, 255), big=None, reference=None,
                                      mirrored=False)
                    cv2.imshow(WINDOW, img)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"{path.name}: {n} frames read, {len(take)} with a confident full body")
    root = project_path(settings.recognizer.recordings_dir)
    return 0 if take.save(root, args.person, stance_id, f"video:{path.name}") else 1


# ------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rth record", description="Record labelled stance examples.")
    ap.add_argument("--person", required=True, help="who is recording (folder name)")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--stance", help=f"stance id from config/stances.yaml, or '{NONE_LABEL}'")
    which.add_argument("--all", action="store_true", help="walk through every stance")
    ap.add_argument("--with-none", action="store_true", help=f"with --all, also record '{NONE_LABEL}'")
    ap.add_argument("--seconds", type=float, default=20.0, help="recording length per stance")
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds after SPACE before recording")
    ap.add_argument("--camera", type=int, help="camera index (default: settings camera.index)")
    ap.add_argument("--from-video", help="process this video file instead of the webcam")
    ap.add_argument("--start", type=float, default=0.0, help="video: skip this many seconds")
    ap.add_argument("--end", type=float, help="video: stop at this time (seconds)")
    ap.add_argument("--show", action="store_true", help="video: show frames while processing")
    ap.add_argument("--recordings", help="output folder (default: settings.recognizer.recordings_dir)")
    ap.add_argument("--settings", default="config/settings.yaml")
    args = ap.parse_args(argv)

    settings = load_settings(args.settings)
    if args.recordings:
        settings.recognizer = settings.recognizer.model_copy(update={"recordings_dir": args.recordings})
    if args.camera is not None:
        settings.camera = settings.camera.model_copy(update={"index": args.camera})
    try:
        safe_name(args.person)
    except ValueError as exc:
        ap.error(str(exc))

    from rth.stances import load_stances

    library = load_stances(settings.stances_path)
    if args.all:
        stance_ids = library.ids() + ([NONE_LABEL] if args.with_none else [])
    else:
        if args.stance != NONE_LABEL and args.stance not in library:
            ap.error(f"unknown stance {args.stance!r}; known: {', '.join(library.ids())}, {NONE_LABEL}")
        stance_ids = [args.stance]

    if args.from_video:
        if args.all:
            ap.error("--from-video needs one --stance (one video per stance)")
        return run_video(settings, args, stance_ids[0])
    return run_webcam(settings, args, stance_ids, library)


if __name__ == "__main__":
    raise SystemExit(main())
