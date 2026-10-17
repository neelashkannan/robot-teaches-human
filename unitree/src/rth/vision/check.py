"""`rth vision-check`: live webcam test of body tracking and limb directions.

With ``--image PATH`` it runs once on a still photo and writes an annotated PNG instead.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from rth.config import load_settings, project_path
from rth.contracts import Limb, PersonPose
from rth.limbs import FORWARD, LEFT, UP, limb_directions

_SHORT = {
    Limb.LEFT_UPPER_ARM: "L upper arm",
    Limb.LEFT_FOREARM: "L forearm",
    Limb.RIGHT_UPPER_ARM: "R upper arm",
    Limb.RIGHT_FOREARM: "R forearm",
    Limb.LEFT_THIGH: "L thigh",
    Limb.LEFT_SHIN: "L shin",
    Limb.RIGHT_THIGH: "R thigh",
    Limb.RIGHT_SHIN: "R shin",
}


def direction_words(limb: Limb, d: np.ndarray | None, threshold: float = 0.45) -> str:
    """Short words for a torso-frame direction, e.g. 'up/out' or 'down'."""
    if d is None:
        return "not seen"
    # "Out" is towards the limb's own side, so it means the same thing for both arms.
    side = 1.0 if limb.side == "left" else -1.0
    parts = [
        (abs(d[UP]), "up" if d[UP] > 0 else "down"),
        (abs(d[FORWARD]), "fwd" if d[FORWARD] > 0 else "back"),
        (abs(d[LEFT]), "out" if d[LEFT] * side > 0 else "in"),
    ]
    parts.sort(key=lambda p: -p[0])
    words = [w for mag, w in parts[:2] if mag >= threshold] or [parts[0][1]]
    return "/".join(words)


def describe(dirs: Mapping[Limb, np.ndarray | None]) -> list[str]:
    return [f"{_SHORT[limb]}: {direction_words(limb, dirs.get(limb))}" for limb in Limb]


def draw_panel(img: np.ndarray, lines: list[str]) -> None:
    """White text on a dark translucent box in the top-left corner."""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, img.shape[0] / 1100)
    thick = max(1, round(scale * 1.6))
    step = int(32 * scale) + 6
    width = max(cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines)
    x1, y1 = min(img.shape[1], width + 24), min(img.shape[0], step * len(lines) + 14)
    roi = img[:y1, :x1]
    roi[:] = (roi * 0.35).astype(img.dtype)
    for i, line in enumerate(lines):
        cv2.putText(
            img, line, (12, step * (i + 1)), font, scale, (255, 255, 255), thick, cv2.LINE_AA
        )


def annotate(img: np.ndarray, pose: PersonPose | None, lines: list[str], *, mirrored: bool) -> None:
    from rth.vision.overlay import draw_skeleton

    draw_skeleton(img, pose, mirrored=mirrored)
    draw_panel(img, lines)


def _resolve_model(arg: str | None, default: str) -> str:
    if arg in ("lite", "full", "heavy"):
        return f"assets/models/pose_landmarker_{arg}.task"
    return arg or default


def run_image(path: Path, out: Path, settings, *, mirrored: bool) -> int:
    from rth.vision.pose import MediaPipePoseTracker

    img = cv2.imread(str(path))
    if img is None:
        print(f"could not read image {path}")
        return 2
    frame = img[:, ::-1].copy() if mirrored else img
    with MediaPipePoseTracker(settings.vision) as tracker:
        t0 = time.perf_counter()
        pose = tracker.process(frame, 0)
        ms = (time.perf_counter() - t0) * 1000
    if pose is None:
        lines = ["nobody found"]
    else:
        dirs = limb_directions(pose.keypoints, pose.visibility, settings.scoring.min_visibility)
        lines = describe(dirs)
    for line in lines:
        print(line)
    print(f"{ms:.0f} ms (first frame, includes warm-up)")
    shown = frame.copy()
    annotate(shown, pose, lines, mirrored=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), shown)
    print(f"wrote {out}")
    return 0 if pose is not None else 1


def run_live(settings) -> int:
    from rth.vision.camera import Camera
    from rth.vision.pose import MediaPipePoseTracker
    from rth.vision.smoothing import DirectionSmoother

    mirrored = settings.camera.mirror_display
    smoother = DirectionSmoother(settings.scoring.smoothing_alpha)
    t_start = time.monotonic()
    fps = 0.0
    with Camera(settings.camera) as cam, MediaPipePoseTracker(settings.vision) as tracker:
        print(f"camera {settings.camera.index} at {cam.size[0]}x{cam.size[1]}; q or Esc quits")
        last = time.perf_counter()
        while True:
            frame = cam.read()
            if frame is None:
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
                continue
            t0 = time.perf_counter()
            pose = tracker.process(frame, int((time.monotonic() - t_start) * 1000))
            track_ms = (time.perf_counter() - t0) * 1000
            if pose is None:
                smoother.reset()
                dirs: Mapping[Limb, np.ndarray | None] = {limb: None for limb in Limb}
            else:
                raw = limb_directions(
                    pose.keypoints, pose.visibility, settings.scoring.min_visibility
                )
                dirs = smoother.update(raw)
            shown = cv2.flip(frame, 1) if mirrored else frame.copy()
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
            last = now
            lines = [f"{fps:4.1f} fps  track {track_ms:.0f} ms"]
            lines += describe(dirs) if pose is not None else ["nobody found - step back"]
            annotate(shown, pose, lines, mirrored=mirrored)
            cv2.imshow("rth vision-check", shown)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rth vision-check", description=__doc__)
    parser.add_argument("--image", type=Path, help="run on a still photo instead of the webcam")
    parser.add_argument("--out", type=Path, help="annotated PNG to write (with --image)")
    parser.add_argument("--model", help="lite, full, heavy or a .task path (default: settings)")
    parser.add_argument(
        "--flip",
        action="store_true",
        help="flip the photo first (shows why tracking needs raw frames)",
    )
    parser.add_argument("--settings", default="config/settings.yaml")
    args = parser.parse_args(argv or [])

    settings = load_settings(args.settings)
    settings.vision.model_path = _resolve_model(args.model, settings.vision.model_path)
    if not project_path(settings.vision.model_path).exists():
        print(f"pose model missing: {settings.vision.model_path}; run `uv run rth fetch-assets`")
        return 2
    if args.image is not None:
        out = args.out or Path(f"{args.image.stem}_vision.png")
        return run_image(args.image, out, settings, mirrored=args.flip)
    return run_live(settings)
