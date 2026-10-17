"""Contact sheet of every stance (normal and mirrored): ``uv run python -m rth.robot.sheet``."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from rth.config import load_settings, project_path
from rth.robot.g1 import G1Teacher
from rth.stances import load_stances


def label(img: np.ndarray, text: str, y: int = 28, scale: float = 0.7) -> np.ndarray:
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA
    )
    return img


def tile(images: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
    return out


def stance_sheet(teacher: G1Teacher, width: int = 320, height: int = 360) -> np.ndarray:
    """Top row: each stance as stored; bottom row: mirrored."""
    ims = []
    for mirrored in (False, True):
        for sid in teacher.stance_ids():
            teacher.set_target(sid, mirrored=mirrored, duration_s=0.0)
            ims.append(label(teacher.render(width, height), sid + (" (m)" if mirrored else "")))
    return tile(ims, len(teacher.stance_ids()))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0]) if argv else project_path("reports/robot_stances.png")
    settings = load_settings()
    with G1Teacher(settings, load_stances(settings.stances_path)) as t:
        sheet = stance_sheet(t)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
