"""Download the MediaPipe Pose Landmarker models into assets/models/ (idempotent)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from rth.config import project_path

MODELS_DIR = "assets/models"
_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
VARIANTS = ("lite", "full", "heavy")


def model_url(variant: str) -> str:
    return f"{_BASE}/pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"


def model_path(variant: str) -> Path:
    return project_path(MODELS_DIR) / f"pose_landmarker_{variant}.task"


def _download(url: str, dest: Path) -> None:
    # Write to a temp name first so an interrupted download never looks complete.
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def fetch(force: bool = False, variants: tuple[str, ...] = VARIANTS) -> None:
    """Download the pose models that are missing (all of them with ``force``)."""
    for v in variants:
        dest = model_path(v)
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"vision: {dest.name} already present")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"vision: downloading {dest.name} ...")
        _download(model_url(v), dest)
        print(f"vision: {dest.name} {dest.stat().st_size / 1e6:.1f} MB")
