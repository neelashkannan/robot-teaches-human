"""LAFAN1 clips retargeted to the G1 (lvhaidong/LAFAN1_Retargeting_Dataset), as MuJoCo qpos.

CSV layout per the dataset card (30 fps, no header, one row per frame):
root ``x y z qx qy qz qw`` then 29 joint angles in ``CSV_JOINT_ORDER``. The
quaternion is xyzw (pinocchio) and MuJoCo wants wxyz. The clips target Unitree's
``g1_29dof_rev_1_0`` URDF, the same robot (and joint zeroes) as Menagerie's g1.
Licence: LAFAN1 motion is CC BY-NC-ND 4.0 (non-commercial).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from rth.config import project_path

FPS = 30.0

# Joint column order from the dataset card ("The Order of Configuration", G1).
CSV_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
CSV_COLUMNS = 7 + len(CSV_JOINT_ORDER)


def clip_path(name: str, clips_dir: str | Path) -> Path:
    """Resolve ``fight1_subject2`` or ``fight1_subject2.csv`` inside ``clips_dir``."""
    stem = name.removesuffix(".csv")
    return project_path(clips_dir) / f"{stem}.csv"


def list_clips(clips_dir: str | Path) -> list[str]:
    d = project_path(clips_dir)
    return sorted(p.stem for p in d.glob("*.csv")) if d.is_dir() else []


def rows_to_qpos(rows: np.ndarray, qpos_adr: dict[str, int], nq: int) -> np.ndarray:
    """(T, 36) CSV rows -> (T, nq) MuJoCo qpos, mapping joints by name."""
    rows = np.atleast_2d(np.asarray(rows, float))
    if rows.shape[1] != CSV_COLUMNS:
        raise ValueError(f"expected {CSV_COLUMNS} columns, got {rows.shape[1]}")
    q = np.zeros((rows.shape[0], nq))
    q[:, 0:3] = rows[:, 0:3]
    q[:, 3] = rows[:, 6]  # xyzw -> wxyz
    q[:, 4:7] = rows[:, 3:6]
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    for col, name in enumerate(CSV_JOINT_ORDER):
        q[:, qpos_adr[name]] = rows[:, 7 + col]
    return q


@lru_cache(maxsize=16)
def _read_rows(path: str) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    rows.setflags(write=False)
    return rows


def load_clip(name: str, clips_dir: str | Path, qpos_adr: dict[str, int], nq: int) -> np.ndarray:
    """Load a clip as (T, nq) MuJoCo qpos at 30 fps (raw: world position and heading kept)."""
    p = clip_path(name, clips_dir)
    if not p.exists():
        raise FileNotFoundError(f"clip {name!r} not found at {p} (run `uv run rth fetch-assets`)")
    return rows_to_qpos(_read_rows(str(p)), qpos_adr, nq)
