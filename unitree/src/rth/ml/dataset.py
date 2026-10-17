"""Labelled stance recordings on disk.

Layout: ``<root>/<person>/<stance_id>/<UTC timestamp>.npz`` with arrays
``landmarks_world`` (N, 33, 3), ``landmarks_visibility`` (N, 33) and
``timestamps_ms`` (N,), plus small string fields (person, stance_id, source).
The folder names are the source of truth for person and label, so a recording
can be relabelled by moving the file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rth.ml.embedding import N_LANDMARKS

NONE_LABEL = "none"  # optional "not a stance" class: walking about, shaking out, random moves


def safe_name(name: str) -> str:
    """Lower-case folder-safe name: letters, digits, '-' and '_'."""
    out = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_")
    if not out:
        raise ValueError(f"unusable name {name!r}")
    return out


def recording_path(root: str | Path, person: str, stance_id: str, when: datetime | None = None) -> Path:
    """A fresh file path for a new recording (never overwrites)."""
    when = when or datetime.now(UTC)
    folder = Path(root) / safe_name(person) / stance_id
    stem = when.astimezone(UTC).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
    path, n = folder / f"{stem}.npz", 1
    while path.exists():
        path, n = folder / f"{stem}-{n}.npz", n + 1
    return path


def save_recording(
    root: str | Path,
    person: str,
    stance_id: str,
    landmarks_world: np.ndarray,
    landmarks_visibility: np.ndarray,
    timestamps_ms: np.ndarray,
    *,
    source: str = "webcam",
) -> Path:
    """Write one recording and return its path."""
    lm = np.asarray(landmarks_world, np.float32)
    vis = np.asarray(landmarks_visibility, np.float32)
    ts = np.asarray(timestamps_ms, np.int64)
    n = len(lm)
    if lm.shape != (n, N_LANDMARKS, 3) or vis.shape != (n, N_LANDMARKS) or ts.shape != (n,):
        raise ValueError(f"inconsistent shapes {lm.shape}, {vis.shape}, {ts.shape}")
    path = recording_path(root, person, stance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        landmarks_world=lm,
        landmarks_visibility=vis,
        timestamps_ms=ts,
        person=np.asarray(safe_name(person)),
        stance_id=np.asarray(stance_id),
        source=np.asarray(source),
    )
    return path


@dataclass
class Recording:
    path: Path
    person: str
    stance_id: str
    landmarks_world: np.ndarray  # (N, 33, 3)
    landmarks_visibility: np.ndarray  # (N, 33)
    timestamps_ms: np.ndarray  # (N,)
    source: str = "unknown"

    @property
    def name(self) -> str:
        return f"{self.person}/{self.stance_id}/{self.path.stem}"


def load_recording(path: str | Path) -> Recording:
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        return Recording(
            path=path,
            person=path.parent.parent.name,
            stance_id=path.parent.name,
            landmarks_world=d["landmarks_world"].astype(np.float64),
            landmarks_visibility=d["landmarks_visibility"].astype(np.float64),
            timestamps_ms=d["timestamps_ms"].astype(np.int64),
            source=str(d["source"]) if "source" in d.files else "unknown",
        )


def list_recordings(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(root.glob("*/*/*.npz")) if root.is_dir() else []


@dataclass
class Dataset:
    """All frames of all recordings, flattened, with who/what/which-recording per frame."""

    landmarks: np.ndarray  # (N, 33, 3)
    visibility: np.ndarray  # (N, 33)
    timestamps_ms: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) stance id
    persons: np.ndarray  # (N,)
    recording: np.ndarray  # (N,) index into ``recordings``
    recordings: list[str]  # "person/stance/stem"
    sources: list[str] = field(default_factory=list)  # per recording: webcam, video:<file>, ...

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def people(self) -> list[str]:
        return sorted(set(self.persons.tolist()))

    @property
    def stances(self) -> list[str]:
        return sorted(set(self.labels.tolist()))

    def subset(self, mask: np.ndarray) -> Dataset:
        return Dataset(
            self.landmarks[mask], self.visibility[mask], self.timestamps_ms[mask],
            self.labels[mask], self.persons[mask], self.recording[mask], self.recordings,
            self.sources,
        )


def load_dataset(root: str | Path, stance_ids: list[str] | None = None) -> Dataset:
    """Load every recording under ``root`` (optionally only these stance ids, plus 'none')."""
    lms, vis, ts, labels, persons, rec_idx, names, sources = [], [], [], [], [], [], [], []
    for path in list_recordings(root):
        if stance_ids is not None and path.parent.name not in {*stance_ids, NONE_LABEL}:
            continue
        r = load_recording(path)
        n = len(r.landmarks_world)
        if n == 0:
            continue
        lms.append(r.landmarks_world)
        vis.append(r.landmarks_visibility)
        ts.append(r.timestamps_ms)
        labels += [r.stance_id] * n
        persons += [r.person] * n
        rec_idx += [len(names)] * n
        names.append(r.name)
        sources.append(r.source)
    if not names:
        return Dataset(
            np.zeros((0, N_LANDMARKS, 3)), np.zeros((0, N_LANDMARKS)), np.zeros(0, np.int64),
            np.zeros(0, str), np.zeros(0, str), np.zeros(0, np.int64), [],
        )
    return Dataset(
        np.concatenate(lms), np.concatenate(vis), np.concatenate(ts),
        np.asarray(labels), np.asarray(persons), np.asarray(rec_idx), names, sources,
    )


def summary_table(ds: Dataset) -> str:
    """Markdown table: frames (and recordings) per person x stance."""
    stances, people = ds.stances, ds.people
    rec_label = {i: n.split("/")[1] for i, n in enumerate(ds.recordings)}
    rec_person = {i: n.split("/")[0] for i, n in enumerate(ds.recordings)}
    lines = [
        "| person | " + " | ".join(stances) + " | total |",
        "| --- | " + " | ".join("---:" for _ in stances) + " | ---: |",
    ]
    for p in people + ["**total**"]:
        cells = []
        for s in stances + [None]:
            pm = np.ones(len(ds), bool) if p == "**total**" else ds.persons == p
            sm = np.ones(len(ds), bool) if s is None else ds.labels == s
            n_rec = sum(
                1 for i in rec_label
                if (p == "**total**" or rec_person[i] == p) and (s is None or rec_label[i] == s)
            )
            frames = int((pm & sm).sum())
            cells.append(f"{frames} ({n_rec})" if frames else "-")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\nCells: frames (recordings)."
