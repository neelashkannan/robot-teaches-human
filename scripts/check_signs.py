"""Validate data/signs.json. Exit non-zero on any failure.

Run: .venv/bin/python scripts/check_signs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hand.vec import HIGH, LOW, N  # noqa: E402

STATUSES = {"exact", "approximated", "not_representable", "unverified"}
USABLE = {"exact", "approximated"}
KEYS = {"id", "gloss", "source_url", "status", "instruction", "target", "hold_seconds", "movement", "notes"}
MAX_INSTRUCTION_WORDS = 12
MIN_USABLE, MAX_USABLE = 5, 6
NEAR_IDENTICAL = 0.3  # max abs difference across the 8 numbers below which two demo signs are too alike
KNOWN_SOURCES = ("https://bslsignbank.ucl.ac.uk/", "https://www.british-sign.co.uk/", "https://signature.org.uk/", "https://bda.org.uk/")

errors: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)


def main() -> int:
    path = ROOT / "data" / "signs.json"
    try:
        signs = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"FAIL: cannot read {path}: {e}")
        return 1
    if not isinstance(signs, list):
        print("FAIL: signs.json must be a list")
        return 1

    seen: set[str] = set()
    usable: list[dict] = []
    for i, s in enumerate(signs):
        sid = s.get("id", f"#{i}") if isinstance(s, dict) else f"#{i}"
        if not isinstance(s, dict):
            fail(f"{sid}: not an object")
            continue
        if set(s) != KEYS:
            fail(f"{sid}: keys differ (missing {sorted(KEYS - set(s))}, extra {sorted(set(s) - KEYS)})")
            continue
        if not isinstance(s["id"], str) or not s["id"]:
            fail(f"{sid}: id must be a non-empty string")
        elif s["id"] in seen:
            fail(f"{sid}: duplicate id")
        seen.add(str(s["id"]))
        if s["status"] not in STATUSES:
            fail(f"{sid}: status {s['status']!r} not in {sorted(STATUSES)}")
        if not isinstance(s["gloss"], str) or not s["gloss"].strip():
            fail(f"{sid}: gloss must be non-empty")
        if not isinstance(s["instruction"], str) or not s["instruction"].strip():
            fail(f"{sid}: instruction must be non-empty")
        elif len(s["instruction"].split()) > MAX_INSTRUCTION_WORDS:
            fail(f"{sid}: instruction has {len(s['instruction'].split())} words, max {MAX_INSTRUCTION_WORDS}")
        if not isinstance(s["notes"], str):
            fail(f"{sid}: notes must be a string")
        hs = s["hold_seconds"]
        if not isinstance(hs, (int, float)) or isinstance(hs, bool) or not 0.5 <= hs <= 10:
            fail(f"{sid}: hold_seconds must be a number in [0.5, 10]")
        mv = s["movement"]
        if mv is not None and not (isinstance(mv, dict) and isinstance(mv.get("type"), str)):
            fail(f"{sid}: movement must be null or an object with a 'type'")

        t = s["target"]
        ok_target = (
            isinstance(t, list)
            and len(t) == N
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) for x in t)
        )
        if not ok_target:
            fail(f"{sid}: target must be {N} finite numbers")
        else:
            a = np.array(t, dtype=float)
            if np.any(a < LOW) or np.any(a > HIGH):
                fail(f"{sid}: target out of range (curl 0..1, spread -1..1): {t}")

        url = s["source_url"]
        if s["status"] != "unverified":
            if not isinstance(url, str) or not url.startswith(KNOWN_SOURCES):
                fail(f"{sid}: non-unverified sign needs a source_url on {KNOWN_SOURCES}")
        if s["status"] in USABLE and ok_target:
            usable.append(s)

    if not MIN_USABLE <= len(usable) <= MAX_USABLE:
        fail(f"usable (exact/approximated) signs: {len(usable)}, want {MIN_USABLE} to {MAX_USABLE}")

    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            d = float(np.max(np.abs(np.array(a["target"]) - np.array(b["target"]))))
            if d < NEAR_IDENTICAL:
                fail(f"{a['id']} and {b['id']}: targets near-identical (max diff {d:.2f} < {NEAR_IDENTICAL})")

    pose_note = "pose-load check skipped (hand/teacher.py not found)"
    if (ROOT / "hand" / "teacher.py").exists():
        pose_note = "pose-load check skipped (not implemented yet, teacher.py exists)"

    counts: dict[str, int] = {}
    for s in signs:
        if isinstance(s, dict):
            counts[s.get("status")] = counts.get(s.get("status"), 0) + 1
    print(f"{len(signs)} signs, usable {len(usable)}, statuses {counts}; {pose_note}")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
