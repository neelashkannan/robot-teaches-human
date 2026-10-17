"""Live diagnostic for finger recognition. Prints numbers only: no frames are saved or sent.

Run in a terminal that has camera access:  .venv/bin/python scripts/diag_fingers.py
You will be asked for four poses, 4 s each. Per pose it reports, per finger, the median raw
bend angle (degrees) and the median normalised curl / spread that the app would use.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hand import hand_pose  # noqa: E402
from hand.vec import FINGERS  # noqa: E402
from hand.vision import HandTracker  # noqa: E402

POSES = [
    ("OPEN hand, all fingers straight", "open"),
    ("INDEX pointing up, other fingers and thumb in a fist", "index_up"),
    ("FIST, everything curled", "fist"),
    ("INDEX curled only, other fingers straight", "index_curled"),
]
GET_READY, HOLD = 2.0, 4.0


def main() -> int:
    cap = cv2.VideoCapture(0)
    ok, _ = cap.read()
    if not ok:
        print("No camera frames (is the app still running, or Camera access missing?)")
        return 1
    tracker = HandTracker()
    t0 = time.time()
    results = {}
    for text, key in POSES:
        print(f"\n>>> {text}   (a beep marks the start of the {HOLD:.0f}s hold)", flush=True)
        raw, vecs, hands, frames, seen, lms = [], [], [], 0, 0, []
        # Wait (up to 25 s) until the hand has been seen for 1 s in a row, so you can get in place.
        visible_since, wait_start, last_note = None, time.time(), 0.0
        while time.time() - wait_start < 25:
            ok, frame = cap.read()
            if not ok:
                break
            lm, _ = tracker.process(cv2.flip(frame, 1), int((time.time() - t0) * 1000))
            if lm is None:
                visible_since = None
            elif visible_since is None:
                visible_since = time.time()
            if time.time() - last_note > 2:
                print("    waiting for your hand ..." if lm is None else "    hand seen, hold the pose", flush=True)
                last_note = time.time()
            if visible_since is not None and time.time() - visible_since > 1.0 + GET_READY:
                break
        subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], check=False)
        start = time.time()
        while time.time() - start < HOLD:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            lm, hand = tracker.process(frame, int((time.time() - t0) * 1000))
            frames += 1
            if lm is None or tracker.world is None:
                continue
            seen += 1
            hands.append(hand)
            lms.append(np.asarray(tracker.world, float).copy())
            w = hand_pose._mirror_left(np.asarray(tracker.world, float), hand)
            raw.append(np.degrees(hand_pose._raw_curl_all(w)))
            vecs.append(hand_pose.landmarks_to_vec(tracker.world, hand))
        results[key] = (raw, vecs, hands, frames, seen, lms)
        print(f"    hand seen in {seen}/{frames} frames, handedness {sorted(set(hands))}", flush=True)
    cap.release()
    tracker.close()

    print("\n=== summary (median over each hold; default calibration) ===")
    print(f"{'pose':14s} " + "  ".join(f"{f:>6s}raw {f:>6s}curl {f:>6s}spr" for f in ("thumb", "index")))
    for _, key in POSES:
        raw, vecs, *_r = results[key]
        if not raw:
            print(f"{key:14s} no hand detected")
            continue
        r, v = np.median(raw, axis=0), np.median(vecs, axis=0)
        cells = "  ".join(f"{r[i]:9.0f} {v[2*i]:10.2f} {v[2*i+1]:9.2f}" for i in (0, 1))
        print(f"{key:14s} {cells}")
    print("\nfull curl/spread per finger (thumb index middle ring):")
    for _, key in POSES:
        raw, vecs, *_r = results[key]
        if vecs:
            print(f"{key:14s} curl {np.round(np.median(vecs, axis=0)[0::2], 2)}  spread {np.round(np.median(vecs, axis=0)[1::2], 2)}")

    # Candidate curl features: which one separates the poses best?
    IDX = {"thumb": (0, 1, 2, 3, 4), "index": (0, 5, 6, 7, 8), "middle": (0, 9, 10, 11, 12), "ring": (0, 13, 14, 15, 16)}

    def ang(a, b):
        return hand_pose._angle_between(a, b)

    def feats(lm, f):
        w, m, p, d, t = (lm[i] for i in IDX[f])
        A = ang(m - p, p - d) * 0  # placeholder, replaced below
        two = ang(p - m, d - p) + ang(d - p, t - d)
        three = two + ang(m - w, p - m)
        ratio = np.linalg.norm(t - w) / max(np.linalg.norm(m - w), 1e-9)
        straight = np.linalg.norm(t - m) / max(np.linalg.norm(p - m) + np.linalg.norm(d - p) + np.linalg.norm(t - d), 1e-9)
        tip_palm = np.linalg.norm(t - lm[9]) / max(np.linalg.norm(lm[9] - w), 1e-9)  # tip to middle MCP, hand-size units
        return {"2joint(old)": np.degrees(two), "3joint": np.degrees(three), "ext_ratio": ratio, "straight": straight, "tip_to_mid_mcp": tip_palm}

    print("\n=== candidate curl features: median open | fist | separation (open vs fist, higher is better) ===")
    for f in ("thumb", "index", "middle", "ring"):
        rows = {}
        for key in ("open", "fist"):
            lms = results[key][5]
            fs = [feats(hand_pose._mirror_left(l, h), f) for l, h in zip(lms, results[key][2])]
            for name in fs[0] if fs else []:
                rows.setdefault(name, {})[key] = np.array([x[name] for x in fs])
        for name, d in rows.items():
            if "open" in d and "fist" in d:
                mo, mf = np.median(d["open"]), np.median(d["fist"])
                sep = abs(mo - mf) / (np.std(d["open"]) + np.std(d["fist"]) + 1e-9)
                print(f"{f:7s} {name:15s} open {mo:7.2f}  fist {mf:7.2f}  sep {sep:5.1f}")
    # index alone: index_up (should be straight) vs index_curled
    rows = {}
    for key in ("index_up", "index_curled"):
        lms = results[key][5]
        for l, h in zip(lms, results[key][2]):
            for name, v in feats(hand_pose._mirror_left(l, h), "index").items():
                rows.setdefault(name, {}).setdefault(key, []).append(v)
    print("--- index: index_up (straight) vs index_curled")
    for name, d in rows.items():
        if "index_up" in d and "index_curled" in d:
            a, b = np.array(d["index_up"]), np.array(d["index_curled"])
            print(f"index   {name:15s} up {np.median(a):7.2f}  curled {np.median(b):7.2f}  sep {abs(np.median(a)-np.median(b))/(a.std()+b.std()+1e-9):5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
