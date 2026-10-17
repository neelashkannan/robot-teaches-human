"""Record a visitor holding stances and report per-limb error against the robot, by axis.

Run from an app with camera access:  uv run python scripts/pose_diagnostic.py
It shows the webcam next to the robot's stance with a countdown, records ~10 s per stance,
and writes reports/pose_diagnostic.json plus a printed table.
"""

from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np

from rth.config import load_settings, project_path
from rth.contracts import Limb
from rth.limbs import angle_deg, limb_directions, limb_error, mirror_directions
from rth.robot.g1 import G1Teacher
from rth.stances import load_stances
from rth.vision.camera import Camera
from rth.vision.overlay import draw_skeleton
from rth.vision.pose import MediaPipePoseTracker

STANCES = sys.argv[1:] or ["ready", "guard", "horse"]
COUNTDOWN_S, RECORD_S = 5.0, 10.0


def flat_angle(p: np.ndarray, r: np.ndarray, forward_weight: float) -> float:
    """Angle after shrinking the forward (depth) component, the noisiest one for a webcam."""
    w = np.array([forward_weight, 1.0, 1.0])
    a, b = p * w, r * w
    return angle_deg(a / np.linalg.norm(a), b / np.linalg.norm(b))


def main() -> int:
    s, lib = load_settings(), load_stances()
    results: dict[str, dict] = {}
    with Camera(s.camera) as cam, MediaPipePoseTracker(s.vision) as tracker, G1Teacher(s, lib) as robot:
        for sid in STANCES:
            robot.snap_to(robot.pose(sid))
            robot.update(0.0)
            target = mirror_directions(limb_directions(robot.keypoints()))
            robot_img = robot.render(480, 540)
            rows: list[dict] = []
            start = time.monotonic()
            while True:
                t = time.monotonic() - start
                if t > COUNTDOWN_S + RECORD_S:
                    break
                frame = cam.read()
                if frame is None:
                    continue
                pose = tracker.process(frame, int(time.monotonic() * 1000))
                show = cv2.flip(frame, 1)
                if pose is not None:
                    draw_skeleton(show, pose, None, mirrored=True)
                    if t >= COUNTDOWN_S:
                        dirs = limb_directions(pose.keypoints, pose.visibility, s.scoring.min_visibility)
                        rows.append({limb.value: (None if dirs[limb] is None else dirs[limb].tolist())
                                     for limb in Limb})
                label = (f"Get into: {lib.get(sid).name}  {COUNTDOWN_S - t:.0f}" if t < COUNTDOWN_S
                         else f"HOLD {lib.get(sid).name}  {COUNTDOWN_S + RECORD_S - t:.0f}")
                show = cv2.resize(show, (960, 540))
                canvas = np.hstack([show, robot_img])
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 56), (20, 20, 20), -1)
                cv2.putText(canvas, label, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1, (80, 220, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow("pose diagnostic", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    return 1
            results[sid] = summarise(rows, target, s.scoring.tolerance_deg)
            results[sid]["rows"] = rows  # raw per-frame directions, for re-tuning offline
    cv2.destroyAllWindows()
    out = project_path("reports/pose_diagnostic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")
    return 0


def summarise(rows: list[dict], target: dict, tol: float) -> dict:
    per_limb = {}
    frames_all_ok = 0
    for row in rows:
        ok = [limb_error(np.array(row[l.value]), target[l]) <= tol
              for l in Limb if row[l.value] is not None and target[l] is not None]
        frames_all_ok += bool(ok) and all(ok) and len(ok) >= 6
    print(f"\n== frames recorded: {len(rows)}, frames with every seen limb green "
          f"(booth score, {tol:g} deg): {frames_all_ok}")
    print(f"   {'limb':16s} {'seen':>5} {'3D err°':>8} {'booth err°':>10} {'err° depth/2':>12}"
          f"  person mean (f, l, u)      robot (f, l, u)")
    for limb in Limb:
        seen = [np.array(r[limb.value]) for r in rows if r[limb.value] is not None]
        tgt = target[limb]
        if not seen or tgt is None:
            per_limb[limb.value] = None
            continue
        err = np.mean([angle_deg(p, tgt) for p in seen])
        half = np.mean([flat_angle(p, tgt, 0.5) for p in seen])
        booth = np.mean([limb_error(p, tgt) for p in seen])
        mean = np.mean(seen, axis=0)
        per_limb[limb.value] = {"seen": len(seen), "err": err, "err_half_depth": half,
                                "err_booth": booth, "person_mean": mean.tolist(),
                                "robot": tgt.tolist()}
        print(f"   {limb.value:16s} {len(seen):5d} {err:8.1f} {booth:10.1f} {half:12.1f}"
              f"  ({mean[0]:+.2f} {mean[1]:+.2f} {mean[2]:+.2f})   ({tgt[0]:+.2f} {tgt[1]:+.2f} {tgt[2]:+.2f})")
    return {"frames": len(rows), "frames_all_green": frames_all_ok, "limbs": per_limb}


if __name__ == "__main__":
    raise SystemExit(main())
