# Schedule — Sat 17 Oct 2026 (hacking ends 16:00). Re-planned 12:10 for Robot Teaches Human.

| Time | Block | Who | Must be true by end |
|---|---|---|---|
| 12:10–13:15 | Foundations, in parallel | `hx-builder`: `teacher.py` (8 numbers -> actuators, offscreen render in an OpenCV window). `hx-data`: `data/signs.json` (5-6 signs) + `scripts/check_signs.py`. `nb-1`: `hand/hand_pose.py` + tests. `nb-2`: `hand/compare.py` + tests. `hx-guard`: privacy rules + 1-page threat model. `hx-pitch`: `pitch/judges.md` | **12:15:** commit with venv, models, vec.py, briefs. **13:13:** sim hand shows a sign on screen; `compare()` and `landmarks_to_vec()` pass their tests on fixtures |
| 12:30–13:15 | Lunch / networking | Alex + hx-pitch questions | 2 judge insights in `pitch/judges.md` |
| 13:15–14:45 | Close the loop | `hx-builder`: `app.py` (webcam + MediaPipe + teacher + marking + 3 s hold + calibration + scorecard). `hx-redteam`: edge-case suite on pose and compare. `hx-guard`: no-network and no-frame-saved checks | **14:43:** webcam -> green/red per finger -> rep counted, end to end on Alex's own hand |
| **14:45** | **FEATURE FREEZE** | | |
| 14:45–15:20 | Harden | Fix only red-team blockers; `hx-pitch` script v1 | Zero open blockers |
| 15:20–15:45 | Ship | Record 60 s backup video; rehearse twice, time it | Pitch <= 3 min, backup video saved |
| 15:45–16:00 | Submit | Alex approves text, submits | Submitted before 16:00 |
| 16:00–17:00 | Judging | Demo to judges | |

Checkpoint reminders fire at 12:12, 13:13, 14:43, 15:18, 15:43, 15:52 (the 11:28 one has passed).

## Cut list (in order, if behind)
1. Movement check, mirror hand (already stretch) -> 2. Scorecard (keep a plain rep count)
-> 3. Per-person calibration (use fixed open/fist defaults) -> 4. Signs from 6 down to 3.
Never cut: teacher hand on screen, per-finger green/red from the webcam, backup video, pitch.
If the thumb mapping is unreliable by 14:00, drop the thumb: signs use index, middle and ring only.
