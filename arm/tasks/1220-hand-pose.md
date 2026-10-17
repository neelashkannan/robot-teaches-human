# 1220 hand pose — nb-1

**Goal:** a pure function turning MediaPipe hand landmarks into the 8-number HandVec, plus per-person
calibration. No camera, no MediaPipe import in this file (it takes plain landmark arrays).

**Read first:** `BRIEF.md` (Interfaces), `hand/vec.py`.

**Files allowed:** `hand/hand_pose.py`, `tests/test_hand_pose.py`. Nothing else.
Use `.venv/bin/python -m pytest tests/test_hand_pose.py`.

**Do**
1. `landmarks_to_vec(lm, handedness, calib=None) -> np.ndarray(8)`. `lm` is 21 rows of (x, y, z) in
   MediaPipe order (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 little). Ignore
   the little finger.
2. Curl per finger from joint angles along the finger (angle between consecutive bone vectors, summed),
   mapped to 0..1 using calibration. Spread per finger from the angle between the finger's base bone
   and the palm's middle-finger axis, mapped to -1..1, positive toward the thumb side. Everything
   must be scale-invariant (hand size, distance) and use 3D coordinates.
3. `handedness` "Left"/"Right" and mirrored webcam: the result for a mirrored left hand must equal
   the right hand's, so spread signs flip when needed. Document the convention in the header.
4. `Calibration` dataclass and `fit_calibration(open_frames, fist_frames)` (lists of landmark arrays
   from a 10 s warm-up): per-finger raw angle for fully open and fully curled; robust to noise
   (use medians). Without calibration, use sensible fixed defaults.
5. Output always shape (8,), in range, finite. Bad input (NaN, wrong shape, all zeros) returns the
   neutral pose or raises a clear `ValueError`; say which in the header.
6. Tests with synthetic landmarks you construct (a straight finger, a curled finger, a spread
   finger, a scaled copy, a mirrored copy). No webcam needed.

**Done when:** tests pass and cover scale, mirror and bad input. Reply with 4 lines max. Commit only
your two files. **Time box:** until 13:10.
