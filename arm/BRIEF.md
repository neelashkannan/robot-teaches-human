# Brief

## Event
Edinburgh AI Hackathon — CompSoc x Edinburgh AI x UoE Informatics. Sponsors: Wordsmith, Skyscanner.
Partners judge during the day. 09:00 doors · 10:00 hacking · 12:30 lunch/networking · 16:00 hacking
ends · 17:00 winners. 6 hours of hacking.

## Challenge — "Flip the Script"
Take something we all assume has to work a certain way and turn it on its head. "Don't just make it
better. Flip the whole idea."

## Chosen idea — Robot Teaches Human: BSL (decided by Alex at 12:10, BSL confirmed 12:20)
(FlipTrip AI was the earlier default and is dropped. Its code never started; nothing to clean up.)

- **The assumption:** in robotics the human is the teacher (imitation learning, teleoperation, mocap).
- **The flip:** the robot hand knows British Sign Language signs, shows one, watches the person
  through a webcam (MediaPipe) and tells them right or wrong, finger by finger. The pipeline that
  turns human motion into robot commands runs backwards, as an examiner.
- **Pitch line:** "Robots spent a decade learning from us. This one teaches you."
- **Shared language:** the visitor's hand and the robot's hand both become the same 8 numbers
  (`hand/vec.py`: curl and spread for 4 fingers), so marking is a direct comparison.

### Honest scope (what we may claim)
- The Pollen AmazingHand has 4 fingers, 8 servos (curl + spread each), no little finger, no wrist,
  one hand. Real BSL also uses two hands (fingerspelling), the little finger, hand location,
  movement, palm direction and facial expression. So the robot shows **approximations of a subset**.
  Every sign in `data/signs.json` carries a `status`: `exact` (the 4-finger handshape is what BSL
  uses), `approximated` (closest the hand can do), or `not_representable` (left out of the demo).
- Every sign cites a source URL (BSL SignBank, https://bslsignbank.ucl.ac.uk/, or the BDA / Signature
  chart). `unverified` signs stay out of the demo. No sign is invented from memory.
- The checker judges **handshape** (finger curl and spread held for a moment) and, for the few signs
  that have it, a simple **movement** path from the wrist landmark. It does not judge location,
  palm direction or facial expression. "Correct" means "matches what this hand and this camera can
  check", not "fluent BSL". Say so on screen and in the pitch.
- A fluent BSL signer has not reviewed the set. Do not describe it as an accredited teaching tool.
- Teacher poses come from the sign table, not a trained policy. The RL teacher is "next", never
  claimed as built.
- Webcam frames stay in memory. Nothing is saved or sent anywhere.

## MVP (must ship, in this order)
1. **Teacher** (`hand/teacher.py`): 8 numbers -> the 8 MuJoCo actuators; renders the sim hand
   offscreen so it can sit in the app window.
2. **Signs** (`data/signs.json`): 5 to 6 verified one-handed BSL signs, each with source, status,
   target HandVec, hold time, optional movement template.
3. **Perception** (`hand/hand_pose.py`): MediaPipe 21 landmarks -> user's HandVec, calibrated from a
   10 s open-hand + fist warm-up.
4. **Marking** (`hand/compare.py`): per-finger right/wrong and one line naming the biggest fix.
5. **App** (`hand/app.py`): one OpenCV window: webcam with overlay | teacher hand. A 3 s hold counts
   as a pass; after the signs, a scorecard.

## Stretch (only if MVP is committed and runs end to end)
- Movement check for signs that move (wrist-path comparison over ~1 s).
- See-through mirror hand copying the visitor.
- Left hand (`AH_Left`, not copied yet).

## Out of scope
RL training, real hardware, two-handed fingerspelling, facial grammar, accounts, cloud, any network
call at runtime.

## Stack
Python 3.12 in `.venv` (repo root, pinned in `requirements.txt`): mujoco 3.16, mediapipe 1.1.1
(Tasks API, `HandLandmarker`, needs a downloaded `.task` model file: pin its sha256), opencv-python,
numpy, pytest. Model files: `models/AH_Right/` (Apache-2.0, Pollen Robotics, see `models/NOTICE`).
Run the app with `.venv/bin/python -m hand.app` (try `mjpython` only if offscreen rendering fails).
No Vercel, no web frontend, no Anthropic API.
Upstream clone for reading only: `~/Desktop/mujoco-amazing-hand/AmazingHand` (its
`Demo/HandTracking` and `AHSimulation/mj_mink_*.py` are useful references). Never run its scripts
(they drive real servos) and never edit it.

## Interfaces (fixed so people can work in parallel)
- `hand/vec.py` (done): `FINGERS`, `NAMES`, `clamp`, `CURL_FULL_DEG`, `SPREAD_FULL_DEG`.
- `hand/hand_pose.py`: `landmarks_to_vec(lm, handedness, calib=None) -> np.ndarray(8)`,
  `Calibration`, `fit_calibration(open_frames, fist_frames) -> Calibration`.
  `lm` is 21 (x, y, z) rows in MediaPipe order. Mirrored/left hand must be handled.
- `hand/compare.py`: `compare(user, target, tol=0.15) -> Result` with `per_finger` (ok, curl_err_deg,
  spread_err_deg), `worst`, `message` ("Ring finger: curl 20 degrees more."), `all_ok`.
- `hand/teacher.py`: `Teacher(model_path)`, `.set_pose(vec)`, `.step(n)`, `.render() -> RGB array`.
- `data/signs.json`: list of `{id, gloss, source_url, status, instruction, target[8], hold_seconds,
  movement|null, notes}`.
