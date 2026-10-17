# 1245 mediapipe integration — hx-builder

**Goal:** the webcam feeds MediaPipe, the landmarks become the user's HandVec, and the app window
shows it next to the teacher hand. First milestone: the user's live HandVec drives a second,
"mirror" render of the sim hand, which proves perception works before marking is added.

**Read first:** `BRIEF.md` (Stack, Interfaces, Honest scope), `hand/hand_pose.py` header (nb-1,
mirrored-image convention), `hand/compare.py` (nb-2), `tasks/threat-model.md` (hx-guard).

**Files allowed:** `hand/vision.py` (new), `hand/app.py`, `tests/test_vision.py`. Do not edit
`hand/hand_pose.py` or `hand/compare.py`; if their interface is wrong, tell the orchestrator.

**Do**
1. `hand/vision.py`: `HandTracker(model_path="models/hand_landmarker.task")`.
   - Before loading, compute the file's sha256 and compare with `models/hand_landmarker.sha256`;
     refuse to run on a mismatch. No network calls at runtime (the file is already downloaded;
     it is gitignored, only the hash is committed).
   - mediapipe 1.1.1 Tasks API: `mediapipe.tasks.python.vision.HandLandmarker`, `num_hands=1`,
     `RunningMode.VIDEO` with `detect_for_video(image, timestamp_ms)`. There is no
     `mp.solutions`.
   - `.process(bgr_frame, t_ms) -> (landmarks 21x3 array in normalised image coords | None,
     handedness "Left"|"Right" | None)`. Use the 3D landmarks MediaPipe provides (world landmarks
     if available) as `hand_pose.landmarks_to_vec` expects; check its header.
2. `hand/app.py`: one OpenCV window, camera index 0, mirrored (selfie view) display. Left panel:
   webcam with 21 landmark dots; right panel: the teacher hand (`Teacher.render()`). Keys: `q`
   quit, `c` start the 10 s calibration (3 s open hand, then 3 s fist, collect frames, call
   `fit_calibration`), `m` toggle the mirror hand (second `Teacher` driven by the user's HandVec).
   Show "No hand detected" when `process` returns None. Frames stay in memory: no `imwrite`,
   no `VideoWriter`, no temp files.
3. Keep the render loop fast: render the teacher only when its pose changes; target >= 15 fps.
4. `tests/test_vision.py`: the hash check refuses a modified file; `process` on a blank frame
   returns None. No webcam in tests.

**Run it** in the `checks` pane: `.venv/bin/python -m hand.app`. macOS will ask for camera access
for the terminal/herdr app; if it is denied the loop gets no frames. Tell the orchestrator so Alex
can grant it (System Settings -> Privacy & Security -> Camera). Do not work around it.

**Done when:** pytest passes, the app shows live landmarks and the mirror hand follows Alex's
fingers, with the exact fps and what you could not verify in your reply (5 lines max). Commit.
**Time box:** until 13:15.
