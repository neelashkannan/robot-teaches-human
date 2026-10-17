# 1220 teacher — hx-builder

**Goal:** the sim hand shows any HandVec on screen, offscreen-rendered into an OpenCV window.

**Read first:** `BRIEF.md` (stack changed to Python; Interfaces section), `hand/vec.py`.

**Files allowed:** `hand/teacher.py`, `hand/__init__.py`, `hand/app.py` (stub only for now),
`tests/test_teacher.py`. Nothing else.

**Do**
1. `Teacher(model_path="models/AH_Right/mjcf/scene.xml")` with `.set_pose(vec)`, `.step(n=200)`,
   `.render(w=480, h=480) -> RGB uint8 array`. Use `mujoco.Renderer` offscreen; a fixed camera that
   frames the whole hand. If offscreen rendering fails on macOS, report it with the error, then try
   `mjpython`.
2. The mapping from the 8 numbers to the 8 actuators (`finger{1..4}_motor{1,2}`, ctrl -1.57..1.57).
   Each finger has a parallel mechanism: two servos move together for curl and in opposition for
   spread. Find the sign and scale by experiment (set ctrl, step, read fingertip positions) and by
   reading the upstream `AHSimulation` and `PythonExample` code (read only, never run). Document the
   rule in the file header.
3. `FINGER_TO_MJCF`: which of finger1..4 plays thumb / index / middle / ring. Check against fingertip
   geometry, not guessing. Write it in the file header.
4. Tests: all-zero pose is open; all-curl pose has every fingertip closer to the palm than open; each
   finger's curl moves only that finger's tip.
5. Show it: a throwaway loop in `hand/app.py` that cycles through open, fist, index-only curl in a
   `cv2.imshow` window (press q to quit). Run it in the `checks` pane, not as a background job.

**Done when:** pytest passes, the cycle is visible, and the header documents the mapping and
FINGER_TO_MJCF. Reply with 5 lines max. Commit.
**Time box:** until 13:10. If the thumb cannot be identified reliably by 13:00, say so and stop on it.
