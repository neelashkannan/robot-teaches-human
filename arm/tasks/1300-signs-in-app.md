# 1300 signs into the app — hx-builder (after tasks/1245-mediapipe.md)

**Goal:** the BSL signs drive everything: the teacher hand shows the sign, the webcam is marked
against that sign, and the learner moves through the set.

**Read first:** `BRIEF.md`, `data/signs.json` (hx-data), `data/sources.md`, `hand/compare.py`.

**Files allowed:** `hand/app.py`, `hand/signs.py` (new), `tests/test_signs.py`.

**Do**
1. `hand/signs.py`: `load_signs(path="data/signs.json")` returns only signs whose `status` is
   `exact` or `approximated` (never `unverified` or `not_representable`); validates the 8-number
   target with `hand.vec.clamp`.
2. App flow per sign: show the gloss, the one-line `instruction`, and the status label on screen
   ("exact" / "approximated: closest this hand can do"). `Teacher.set_pose(target)`. Compare the
   user's HandVec to the target with `compare()`. Per-finger right/wrong on the webcam panel, the
   `message` line underneath, a 3 s `Hold` ring. On `done`, mark passed and move to the next sign
   (key `n` skips, `b` goes back, `r` repeats).
3. Always-visible caption: "Checks handshape only, not location, movement or facial expression."
   This sentence is required by the brief's honest scope.
4. Scorecard at the end: per sign, passed or skipped, and the finger that was off most often.
   Plain text, no emoji.
5. If a sign has a `movement` template, ignore it for now (stretch). Do not fake a movement check.

**Done when:** every verified sign in `data/signs.json` shows on the teacher hand, and a full pass
through the set works on Alex's own hand. Reply 5 lines max. Commit.
**Time box:** until 14:00.
