# 1220 compare — nb-2

**Goal:** a pure function that marks a user's HandVec against a target and says in one line what to fix.

**Read first:** `BRIEF.md` (Interfaces), `hand/vec.py`.

**Files allowed:** `hand/compare.py`, `tests/test_compare.py`. Nothing else.
Use `.venv/bin/python -m pytest tests/test_compare.py`.

**Do**
1. `compare(user, target, tol=0.15) -> Result` (dataclass). `per_finger`: for each of the 4 fingers
   `ok`, `curl_err_deg`, `spread_err_deg` (signed: positive means the user needs more).
   Degrees from `CURL_FULL_DEG` and `SPREAD_FULL_DEG`. A finger is ok when both normalised errors
   are within `tol`. `all_ok`, `worst` (finger with the largest normalised error, None if all ok).
2. `message`: one short plain sentence naming the worst finger and what to do, for example
   "Ring finger: curl 20 degrees more." / "Index finger: straighten by 15 degrees." /
   "Middle finger: spread 10 degrees toward the thumb." Round to the nearest 5 degrees. When all ok:
   "Handshape matches." No emoji.
3. `Hold` helper: `Hold(seconds=3.0)` with `.update(all_ok, now) -> progress 0..1, done bool`; resets
   when `all_ok` goes false for more than 0.3 s (so a flicker does not reset it).
4. Treat NaN/wrong shape as not ok with message "No hand detected." No exceptions.
5. Tests: identical passes; one-finger error flagged on that finger only; worst-finger choice;
   sign of the error; tolerance edge; Hold flicker tolerance and completion; bad input.

**Done when:** tests pass. Reply with 4 lines max. Commit only your two files. **Time box:** until 13:10.
