# Threat model — Robot Teaches Human (hx-guard, 17 Oct 2026)

**One line for the pitch:** the camera never leaves memory. The app has no network code, saves no
frame, and checks the one file it downloads against a pinned hash.

Everything runs locally: webcam -> MediaPipe -> 8 numbers -> compare -> screen. No server, accounts,
cloud call or LLM at runtime. That removes most classic web threats; what is left is below.

| Asset / surface | Threat | Mitigation | Status |
|---|---|---|---|
| **Webcam frames** (a visitor's face and hand) | A frame is written to disk, encoded, or sent off the machine, by us or by a careless edit. | Frames live only in numpy arrays inside the loop. `scripts/security_check.py` fails on `imwrite`/`imencode`/`VideoWriter`/`tempfile`/write-mode `open()`/`np.save`/`pickle.dump`, and on any network or subprocess import in `hand/`. | Check written and self-tested. Re-run after every `hand/` change and before the 15:20 ship. |
| **Visitor's hand shape** (the 8 numbers, calibration) | Reused to identify someone, or logged. | Only the 8-number vector and a rolling hold timer are kept, in RAM, for the session. Scorecard shows rep counts, not images. Nothing is persisted. | By design. Builder must not add logging of vectors to files (the check blocks file writes in `hand/`). |
| **MediaPipe `.task` model** (7.8 MB, downloaded) | Swapped or tampered file is loaded (it is parsed by native code). | One download helper, `hand/model_fetch.py`, the only file allowed to use `urllib.request`. It fetches the versioned official Google URL and refuses any file whose sha256 differs from `models/hand_landmarker.sha256`. The check enforces: helper exists once, verifies sha256, uses only `storage.googleapis.com/mediapipe-models/` URLs, never imports cv2/mediapipe. | Hash recorded. **Open:** `hand/model_fetch.py` not written yet (hx-builder). Until it exists the check cannot test it. |
| **pip dependencies** | Typosquat or a malicious new release on install. | Exact `==` pins in `requirements.txt`; the check rejects unpinned lines. Installed versions match the pins today. Install only from that file. | OK now. **Open:** no per-file `--hash` pins (not worth the time). Do not `pip install` anything ad hoc at the venue. |
| **Upstream clone** `~/Desktop/mujoco-amazing-hand/AmazingHand` | Running its demo scripts drives real servos on serial ports (Rust `AHControl`, `*_real*.yml`), or dependencies get pulled in. | Reading only, never executed or edited (BRIEF rule). We copy only the Apache-2.0 sim model into `models/` with `NOTICE`. `hand/` has no serial or subprocess imports (the check blocks `subprocess`/`os.system`). | OK. No hardware is attached or in scope. |
| **Nebius / proxy key** in `.env.local` | Key committed, printed, or baked into a file we publish. | `.env*` is gitignored. Check fails if any `.env*` file is tracked or a key-shaped string (`sk-`, `eyJ...` JWT, `AKIA`, `api_key = "..."`) is in a tracked file. Git history has no JWT-shaped key. The app itself needs no key; it is only for the dev proxy (`scripts/nebius/litellm.yaml` reads it from env). | OK. **Open:** `.env.local` is mode 644 (any local user can read it); suggest `chmod 600 .env.local`. Nothing was changed. |
| **Submission / demo video** | The 60 s backup video or screenshots show a real face or a visitor who did not agree. | Record with Alex's own hand only, or crop to the hand. Say on screen that no video is stored. | Pitch/ship step, owner Alex. |
| **Claims** | Overstating what the checker or the privacy gives. | Say "handshape check, not fluent BSL" (BRIEF). Privacy claim is limited to "no frame saved, no network in the app", which the script backs. We do not claim the OS, webcam driver or MediaPipe internals are audited. | Wording agreed with BRIEF. |

## What the automated check proves, and what it does not
- **Proves:** no banned call or import in `hand/` today; only one download helper, constrained; no tracked
  env file or key-shaped string; pins are exact; the model on disk matches the pinned hash.
- **Does not prove:** that nothing is ever saved (static check on names; dynamic tricks like `getattr`
  or `exec` could slip past), or that third-party libraries are silent. Backstop for the demo: run
  the app once with the network off (Wi-Fi off) and `lsof -i -a -p <pid>` showing no sockets, then say so.

## Run it
```
.venv/bin/python scripts/security_check.py             # exit 0 = clean
.venv/bin/python scripts/security_check.py --selftest  # plants bad code, must report every one
```
