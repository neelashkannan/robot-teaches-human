# 1220 guard privacy — hx-guard

**Goal:** a one-page threat model and an automated check that the app never keeps or sends camera data.

**Read first:** `BRIEF.md` (Honest scope, Stack), `.claude/agents/hx-guard.md`.

**Files allowed:** `tasks/threat-model.md`, `scripts/security_check.py`, `requirements.txt` (pins
only). Do not edit `hand/` without the orchestrator.

**Do**
1. `tasks/threat-model.md` (one page): assets (webcam frames, the visitor's hand shape), the
   downloaded MediaPipe model, pip dependencies, the upstream clone with real-servo scripts, the
   Nebius key in `.env.local`. For each: threat, mitigation, status. Plain language; it feeds the
   pitch ("the camera never leaves memory").
2. `scripts/security_check.py`: fail if anything under `hand/` uses `imwrite`, `VideoWriter`,
   `tempfile`, `open(... 'w'/'wb')` on image data, or imports `requests`, `urllib`, `socket`,
   `http`, `httpx`. Allow the single model-download helper if it is isolated in
   `hand/model_fetch.py` with the pinned sha256; the check must verify that exception is the only one.
   Also fail if a tracked file matches an API-key pattern or `git ls-files | grep -E '\.env'` is
   non-empty.
3. Work out how the HandLandmarker `.task` file will be obtained (mediapipe 1.1.1 Tasks API has no
   bundled model). Find the official URL, download it once, record its sha256 in
   `models/hand_landmarker.sha256`, and tell the orchestrator the exact path and URL for hx-builder.
   Do not commit the binary if it is large; commit the hash and the fetch step.

**Done when:** the threat model exists, `security_check.py` runs (it can pass trivially now and
must fail on a planted bad example, show that), and the model hash is recorded. Reply with 5 lines max.
**Time box:** until 13:10.
