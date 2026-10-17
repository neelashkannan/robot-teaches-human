# Robot Teaches Human

Built at the Edinburgh AI Hackathon (17 October 2026), challenge "Flip the Script". In robotics the
human usually teaches the robot (imitation learning, teleoperation). Here the robot teaches you:
it shows a pose, watches you through a webcam and tells you, joint by joint, what to fix.

Two versions, one idea:

| Folder | Robot | Teaches |
| --- | --- | --- |
| [`arm/`](arm/) | Simulated Pollen AmazingHand (MuJoCo) | British Sign Language handshapes, and short statements you type |
| [`unitree/`](unitree/) | Simulated Unitree G1 humanoid (MuJoCo) | Martial arts stances, scored limb by limb |

Both run on one laptop with a webcam. Frames stay in memory; nothing is saved or uploaded.

## arm: BSL robot hand

Python 3.12.

```bash
cd arm && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
curl -fsSL -o arm/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

```bash
cd arm && .venv/bin/python -m hand.app
```

The model file is checked against `models/hand_landmarker.sha256` before it loads. Keys in the app:
`s` say a statement (type it, Enter), `t` browse / practice, `c` calibrate, `g` strict view check,
`q` quit. `--say "good morning"` starts on a statement, `--debug` prints the raw finger angles.
Signs and their sources are in `arm/data/` (`sources.md` says what was checked and what was not).
The hand checks handshape only: it is not an accredited BSL teaching tool.

## unitree: G1 stance teacher

Python 3.12 with [uv](https://docs.astral.sh/uv/).

```bash
cd unitree && uv sync && uv run rth fetch-assets
```

```bash
cd unitree && uv run rth demo
```

Optional AI sensei on Nebius Token Factory: copy `.env.example` to `.env`, set `NEBIUS_API_KEY`,
then `uv run rth check-nebius`. The demo also works with no key. `uv run rth --help` lists the
other commands (physics sim, vision check, recording and training the stance recogniser).
`unitree/docs/ARCHITECTURE.md` explains the pipeline; `unitree/presentation/` is the pitch video.

## Credits

The `arm/` history is the hackathon team repository's history, kept as it was. Robot hand model:
Pollen Robotics AmazingHand, Apache-2.0 (`arm/models/NOTICE`). Hand and pose tracking: Google
MediaPipe.
