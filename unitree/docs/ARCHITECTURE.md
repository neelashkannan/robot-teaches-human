# Architecture

A simulated Unitree G1 shows a martial arts stance, the visitor copies it in front of a
webcam, and every limb is scored against the robot's. An AI sensei on Nebius decides what
comes next. One Python process, one laptop.

```
webcam ─► vision (MediaPipe Pose) ─► limbs.limb_directions ─┐
                                                              ├─► limbs.compare ─► app (session, UI)
G1 in MuJoCo (robot) ───────────► limbs.limb_directions ─────┘          │
                                                                         ├─► ml recogniser (every frame, local)
                                                                         └─► coach on Nebius (once per request / rep,
                                                                             background thread, offline fallback)
```

**The shared language:** both bodies become 8 limb directions (upper arm, forearm, thigh,
shin on each side) in their own torso frame (forward, left, up). Directions ignore limb
length, so the G1's proportions don't matter. See `src/rth/limbs.py`.

**Rule:** everything per-frame runs on the laptop. Nebius is called once per request or per
rep, never per frame, and the demo must work with no key at all (offline fallbacks).

## Shared files (change only by agreement)

| File | What |
| --- | --- |
| `src/rth/contracts.py` | `Joint`, `Limb`, `PersonPose`, `RepResult`, `SenseiDecision`, `PickResult`, and the `PoseTracker`, `RobotTeacher`, `StanceRecognizer`, `MovePicker`, `Sensei`, `VisionCoach`, `Voice` protocols |
| `src/rth/limbs.py` | torso frame, limb directions, mirroring, scoring, correction hints |
| `src/rth/config.py`, `config/settings.yaml` | settings, one section per module (add fields to your own section only) |
| `src/rth/stances.py`, `config/stances.yaml` | the stance library; `pose:` fields belong to the robot module |
| `src/rth/cli.py` | `rth <command>` dispatcher |

## Modules and their public names

| Module | Owner role | Public names the others use |
| --- | --- | --- |
| `rth.robot` | Robot | `rth.robot.g1.G1Teacher(settings, stances)` implements `RobotTeacher`; `rth.robot.assets.fetch(force=False)`; `rth.robot.viewer.main(argv)` |
| `rth.robot.factory` | Robot | `make_teacher(settings, stances, *, mode=None)`: the app's only way to get a robot; `settings.robot.mode` is `physics` or `kinematic` |
| `rth.sim` | Robot | `rth.sim.teacher.PhysicsTeacher(settings, stances)` implements `RobotTeacher` with real physics (plus `.model`, `.data`, `.push()`, `.reset()`, `.fallen`, `.status`); `rth.sim.scene.build_spec(settings) -> mujoco.MjSpec` (G1 in the dojo); `rth.sim.viewer.main(argv)` (`rth sim`) |
| `rth.vision` | Vision | `rth.vision.camera.Camera(settings.camera)` with `.read() -> ndarray \| None` (unmirrored BGR), `.close()`, context manager; `rth.vision.pose.MediaPipePoseTracker(settings.vision)` implements `PoseTracker`; `rth.vision.overlay.draw_skeleton(img, pose, scores=None, *, mirrored)`; `rth.vision.smoothing.DirectionSmoother(alpha)` with `.update(dirs)`, `.reset()`; `rth.vision.assets.fetch(force=False)`; `rth.vision.check.main(argv)` |
| `rth.ml` | ML | `rth.ml.recognizer.load_recognizer(settings) -> StanceRecognizer \| None`; `rth.ml.recorder.main(argv)`; `rth.ml.train.main(argv)` |
| `rth.coach` | Demo | `rth.coach.picker.make_picker(settings, stances)`, `OfflineMovePicker(stances)`; `rth.coach.sensei.make_sensei(settings, stances)`, `RuleSensei(settings, stances)`; `rth.coach.vision_tip.make_vision_coach(settings) -> VisionCoach \| None`; `rth.coach.voice.make_voice(settings)`; `rth.coach.check.main(argv)` |
| `rth.app` | Demo | `rth.app.main.main(argv)` (`rth demo`) |

## Conventions

- Run everything with `uv run` (Python 3.12; MediaPipe has no wheels for newer Pythons).
- MediaPipe runs on the **unmirrored** camera frame. Mirror only for display.
- Tests: `uv run pytest`. Default tests need no camera, network or downloaded assets; mark
  others with `@pytest.mark.assets`, `camera` or `nebius`.
