# Robot Teaches Human

> Robots learned kung fu from us. This one teaches it back.

Humanoid robots learn martial arts by copying human motion capture. We flip it: a simulated
Unitree G1 shows you a stance, watches you copy it through a webcam and corrects you limb by
limb. An AI sensei on Nebius picks what comes next.

- **Shared language:** you and the robot both become 8 limb directions (upper arm, forearm,
  thigh, shin on each side) in your own torso frame, so you're scored directly against the
  robot whatever your height or where you stand.
- **Our own ML:** a stance recogniser (pose embeddings + nearest neighbours) tested on people it
  has never seen.
- **Nebius Token Factory:** an embedding model picks a stance from your words, an LLM sensei
  plans the next rep and says one line, and an optional vision model adds a tip.
- **Works offline:** with no key, a timeout or a bad reply, rule-based fallbacks take over and
  the demo carries on.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the pieces fit.

## Setup (once per laptop)

Needs macOS or Linux, [uv](https://docs.astral.sh/uv/) and Python 3.12. MediaPipe has no
wheels for newer Pythons; uv installs 3.12 for you.

```bash
uv sync
```

```bash
uv run rth fetch-assets
```

That downloads the G1 model (MuJoCo Menagerie), a few LAFAN1 clips and the MediaPipe pose
models into `assets/`, about 120 MB.

For the AI roles, copy `.env.example` to `.env` and set `NEBIUS_API_KEY`. `.env` is
git-ignored. Then check it:

```bash
uv run rth check-nebius
```

On macOS, the camera only works when you run the demo from an app that has camera access
(System Settings → Privacy & Security → Camera). Launch it from VS Code's terminal or the
Claude app's terminal panel if those are switched on there. On the booth laptop, macOS never
showed Terminal.app a camera prompt, so Terminal never got access. While a permission prompt
is open, the demo keeps retrying for up to `camera.permission_wait_s` seconds.

## Run the booth

```bash
uv run rth demo
```

The visitor stands on the floor mark, full body in view. Type a request (T or Enter, e.g.
"something for balance") or press 1–7 to pick a stance. The G1 shows it. The visitor copies it
until every limb is green for the hold time. The sensei picks the next stance, and after 5
stances a scorecard compares the first try with the last.

| Key | Does |
| --- | --- |
| T or Enter | type a request (Enter sends, Esc cancels) |
| 1–7 | pick a stance directly |
| Space | skip |
| M | toggle mirror view (copy the robot like a mirror) |
| R | restart |
| F | fullscreen |
| Q or Esc | quit |

Useful options:

- `--offline`: never call Nebius.
- `--fake-person`: no webcam; a synthetic visitor copies the robot.
- `--robot physics|kinematic`: physics-simulated or directly posed G1.
- `--camera INDEX`.
- `--headless N --out DIR`: run N frames with no window and save PNGs (for testing).

## Physics sim: `uv run rth sim`

The G1 in a small dojo with real physics, in MuJoCo's interactive viewer. On macOS it
re-launches itself under `mjpython` automatically.

```bash
uv run rth sim
```

- `uv run rth sim --passive`: no controller; the robot holds "stand", so a push knocks it over.
- `uv run rth sim --scene plain --stance horse --tour`: the plain scene, starting from horse,
  moving through the stances on a timer.

| Input | Does |
| --- | --- |
| 1–7 | stance |
| M | mirrored |
| P / B | random push / big push |
| R | reset to ready |
| N | tour (next stance every 6 s) |
| Space | pause |
| Double-click a body, then Ctrl+right-drag | push it (Ctrl+left-drag twists) |
| Left-drag / right-drag / scroll | rotate / move / zoom the view |
| Tab, Shift+Tab | hide the side panels |
| F1 | MuJoCo's own help |

The Camera list on the left has "booth" (the visitor's view) and "wide". Try the props: punch
the heavy bag, shove the wooden dummy, topple the foam blocks. For stills and render timing,
run `uv run python -m rth.sim.scene`.

## Team jobs today

| Role | Command | What to do |
| --- | --- | --- |
| Robot | `uv run rth robot-view` | Check each stance looks right (n/p stance, m mirror, c clip mode, y/j print a `pose:` snippet for `config/stances.yaml`) |
| Vision | `uv run rth vision-check` | Stand where visitors will; check all 8 limbs track and the fps; try the lite model if slow |
| ML | `uv run rth record --person NAME --all` | Every teammate records every stance (SPACE, 3 s countdown, hold ~20 s while moving a little). Then `uv run rth train-recognizer` and put `reports/recognizer.md` and `reports/recognizer_confusion.png` on the poster |
| Demo | `uv run rth demo` | Booth flow, tolerance tuning, poster and pitch, a backup video (`--headless` frames or a screen recording) |

**The recogniser has no real-person accuracy yet.** It has only been tested on synthetic data.
Record at least 3 people; the report then tests on each person while training on the others.
Phone videos work too:
`uv run rth record --person bob --stance horse --from-video clip.mp4 --start 2 --end 18`.

## Tuning at the venue

If correct poses don't turn green, measure them instead of guessing. Run this from an app
with camera access:

```bash
uv run python scripts/pose_diagnostic.py ready guard horse
```

Hold each stance for 10 s. It prints each limb's error against the robot and saves the raw
frames to `reports/pose_diagnostic.json`. Scoring is face-on wherever a limb lies across the
camera's view and full 3D only where it points at or away from the camera, because webcam depth
is unreliable (see `rth.limbs.limb_error`).

All in `config/settings.yaml`:

- `scoring.tolerance_deg` (30): lower is stricter.
- `session.default_hold_s` (3): seconds to hold a stance.
- `vision.model_path`: `pose_landmarker_full.task` is about 22 ms a frame; `lite` is about
  15 ms if the booth drops below 25 fps.
- `nebius.sensei_model`: `Qwen/Qwen3-235B-A22B-Instruct-2507` answered in 0.8–1.7 s on 17 Oct.
  The smaller `Qwen3-30B-A3B` was overloaded and timed out. Any reply later than
  `timeout_s + 1` uses the rule-based sensei.
- `robot.mode`: `physics` or `kinematic`.

Nebius calls happen once per request or per rep, never per frame. Each call is logged with
latency and tokens in `logs/nebius_usage.jsonl`.

## Tests

```bash
uv run pytest
```

Asset tests (G1 model, pose model, real photos) skip themselves if `rth fetch-assets` hasn't
run. The live Nebius test runs only when `NEBIUS_API_KEY` is set in the environment.

## Data and licences

- **G1 model:** MuJoCo Menagerie, BSD-3-Clause (copyright Unitree Robotics).
- **Motion clips:** LAFAN1 retargeted to the G1 (lvhaidong/LAFAN1_Retargeting_Dataset), CC
  BY-NC-ND 4.0, non-commercial. They're fine for this student demo but not for a product. The
  stances in `config/stances.yaml` are hand-authored joint angles, not clip frames.
- **Test photos:** sources and licences are in `tests/data/vision/SOURCES.md`.
- **Coaching only:** this is exercise coaching, with no medical claims.
