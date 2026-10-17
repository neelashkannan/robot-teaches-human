# Pitch video (Remotion)

A 1:59 pitch video with voiceover and music for the 2-minute slot. The voiceover transcript and
Q&A notes are in [SCRIPT.md](SCRIPT.md).

```bash
npm install
```

```bash
npm run studio
```

```bash
npm run render
```

`studio` opens a live preview; every scene is also listed on its own under "Scenes". `render`
writes `out/pitch.mp4`.

## Voice and music

- Lines: `src/voice/lines.json` (`at` is the earliest start in seconds within the scene). Voice them
  with `python3 scripts/voiceover.py` (needs `OPENROUTER_API_KEY`; only changed lines are re-voiced).
  Clips land in `public/voice/`, lengths in `src/voice/durations.json`, and each scene stretches to
  fit its lines.
- Music: `public/music/bed.mp3` (Lyria 3, `audio-src/music-raw.mp3` is the raw take), ducked under
  the voice in `src/Pitch.tsx`. Render without it with `--props='{"music":false}'`.

## Swapping footage

Everything about the recordings is in `src/footage.ts`. Put videos in `public/footage/` (they're
git-ignored, like every `.mp4`).

- **Robot hand demo:** `public/footage/arm.mp4` is the 14:42 recording, cropped to the webcam and
  teacher panels: `-vf "crop=2100:979:0:0,scale=1920:896,fps=30"`. `ARM.clips` and `ARM.steps` pick
  the moments.
- **Unitree demo:** `public/footage/unitree.mp4` is the 13:49 screen recording re-encoded to 1080p:

  ```bash
  ffmpeg -i "<recording>.mov" -vf "scale=1920:1072,fps=30" -c:v libx264 -crf 18 -pix_fmt yuv420p -an public/footage/unitree.mp4
  ```

  `UNITREE.clips` picks three moments (ready and guard, horse with a correction, the scorecard);
  `zoom` adds a slow push-in.

To check a few frames quickly without a full render:

```bash
node scripts/stills.mjs UnitreeDemo 100 450
```

## Scenes

| Scene | What it does for the judging rubric |
| --- | --- |
| Flip | Old assumption → flip, shown as an animation (creativity) |
| Title | The flip in one sentence (presentation) |
| Why | Why the flip was chosen (presentation) |
| Virtual | No robot access at the venue, so we built a virtual dojo (understanding) |
| ArmDemo, Hype, UnitreeDemo | Real recordings with a step list, escalating from the hand to the humanoid with a flash reveal (execution) |
| Trick | How a small robot marks a tall person: 8 arrows, direction only (understanding) |
| Stack | Each part and why, plus AI tools disclosed (tech stack) |
| Limits | What's real and what's simplified (understanding) |
| Next | Concept: a real robot walks over and fixes your pose, labelled "not built yet" (fit and ambition) |
| Close | The line to remember |
