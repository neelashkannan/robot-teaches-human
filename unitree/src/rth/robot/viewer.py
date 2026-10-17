"""`rth robot-view`: look at the G1 in each stance and scrub LAFAN1 clips for new poses.

The key handling is a small pure state machine (``ViewerState.handle_key``) so it
can be tested without a window; ``main`` only wires it to OpenCV.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from rth.config import load_settings
from rth.robot.clips import FPS, list_clips, load_clip
from rth.robot.poses import clip_pose_snippet, joints_pose_snippet

HELP = """\
G1 viewer keys
  n / p         next / previous stance
  m             toggle mirrored (left and right swapped)
  c             toggle clip-scrubbing mode
  ] / [         next / previous clip            (clip mode)
  . / ,         step +1 / -1 frame              (clip mode)
  > / <         step +30 / -30 frames (1 s)     (clip mode)
  space         play / pause the clip           (clip mode)
  y             print the YAML pose: for this clip frame
  j             print the current pose as hand-authored joint angles (degrees)
  h             print this help
  q / Esc       quit"""

STANCE_MOVE_S = 1.2
BIG_STEP = 30


@dataclass
class ViewerState:
    stances: list[str]
    clips: list[str]
    clip_lengths: dict[str, int] = field(default_factory=dict)
    mode: str = "stances"  # or "clip"
    stance_index: int = 0
    mirrored: bool = False
    clip_index: int = 0
    frame: int = 0
    playing: bool = False

    @property
    def stance(self) -> str | None:
        return self.stances[self.stance_index] if self.stances else None

    @property
    def clip(self) -> str | None:
        return self.clips[self.clip_index] if self.clips else None

    def clip_length(self) -> int:
        return self.clip_lengths.get(self.clip or "", 1)

    def step(self, n: int) -> None:
        self.frame = int(np.clip(self.frame + n, 0, self.clip_length() - 1))

    def handle_key(self, key: int) -> str | None:
        """Apply one key press. Returns an action for the caller: 'quit', 'help',
        'yaml', 'joints', 'pose' (the shown pose changed) or None."""
        if key < 0:
            return None
        ch = chr(key & 0xFF)
        if ch in ("q", "\x1b"):
            return "quit"
        if ch == "h":
            return "help"
        if ch == "j":
            return "joints"
        if ch == "y":
            return "yaml"
        if ch == "m":
            self.mirrored = not self.mirrored
            return "pose"
        if ch == "c":
            if not self.clips:
                return "noclips"
            self.mode = "stances" if self.mode == "clip" else "clip"
            self.playing = False
            return "pose"
        if self.mode == "stances" and ch in ("n", "p") and self.stances:
            self.stance_index = (self.stance_index + (1 if ch == "n" else -1)) % len(self.stances)
            return "pose"
        if self.mode == "clip":
            if ch in ("]", "["):
                self.clip_index = (self.clip_index + (1 if ch == "]" else -1)) % len(self.clips)
                self.frame = 0
                return "pose"
            steps = {".": 1, ",": -1, ">": BIG_STEP, "<": -BIG_STEP}
            if ch in steps:
                self.step(steps[ch])
                return "pose"
            if ch == " ":
                self.playing = not self.playing
                return None
        return None

    def caption(self) -> str:
        mirror = " (mirrored)" if self.mirrored else ""
        if self.mode == "clip":
            play = " playing" if self.playing else ""
            return f"clip {self.clip} frame {self.frame}/{self.clip_length() - 1}{mirror}{play}"
        return f"stance {self.stance}{mirror}"


def overlay(img: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    for i, text in enumerate(lines):
        y = 26 + 24 * i
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
    return img


def limb_lines(dirs: dict) -> list[str]:
    """Compact limb-direction readout (forward, left, up) for the overlay."""
    out = []
    for limb, d in dirs.items():
        if d is not None:
            out.append(f"{limb.value:16s} {d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rth robot-view", description=HELP.splitlines()[0])
    parser.add_argument("--stance", help="start on this stance id")
    parser.add_argument("--clip", help="start in clip mode on this clip")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-limbs", action="store_true", help="hide the limb-direction readout")
    args = parser.parse_args(argv or [])

    import cv2

    from rth.robot.g1 import G1Teacher
    from rth.stances import load_stances

    settings = load_settings()
    rs = settings.robot
    width, height = args.width or rs.render_width, args.height or rs.render_height
    teacher = G1Teacher(settings, load_stances(settings.stances_path))
    clips = list_clips(rs.clips_dir)
    adr, nq = teacher.builder.kin.index.qpos_adr, teacher.model.nq
    lengths = {c: len(load_clip(c, rs.clips_dir, adr, nq)) for c in clips}
    state = ViewerState(teacher.stance_ids(), clips, lengths)
    if args.stance in state.stances:
        state.stance_index = state.stances.index(args.stance)
    if args.clip:
        if args.clip not in clips:
            print(f"unknown clip {args.clip!r}; have: {clips}")
            return 2
        state.mode, state.clip_index = "clip", clips.index(args.clip)
        state.step(args.frame)
    print(HELP)
    if not clips:
        print(f"(no clips in {rs.clips_dir}; run `uv run rth fetch-assets` for clip mode)")

    def show_pose(duration: float) -> None:
        if state.mode == "clip" and state.clip:
            q = teacher.builder.clip_frame(state.clip, state.frame)
            if state.mirrored:
                from rth.robot.poses import mirror_qpos

                q = mirror_qpos(q)
            teacher.snap_to(q)
        elif state.stance:
            teacher.set_target(state.stance, mirrored=state.mirrored, duration_s=duration)

    window = "G1 teacher"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    show_pose(0.0)
    last = time.perf_counter()
    play_acc = 0.0
    try:
        while True:
            now = time.perf_counter()
            dt, last = now - last, now
            if state.mode == "clip" and state.playing:
                play_acc += dt * FPS
                if play_acc >= 1.0:
                    state.step(int(play_acc))
                    play_acc -= int(play_acc)
                    show_pose(0.0)
            teacher.update(dt)
            img = teacher.render(width, height)
            lines = [state.caption()]
            if not args.no_limbs:
                lines += limb_lines(teacher.limb_directions())
            cv2.imshow(window, overlay(img, lines))
            action = state.handle_key(cv2.waitKey(15))
            if action == "quit":
                break
            if action == "help":
                print(HELP)
            elif action == "noclips":
                print(f"no clips in {rs.clips_dir}; run `uv run rth fetch-assets`")
            elif action == "yaml":
                if state.mode == "clip" and state.clip:
                    print(clip_pose_snippet(state.clip, state.frame, mirror=state.mirrored))
                else:
                    print("(y prints a clip frame; press c for clip mode, or j for joint angles)")
            elif action == "joints":
                print(joints_pose_snippet(teacher.qpos))
            elif action == "pose":
                show_pose(STANCE_MOVE_S)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        teacher.close()
        cv2.destroyAllWindows()
    return 0
