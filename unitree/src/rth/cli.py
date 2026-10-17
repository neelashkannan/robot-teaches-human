"""`rth <command>`: one entry point for the whole team. Commands import lazily."""

from __future__ import annotations

import argparse
import importlib
import sys

# command -> ("module:function", help). Each function takes argv (list[str]) and returns an exit code.
COMMANDS: dict[str, tuple[str, str]] = {
    "fetch-assets": ("rth.assets:main", "download the G1 model, motion clips and pose model"),
    "demo": ("rth.app.main:main", "run the booth demo"),
    "robot-view": ("rth.robot.viewer:main", "look at the G1 in each stance; scrub clips to pick frames"),
    "sim": ("rth.sim.viewer:main", "interactive physics sim of the G1 in the dojo: push it, switch stances"),
    "vision-check": ("rth.vision.check:main", "live webcam test of body tracking and limb directions"),
    "record": ("rth.ml.recorder:main", "record labelled stance examples for the recogniser"),
    "train-recognizer": ("rth.ml.train:main", "fit the stance recogniser and run leave-one-person-out evaluation"),
    "check-nebius": ("rth.coach.check:main", "check the Nebius key, model IDs and each AI role"),
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="rth", description="Robot Teaches Human")
    parser.add_argument("command", choices=sorted(COMMANDS), help="; ".join(f"{k}: {v[1]}" for k, v in COMMANDS.items()))
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    target, _ = COMMANDS[args.command]
    module, func = target.split(":")
    return int(getattr(importlib.import_module(module), func)(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
