"""`rth fetch-assets`: download everything the demo needs into assets/ (idempotent)."""

from __future__ import annotations

import argparse


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rth fetch-assets")
    parser.add_argument("--force", action="store_true", help="download again even if present")
    args = parser.parse_args(argv)

    from rth.robot.assets import fetch as fetch_robot
    from rth.vision.assets import fetch as fetch_vision

    fetch_robot(force=args.force)
    fetch_vision(force=args.force)
    return 0
