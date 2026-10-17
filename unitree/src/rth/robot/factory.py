"""Pick the robot teacher the booth runs: physics simulation or kinematic playback."""

from __future__ import annotations

from rth.config import Settings
from rth.contracts import RobotTeacher
from rth.stances import StanceLibrary


def make_teacher(settings: Settings, stances: StanceLibrary, *, mode: str | None = None) -> RobotTeacher:
    """``mode`` overrides ``settings.robot.mode`` ("physics" or "kinematic")."""
    mode = mode or settings.robot.mode
    if mode == "physics":
        from rth.sim.teacher import PhysicsTeacher

        return PhysicsTeacher(settings, stances)
    if mode == "kinematic":
        from rth.robot.g1 import G1Teacher

        return G1Teacher(settings, stances)
    raise ValueError(f"unknown robot mode {mode!r}; use 'physics' or 'kinematic'")
