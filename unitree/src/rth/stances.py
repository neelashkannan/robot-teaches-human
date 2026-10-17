"""The stance library (config/stances.yaml), shared by the robot, recogniser, coach and app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from rth.config import project_path


class Stance(BaseModel):
    id: str
    name: str
    description: str  # what the visitor does, in plain words; also embedded by the move picker
    tags: list[str] = Field(default_factory=list)
    difficulty: int = Field(1, ge=1, le=3)
    hold_s: float = 3.0
    # Where the robot's pose comes from. Owned by rth.robot, which validates it.
    pose: dict[str, Any] | None = None


class StanceLibrary(BaseModel):
    stances: list[Stance]

    def ids(self) -> list[str]:
        return [s.id for s in self.stances]

    def get(self, stance_id: str) -> Stance:
        for s in self.stances:
            if s.id == stance_id:
                return s
        raise KeyError(f"unknown stance {stance_id!r}; known: {self.ids()}")

    def __contains__(self, stance_id: object) -> bool:
        return stance_id in self.ids()


def load_stances(path: str | Path = "config/stances.yaml") -> StanceLibrary:
    lib = StanceLibrary.model_validate(yaml.safe_load(project_path(path).read_text()))
    ids = lib.ids()
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate stance ids in {path}")
    return lib
