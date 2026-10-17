"""Settings, loaded from config/settings.yaml with environment overrides from .env.

Each module owns its own section. To add a setting, add a field with a default
to your section here and to config/settings.yaml.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


def project_path(p: str | Path) -> Path:
    """Resolve a settings path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class CameraSettings(BaseModel):
    index: int = 0
    width: int = 1280
    height: int = 720
    mirror_display: bool = True  # flip the webcam image for display only; tracking uses the raw frame
    fps: int = 30  # requested capture rate; the camera may ignore it
    # macOS answers the first open with a permission prompt and a failure; keep retrying
    # while the visitor-facing operator clicks Allow instead of quitting.
    permission_wait_s: float = 30.0


class ScoringSettings(BaseModel):
    tolerance_deg: float = 30.0  # a limb is green within this angle of the robot's
    min_visibility: float = 0.5
    min_limbs_seen: int = 6
    mirror_view: bool = True  # the visitor copies the robot like a mirror
    smoothing_alpha: float = 0.35  # EMA weight of the newest frame for limb directions


class SessionSettings(BaseModel):
    stances_per_session: int = 5
    default_hold_s: float = 3.0
    default_transition_s: float = 2.0
    give_up_s: float = 20.0  # end the rep as not held after this long
    hold_grace_s: float = 0.3  # brief drop-outs allowed during a hold
    start_stance: str = "ready"
    auto_start_s: float = 2.0  # a visitor seen head to toe this long starts a session; 0 = keys only
    absent_reset_s: float = 12.0  # visitor gone this long mid-session: reset for the next one
    result_pause_s: float = 1.5  # show how the rep went before the next stance starts
    scorecard_s: float = 15.0  # how long the scorecard stays up before the next visitor
    arrive_timeout_s: float = 4.0  # robot still not at the stance this long after duration_s: go on


class RobotSettings(BaseModel):
    # "physics": the G1 is simulated (gravity, contacts, motor limits) by rth.sim.
    # "kinematic": joint angles are set directly, a fallback if physics misbehaves at the booth.
    mode: str = "kinematic"
    model_path: str = "assets/mujoco_menagerie/unitree_g1/scene.xml"
    clips_dir: str = "assets/lafan1/g1"
    render_width: int = 640
    render_height: int = 720
    camera_distance: float = 2.5  # metres from the look-at point
    camera_elevation_deg: float = -5.0
    camera_lookat_z: float = 0.78
    camera_azimuth_deg: float = 200.0  # 180 = dead front; a slight turn shows depth
    render_shadows: bool = False  # shadows and floor reflection roughly triple the frame time
    render_samples: int = 4  # MSAA samples for the offscreen buffer


class SimSettings(BaseModel):
    """Physics simulation of the G1 (rth.sim). Owners add fields here."""

    scene: str = "dojo"  # "dojo" (rth.sim.scene) or "plain" (Menagerie floor only)
    realtime_factor: float = 1.0
    max_substeps_per_frame: int = 40  # caps catch-up after a slow frame so the loop never spirals
    # Safety harness: a rope from the gantry to the torso, slack unless the robot drops.
    harness: bool = True
    harness_anchor_z: float = 2.6  # only used when the scene has no gantry anchor of its own
    harness_slack_m: float = 0.15  # torso drop below its planned height before the rope goes taut
    fall_pelvis_z: float = 0.55  # fallen below this (or harness_slack_m under a low stance's own height)
    fall_tilt_deg: float = 40.0  # fallen when the torso leans further than this
    recover_wait_s: float = 1.5  # hang in the harness this long, then re-spawn standing
    # Stances the robot can't yet hold unaided: the winch keeps the rope short so it carries
    # part of the weight (shown as assist=true in status), like a lab gantry.
    assist_stances: list[str] = Field(default_factory=lambda: ["bow"])
    assist_slack_m: float = 0.05  # rope slack for those: it takes weight after a small sag
    # Stepping between stances with different foot placements.
    # Off: feet stay where they are and only the body moves (fall-free fallback for the
    # booth). On: real steps, still unreliable for guard, horse, archer and bow.
    stepping: bool = False
    max_stance_width_m: float = 0.55  # wider foot placements are narrowed: a static step can't reach them
    step_height_m: float = 0.09  # swing foot clearance
    step_swing_s: float = 1.0  # lift, move and lower one foot
    weight_shift_s: float = 0.7  # move the COM over the stance foot (or back between the feet)
    # Interactive viewer (`rth sim`): keyboard pushes on the torso and the stance tour.
    viewer_push_n: float = 120.0  # P: about 0.5 m/s of impulse on the 35 kg robot over viewer_push_s
    viewer_big_push_n: float = 320.0  # B: enough to need a step or the harness
    viewer_push_s: float = 0.15
    viewer_tour_s: float = 6.0  # N: seconds per stance in tour mode


class VisionSettings(BaseModel):
    model_path: str = "assets/models/pose_landmarker_full.task"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class RecognizerSettings(BaseModel):
    recordings_dir: str = "data/recordings"
    model_path: str = "data/recognizer.npz"
    report_dir: str = "reports"
    k: int = 10
    smoothing_alpha: float = 0.2
    min_confidence: float = 0.6
    enabled: bool = True
    top_n_by_max: int = 30  # k-NN shortlist size before the mean-distance vote (MediaPipe's top_n)
    forward_weight: float = 0.5  # weight of the depth-like torso axis, the noisiest one for a webcam
    reject_quantile: float = 0.95  # rejection threshold from held-out-person neighbour distances
    reject_margin: float = 1.1
    max_frames_per_recording: int = 300  # consecutive frames are near-duplicates; cap per recording
    record_min_visibility: float = 0.6  # every body joint must be this visible to keep a frame


class NebiusSettings(BaseModel):
    base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    api_key_env: str = "NEBIUS_API_KEY"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    sensei_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    vision_model: str = "openbmb/MiniCPM-V-4_5"
    timeout_s: float = 4.0
    enable_picker: bool = True
    enable_sensei: bool = True
    enable_vision_tip: bool = False
    usage_log: str = "logs/nebius_usage.jsonl"
    embedding_cache: str = "data/stance_embeddings.json"
    structured_output: str = "json_schema"  # json_schema | json_object | none (prompt-only JSON)
    sensei_max_tokens: int = 160
    sensei_temperature: float = 0.6
    vision_max_tokens: int = 60
    vision_max_px: int = 512  # longest side of the snapshot sent to the vision model

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


class VoiceSettings(BaseModel):
    enabled: bool = True
    voice: str | None = None  # macOS `say -v` voice name; None = system default
    rate: int = 185


class Settings(BaseModel):
    stances_path: str = "config/stances.yaml"
    camera: CameraSettings = Field(default_factory=CameraSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    robot: RobotSettings = Field(default_factory=RobotSettings)
    sim: SimSettings = Field(default_factory=SimSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    recognizer: RecognizerSettings = Field(default_factory=RecognizerSettings)
    nebius: NebiusSettings = Field(default_factory=NebiusSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    load_dotenv(ROOT / ".env")
    p = project_path(path)
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    return Settings.model_validate(data or {})


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings from the default file."""
    return load_settings()
