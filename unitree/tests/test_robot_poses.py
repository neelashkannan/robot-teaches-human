"""Robot pose maths that needs no downloaded assets: specs, mirroring, interpolation, viewer keys."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from rth.robot.clips import CSV_COLUMNS, CSV_JOINT_ORDER, rows_to_qpos
from rth.robot.g1 import blend_qpos, min_jerk, slerp
from rth.robot.model import JOINT_NAMES, ROOT_NQ
from rth.robot.poses import (
    ClipPose,
    JointsPose,
    axis_quat,
    clip_pose_snippet,
    joints_pose_snippet,
    mirror_qpos,
    parse_pose_spec,
    quat_mul,
    quat_rpy,
    rpy_quat,
)
from rth.robot.viewer import ViewerState
from rth.stances import load_stances

NQ = ROOT_NQ + len(JOINT_NAMES)
ADR = {n: ROOT_NQ + i for i, n in enumerate(JOINT_NAMES)}


def random_qpos(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=NQ)
    q[3:7] /= np.linalg.norm(q[3:7])
    return q


# --------------------------------------------------------------------------- specs


def test_joint_spec_parses():
    spec = parse_pose_spec({"source": "joints", "degrees": True, "joints": {"left_knee_joint": 30}})
    assert isinstance(spec, JointsPose)
    assert spec.joints == {"left_knee_joint": 30}
    assert spec.scale() == pytest.approx(math.pi / 180)


def test_clip_spec_parses():
    spec = parse_pose_spec({"source": "clip", "file": "fight1_subject2", "frame": 12})
    assert isinstance(spec, ClipPose)
    assert (spec.file, spec.frame, spec.mirror) == ("fight1_subject2", 12, False)


@pytest.mark.parametrize(
    "bad",
    [
        {"source": "joints", "joints": {"left_knee": 1.0}},  # unknown joint
        {"source": "clip", "file": "x"},  # missing frame
        {"source": "clip", "file": "x", "frame": -1},
        {"source": "video", "file": "x"},
        {"source": "joints", "joints": {}, "colour": "red"},  # unknown key
    ],
)
def test_bad_specs_rejected(bad):
    with pytest.raises(ValidationError):
        parse_pose_spec(bad)


def test_every_stance_has_a_valid_pose_spec():
    for stance in load_stances().stances:
        assert stance.pose, f"{stance.id} has no pose"
        parse_pose_spec(stance.pose)


def test_snippets_round_trip_through_parser():
    import yaml

    clip = yaml.safe_load(clip_pose_snippet("fight1_subject2", 7, mirror=True))["pose"]
    assert parse_pose_spec(clip) == ClipPose(
        source="clip", file="fight1_subject2", frame=7, mirror=True
    )
    q = np.zeros(NQ)
    q[3:7] = rpy_quat(0.1, -0.05, 0.0)
    q[ADR["left_knee_joint"]] = math.radians(40)
    spec = parse_pose_spec(yaml.safe_load(joints_pose_snippet(q))["pose"])
    assert spec.joints == {"left_knee_joint": 40.0}
    assert spec.root_rpy == pytest.approx((5.7, -2.9, 0.0))


# --------------------------------------------------------------------------- quaternions


def test_rpy_round_trip():
    for rpy in [(0.1, -0.2, 0.3), (0.0, 0.0, 0.0), (-0.4, 0.5, -2.0)]:
        assert quat_rpy(rpy_quat(*rpy)) == pytest.approx(rpy)


def test_quat_mul_composes_rotations():
    a = quat_mul(axis_quat(2, 0.3), axis_quat(2, 0.4))
    assert a == pytest.approx(axis_quat(2, 0.7))


# --------------------------------------------------------------------------- mirroring


def test_mirror_is_involution():
    for seed in range(5):
        q = random_qpos(seed)
        assert mirror_qpos(mirror_qpos(q)) == pytest.approx(q)


def test_mirror_swaps_sides_with_sign_rules():
    q = np.zeros(NQ)
    q[3] = 1.0
    q[ADR["left_hip_pitch_joint"]] = 0.1
    q[ADR["left_hip_roll_joint"]] = 0.2
    q[ADR["left_shoulder_yaw_joint"]] = 0.3
    q[ADR["left_elbow_joint"]] = 0.4
    q[ADR["waist_yaw_joint"]] = 0.5
    q[ADR["waist_roll_joint"]] = 0.6
    q[ADR["waist_pitch_joint"]] = 0.7
    m = mirror_qpos(q)
    assert m[ADR["right_hip_pitch_joint"]] == pytest.approx(0.1)
    assert m[ADR["right_hip_roll_joint"]] == pytest.approx(-0.2)
    assert m[ADR["right_shoulder_yaw_joint"]] == pytest.approx(-0.3)
    assert m[ADR["right_elbow_joint"]] == pytest.approx(0.4)
    assert m[ADR["left_hip_roll_joint"]] == 0.0
    assert m[ADR["waist_yaw_joint"]] == pytest.approx(-0.5)
    assert m[ADR["waist_roll_joint"]] == pytest.approx(-0.6)
    assert m[ADR["waist_pitch_joint"]] == pytest.approx(0.7)


def test_mirror_reflects_root_orientation():
    q = np.zeros(NQ)
    q[3:7] = rpy_quat(0.2, 0.3, 0.4)
    q[1] = 0.5
    m = mirror_qpos(q)
    # Reflecting through x-z keeps pitch and negates roll and yaw.
    assert quat_rpy(m[3:7]) == pytest.approx((-0.2, 0.3, -0.4))
    assert m[1] == pytest.approx(-0.5)


# --------------------------------------------------------------------------- interpolation


def test_min_jerk_profile():
    assert min_jerk(0.0) == 0.0
    assert min_jerk(1.0) == pytest.approx(1.0)
    assert min_jerk(0.5) == pytest.approx(0.5)
    assert min_jerk(-1.0) == 0.0 and min_jerk(2.0) == pytest.approx(1.0)
    h = 1e-4
    assert (min_jerk(h) - min_jerk(0)) / h == pytest.approx(0.0, abs=1e-3)  # starts at rest
    xs = [min_jerk(t) for t in np.linspace(0, 1, 50)]
    assert np.all(np.diff(xs) >= 0)


def test_slerp_endpoints_and_unit_norm():
    a, b = axis_quat(2, 0.0), axis_quat(2, 1.0)
    assert slerp(a, b, 0.0) == pytest.approx(a)
    assert slerp(a, b, 1.0) == pytest.approx(b)
    assert slerp(a, b, 0.5) == pytest.approx(axis_quat(2, 0.5))
    assert np.linalg.norm(slerp(a, -b, 0.3)) == pytest.approx(1.0)


def test_blend_hits_endpoints():
    a, b = random_qpos(1), random_qpos(2)
    assert blend_qpos(a, b, 0.0) == pytest.approx(a)
    assert blend_qpos(a, b, 1.0)[:3] == pytest.approx(b[:3])
    assert blend_qpos(a, b, 1.0)[7:] == pytest.approx(b[7:])
    q = blend_qpos(a, b, 1.0)[3:7]
    assert abs(np.dot(q, b[3:7])) == pytest.approx(1.0)  # same rotation (q or -q)


# --------------------------------------------------------------------------- clips


def test_rows_to_qpos_maps_quaternion_and_joints_by_name():
    row = np.arange(CSV_COLUMNS, dtype=float) / 100.0
    row[3:7] = (0.1, 0.2, 0.3, 0.9)  # xyzw
    q = rows_to_qpos(row, ADR, NQ)[0]
    quat = np.array([0.9, 0.1, 0.2, 0.3]) / np.linalg.norm([0.9, 0.1, 0.2, 0.3])
    assert q[3:7] == pytest.approx(quat)
    assert q[0:3] == pytest.approx(row[0:3])
    for col, name in enumerate(CSV_JOINT_ORDER):
        assert q[ADR[name]] == pytest.approx(row[7 + col])


def test_rows_to_qpos_rejects_wrong_width():
    with pytest.raises(ValueError):
        rows_to_qpos(np.zeros((2, 30)), ADR, NQ)


def test_csv_order_covers_every_joint_once():
    assert sorted(CSV_JOINT_ORDER) == sorted(JOINT_NAMES)


# --------------------------------------------------------------------------- viewer keys


def test_viewer_state_keys():
    s = ViewerState(["ready", "horse"], ["clip_a", "clip_b"], {"clip_a": 100, "clip_b": 50})
    assert s.handle_key(-1) is None
    assert s.handle_key(ord("n")) == "pose" and s.stance == "horse"
    assert s.handle_key(ord("n")) == "pose" and s.stance == "ready"
    assert s.handle_key(ord("p")) == "pose" and s.stance == "horse"
    assert s.handle_key(ord("m")) == "pose" and s.mirrored
    assert s.handle_key(ord("c")) == "pose" and s.mode == "clip"
    s.handle_key(ord(">"))
    s.handle_key(ord("."))
    assert s.frame == 31
    s.handle_key(ord("<"))
    s.handle_key(ord("<"))
    assert s.frame == 0  # clamped
    for _ in range(5):
        s.handle_key(ord(">"))
    assert s.frame == 99  # clamped to clip length
    assert s.handle_key(ord("]")) == "pose" and s.clip == "clip_b" and s.frame == 0
    assert s.handle_key(ord("[")) == "pose" and s.clip == "clip_a"
    s.handle_key(ord(" "))
    assert s.playing
    assert "clip_a" in s.caption() and "mirrored" in s.caption()
    assert s.handle_key(ord("y")) == "yaml"
    assert s.handle_key(ord("j")) == "joints"
    assert s.handle_key(ord("q")) == "quit"
    assert s.handle_key(27) == "quit"


def test_viewer_without_clips():
    s = ViewerState(["ready"], [])
    assert s.handle_key(ord("c")) == "noclips" and s.mode == "stances"
