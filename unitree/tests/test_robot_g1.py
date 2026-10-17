"""G1Teacher with the real Menagerie model and stance library (needs `rth fetch-assets`)."""

from __future__ import annotations

import numpy as np
import pytest

from rth.config import load_settings, project_path
from rth.contracts import Joint
from rth.contracts import Limb as L
from rth.limbs import angle_deg, limb_directions, mirror_directions, torso_frame
from rth.robot.model import FOOT_BODIES
from rth.stances import load_stances

SETTINGS = load_settings()
pytestmark = [
    pytest.mark.assets,
    pytest.mark.skipif(
        not project_path(SETTINGS.robot.model_path).exists(),
        reason="G1 model not downloaded (uv run rth fetch-assets)",
    ),
]
FORWARD, LEFT, UP = 0, 1, 2
STANCES = load_stances(SETTINGS.stances_path)
DOUBLE_SUPPORT = {"ready", "horse", "bow", "guard", "archer"}


@pytest.fixture(scope="module")
def teacher():
    from rth.robot.g1 import G1Teacher

    t = G1Teacher(SETTINGS, STANCES)
    yield t
    t.close()


def dirs_of(teacher, sid: str, mirrored: bool = False) -> dict:
    return limb_directions(teacher.builder.kin.keypoints(teacher.pose(sid, mirrored=mirrored)))


def foot_heights(teacher, qpos) -> list[float]:
    """Lowest sole point of (left, right) foot."""
    m, d = teacher.model, teacher.builder.kin.run(qpos)
    out = []
    for name in FOOT_BODIES:
        bid = m.body(name).id
        g = [i for i in teacher.builder.kin.index.foot_spheres if m.geom_bodyid[i] == bid]
        out.append(float(np.min(d.geom_xpos[g, 2] - m.geom_size[g, 0])))
    return out


def all_poses(teacher):
    for sid in teacher.stance_ids():
        for mirrored in (False, True):
            yield sid, mirrored, teacher.pose(sid, mirrored=mirrored)


def test_every_stance_loads(teacher):
    assert teacher.load_errors == {}
    assert teacher.stance_ids() == STANCES.ids()


def test_model_layout(teacher):
    m = teacher.model
    assert (m.nq, m.nv) == (36, 35)
    idx = teacher.builder.kin.index
    assert sorted(idx.qpos_adr.values()) == list(range(7, 36))


def test_joint_limits_respected(teacher):
    idx = teacher.builder.kin.index
    for sid, mirrored, q in all_poses(teacher):
        assert np.all(q >= idx.lo - 1e-9) and np.all(q <= idx.hi + 1e-9), (sid, mirrored)


def test_feet_grounded(teacher):
    for sid, mirrored, q in all_poses(teacher):
        left, right = foot_heights(teacher, q)
        assert min(left, right) == pytest.approx(0.0, abs=1e-6), (sid, mirrored)
        if sid in DOUBLE_SUPPORT:
            assert max(left, right) < 0.01, (sid, mirrored, left, right)
        assert q[0] == 0.0 and q[1] == 0.0


def test_crane_lifts_one_foot(teacher):
    left, right = foot_heights(teacher, teacher.pose("crane"))
    assert right == pytest.approx(0.0, abs=1e-6) and left > 0.15
    left, right = foot_heights(teacher, teacher.pose("crane", mirrored=True))
    assert left == pytest.approx(0.0, abs=1e-6) and right > 0.15


def test_faces_camera_and_torso_upright(teacher):
    for sid, mirrored, q in all_poses(teacher):
        frame = torso_frame(teacher.builder.kin.keypoints(q))
        assert frame[0] @ np.array([1.0, 0.0, 0.0]) > 0.99, (sid, mirrored)  # faces +x
        assert frame[2] @ np.array([0.0, 0.0, 1.0]) > np.cos(np.radians(15)), (sid, mirrored)


def test_mirror_matches_limbs_mirror_directions(teacher):
    for sid in teacher.stance_ids():
        expected = mirror_directions(dirs_of(teacher, sid))
        got = dirs_of(teacher, sid, mirrored=True)
        for limb in L:
            assert angle_deg(got[limb], expected[limb]) < 2.0, (sid, limb)


def test_ready_limbs_point_down_and_frame_faces_forward(teacher):
    teacher.set_target("ready", duration_s=0.0)
    kp = teacher.keypoints()
    assert set(kp) == set(Joint)
    for limb, d in limb_directions(kp).items():
        assert d[UP] < -0.95, limb
    assert torso_frame(kp)[0] == pytest.approx([1.0, 0.0, 0.0], abs=0.02)
    assert kp[Joint.LEFT_SHOULDER][1] > kp[Joint.RIGHT_SHOULDER][1]  # left is +y


def bend(d: dict, a: L, b: L) -> float:
    return angle_deg(d[a], d[b])


def test_horse(teacher):
    d = dirs_of(teacher, "horse")
    assert d[L.LEFT_THIGH][LEFT] > 0.4 and d[L.RIGHT_THIGH][LEFT] < -0.4  # out
    assert d[L.LEFT_THIGH][UP] < -0.3 and d[L.RIGHT_THIGH][UP] < -0.3  # down
    assert bend(d, L.LEFT_THIGH, L.LEFT_SHIN) > 50 and bend(d, L.RIGHT_THIGH, L.RIGHT_SHIN) > 50
    assert d[L.LEFT_UPPER_ARM][FORWARD] < -0.3 and d[L.RIGHT_UPPER_ARM][FORWARD] < -0.3
    assert d[L.LEFT_FOREARM][FORWARD] > 0.8 and d[L.RIGHT_FOREARM][FORWARD] > 0.8


def test_bow(teacher):
    d = dirs_of(teacher, "bow")
    assert d[L.LEFT_THIGH][FORWARD] > 0.5 and bend(d, L.LEFT_THIGH, L.LEFT_SHIN) > 30
    assert d[L.RIGHT_THIGH][FORWARD] < -0.2 and bend(d, L.RIGHT_THIGH, L.RIGHT_SHIN) < 10
    assert d[L.RIGHT_UPPER_ARM][FORWARD] > 0.9 and d[L.RIGHT_FOREARM][FORWARD] > 0.9


def test_guard(teacher):
    d = dirs_of(teacher, "guard")
    assert d[L.LEFT_FOREARM][UP] > 0.8 and d[L.RIGHT_FOREARM][UP] > 0.8
    assert d[L.LEFT_UPPER_ARM][UP] < -0.5 and d[L.RIGHT_UPPER_ARM][UP] < -0.5
    assert d[L.LEFT_THIGH][FORWARD] > d[L.RIGHT_THIGH][FORWARD] + 0.15  # left foot forward
    kp = teacher.builder.kin.keypoints(teacher.pose("guard"))
    for side in ("LEFT", "RIGHT"):
        wrist, elbow = kp[Joint[f"{side}_WRIST"]], kp[Joint[f"{side}_ELBOW"]]
        # The G1's hands reach ~0.1 m past the wrist keypoint, so wrists at shoulder
        # height put the fists by the face.
        assert wrist[2] > kp[Joint[f"{side}_SHOULDER"]][2] - 0.05
        assert wrist[2] > elbow[2] + 0.1  # forearm (elbow to wrist) is ~0.14 m


def test_crane(teacher):
    d = dirs_of(teacher, "crane")
    assert abs(d[L.LEFT_THIGH][UP]) < 0.35 and d[L.LEFT_THIGH][FORWARD] > 0.85
    assert d[L.RIGHT_THIGH][UP] < -0.95 and d[L.RIGHT_SHIN][UP] < -0.95
    assert d[L.LEFT_UPPER_ARM][LEFT] > 0.8 and d[L.RIGHT_UPPER_ARM][LEFT] < -0.8


def test_white_crane(teacher):
    d = dirs_of(teacher, "white_crane")
    kp = teacher.builder.kin.keypoints(teacher.pose("white_crane"))
    head_z = kp[Joint.RIGHT_SHOULDER][2] + 0.25
    assert kp[Joint.RIGHT_WRIST][2] > head_z  # right hand above the head
    mid_torso = (kp[Joint.LEFT_SHOULDER][2] + kp[Joint.LEFT_HIP][2]) / 2
    assert kp[Joint.LEFT_WRIST][2] < mid_torso  # left hand pressing down by the hip
    assert d[L.RIGHT_UPPER_ARM][UP] > 0.5
    assert bend(d, L.RIGHT_THIGH, L.RIGHT_SHIN) > 30  # weight sinks onto the bent back leg


def test_archer(teacher):
    d = dirs_of(teacher, "archer")
    assert d[L.LEFT_UPPER_ARM][LEFT] > 0.9 and d[L.LEFT_FOREARM][LEFT] > 0.9  # straight out
    assert d[L.RIGHT_UPPER_ARM][LEFT] < -0.8 and d[L.RIGHT_FOREARM][LEFT] > 0.5  # drawn across
    assert d[L.LEFT_THIGH][LEFT] > 0.3 and d[L.RIGHT_THIGH][LEFT] < -0.3  # wide stance


def test_no_self_intersection(teacher):
    """Limbs don't pass through the trunk or each other (convex hulls of the visual meshes)."""
    import mujoco

    m = teacher.model
    groups = {}
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = m.body(m.geom_bodyid[g]).name
        side = name.split("_")[0] if name.startswith(("left_", "right_")) else "trunk"
        part = name.split("_")[1] if side != "trunk" else "trunk"
        kind = "arm" if part in ("shoulder", "elbow", "wrist") else "leg"
        groups[g] = ("trunk", None) if side == "trunk" else (f"{side}_{kind}", part)
    geoms = list(groups)
    pairs = []
    for i, a in enumerate(geoms):
        for b in geoms[i + 1 :]:
            (ga, pa), (gb, pb) = groups[a], groups[b]
            if ga == gb:
                continue
            if "trunk" in (ga, gb) and (pa or pb) in ("shoulder", "hip"):
                continue  # limb roots touch the trunk by design
            pairs.append((a, b))
    for sid, mirrored, q in all_poses(teacher):
        d = teacher.builder.kin.run(q)
        for a, b in pairs:
            dist = mujoco.mj_geomDistance(m, d, a, b, 0.05, None)
            assert dist > -0.005, (
                sid,
                mirrored,
                m.body(m.geom_bodyid[a]).name,
                m.body(m.geom_bodyid[b]).name,
                dist,
            )


def test_interpolation_hits_endpoints(teacher):
    teacher.set_target("ready", duration_s=0.0)
    start = teacher.qpos
    teacher.set_target("horse", duration_s=2.0)
    teacher.update(0.0)
    assert teacher.qpos == pytest.approx(start, abs=1e-9)
    assert not teacher.at_target()
    teacher.update(1.0)
    mid = teacher.qpos
    assert not teacher.at_target()
    assert min(foot_heights(teacher, mid)) == pytest.approx(0.0, abs=1e-6)  # grounded mid-move
    knee = teacher.builder.kin.index.qpos_adr["left_knee_joint"]
    target = teacher.pose("horse")
    assert mid[knee] == pytest.approx((start[knee] + target[knee]) / 2, abs=1e-6)  # s(0.5) = 0.5
    teacher.update(1.0)
    assert teacher.at_target()
    assert teacher.qpos == pytest.approx(target, abs=1e-9)
    teacher.go_neutral(duration_s=0.5)
    teacher.update(0.6)
    assert teacher.at_target() and teacher.qpos == pytest.approx(start, abs=1e-9)


def test_retarget_mid_move_starts_from_current_pose(teacher):
    teacher.set_target("ready", duration_s=0.0)
    teacher.set_target("crane", duration_s=1.0)
    teacher.update(0.4)
    here = teacher.qpos
    teacher.set_target("archer", mirrored=True, duration_s=1.0)
    teacher.update(0.0)
    assert teacher.qpos == pytest.approx(here, abs=1e-9)
    teacher.update(1.0)
    assert teacher.qpos == pytest.approx(teacher.pose("archer", mirrored=True), abs=1e-9)


def test_unknown_stance_raises(teacher):
    with pytest.raises(KeyError):
        teacher.set_target("moonwalk")


def test_render_shape_and_camera(teacher):
    teacher.set_target("ready", duration_s=0.0)
    img = teacher.render(320, 360)
    assert img.shape == (360, 320, 3) and img.dtype == np.uint8
    assert img.std() > 10  # not blank
    img = teacher.render(160, 120)  # size change recreates the renderer
    assert img.shape == (120, 160, 3)
    cam = teacher._renderer.scene.camera[0]
    assert (np.asarray(cam.pos) - np.asarray(cam.forward))[0] > 1.0  # camera is in front (+x)


def test_clip_pose_loads(teacher):
    from rth.robot.clips import list_clips

    clips = list_clips(SETTINGS.robot.clips_dir)
    if not clips:
        pytest.skip("no LAFAN1 clips downloaded")
    q = teacher.builder.build({"source": "clip", "file": clips[0], "frame": 10, "mirror": True})
    assert q.shape == (teacher.model.nq,)
    assert min(foot_heights(teacher, q)) == pytest.approx(0.0, abs=1e-6)
    frame = torso_frame(teacher.builder.kin.keypoints(q))
    assert frame[0][0] > 0.99  # turned to face the camera


def test_viewer_rejects_unknown_clip_before_opening_a_window():
    from rth.robot.viewer import main

    assert main(["--clip", "no_such_clip"]) == 2
