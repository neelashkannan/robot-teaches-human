"""The dojo scene (rth.sim.scene) and its textures."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rth.config import load_settings, project_path
from rth.sim import textures

SETTINGS = load_settings()
needs_model = pytest.mark.skipif(
    not project_path(SETTINGS.robot.model_path).exists(),
    reason="G1 model not downloaded (uv run rth fetch-assets)",
)


def settings_for(scene: str):
    s = SETTINGS.model_copy(deep=True)
    s.sim.scene = scene
    return s


# --------------------------------------------------------------------------- textures (no model needed)


def test_textures_are_seeded_and_sized():
    a, b = textures.tatami(), textures.tatami()
    assert a.dtype == np.uint8 and a.shape == (512, 1024, 3)
    assert np.array_equal(a, b)
    # Straw is a warm yellow-brown: red >= green > blue on average.
    r, g, bl = a.reshape(-1, 3).mean(axis=0)
    assert r >= g > bl


def test_ensure_textures_writes_missing_files(tmp_path):
    paths = textures.ensure_textures(tmp_path)
    assert set(paths) == set(textures.GENERATORS)
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths.values())


def test_unknown_scene_is_rejected():
    from rth.sim.scene import build_spec

    with pytest.raises(ValueError, match="unknown sim scene"):
        build_spec(settings_for("moon"))


# --------------------------------------------------------------------------- the model


@pytest.fixture(scope="module")
def dojo():
    from rth.sim.scene import build_model

    return build_model(settings_for("dojo"))


@pytest.fixture(scope="module")
def plain():
    from rth.sim.scene import build_model

    return build_model(settings_for("plain"))


@pytest.mark.assets
@needs_model
@pytest.mark.parametrize("scene", ["dojo", "plain"])
def test_both_scenes_compile_with_cameras(scene):
    from rth.sim.scene import build_model, build_spec

    spec = build_spec(settings_for(scene))
    assert isinstance(spec, mujoco.MjSpec)
    m = build_model(settings_for(scene))
    assert m.nu == 29 and m.key("stand") is not None
    for cam in ("booth", "wide"):
        assert m.camera(cam).id >= 0
    assert m.vis.global_.offwidth >= 1280 and m.vis.global_.offheight >= 720


@pytest.mark.assets
@needs_model
def test_floor_is_menagerie_floor(dojo, plain):
    for m in (dojo, plain):
        f = m.geom("floor")
        assert f.type[0] == mujoco.mjtGeom.mjGEOM_PLANE
        assert f.pos[2] == 0.0
        np.testing.assert_allclose(f.friction, plain.geom("floor").friction)
    # The robot's soles touch z = 0 in the stand keyframe in both scenes.
    from rth.sim.scene import standing_data

    for m in (dojo, plain):
        d = standing_data(m)
        feet = [
            g
            for g in range(m.ngeom)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and "ankle_roll" in m.body(m.geom_bodyid[g]).name
        ]
        low = float(np.min(d.geom_xpos[feet, 2] - m.geom_size[feet, 0]))
        assert abs(low) < 0.01


@pytest.mark.assets
@needs_model
def test_nothing_solid_inside_the_clear_circle(dojo):
    from rth.sim.scene import CLEAR_RADIUS_M, robot_body_ids, standing_data

    d = standing_data(dojo)
    robot = robot_body_ids(dojo)
    for g in range(dojo.ngeom):
        if dojo.geom_bodyid[g] in robot or dojo.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if dojo.geom_contype[g] == 0 and dojo.geom_conaffinity[g] == 0:
            continue  # visual only (gantry, rails, hook)
        r = np.hypot(*d.geom_xpos[g, :2]) - dojo.geom_rbound[g]
        assert r > CLEAR_RADIUS_M, dojo.geom(g).name


@pytest.mark.assets
@needs_model
def test_robot_stands_3s_with_no_prop_contacts(dojo):
    from rth.sim.scene import robot_body_ids, standing_data

    d = standing_data(dojo)
    d.ctrl[:] = dojo.key_ctrl[dojo.key("stand").id]
    robot = robot_body_ids(dojo)
    pelvis = dojo.body("pelvis").id
    start = d.xpos[pelvis].copy()
    floor = dojo.geom("floor").id
    for _ in range(int(3.0 / dojo.opt.timestep)):
        mujoco.mj_step(dojo, d)
        for c in d.contact[: d.ncon]:
            ours = [dojo.geom_bodyid[c.geom1] in robot, dojo.geom_bodyid[c.geom2] in robot]
            if any(ours) and not all(ours):
                other = c.geom2 if ours[0] else c.geom1
                assert other == floor, dojo.geom(other).name
    assert np.linalg.norm(d.xpos[pelvis] - start) < 0.01


@pytest.mark.assets
@needs_model
def test_props_are_dynamic_and_settle(dojo):
    from rth.sim.scene import PROP_BODIES, prop_body_ids, standing_data

    props = prop_body_ids(dojo)
    assert {dojo.body(b).name for b in props} == set(PROP_BODIES)
    for b in props:
        assert dojo.body_dofnum[b] > 0, dojo.body(b).name
        assert 1.0 <= dojo.body_subtreemass[b] <= 60.0
    d = standing_data(dojo)
    d.ctrl[:] = dojo.key_ctrl[dojo.key("stand").id]
    start = {b: d.xpos[b].copy() for b in props}
    for _ in range(int(3.0 / dojo.opt.timestep)):
        mujoco.mj_step(dojo, d)
    for b in props:
        assert np.linalg.norm(d.xpos[b] - start[b]) < 0.02, dojo.body(b).name
        assert np.linalg.norm(d.cvel[b]) < 0.05, dojo.body(b).name


def _swing_speed(m, d, dof) -> float:
    return float(np.linalg.norm(d.qvel[dof : dof + 3]))


@pytest.mark.assets
@needs_model
def test_heavy_bag_swings_and_damps(dojo):
    from rth.sim.scene import standing_data

    d = standing_data(dojo)
    d.ctrl[:] = dojo.key_ctrl[dojo.key("stand").id]
    bag = dojo.body("heavy_bag").id
    dof = dojo.jnt_dofadr[dojo.body_jntadr[bag]]
    assert dojo.jnt_type[dojo.body_jntadr[bag]] == mujoco.mjtJoint.mjJNT_BALL
    d.qvel[dof] = 1.0  # a good punch: about 1 m/s at the bag's middle
    low = d.xpos[bag].copy()
    peak = 0.0
    for _ in range(int(2.0 / dojo.opt.timestep)):
        mujoco.mj_step(dojo, d)
        bag_mid = d.geom_xpos[dojo.geom("bag_body").id]
        peak = max(peak, float(np.hypot(*(bag_mid[:2] - low[:2]))))
    assert peak > 0.2  # it really swings
    for _ in range(int(16.0 / dojo.opt.timestep)):
        mujoco.mj_step(dojo, d)
    assert _swing_speed(dojo, d, dof) < 0.1  # and has mostly died away 18 s later


@pytest.mark.assets
@needs_model
def test_booth_camera_matches_free_camera(dojo):
    from rth.sim.scene import booth_mjv_camera, standing_data

    rs = SETTINGS.robot
    d = standing_data(dojo)
    with mujoco.Renderer(dojo, height=180, width=160) as r:
        r.update_scene(d, camera="booth")
        named = r.render().astype(float)
        r.update_scene(d, camera=booth_mjv_camera(rs))
        free = r.render().astype(float)
    assert np.mean(np.abs(named - free)) < 2.0


def _seg(model, data, camera: str, w: int, h: int) -> np.ndarray:
    with mujoco.Renderer(model, height=h, width=w) as r:
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=camera)
        return r.render()


@pytest.mark.assets
@needs_model
def test_robot_full_body_and_unobstructed_in_booth_view(dojo, plain):
    """For every stance: robot fully in frame, no prop in front of it or next to its outline."""
    import cv2

    from rth.robot.model import load_model
    from rth.robot.poses import PoseBuilder
    from rth.sim.scene import prop_body_ids, robot_body_ids, standing_data
    from rth.stances import load_stances

    w, h = 320, 360
    builder = PoseBuilder(
        load_model(project_path(SETTINGS.robot.model_path)), SETTINGS.robot.clips_dir
    )
    poses = [builder.stance(s) for s in load_stances(SETTINGS.stances_path).stances if s.pose]
    robot, props = robot_body_ids(dojo), prop_body_ids(dojo)
    geom = mujoco.mjtObj.mjOBJ_GEOM
    d_dojo, d_plain = standing_data(dojo), standing_data(plain)
    for q in poses:
        n = len(q)
        counts = []
        for m, d in ((dojo, d_dojo), (plain, d_plain)):
            d.qpos[:n] = q
            mujoco.mj_forward(m, d)
            seg = _seg(m, d, "booth", w, h)
            is_geom = seg[..., 1] == geom
            ids = np.where(is_geom, seg[..., 0], -1)
            body = np.where(ids >= 0, m.geom_bodyid[np.clip(ids, 0, None)], -1)
            mask = np.isin(body, list(robot))
            counts.append(int(mask.sum()))
            if m is dojo:
                border = np.concatenate(
                    [mask[:3].ravel(), mask[-3:].ravel(), mask[:, :3].ravel(), mask[:, -3:].ravel()]
                )
                assert not border.any(), "robot cut off by the frame"
                near = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                assert not np.isin(body[near], list(props)).any(), "prop next to the robot"
        assert counts[1] > 0.05 * w * h  # the robot really is in the picture
        assert counts[0] >= 0.99 * counts[1], "something in front of the robot"
    assert len(poses) >= 7


@pytest.mark.assets
@needs_model
def test_render_shape_and_time_vs_plain():
    from rth.sim.scene import booth_frame_ms, booth_render, build_model, standing_data

    rs = SETTINGS.robot
    m = build_model(settings_for("dojo"))
    with mujoco.Renderer(m, height=rs.render_height, width=rs.render_width) as r:
        img = booth_render(r, standing_data(m), rs)
    assert img.shape == (rs.render_height, rs.render_width, 3) and img.dtype == np.uint8
    # Interleaved, best of two each: the laptop's load drifts more than the scenes differ.
    plain_ms = min(booth_frame_ms(settings_for("plain"), 25) for _ in range(2))
    dojo_ms = min(booth_frame_ms(settings_for("dojo"), 25) for _ in range(2))
    assert dojo_ms < 1.3 * plain_ms + 2.0, (dojo_ms, plain_ms)
