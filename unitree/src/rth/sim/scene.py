"""The G1's physics world: Menagerie's G1 in a small dojo, built with MjSpec.

``build_spec(settings)`` returns an uncompiled spec so the controller can add
its own parts (the safety-harness rope) before compiling; ``build_model``
compiles it as is. ``settings.sim.scene`` picks ``"dojo"`` or ``"plain"``
(Menagerie's checker floor only). Both have the ``booth`` and ``wide`` cameras.

Dojo layout (robot at the origin facing +x, the booth camera in front of it):

* tatami floor, top at z = 0; back wall at x = -3.6, side walls at y = +-4,
  open towards +x so the cameras can look in;
* a wooden gantry over the robot (posts at y = +-1.8, anchor 2.6 m up);
* props kept outside a 1.5 m circle and outside the booth camera's view of the
  robot: a 35 kg heavy bag on a ball joint (+y), a springy wooden dummy (-y),
  a stack of light foam blocks and a roller (front right).

Collision bits: the robot and floor use bit 1; walls use bit 2 only, so the
robot never pays for wall contacts; props use both, so they hit everything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from rth.config import RobotSettings, Settings, project_path
from rth.sim.textures import MAT_L, MAT_W, SHOJI_BAY, WOOD_TILE_M, ensure_textures

SCENES = ("dojo", "plain")
ROBOT_ROOT = "pelvis"
CLEAR_RADIUS_M = 1.5  # nothing but the floor (and the harness rope) inside this circle
ANCHOR_POS = (0.0, 0.0, 2.6)  # harness anchor under the gantry beam
ANCHOR_SITE = "gantry_anchor"

BACK_X, SIDE_Y, FRONT_X = -3.6, 4.0, 4.4  # room extents; the side walls end at FRONT_X
FLOOR_FRONT_X = 5.4  # the tatami runs on past the walls, under the wide camera
WALL_H, SILL_Z, LINTEL_Z = 3.6, 0.9, 2.7

ROBOT_BITS, WALL_BITS, PROP_BITS = 1, 2, 3

PROP_BODIES = (
    "heavy_bag",
    "wooden_dummy",
    "foam_block_left",
    "foam_block_right",
    "foam_block_top",
    "foam_cube",
    "foam_roller",
)

WIDE_CAMERA = {"pos": (5.6, 3.3, 3.0), "lookat": (-0.9, -0.5, 0.95), "fovy": 58.0}


# --------------------------------------------------------------------------- small helpers


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def frame_quat(x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """wxyz quaternion of the frame whose x and y axes are given (z = x cross y)."""
    x = _unit(x_axis)
    y = _unit(np.asarray(y_axis, float) - np.dot(y_axis, x) * x)
    z = np.cross(x, y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


def look_at_quat(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera orientation looking from ``pos`` at ``target`` with world z up.

    MuJoCo cameras look along their -z axis with +y up in the image.
    """
    fwd = _unit(np.asarray(target, float) - np.asarray(pos, float))
    right = _unit(np.cross(fwd, (0.0, 0.0, 1.0)))
    return frame_quat(right, np.cross(right, fwd))


def free_camera_pose(rs: RobotSettings) -> tuple[np.ndarray, np.ndarray]:
    """Position and look-at target of the booth view the kinematic teacher uses.

    Same convention as ``mjvCamera``: the camera sits ``distance`` back from
    ``lookat`` along the azimuth/elevation viewing direction.
    """
    az, el = math.radians(rs.camera_azimuth_deg), math.radians(rs.camera_elevation_deg)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    lookat = np.array([0.0, 0.0, rs.camera_lookat_z])
    return lookat - rs.camera_distance * fwd, lookat


def booth_mjv_camera(rs: RobotSettings) -> mujoco.MjvCamera:
    """Free camera equal to the ``booth`` camera (for renderers that take an MjvCamera)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, rs.camera_lookat_z)
    cam.distance = rs.camera_distance
    cam.azimuth = rs.camera_azimuth_deg
    cam.elevation = rs.camera_elevation_deg
    return cam


@dataclass(frozen=True)
class Surface:
    """Collision settings for one kind of geom."""

    contype: int
    conaffinity: int
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    solref: tuple[float, float] = (0.02, 1.0)


VISUAL = Surface(0, 0)
WALL = Surface(WALL_BITS, WALL_BITS)
PROP = Surface(PROP_BITS, PROP_BITS, (0.8, 0.005, 0.0001))
FOAM = Surface(PROP_BITS, PROP_BITS, (0.9, 0.01, 0.001), (0.04, 1.0))  # soft: sinks in a little


def _geom(
    parent: mujoco.MjsBody,
    name: str,
    kind: mujoco.mjtGeom,
    size,
    *,
    surface: Surface,
    pos=(0, 0, 0),
    quat=None,
    fromto=None,
    material: str = "",
    rgba=None,
    mass: float | None = None,
    group: int = 0,
) -> mujoco.MjsGeom:
    g = parent.add_geom(name=name, type=kind, size=list(size) + [0.0] * (3 - len(size)))
    if fromto is not None:
        g.fromto = list(fromto)
    else:
        g.pos = list(pos)
        if quat is not None:
            g.quat = list(quat)
    g.contype, g.conaffinity = surface.contype, surface.conaffinity
    g.friction = list(surface.friction)
    g.solref = list(surface.solref)
    g.group = group
    if material:
        g.material = material
    if rgba is not None:
        g.rgba = list(rgba)
    if mass is not None:
        g.mass = mass
    return g


def _box(parent, name, center, half, *, surface=VISUAL, **kw) -> mujoco.MjsGeom:
    return _geom(parent, name, mujoco.mjtGeom.mjGEOM_BOX, half, surface=surface, pos=center, **kw)


# --------------------------------------------------------------------------- spec entry points


def build_spec(settings: Settings) -> mujoco.MjSpec:
    """The G1 in the scene ``settings.sim.scene`` picks, uncompiled."""
    scene = settings.sim.scene
    if scene not in SCENES:
        raise ValueError(f"unknown sim scene {scene!r}; use one of {SCENES}")
    spec = mujoco.MjSpec.from_file(str(project_path(settings.robot.model_path)))
    _render_settings(spec, settings.robot)
    if scene == "dojo":
        add_dojo(spec)
        spec.modelname = "G1 dojo"  # the viewer's window title
    _add_cameras(spec, settings.robot)
    return spec


def build_model(settings: Settings) -> mujoco.MjModel:
    return build_spec(settings).compile()


def _render_settings(spec: mujoco.MjSpec, rs: RobotSettings) -> None:
    g = spec.visual.global_
    # Room for the 640x720 booth frame and 1280x720 stills of the wide camera.
    g.offwidth = max(g.offwidth, 1280, rs.render_width)
    g.offheight = max(g.offheight, 1280, rs.render_height)
    spec.visual.quality.offsamples = rs.render_samples


def _add_cameras(spec: mujoco.MjSpec, rs: RobotSettings) -> None:
    pos, lookat = free_camera_pose(rs)
    spec.worldbody.add_camera(
        name="booth",
        pos=list(pos),
        quat=list(look_at_quat(pos, lookat)),
        fovy=float(spec.visual.global_.fovy),
    )
    w = WIDE_CAMERA
    spec.worldbody.add_camera(
        name="wide",
        pos=list(w["pos"]),
        quat=list(look_at_quat(w["pos"], w["lookat"])),
        fovy=w["fovy"],
    )


# --------------------------------------------------------------------------- the dojo


def add_dojo(spec: mujoco.MjSpec) -> None:
    """Turn Menagerie's G1 scene into the dojo (floor, walls, light, gantry, props)."""
    # Props at rest go to sleep, so a quiet dojo costs about what the plain floor does.
    # The actuated robot never sleeps; a contact or applied force wakes a prop.
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    _materials(spec)
    _atmosphere(spec)
    _floor(spec)
    _walls(spec)
    _gantry(spec)
    _heavy_bag(spec)
    _wooden_dummy(spec)
    _foam(spec)


def _materials(spec: mujoco.MjSpec) -> None:
    tex = ensure_textures()
    # name -> (texture, metres per repeat (u, v) or None to stretch once over the geom)
    textured = {
        "dojo_tatami": ("tatami", (2 * MAT_L, 2 * MAT_W)),
        "dojo_wood": ("wood", (WOOD_TILE_M, WOOD_TILE_M)),
        "dojo_dark_wood": ("dark_wood", (WOOD_TILE_M, WOOD_TILE_M)),
        "dojo_shoji": ("shoji", SHOJI_BAY),
        "dojo_plaster": ("plaster", (1.5, 1.5)),
        "dojo_scroll": ("scroll", None),
    }
    for mat_name, (tex_name, tile) in textured.items():
        spec.add_texture(
            name=f"{mat_name}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex[tex_name])
        )
        m = spec.add_material(name=mat_name)
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = f"{mat_name}_tex"
        if tile is None:
            m.texuniform, m.texrepeat = False, [1.0, 1.0]
        else:
            m.texuniform, m.texrepeat = True, [1.0 / tile[0], 1.0 / tile[1]]
        m.specular, m.shininess, m.reflectance = 0.08, 0.1, 0.0
    spec.material("dojo_shoji").emission = 0.12  # daylight through the paper
    flat = {
        "dojo_leather": ((0.36, 0.13, 0.09, 1), 0.25),
        "dojo_steel": ((0.22, 0.22, 0.23, 1), 0.5),
        "dojo_foam_blue": ((0.27, 0.38, 0.52, 1), 0.05),
        "dojo_foam_clay": ((0.70, 0.40, 0.28, 1), 0.05),
        "dojo_foam_sand": ((0.80, 0.70, 0.50, 1), 0.05),
    }
    for name, (rgba, spec_) in flat.items():
        spec.add_material(name=name, rgba=list(rgba), specular=spec_, shininess=0.3)


def _atmosphere(spec: mujoco.MjSpec) -> None:
    """Warm key light, cool fill, dim headlight; a dark warm sky in place of the blue one."""
    for t in spec.textures:
        if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            t.rgb1, t.rgb2 = [0.16, 0.12, 0.09], [0.03, 0.025, 0.02]
    spec.visual.rgba.haze = [0.12, 0.09, 0.07, 1.0]
    h = spec.visual.headlight
    h.diffuse, h.ambient, h.specular = [0.32, 0.31, 0.30], [0.22, 0.20, 0.18], [0.1, 0.1, 0.1]
    # Menagerie's overhead light becomes the key light: high, in front, to the robot's left.
    # It is the only one that casts shadows (and only where shadows are switched on).
    # A spot just under the rafters, so the rafters and walls don't stripe the room with shadow.
    key = spec.lights[0]
    key.type = mujoco.mjtLightType.mjLIGHT_SPOT
    key_pos = np.array([3.4, 1.4, 3.35])
    key.pos, key.dir = list(key_pos), list(_unit(np.array([-0.6, -0.3, 0.5]) - key_pos))
    key.cutoff, key.exponent, key.attenuation = 70.0, 3.0, [1.0, 0.0, 0.0]
    key.diffuse, key.specular, key.ambient = [0.70, 0.62, 0.50], [0.15, 0.14, 0.12], [0, 0, 0]
    key.castshadow = True
    spec.worldbody.add_light(
        name="fill",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[2.0, -3.0, 3.0],
        dir=list(_unit((-0.5, 0.65, -0.55))),
        diffuse=[0.18, 0.20, 0.24],
        specular=[0, 0, 0],
        castshadow=False,
    )
    # The extent also sizes the directional shadow map: it must cover the whole room.
    spec.stat.center, spec.stat.extent = [0.4, 0.0, 1.2], 7.0


def _floor(spec: mujoco.MjSpec) -> None:
    """Same plane as Menagerie's (z = 0, same friction), tatami-textured and sized to the room."""
    f = spec.geom("floor")
    f.material = "dojo_tatami"
    f.size = [(FLOOR_FRONT_X - BACK_X) / 2, SIDE_Y, 0.05]
    f.pos = [(FLOOR_FRONT_X + BACK_X) / 2, 0.0, 0.0]


def _wall(spec: mujoco.MjSpec, name: str, origin, x_axis, half_len: float) -> None:
    """One wall as three textured planes (wainscot, shoji, plaster) facing into the room."""
    wb = spec.worldbody
    q = frame_quat(np.asarray(x_axis, float), np.array([0.0, 0.0, 1.0]))
    o = np.asarray(origin, float)
    bands = (
        ("wainscot", 0.0, SILL_Z, "dojo_dark_wood"),
        ("shoji", SILL_Z, LINTEL_Z, "dojo_shoji"),
        ("frieze", LINTEL_Z, WALL_H, "dojo_plaster"),
    )
    for band, z0, z1, mat in bands:
        surface = WALL if band == "wainscot" else VISUAL  # one collider per wall is enough
        _geom(
            wb,
            f"{name}_{band}",
            mujoco.mjtGeom.mjGEOM_PLANE,
            (half_len, (z1 - z0) / 2, 0.5),
            surface=surface,
            pos=o + (0, 0, (z0 + z1) / 2),
            quat=q,
            material=mat,
        )
    # Rails and baseboard in dark wood, proud of the wall so they catch the light.
    n = _unit(np.cross(x_axis, (0, 0, 1)))
    along = _unit(x_axis)
    for rail, z, hh in (("base", 0.05, 0.05), ("sill", SILL_Z, 0.035), ("lintel", LINTEL_Z, 0.06)):
        half = np.abs(along) * half_len + np.abs(n) * 0.03 + (0, 0, hh)
        _box(wb, f"{name}_{rail}", o + n * 0.03 + (0, 0, z), half, material="dojo_dark_wood")


def _pillar(spec: mujoco.MjSpec, name: str, x: float, y: float) -> None:
    _box(spec.worldbody, name, (x, y, WALL_H / 2), (0.07, 0.07, WALL_H / 2), material="dojo_wood")


def _walls(spec: mujoco.MjSpec) -> None:
    mid_x, half_x = (FRONT_X + BACK_X) / 2, (FRONT_X - BACK_X) / 2
    _wall(spec, "wall_back", (BACK_X, 0.0, 0.0), (0, 1, 0), SIDE_Y)
    _wall(spec, "wall_left", (mid_x, SIDE_Y, 0.0), (1, 0, 0), half_x)
    _wall(spec, "wall_right", (mid_x, -SIDE_Y, 0.0), (-1, 0, 0), half_x)
    for i, (x, y) in enumerate(
        [
            (BACK_X, SIDE_Y),
            (BACK_X, -SIDE_Y),
            (BACK_X, 2.7),
            (BACK_X, -2.7),
            (0.0, SIDE_Y),
            (0.0, -SIDE_Y),
            (FRONT_X, SIDE_Y),
            (FRONT_X, -SIDE_Y),
        ]
    ):
        _pillar(spec, f"pillar_{i}", x, y)
    # Rafters across the open top; they give the dark sky above some structure.
    for i, x in enumerate(np.arange(BACK_X + 0.6, FRONT_X + 0.01, 1.2)):
        _box(
            spec.worldbody,
            f"rafter_{i}",
            (x, 0.0, WALL_H + 0.08),
            (0.06, SIDE_Y, 0.08),
            material="dojo_dark_wood",
        )
    _box(
        spec.worldbody,
        "ridge_beam",
        (0.4, 0.0, WALL_H + 0.24),
        (4.0, 0.08, 0.08),
        material="dojo_dark_wood",
    )
    # A hanging scroll on the back wall, to the right of the robot in the booth view.
    _geom(
        spec.worldbody,
        "scroll",
        mujoco.mjtGeom.mjGEOM_PLANE,
        (0.25, 0.65, 0.5),
        surface=VISUAL,
        pos=(BACK_X + 0.012, 1.15, 1.72),
        quat=frame_quat((0, 1, 0), (0, 0, 1)),
        material="dojo_scroll",
    )
    _box(
        spec.worldbody,
        "scroll_rod",
        (BACK_X + 0.02, 1.15, 1.05),
        (0.015, 0.29, 0.015),
        material="dojo_dark_wood",
    )


def _gantry(spec: mujoco.MjSpec) -> None:
    """The harness frame: two posts and a beam, with the anchor site under the beam's middle."""
    wb = spec.worldbody
    post_y, top = 1.8, ANCHOR_POS[2] + 0.18
    for side, y in (("left", post_y), ("right", -post_y)):
        _box(
            wb, f"gantry_post_{side}", (0, y, top / 2), (0.07, 0.07, top / 2), material="dojo_wood"
        )
        _box(wb, f"gantry_foot_{side}", (0, y, 0.03), (0.25, 0.12, 0.03), material="dojo_dark_wood")
    _box(wb, "gantry_beam", (0, 0, top + 0.06), (0.09, post_y + 0.25, 0.08), material="dojo_wood")
    _box(wb, "gantry_tie", (0, 0, top - 0.25), (0.05, post_y, 0.05), material="dojo_dark_wood")
    hook = (ANCHOR_POS[0], ANCHOR_POS[1])
    _geom(
        wb,
        "gantry_hook",
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (0.012,),
        surface=VISUAL,
        fromto=(*hook, top - 0.20, *hook, ANCHOR_POS[2] + 0.015),
        material="dojo_steel",
    )
    wb.add_site(name=ANCHOR_SITE, pos=list(ANCHOR_POS), size=[0.025, 0, 0], rgba=[0.6, 0.6, 0.6, 1])


def _heavy_bag(spec: mujoco.MjSpec) -> None:
    """35 kg bag on a 0.5 m chain from a wall bracket; a damped ball joint at the bracket.

    Damping 16 N m s/rad gives a damping ratio of about 0.06: it swings for a
    while and settles within roughly 20 s.
    """
    wb = spec.worldbody
    px, py, pz = -1.3, 2.5, 2.6
    _box(
        wb,
        "bag_bracket",
        (px, (py + SIDE_Y) / 2 - 0.05, pz + 0.07),
        (0.05, (SIDE_Y - py) / 2 + 0.1, 0.05),
        material="dojo_steel",
    )
    _geom(
        wb,
        "bag_brace",
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (0.02,),
        surface=VISUAL,
        fromto=(px, py + 0.5, pz + 0.07, px, SIDE_Y - 0.02, pz - 0.6),
        material="dojo_steel",
    )
    bag = wb.add_body(name="heavy_bag", pos=[px, py, pz])
    bag.add_joint(
        name="heavy_bag_swing",
        type=mujoco.mjtJoint.mjJNT_BALL,
        damping=[16.0, 0.0, 0.0],
        armature=0.01,
    )
    chain, length, radius = 0.5, 1.2, 0.17
    _geom(
        bag,
        "bag_chain",
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (0.01,),
        surface=VISUAL,
        fromto=(0, 0, 0, 0, 0, -chain),
        material="dojo_steel",
        mass=1.0,
    )
    _geom(
        bag,
        "bag_body",
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (radius, length / 2 - radius),
        surface=PROP,
        pos=(0, 0, -chain - length / 2),
        material="dojo_leather",
        mass=35.0,
    )
    _geom(
        bag,
        "bag_cap",
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (radius * 0.9, 0.03),
        surface=VISUAL,
        pos=(0, 0, -chain - 0.06),
        rgba=(0.12, 0.10, 0.09, 1),
        mass=0.1,
    )


def _wooden_dummy(spec: mujoco.MjSpec) -> None:
    """A wing chun dummy: pine trunk, three arms and a leg, on a stiff springy mount.

    Two hinges (pitch, roll) with stiffness 5000 N m/rad and damping 100 N m s/rad:
    a firm push tips it a few degrees and it rings back at about 2 Hz.
    """
    wb = spec.worldbody
    x, y = -1.6, -3.3
    yaw = math.atan2(-y, -x)  # arms face the middle of the room
    face = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    side = np.array([-face[1], face[0], 0.0])
    _box(wb, "dummy_base", (x, y, 0.025), (0.32, 0.32, 0.025), material="dojo_dark_wood")
    _geom(
        wb,
        "dummy_collar",
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (0.15, 0.04),
        surface=VISUAL,
        pos=(x, y, 0.09),
        material="dojo_dark_wood",
    )
    body = spec.worldbody.add_body(name="wooden_dummy", pos=[x, y, 0.1])
    for axis_name, axis in (("pitch", side), ("roll", face)):
        body.add_joint(
            name=f"wooden_dummy_{axis_name}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=list(axis),
            stiffness=[5000.0, 0.0, 0.0],
            damping=[100.0, 0.0, 0.0],
            armature=0.2,
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            range=[-0.15, 0.15],
        )
    r, h = 0.12, 1.55
    _geom(
        body,
        "dummy_trunk",
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (r, h / 2),
        surface=PROP,
        pos=(0, 0, h / 2),
        material="dojo_wood",
        mass=38.0,
    )

    def peg(name: str, z: float, direction: np.ndarray, length: float) -> None:
        d = _unit(direction)
        start = np.array([0, 0, z]) + d * (r - 0.02)
        _geom(
            body,
            name,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            (0.028,),
            surface=PROP,
            fromto=(*start, *(start + d * length)),
            material="dojo_wood",
            mass=0.9,
        )

    spread = math.radians(22)
    peg(
        "dummy_arm_left",
        1.18,
        face * math.cos(spread) + side * math.sin(spread) + (0, 0, 0.12),
        0.48,
    )
    peg(
        "dummy_arm_right",
        1.18,
        face * math.cos(spread) - side * math.sin(spread) + (0, 0, 0.12),
        0.48,
    )
    peg("dummy_arm_middle", 0.95, face, 0.42)
    peg("dummy_leg", 0.45, face - (0, 0, 0.6), 0.5)


def _foam(spec: mujoco.MjSpec) -> None:
    """Free-floating foam (about 25 kg/m^3): a 2 + 1 stack of blocks, a cube and a roller.

    Each starts 1 mm above its support so it settles without a jolt. The small
    free-joint damping stands in for air drag, which matters for foam this light.
    """
    wb = spec.worldbody
    cx, cy = 1.7, -2.9
    bx, by, bz = 0.25, 0.25, 0.15  # half sizes of a 0.5 x 0.5 x 0.3 m block
    gap = 0.001

    def block(
        name: str, pos, half, mat: str, mass: float, kind=mujoco.mjtGeom.mjGEOM_BOX, quat=None
    ) -> None:
        b = wb.add_body(name=name, pos=list(pos))
        if quat is not None:
            b.quat = list(quat)
        b.add_freejoint(name=f"{name}_free")
        b.joints[0].damping = [0.02, 0.0, 0.0]
        _geom(b, f"{name}_geom", kind, half, surface=FOAM, material=mat, mass=mass)

    block("foam_block_left", (cx, cy + by + 0.01, bz + gap), (bx, by, bz), "dojo_foam_blue", 2.0)
    block("foam_block_right", (cx, cy - by - 0.01, bz + gap), (bx, by, bz), "dojo_foam_clay", 2.0)
    block("foam_block_top", (cx, cy, 3 * bz + 2 * gap), (bx, by, bz), "dojo_foam_sand", 2.0)
    block(
        "foam_cube",
        (cx - 0.75, cy - 0.35, 0.2 + gap),
        (0.2, 0.2, 0.2),
        "dojo_foam_blue",
        1.5,
        quat=frame_quat((math.cos(0.4), math.sin(0.4), 0), (-math.sin(0.4), math.cos(0.4), 0)),
    )
    # Lying along y, so a push from the robot's side rolls it.
    block(
        "foam_roller",
        (cx - 0.9, cy + 0.55, 0.15 + gap),
        (0.15, 0.45),
        "dojo_foam_clay",
        1.5,
        kind=mujoco.mjtGeom.mjGEOM_CYLINDER,
        quat=frame_quat((1, 0, 0), (0, 0, 1)),
    )


# --------------------------------------------------------------------------- queries


def robot_body_ids(model: mujoco.MjModel) -> set[int]:
    """Bodies of the G1 (everything under the pelvis)."""
    root = model.body(ROBOT_ROOT).id
    return {b for b in range(model.nbody) if model.body_rootid[b] == root}


def prop_body_ids(model: mujoco.MjModel) -> set[int]:
    """Dynamic dojo props present in this model (none in the plain scene)."""
    names = {model.body(b).name for b in range(model.nbody)}
    return {model.body(n).id for n in PROP_BODIES if n in names}


# --------------------------------------------------------------------------- stills and timing


def standing_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Data with the robot in Menagerie's "stand" keyframe and props where they start."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key("stand").id)
    mujoco.mj_forward(model, d)
    return d


def booth_render(renderer: mujoco.Renderer, data: mujoco.MjData, rs: RobotSettings) -> np.ndarray:
    """One RGB booth frame, with the same shadow and reflection flags the teacher uses."""
    renderer.update_scene(data, camera="booth")
    on = int(rs.render_shadows)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = on
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = on
    return renderer.render()


def booth_frame_ms(settings: Settings, frames: int = 60) -> float:
    """Median time of update_scene + render + read-back for one booth frame."""
    import time

    rs = settings.robot
    model = build_model(settings)
    data = standing_data(model)
    with mujoco.Renderer(model, height=rs.render_height, width=rs.render_width) as r:
        times = []
        for _ in range(frames + 5):
            t = time.perf_counter()
            booth_render(r, data, rs)
            times.append(time.perf_counter() - t)
    return float(np.median(times[5:]) * 1e3)  # the first frames include GL warm-up


def render_stills(settings: Settings, out_dir: str = "reports") -> dict[str, str]:
    """Write <scene>_booth.png (booth size) and <scene>_wide.png (1280x720, shadows on)."""
    import cv2

    rs = settings.robot
    model = build_model(settings)
    data = standing_data(model)
    out = project_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    name = settings.sim.scene
    with mujoco.Renderer(model, height=rs.render_height, width=rs.render_width) as r:
        paths["booth"] = str(out / f"{name}_booth.png")
        cv2.imwrite(paths["booth"], booth_render(r, data, rs)[:, :, ::-1])
    with mujoco.Renderer(model, height=720, width=1280) as r:
        r.update_scene(data, camera="wide")
        paths["wide"] = str(out / f"{name}_wide.png")
        cv2.imwrite(paths["wide"], r.render()[:, :, ::-1])
    return paths


def main(argv: list[str] | None = None) -> int:
    """``python -m rth.sim.scene``: dojo stills into reports/ and booth frame time, plain vs dojo."""
    from rth.config import load_settings

    settings = load_settings()
    settings.sim.scene = "dojo"
    for kind, path in render_stills(settings).items():
        print(f"{kind:6s} {path}")
    rs = settings.robot
    for scene in ("plain", "dojo", "plain", "dojo"):  # interleaved: the machine's load drifts
        settings.sim.scene = scene
        ms = booth_frame_ms(settings)
        print(f"booth {rs.render_width}x{rs.render_height} {scene:5s} {ms:6.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANCHOR_POS",
    "ANCHOR_SITE",
    "CLEAR_RADIUS_M",
    "PROP_BODIES",
    "SCENES",
    "add_dojo",
    "booth_frame_ms",
    "booth_mjv_camera",
    "booth_render",
    "build_model",
    "build_spec",
    "free_camera_pose",
    "look_at_quat",
    "prop_body_ids",
    "render_stills",
    "robot_body_ids",
    "standing_data",
]
