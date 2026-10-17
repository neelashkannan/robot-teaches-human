"""The physics world: the dojo scene (or the plain floor), the safety harness, and G1 lookups.

The dojo comes from ``rth.sim.scene.build_spec`` when it loads; otherwise the
Menagerie scene is used as is, so the controller never depends on it. The
harness is a rope from an overhead anchor to a lifting eye on the torso: a
limited spatial tendon, slack while the robot stands and taut only if it drops.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from rth.config import Settings, project_path
from rth.contracts import Joint
from rth.robot.model import (
    FOOT_BODIES,
    JOINT_NAMES,
    KEYPOINT_BODIES,
    brighten_headlight,
    configure_offscreen,
)

ANCHOR_SITE = "gantry_anchor"
HARNESS_SITE = "harness_eye"
ROPE = "harness_rope"
# Lifting eye on the upper back, below the head (torso_link frame).
HARNESS_EYE_POS = (-0.06, 0.0, 0.40)
ROPE_RGBA = (0.82, 0.72, 0.52, 1.0)
ROPE_WIDTH = 0.007


def load_spec(settings: Settings) -> tuple[mujoco.MjSpec, str]:
    """(spec, scene name actually used). Falls back to the plain Menagerie scene."""
    if settings.sim.scene != "plain":
        try:
            from rth.sim.scene import build_spec

            return build_spec(settings), settings.sim.scene
        except Exception as e:  # noqa: BLE001 - any dojo problem must not stop the robot
            print(f"[sim] dojo scene unavailable ({type(e).__name__}: {e}); using the plain floor")
    rs = settings.robot
    spec = mujoco.MjSpec.from_file(str(project_path(rs.model_path)))
    configure_offscreen(spec, (max(1280, rs.render_width), max(1280, rs.render_height)), rs.render_samples)
    return spec, "plain"


def _find_site(spec: mujoco.MjSpec, name: str) -> mujoco.MjsSite | None:
    for s in spec.sites:
        if s.name == name:
            return s
    return None


def add_harness(spec: mujoco.MjSpec, anchor_z: float, length: float) -> None:
    """Rope from the anchor (the dojo's, or one added here) to the torso's lifting eye."""
    if _find_site(spec, ANCHOR_SITE) is None:
        spec.worldbody.add_site(name=ANCHOR_SITE, pos=[0.0, 0.0, anchor_z], size=[0.02, 0, 0],
                                rgba=[0.6, 0.6, 0.6, 1.0])
    torso = spec.body("torso_link")
    torso.add_site(name=HARNESS_SITE, pos=list(HARNESS_EYE_POS), size=[0.012, 0, 0], rgba=[0.3, 0.3, 0.3, 1.0],
                   group=0)
    rope = spec.add_tendon(name=ROPE)
    rope.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    rope.range = [0.0, length]
    rope.width = ROPE_WIDTH
    rope.rgba = list(ROPE_RGBA)
    # A little stretch, like a real lanyard, so the catch isn't a hard jolt.
    rope.solref_limit = [0.04, 1.0]
    rope.wrap_site(ANCHOR_SITE)
    rope.wrap_site(HARNESS_SITE)


def build_world(settings: Settings, rope_length: float | None) -> tuple[mujoco.MjModel, str]:
    """Compiled sim model; ``rope_length`` None leaves the harness out."""
    spec, scene = load_spec(settings)
    if rope_length is not None:
        add_harness(spec, settings.sim.harness_anchor_z, rope_length)
    model = spec.compile()
    if scene == "plain":
        brighten_headlight(model)
    return model, scene


@dataclass(frozen=True)
class G1Map:
    """Where the G1's state lives inside a (possibly larger) sim model.

    ``qpos``/``qvel`` index the sim arrays in the G1-only layout (36 and 35 long),
    so G1 poses built on the bare model drop straight in.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    actuators: np.ndarray  # (29,) actuator id per joint, JOINT_NAMES order
    kp: np.ndarray  # (29,) position gain of each actuator
    force_limit: np.ndarray  # (29,) |torque| the joint can deliver
    ctrl_lo: np.ndarray
    ctrl_hi: np.ndarray
    pelvis: int
    torso: int
    feet: tuple[int, int]
    foot_geoms: tuple[frozenset[int], frozenset[int]]  # sole spheres per foot
    robot_bodies: frozenset[int]
    keypoint_bodies: dict[Joint, int]
    rope: int  # tendon id, -1 without a harness
    eye_site: int
    anchor_site: int

    @staticmethod
    def of(model: mujoco.MjModel) -> G1Map:
        root = model.joint("floating_base_joint")
        qadr = [int(root.qposadr[0]) + i for i in range(7)]
        vadr = [int(root.dofadr[0]) + i for i in range(6)]
        acts = np.zeros(len(JOINT_NAMES), int)
        by_joint = {
            int(model.actuator_trnid[a, 0]): a
            for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
        }
        for i, name in enumerate(JOINT_NAMES):
            j = model.joint(name)
            qadr.append(int(j.qposadr[0]))
            vadr.append(int(j.dofadr[0]))
            acts[i] = by_joint[j.id]
        jids = model.actuator_trnid[acts, 0]
        limit = np.where(
            model.jnt_actfrclimited[jids], np.abs(model.jnt_actfrcrange[jids]).min(axis=1), np.inf
        )
        pelvis = model.body("pelvis").id
        robot = frozenset(b for b in range(model.nbody) if model.body_rootid[b] == pelvis)
        feet = tuple(model.body(n).id for n in FOOT_BODIES)
        spheres = tuple(
            frozenset(
                g
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == f and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            )
            for f in feet
        )
        rope = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, ROPE)
        return G1Map(
            qpos=np.array(qadr),
            qvel=np.array(vadr),
            actuators=acts,
            kp=model.actuator_gainprm[acts, 0].copy(),
            force_limit=limit,
            ctrl_lo=model.actuator_ctrlrange[acts, 0].copy(),
            ctrl_hi=model.actuator_ctrlrange[acts, 1].copy(),
            pelvis=pelvis,
            torso=model.body("torso_link").id,
            feet=(feet[0], feet[1]),
            foot_geoms=(spheres[0], spheres[1]),
            robot_bodies=robot,
            keypoint_bodies={j: model.body(b).id for j, b in KEYPOINT_BODIES.items()},
            rope=rope,
            eye_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, HARNESS_SITE),
            anchor_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ANCHOR_SITE),
        )
