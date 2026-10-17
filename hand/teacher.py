"""The teacher: the Pollen AmazingHand in MuJoCo, driven by the shared 8-number HandVec.

Mapping rule (found by experiment: set ctrl, step 800 times, read the tip sites; matches the
upstream PythonExample servo code, where open = (-35, +35) deg and closed = (+90, -90) deg):
  Each finger has two servos, motor1 and motor2, in a parallel mechanism.
    curl   = the two servos move in OPPOSITION:  a = OPEN_RAD + curl * (CLOSED_RAD - OPEN_RAD)
    spread = the two servos move TOGETHER:       s = SPREAD_RAD * sign * spread
    ctrl   = [motor1, motor2] = [a + s, -a + s], clipped to the actuator range +-1.57 rad.
  OPEN_RAD = -0.61 (-35 deg), CLOSED_RAD = +1.57 (+90 deg). curl 0 = straight, 1 = fully curled.
  The all-zero ctrl is NOT the open hand (it is half curled); set_pose(zeros) is open.
  Spread sign (checked on the tip y coordinate): for index/middle/ring, positive ctrl moves the
  tip toward the thumb side (+y), which is HandVec's "+1 = toward the thumb". For the thumb,
  positive ctrl moves it toward the other fingers, so its sign is flipped: thumb spread +1 = away
  from the fingers (abducted). Spread only has an effect while the finger is not fully curled.

FINGER_TO_MJCF (checked on fingertip geometry at zero pose, not guessed):
  finger1, finger2, finger3 are a row along y (0.024, -0.015, -0.054) with the tips at z ~0.15;
  finger4 sits off to the side and lower (x 0.081, y 0.044, z 0.090) and is the only one on the
  thumb side, so it is the thumb. Next to the thumb, finger1 is the index, then finger2 middle,
  finger3 ring. Upstream agrees (servos 1-2 index, 3-4 middle, 5-6 ring, 7-8 thumb).
  Using the thumb with the same curl/spread law as the fingers works geometrically (its tip goes
  from open (0.11, 0.05, 0.04) to closed (0.01, 0.02, 0.10), next to the palm), but it moves on a
  different arc, so a "thumb curl" is only an approximation of a human thumb.

The red mocap target spheres and the floor of scene.xml are hidden in the loaded model (the file
on disk is not changed). Frames are rendered offscreen and returned as arrays; nothing is saved.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from hand.vec import FINGERS, N, clamp

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO / "models" / "AH_Right" / "mjcf" / "scene.xml"

# HandVec finger name -> MJCF finger number (finger{n}_motor{1,2}, tip{n}).
FINGER_TO_MJCF = {"thumb": 4, "index": 1, "middle": 2, "ring": 3}
# +1 for index/middle/ring (toward thumb side); -1 for the thumb (see header).
SPREAD_SIGN = {"thumb": -1.0, "index": 1.0, "middle": 1.0, "ring": 1.0}

OPEN_RAD = np.deg2rad(-35.0)
CLOSED_RAD = np.deg2rad(90.0)
SPREAD_RAD = 0.6
CTRL_LIMIT = 1.5707963267948966

# Free camera framing the whole hand (fingers point up the image, palm facing the viewer's side).
CAM_LOOKAT = (0.05, -0.01, 0.085)
CAM_DISTANCE = 0.38
CAM_AZIMUTH = 150.0
CAM_ELEVATION = -15.0


def vec_to_ctrl(vec) -> np.ndarray:
    """8 HandVec numbers -> 8 actuator targets in model actuator order."""
    v = clamp(vec)
    ctrl = np.zeros(N)
    for i, finger in enumerate(FINGERS):
        a = OPEN_RAD + v[2 * i] * (CLOSED_RAD - OPEN_RAD)
        s = SPREAD_RAD * SPREAD_SIGN[finger] * v[2 * i + 1]
        n = FINGER_TO_MJCF[finger]
        ctrl[2 * (n - 1)] = a + s
        ctrl[2 * (n - 1) + 1] = -a + s
    return np.clip(ctrl, -CTRL_LIMIT, CTRL_LIMIT)


class Teacher:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._hide_helpers()
        self._act = [self.model.actuator(f"finger{n}_motor{k}").id for n in range(1, 5) for k in (1, 2)]
        self._tip = {f: self.model.site(f"tip{n}").id for f, n in FINGER_TO_MJCF.items()}
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = CAM_LOOKAT
        self.cam.distance = CAM_DISTANCE
        self.cam.azimuth = CAM_AZIMUTH
        self.cam.elevation = CAM_ELEVATION
        self._renderer = None
        self._size = None
        self.set_pose(np.zeros(N))
        self.step(400)

    def _hide_helpers(self) -> None:
        m = self.model
        for g in range(m.ngeom):
            body = m.geom_bodyid[g]
            if m.body_mocapid[body] >= 0 or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                m.geom_rgba[g, 3] = 0.0

    def set_pose(self, vec) -> None:
        ctrl = vec_to_ctrl(vec)
        for k, a in enumerate(self._act):
            self.data.ctrl[a] = ctrl[k]

    def step(self, n: int = 200) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def tip_positions(self) -> dict[str, np.ndarray]:
        """World position of each fingertip, keyed by HandVec finger name."""
        return {f: self.data.site_xpos[s].copy() for f, s in self._tip.items()}

    def render(self, w: int = 480, h: int = 480) -> np.ndarray:
        if self._renderer is None or self._size != (w, h):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._size = (w, h)
        self._renderer.update_scene(self.data, camera=self.cam)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
