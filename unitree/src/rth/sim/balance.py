"""Balance: capture-point feedback on the COM, turned into ankle and hip corrections.

The capture point xi = c + c_dot / omega says where the robot would have to put its
weight to come to rest. Its error against the plan is fed back as a COM shift
relative to the planted feet. Small shifts use the ankle strategy (a joint map in
which ankles are the cheap joints: the body leans over its feet); what's left
beyond ``ANKLE_MAX_M`` uses the hip strategy (ankles expensive, hips, knees and
waist cheap). Both maps keep the planted feet still and are recomputed for each
new reference pose.
"""

from __future__ import annotations

import math

import numpy as np

from rth.sim.ik import NJ, WholeBodyIK
from rth.sim.targets import joint_costs

GRAVITY = 9.81
K_XI = 0.6  # COM shift per metre of capture-point error, double support (higher slips a foot)
K_XI_SINGLE = 1.5  # on one 6 cm wide foot the COM must be caught before the foot rolls
K_I = 1.5  # 1/s, slowly removes a steady COM offset (model mismatch, contact settling)
I_MAX_M = 0.03
ANKLE_MAX_M = 0.035  # beyond this the hips take over
SHIFT_MAX_M = 0.10

ANKLE_COST = joint_costs(ankle=1.0, leg=10.0, hip_yaw=100.0, waist=100.0, arm=1000.0)
HIP_COST = joint_costs(ankle=50.0, leg=1.0, hip_yaw=100.0, waist=2.0, arm=1000.0)


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= limit else v * (limit / n)


class Balance:
    """COM feedback for one robot; ``linearise`` once per reference, ``correction`` every tick."""

    def __init__(self, ik: WholeBodyIK) -> None:
        self.ik = ik
        self.ankle_map = np.zeros((NJ, 2))
        self.hip_map = np.zeros((NJ, 2))
        self.torques = np.zeros(NJ)  # static gravity torques of the reference pose
        self.integral = np.zeros(2)
        self.last_error = np.zeros(2)
        self.single = False

    def reset(self) -> None:
        self.integral[:] = 0.0
        self.last_error[:] = 0.0

    def linearise(self, qpos: np.ndarray, planted: tuple[bool, bool], left_share: float) -> None:
        if not any(planted):
            self.ankle_map[:] = 0.0
            self.hip_map[:] = 0.0
            self.torques[:] = 0.0
            return
        self.ankle_map = self.ik.com_shift_map(qpos, planted, ANKLE_COST)
        self.hip_map = self.ik.com_shift_map(qpos, planted, HIP_COST)
        self.torques = self.ik.static_torques(qpos, planted, left_share)
        self.single = planted[0] != planted[1]

    def correction(
        self, com: np.ndarray, vel: np.ndarray, com_ref: np.ndarray, vel_ref: np.ndarray, dt: float
    ) -> np.ndarray:
        """(29,) joint target offsets for the measured COM state against the plan."""
        omega = math.sqrt(GRAVITY / max(float(com[2]), 0.3))
        err = (com[:2] - com_ref) + (vel[:2] - vel_ref) / omega
        self.last_error = err
        self.integral = _clip_norm(self.integral + (com[:2] - com_ref) * dt, I_MAX_M / max(K_I, 1e-6))
        k = K_XI_SINGLE if self.single else K_XI
        shift = _clip_norm(-(k * err + K_I * self.integral), SHIFT_MAX_M)
        ankle = _clip_norm(shift, ANKLE_MAX_M)
        return self.ankle_map @ ankle + self.hip_map @ (shift - ankle)
