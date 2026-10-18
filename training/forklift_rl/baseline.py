"""Classical docking controller (AI-07 reference): visual servoing on the ArUco pallet pose.

It sees exactly the observation the RL policy sees, so the comparison is fair. Strategy:
  1. drive the fork height to the opening while approaching;
  2. track the pallet axis with a second-order lateral controller in the distance domain
     (curvature = -(2·ζ·λ·heading + λ²·lateral)), slowing down near the face;
  3. if the fork tips reach the entrance misaligned (or at the wrong height), back off and try again.
"""

from __future__ import annotations

import math

import numpy as np


class BaselineController:
    def __init__(self, cfg: dict):
        f, c = cfg["forklift"], cfg["control"]
        self.L = f["wheelbase"]
        self.fork = f["fork_length"]
        self.v_max = c["v_max"]
        self.steer_max = math.radians(c["steer_max_deg"])
        self.dt = 1.0 / c["rate_hz"]
        self.lam = 11.0  # spatial bandwidth, 1/m (settles in ≈ 0.4 m of travel)
        self.reset()

    def reset(self) -> None:
        self.backoff = 0.0  # seconds of reversing left
        self.tries = 0

    @staticmethod
    def decode(obs: np.ndarray) -> dict:
        dx, dy = obs[0] * 0.6, obs[1] * 0.15
        phi = math.atan2(obs[2], obs[3])  # pallet yaw relative to the forklift
        c, s = math.cos(-phi), math.sin(-phi)
        # forklift carriage expressed in the pallet frame (x along the insertion axis, 0 = pallet face)
        px, py = c * -dx - s * -dy, s * -dx + c * -dy
        heading = -phi
        tip_x = px + math.cos(heading) * 0.18
        tip_y = py + math.sin(heading) * 0.18
        return {"s": px, "e": py, "heading": heading, "tip_s": tip_x, "tip_e": tip_y, "z_err": obs[4] * 0.05}

    def act(self, obs: np.ndarray) -> np.ndarray:
        st = self.decode(obs)
        lift = float(np.clip(st["z_err"] / 0.015, -1, 1))
        if self.backoff > 0:
            self.backoff -= self.dt
            return np.array([-1.0, 0.0, lift], dtype=np.float32)  # straight back, re-converge going forward

        misaligned = abs(st["tip_e"]) > 0.008 or abs(st["e"]) > 0.01 or abs(st["heading"]) > math.radians(3) \
            or abs(st["z_err"]) > 0.004
        if -0.03 < st["tip_s"] < 0.005 and misaligned and self.tries < 4:
            self.tries += 1
            self.backoff = 2.0
            return np.array([0.0, 0.0, lift], dtype=np.float32)

        # lateral control about the front (non-steered) axle, where ė = v·sin(heading) holds exactly
        e_axle = st["e"] - 0.03 * math.sin(st["heading"])
        lam, zeta = self.lam, 0.9
        curvature = -(2 * zeta * lam * math.sin(st["heading"]) + lam * lam * e_axle)
        steer = math.atan(-curvature * self.L)  # rear-wheel steering: κ = -tan(δ)/L
        if st["tip_s"] < -0.15:
            v = 1.0
        elif st["tip_s"] < -0.01:
            v = 0.55
        else:
            v = 0.4
        if abs(st["z_err"]) > 0.004 and st["tip_s"] > -0.06:
            v = 0.0  # wait for the forks to reach the opening height before entering
        return np.array([v, float(np.clip(steer / self.steer_max, -1, 1)), lift], dtype=np.float32)
