"""Run every policy command through the robot's two safety layers before it reaches the simulated motors (AI-06).

    policy action -> (v, omega) exactly like forklift_rl/docking_core.action_to_command on the Jetson
      -> Jetson supervisor  : forklift_safety.safety_core.SafetySupervisor (the real module, imported from ros2_ws)
      -> ESP32 limiter      : forklift_rl.firmware_limiter.DriveLimiter (1:1 port of lib/forklift_core, see there)
      -> simulated motors   : env action [v / v_max, steering, lift]

Sensors fed to both layers are the ones the policy sees: the three front ToF readings (obs 9-11 -> front_left,
front, front_right; the side and rear sensors are not simulated and report "no target"), the fork height (obs 8)
and, for the supervisor, the ESP32 state byte of the previous tick (as on the robot). The source is "dock", so
SF-03 docking mode is on: speed cap 0.1 m/s and the front sensors in `docking_ignore` are not looked at.

Like the robot's docking node (docking_core.py) the policy sees its own previous action in the observation, not the
supervised one, and the episode is aborted ("safety_stop") once the supervisor has held the forklift stopped for
`abort_after_s`. When the supervised speed is 0 the steering angle of the policy is kept (the ESP32 holds the
rear-wheel angle while standing; Twist omega = 0 carries no steering information).
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np

from . import firmware_limiter as fw

_SAFETY_PKG = Path(__file__).resolve().parents[2] / "ros2_ws" / "src" / "forklift_safety"
if str(_SAFETY_PKG) not in sys.path:
    sys.path.insert(0, str(_SAFETY_PKG))
from forklift_safety import safety_core as sc  # noqa: E402  (rclpy-free core)

# which front sensors SF-03 ignores while docking
DOCKING_IGNORE = {
    "firmware": ("front", "front_left", "front_right"),  # = config.h TOF_MASK_DOCKING_IGNORE (all three front sensors)
    "front_all": ("front", "front_left", "front_right"),  # kept as an alias for the earlier evaluation
    "centre_only": ("front",),                            # the original mask that blocked every docking (history)
}
_FW_BITS = {"front": 1 << fw.TOF_FRONT, "front_left": 1 << fw.TOF_FRONT_LEFT, "front_right": 1 << fw.TOF_FRONT_RIGHT}

RULES = ("proposed_docking_stop", "jetson_tof_stop", "jetson_tof_slow", "jetson_docking_cap", "jetson_fork_cap", "jetson_ramp",
         "esp32_tof_stop", "esp32_tof_slow", "esp32_docking_cap", "esp32_fork_cap", "esp32_ramp")


class SafetyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, docking_ignore: str = "firmware", jetson: bool = True, esp32: bool = True,
                 abort_after_s: float | None = 3.0, docking_stop_m: float | None = None):
        super().__init__(env)
        base = env.unwrapped
        self.cfg = base.cfg
        f, c = self.cfg["forklift"], self.cfg["control"]
        self.v_max, self.steer_max = float(c["v_max"]), math.radians(c["steer_max_deg"])
        self.L, self.lift_max, self.dt = float(f["wheelbase"]), float(f["lift_max"]), 1.0 / c["rate_hz"]
        ignored = DOCKING_IGNORE[docking_ignore]
        self.params = sc.SafetyParams(docking_ignored_tof=list(ignored))
        mask = 0
        for n in ignored:
            mask |= _FW_BITS[n]
        self.fw_cfg = fw.SafetyConfig(mask_docking_ignore=mask)
        self.use_jetson, self.use_esp32 = jetson, esp32
        self.abort_after_s = abort_after_s
        # PROPOSAL, not on the robot: while docking, stop forward motion when a front-corner ToF reads below this
        # (forks fully inserted / mast at the rack). Off (None) = exactly the robot's rules.
        self.docking_stop_m = docking_stop_m
        self._obs: np.ndarray | None = None

    # ------------------------------------------------------------------ gym API
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.sup = sc.SafetySupervisor(self.params)
        self.esp = fw.DriveLimiter(self.fw_cfg)
        self.esp_byte = sc.encode_esp32(sc.ESP_IDLE)
        self.t = 0.0
        self.stopped_for = 0.0
        self.rule_steps: Counter = Counter()
        self.modified_steps = 0
        self.steps = 0
        self.last: dict = {}
        self._obs = obs
        info["safety"] = self.summary()
        return obs, info

    def step(self, action):
        raw = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1, 1)
        safe = self.supervise(raw, self._obs)
        obs, reward, terminated, truncated, info = self.env.step(safe)
        obs = obs.copy()
        obs[14:17] = raw  # docking_core feeds back the policy's own action
        self.env.unwrapped._prev_action = raw
        self._obs = obs
        self.steps += 1
        aborted = False
        if self.abort_after_s is not None and not terminated and self.stopped_for >= self.abort_after_s:
            truncated, aborted = True, True
        info["safety_abort"] = aborted
        info["safety"] = self.summary()
        return obs, reward, terminated, truncated, info

    def summary(self) -> dict:
        return {"rule_steps": {r: self.rule_steps.get(r, 0) for r in RULES}, "modified_steps": self.modified_steps,
                "steps": self.steps}

    # ------------------------------------------------------------------ the two layers
    @staticmethod
    def sensors(obs: np.ndarray, lift_max: float) -> tuple[dict[str, float], float]:
        tof = {"front_left": float(obs[9]) * 1.2, "front": float(obs[10]) * 1.2, "front_right": float(obs[11]) * 1.2}
        return tof, float(obs[8]) * lift_max * 1000.0

    def supervise(self, raw: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Policy action -> supervised env action. Records which rule limited the command in `self.last`."""
        tof, fork_mm = self.sensors(obs, self.lift_max)
        v_cmd = float(raw[0]) * self.v_max
        steer = float(raw[1]) * self.steer_max
        w_cmd = -v_cmd * math.tan(steer) / self.L
        fired: set[str] = set()
        p = self.params
        v, w = v_cmd, w_cmd
        proposed_stop = (self.docking_stop_m is not None and v > 0
                         and min(tof["front_left"], tof["front_right"]) < self.docking_stop_m)

        if self.use_jetson:
            t = self.t
            s = self.sup
            s.set_cmd("dock", v, w, t)
            s.set_tof([tof["front"], tof["front_left"], tof["front_right"], fw.TOF_NO_TARGET_M, fw.TOF_NO_TARGET_M,
                       fw.TOF_NO_TARGET_M], t)
            s.set_fork(fork_mm)
            s.set_fleet(t, "auto", False, True, 0.0, True)
            s.set_esp32(self.esp_byte)
            out = s.step(t, self.dt)
            moving = abs(v) > 1e-6
            if moving and out.reason == "tof_stop":
                fired.add("jetson_tof_stop")
            elif moving:
                if out.reason == "tof_slow" and abs(v) > p.max_speed * p.tof_slow_factor + 1e-9:
                    fired.add("jetson_tof_slow")
                if abs(v) > p.docking_speed + 1e-9:
                    fired.add("jetson_docking_cap")
                if out.forks_raised and abs(v) > p.raised_speed + 1e-9:
                    fired.add("jetson_fork_cap")
                if abs(out.v) + 1e-9 < min(abs(v), out.speed_limit):
                    fired.add("jetson_ramp")
            v, w = out.v, out.w
            j_state = out.state
        else:
            j_state = sc.NORMAL

        if self.use_esp32:
            now_ms = int(round(self.t * 1000))
            dist = [tof["front"], tof["front_left"], tof["front_right"], fw.TOF_NO_TARGET_M, fw.TOF_NO_TARGET_M,
                    fw.TOF_NO_TARGET_M]
            readings = [fw.ToFReading(d, True, now_ms) for d in dist]
            o = self.esp.update(v, w, readings, now_ms, fork_mm, True, True, self.dt)
            moving = abs(v) > self.fw_cfg.cmd_eps_mps
            if moving and o.state == fw.STOP:
                fired.add("esp32_tof_stop")
            elif moving:
                if o.tof.slow and abs(v) > o.speed_cap + 1e-9 and o.cap_reason == fw.R_TOF_SLOW:
                    fired.add("esp32_tof_slow")
                if abs(v) > self.fw_cfg.docking_max_mps + 1e-9:
                    fired.add("esp32_docking_cap")
                if fork_mm > self.fw_cfg.fork_raised_mm and abs(v) > self.fw_cfg.raised_fork_max_mps + 1e-9:
                    fired.add("esp32_fork_cap")
                if abs(o.v) + 1e-9 < min(abs(v), o.speed_cap):
                    fired.add("esp32_ramp")
            self.esp_byte = o.byte
            v = o.v
            esp_stop = o.state == fw.STOP
        else:
            esp_stop = False

        if proposed_stop:  # hard stop like tof_stop: no ramp in either layer
            fired.add("proposed_docking_stop")
            v = 0.0
            self.sup.v_out = self.sup.w_out = 0.0
            self.esp.v_out = 0.0
        self.t += self.dt
        held = j_state >= sc.STOP or esp_stop or proposed_stop
        self.stopped_for = self.stopped_for + self.dt if held else 0.0
        for r in fired:
            self.rule_steps[r] += 1
        modified = abs(v - v_cmd) > 1e-4
        self.modified_steps += modified
        self.last = {"v_cmd": v_cmd, "v": v, "fired": fired, "modified": modified, "stopped": held}
        return np.array([v / self.v_max, raw[1], raw[2]], dtype=np.float32)
