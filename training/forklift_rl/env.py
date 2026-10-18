"""Gymnasium environment: final approach to a pallet and fork insertion (requirements §8, AI-01..AI-04).

Observation (18, float32, roughly in [-1, 1]) — only what the real forklift can sense:
  0-1  pallet face centre in the fork-carriage frame (dx, dy), from the ArUco marker (PC-04/PC-05)
  2-3  sin/cos of the pallet yaw relative to the forklift
  4    fork height error to the pallet opening (from the marker's 3-D pose)
  5    marker visible this step (else the estimate is carried by odometry)
  6-8  measured speed, steering angle, fork height
  9-11 front ToF ring: left corner, centre, right corner (VL53L1X)
  12-13 TCRT5000 on the left / right fork tip (pallet above the tip)
  14-16 previous action
  17   time left in the episode
Action (3, in [-1, 1]): linear speed (±0.1 m/s), rear steering angle (±30°), fork lift rate.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from .model import build_xml

CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path | str = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


class ForkliftDockingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, config: dict | None = None, randomize: bool = True, render_mode: str | None = None,
                 render_size: tuple[int, int] = (720, 1280)):
        self.cfg = config or load_config()
        self.randomize = randomize
        self.render_mode = render_mode
        self.render_size = render_size
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.cfg))
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = lambda n: m.joint(n).id  # noqa: E731
        self._qx, self._qy, self._qyaw = (m.jnt_qposadr[jid(n)] for n in ("x", "y", "yaw"))
        self._qlift = m.jnt_qposadr[jid("lift")]
        self._qpal = m.jnt_qposadr[jid("pallet")]
        self._pallet_body = m.body("pallet").id
        self._shelf_mocap = m.body_mocapid[m.body("shelf").id]
        self._robot_geoms = {i for i in range(m.ngeom) if m.geom_group[i] == 1}
        self._obstacle_geoms = {m.geom(n).id for n in ("deck", "str_l", "str_c", "str_r", "load", "shelf")}
        self._tof_sites = [m.site(n).id for n in ("tof_l", "tof_c", "tof_r")]
        self._tip_sites = [m.site(n).id for n in ("tip_l", "tip_r")]
        self._floor_geom = m.geom("floor").id
        self._pallet_geoms = [m.geom(n).id for n in ("deck", "str_l", "str_c", "str_r")]
        self._act = {n: m.actuator(n).id for n in ("vx", "vy", "wz", "steer", "lift")}
        self._base_pallet_mass = float(m.body_mass[self._pallet_body])
        self._geomgroup = np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8)  # rays ignore robot (1) and decor (3)

        c = self.cfg["control"]
        self.dt = 1.0 / c["rate_hz"]
        self.n_sub = int(round(self.dt / c["physics_dt"]))
        self.v_max = float(c["v_max"])
        self.steer_max = math.radians(c["steer_max_deg"])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(18,), dtype=np.float32)
        self._renderer: mujoco.Renderer | None = None
        self.camera = mujoco.MjvCamera()
        self.difficulty = 1.0  # curriculum: 0 = close and aligned starts, 1 = the full task ranges

    def set_difficulty(self, value: float) -> None:
        self.difficulty = float(np.clip(value, 0.0, 1.0))

    # ------------------------------------------------------------------ geometry helpers
    def robot_pose(self) -> tuple[np.ndarray, float]:
        q = self.data.qpos
        return np.array([q[self._qx], q[self._qy]]), float(q[self._qyaw])

    def pallet_pose(self) -> tuple[np.ndarray, float, float]:
        q = self.data.qpos[self._qpal:self._qpal + 7]
        w, x, y, z = q[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array(q[:2]), yaw, float(q[2])

    def _points(self) -> dict:
        f = self.cfg["forklift"]
        b, th = self.robot_pose()
        r = rot(th)
        carriage = b + r @ np.array([f["carriage_x"], 0.0])
        tip = b + r @ np.array([f["carriage_x"] + f["fork_length"], 0.0])
        return {"base": b, "yaw": th, "carriage": carriage, "tip": tip}

    def errors(self) -> dict:
        """Ground-truth docking errors (used for reward, success and the classical baseline's tests)."""
        f, p = self.cfg["forklift"], self.cfg["pallet"]
        pts = self._points()
        P, psi, pz = self.pallet_pose()
        u = np.array([math.cos(psi), math.sin(psi)])
        n = np.array([-math.sin(psi), math.cos(psi)])
        support = self._elevation
        fork_z = f["fork_clearance"] + float(self.data.qpos[self._qlift])
        opening_lo = support
        opening_hi = support + p["height"] - p["deck_thickness"]
        target_z = opening_lo + (opening_hi - opening_lo - f["fork_thickness"]) / 2
        goal = P + u * (self.cfg["task"]["success_insertion"] * f["fork_length"])
        return {
            "depth": float(u @ (pts["tip"] - P)),
            "lat_tip": float(n @ (pts["tip"] - P)),
            "lat_carriage": float(n @ (pts["carriage"] - P)),
            "heading": wrap(pts["yaw"] - psi),
            "dist_goal": float(np.linalg.norm(pts["tip"] - goal)),
            "dist_face": float(np.linalg.norm(pts["tip"] - P)),
            "z_err": target_z - fork_z,
            "height_ok": opening_lo + 0.0015 < fork_z and fork_z + f["fork_thickness"] < opening_hi - 0.0015,
            "pallet_moved": float(np.linalg.norm(P - self._pallet_start[:2])) + abs(pz - self._pallet_start[2]),
        }

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        t, rz, f = self.cfg["task"], self.cfg["randomization"], self.cfg["forklift"]
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        opts = options or {}

        u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
        self._elevation = float(opts.get("elevation", rng.choice(t["pallet_elevation"])))
        if self.randomize:
            self._delay_sub = int(round(u(rz["motor_delay_ms"]) / 1000 / self.cfg["control"]["physics_dt"]))
            self._slip = u(rz["wheel_slip"])
            self._steer_bias = math.radians(u(rz["steer_bias_deg"]))
            self._aruco_bias = rng.uniform(-rz["aruco_bias_m"], rz["aruco_bias_m"], size=2)
            fr = u(rz["floor_friction"])
            m.body_mass[self._pallet_body] = self._base_pallet_mass + u(rz["pallet_load_kg"])
        else:
            self._delay_sub, self._slip, self._steer_bias, self._aruco_bias, fr = 0, 1.0, 0.0, np.zeros(2), 0.9
            m.body_mass[self._pallet_body] = self._base_pallet_mass
        m.geom_friction[self._floor_geom, 0] = fr
        for g in self._pallet_geoms:
            m.geom_friction[g, 0] = fr

        # pallet at the origin facing +x; shelf under it when elevated
        d.qpos[self._qpal:self._qpal + 7] = [0.0, 0.0, self._elevation, 1, 0, 0, 0]
        d.mocap_pos[self._shelf_mocap] = [0.0, 0.0, self._elevation - 0.03] if self._elevation > 0 else [0.0, 0.0, -1.0]

        k = self.difficulty
        d_lo, d_hi = t["start_distance"]
        dist = float(opts.get("distance", u((d_lo, d_lo + (d_hi - d_lo) * k))))
        lat = float(opts.get("lateral", u([x * k for x in t["start_lateral"]])))
        yaw = math.radians(float(opts.get("yaw_deg", u([x * k for x in t["start_yaw_deg"]]))))
        tip = np.array([-dist, lat])
        base = tip - rot(yaw) @ np.array([f["carriage_x"] + f["fork_length"], 0.0])
        d.qpos[self._qx], d.qpos[self._qy], d.qpos[self._qyaw] = base[0], base[1], yaw
        lift0 = float(opts.get("lift", rng.uniform(0.0, 0.08)))
        d.qpos[self._qlift] = lift0
        mujoco.mj_forward(m, d)
        self._pallet_start = d.qpos[self._qpal:self._qpal + 3].copy()

        self._lift_target = lift0
        self._v = 0.0
        self._steer = 0.0
        self._cmd_queue: deque = deque([(0.0, 0.0)] * self._delay_sub)
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._steps = 0
        self._meas: deque = deque(maxlen=5)  # PC-05 moving average
        self._est_world: np.ndarray | None = None
        self._visible = False
        self._update_estimate()
        self._prev_err = self.errors()
        return self._obs(), self._info(self._prev_err, False, False)

    # ------------------------------------------------------------------ sensing
    def _update_estimate(self) -> None:
        """Simulated ArUco detection from the fork camera; world-frame estimate kept between sightings."""
        rz = self.cfg["randomization"]
        pts = self._points()
        P, psi, pz = self.pallet_pose()
        rel = rot(-pts["yaw"]) @ (P - pts["carriage"])
        bearing = math.atan2(rel[1], rel[0])
        rel_yaw = wrap(psi - pts["yaw"])
        self._visible = (rel[0] > rz["marker_min_range"] and abs(bearing) < math.radians(rz["camera_fov_deg"]) / 2
                         and abs(rel_yaw) < math.radians(60))
        if self._visible:
            noise = self.np_random.normal(0, rz["aruco_noise_m"] / 2, size=2) if self.randomize else np.zeros(2)
            yaw_noise = self.np_random.normal(0, rz["aruco_noise_rad"] / 2) if self.randomize else 0.0
            meas_rel = rel + noise + self._aruco_bias
            world = pts["carriage"] + rot(pts["yaw"]) @ meas_rel
            self._meas.append(np.array([world[0], world[1], psi + yaw_noise]))
            self._est_world = np.mean(np.stack(self._meas), axis=0)

    def _tof(self) -> list[float]:
        b, th = self.robot_pose()
        direction = np.array([math.cos(th), math.sin(th), 0.0])
        out = []
        geomid = np.zeros(1, dtype=np.int32)
        for sid in self._tof_sites:
            dist = mujoco.mj_ray(self.model, self.data, self.data.site_xpos[sid], direction, self._geomgroup, 1, -1, geomid)
            dist = 1.2 if dist < 0 else min(dist, 1.2)
            if self.randomize:
                dist = float(np.clip(dist + self.np_random.normal(0, 0.005), 0, 1.2))
            out.append(dist)
        return out

    def _tcrt(self) -> list[float]:
        p = self.cfg["pallet"]
        P, psi, _ = self.pallet_pose()
        res = []
        for sid in self._tip_sites:
            tip = self.data.site_xpos[sid][:2]
            local = rot(-psi) @ (tip - P)
            res.append(1.0 if 0.005 < local[0] < p["depth"] and abs(local[1]) < p["width"] / 2 else 0.0)
        return res

    def _obs(self) -> np.ndarray:
        f, p = self.cfg["forklift"], self.cfg["pallet"]
        pts = self._points()
        if self._est_world is None:
            rel, rel_yaw = np.array([0.6, 0.0]), 0.0
        else:
            rel = rot(-pts["yaw"]) @ (self._est_world[:2] - pts["carriage"])
            rel_yaw = wrap(self._est_world[2] - pts["yaw"])
        fork_z = f["fork_clearance"] + float(self.data.qpos[self._qlift])
        target_z = self._elevation + (p["height"] - p["deck_thickness"] - f["fork_thickness"]) / 2
        z_err = target_z - fork_z + (self.np_random.normal(0, 0.001) if self.randomize else 0.0)
        speed_meas = self._v + (self.np_random.normal(0, 0.003) if self.randomize else 0.0)
        tof = self._tof()
        obs = np.array([
            rel[0] / 0.6, rel[1] / 0.15, math.sin(rel_yaw), math.cos(rel_yaw), z_err / 0.05, float(self._visible),
            speed_meas / self.v_max, self._steer / self.steer_max, float(self.data.qpos[self._qlift]) / f["lift_max"],
            tof[0] / 1.2, tof[1] / 1.2, tof[2] / 1.2, *self._tcrt(), *self._prev_action,
            1.0 - self._steps / self.cfg["control"]["max_steps"],
        ], dtype=np.float32)
        return np.clip(obs, -5, 5)

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        c, f = self.cfg["control"], self.cfg["forklift"]
        v_cmd, steer_cmd = float(a[0]) * self.v_max, float(a[1]) * self.steer_max
        self._lift_target = float(np.clip(self._lift_target + a[2] * c["lift_rate"] * self.dt, 0, f["lift_max"]))
        m, d = self.model, self.data
        alpha = min(1.0, c["physics_dt"] / c["actuator_tau"])
        L = f["wheelbase"]
        for _ in range(self.n_sub):
            self._cmd_queue.append((v_cmd, steer_cmd))
            dv, ds = self._cmd_queue.popleft()
            self._v += alpha * (dv - self._v)
            self._steer += alpha * (ds - self._steer)
            _, th = self.robot_pose()
            v = self._v * self._slip
            delta = self._steer + self._steer_bias
            omega = -v * math.tan(delta) / L  # rear-wheel steering: steer left → yaw right
            d.ctrl[self._act["vx"]] = v * math.cos(th)
            d.ctrl[self._act["vy"]] = v * math.sin(th)
            d.ctrl[self._act["wz"]] = omega
            d.ctrl[self._act["steer"]] = self._steer
            d.ctrl[self._act["lift"]] = self._lift_target
            mujoco.mj_step(m, d)
        self._steps += 1
        self._update_estimate()

        err = self.errors()
        collision = self._touching() or err["pallet_moved"] > self.cfg["task"]["pallet_moved_limit"]
        t = self.cfg["task"]
        success = (not collision and err["depth"] >= t["success_insertion"] * f["fork_length"]
                   and abs(err["lat_tip"]) <= t["success_lateral"] and abs(err["lat_carriage"]) <= t["success_lateral"]
                   and err["height_ok"])
        out = err["dist_face"] > t["out_of_range"] or err["depth"] < -0.9

        rw = self.cfg["reward"]
        pe = self._prev_err
        reward = (rw["progress"] * (pe["dist_goal"] - err["dist_goal"])
                  + rw["lateral"] * (abs(pe["lat_tip"]) - abs(err["lat_tip"]))
                  + rw["heading"] * (abs(pe["heading"]) - abs(err["heading"]))
                  + rw["height"] * (abs(pe["z_err"]) - abs(err["z_err"]))
                  + rw["step"]
                  + rw["steer_change"] * abs(float(a[1] - self._prev_action[1])))
        if success:
            reward += rw["success"]
        if collision:
            reward += rw["collision"]
        elif out:
            reward += rw["out_of_range"]
        self._prev_err = err
        self._prev_action = a
        terminated = success or collision or out
        truncated = self._steps >= c["max_steps"] and not terminated
        return self._obs(), float(reward), terminated, truncated, self._info(err, success, collision)

    def _touching(self) -> bool:
        d = self.data
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if (g1 in self._robot_geoms and g2 in self._obstacle_geoms) or (g2 in self._robot_geoms and g1 in self._obstacle_geoms):
                return True
        return False

    def _info(self, err: dict, success: bool, collision: bool) -> dict:
        return {**err, "success": success, "collision": collision, "steps": self._steps, "visible": self._visible,
                "elevation": self._elevation}

    # ------------------------------------------------------------------ render
    def render(self, view: str = "side"):
        """`side`: oblique view of forklift + pallet, `top`: plan view, `fork`: the fork-carriage camera."""
        if self._renderer is None:
            h, w = self.render_size
            self._renderer = mujoco.Renderer(self.model, h, w)
            self._scene_opt = mujoco.MjvOption()
            self._scene_opt.geomgroup[3] = 1  # show decor (wheels, roof guard, LiDAR)
            self._scene_opt.sitegroup[0] = 0
        if view == "fork":
            self._renderer.update_scene(self.data, camera="fork_cam", scene_option=self._scene_opt)
            return self._renderer.render()
        pts = self._points()
        mid = (pts["base"] + np.array([0.12, 0.0])) / 2
        self.camera.lookat[:] = [mid[0], mid[1], 0.07 + self._elevation / 2]
        if view == "top":
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.25, 90, -89.9
        else:
            self.camera.distance, self.camera.azimuth, self.camera.elevation = 1.05, -100, -24
        self._renderer.update_scene(self.data, camera=self.camera, scene_option=self._scene_opt)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="ForkliftDocking-v0", entry_point="forklift_rl.env:ForkliftDockingEnv")
