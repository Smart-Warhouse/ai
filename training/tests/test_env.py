import math

import numpy as np
import pytest

from forklift_rl.baseline import BaselineController
from forklift_rl.env import ForkliftDockingEnv


@pytest.fixture()
def env():
    e = ForkliftDockingEnv(randomize=False)
    yield e
    e.close()


def test_spaces_and_reset(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert env.action_space.shape == (3,)
    assert not info["success"] and not info["collision"]
    assert 0.30 <= -info["depth"] <= 0.60 + 1e-6


def test_reset_is_deterministic(env):
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    assert np.allclose(a, b)


def test_forward_drive_moves_along_heading(env):
    env.reset(seed=0, options={"distance": 0.5, "lateral": 0.0, "yaw_deg": 0, "elevation": 0.0, "lift": 0.0})
    start = env.errors()["depth"]
    for _ in range(20):  # 1 s at 0.1 m/s
        env.step(np.array([1.0, 0.0, 0.0]))
    moved = env.errors()["depth"] - start
    assert 0.08 < moved < 0.11


def test_rear_steering_sign(env):
    """Rear wheels turned left (positive steer) must yaw the forklift right (FW-01, Ackermann on the rear axle)."""
    env.reset(seed=0, options={"distance": 0.6, "lateral": 0.0, "yaw_deg": 0, "elevation": 0.0, "lift": 0.0})
    for _ in range(20):
        env.step(np.array([1.0, 1.0, 0.0]))
    assert env.robot_pose()[1] < -0.05


def test_ramming_the_stringer_is_a_collision(env):
    # forks offset by half an opening width: the left fork hits the centre stringer
    env.reset(seed=0, options={"distance": 0.15, "lateral": 0.045, "yaw_deg": 0, "elevation": 0.0, "lift": 0.002})
    info = {}
    for _ in range(60):
        _, r, term, _, info = env.step(np.array([1.0, 0.0, 0.0]))
        if term:
            break
    assert info["collision"] and r < -50


def test_lift_reaches_rack_height(env):
    env.reset(seed=0, options={"elevation": 0.06, "lift": 0.0})
    for _ in range(40):
        env.step(np.array([0.0, 0.0, 1.0]))
    assert env.data.qpos[env._qlift] > 0.06


def test_baseline_docks_easy_case(env):
    ctrl = BaselineController(env.cfg)
    obs, _ = env.reset(seed=0, options={"distance": 0.45, "lateral": 0.02, "yaw_deg": 5, "elevation": 0.0})
    info = {}
    for _ in range(300):
        obs, _, term, trunc, info = env.step(ctrl.act(obs))
        if term or trunc:
            break
    assert info["success"], info
    assert abs(info["lat_tip"]) <= 0.01 and info["depth"] >= 0.144


def test_marker_out_of_view_keeps_estimate(env):
    env.reset(seed=0, options={"distance": 0.4, "lateral": 0.0, "yaw_deg": 0, "elevation": 0.0, "lift": 0.0})
    obs, *_ = env.step(np.zeros(3))
    assert obs[5] == 1.0
    env.reset(seed=0, options={"distance": 0.4, "lateral": 0.0, "yaw_deg": 0, "elevation": 0.0})
    env._est_world = np.array([0.0, 0.0, 0.0])
    env._visible = False
    assert math.isfinite(env._obs()[0])
