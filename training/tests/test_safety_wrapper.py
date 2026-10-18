"""SafetyWrapper: each rule of the Jetson supervisor and the ESP32 limiter triggers on constructed sensor states."""

from __future__ import annotations

import numpy as np
import pytest

from forklift_rl.env import ForkliftDockingEnv
from forklift_rl.safety_wrapper import SafetyWrapper


@pytest.fixture(scope="module")
def base():
    e = ForkliftDockingEnv(randomize=False)
    yield e
    e.close()


def make(base, **kw) -> SafetyWrapper:
    w = SafetyWrapper(base, **kw)
    w.reset(seed=0, options={"distance": 0.5, "lateral": 0.0, "yaw_deg": 0, "elevation": 0.0, "lift": 0.0})
    return w


def obs(tof=(1.2, 1.2, 1.2), lift_frac=0.0) -> np.ndarray:
    """Observation with given front ToF (left corner, centre, right corner) in metres and fork height / lift_max."""
    o = np.zeros(18, dtype=np.float32)
    o[8] = lift_frac
    o[9:12] = np.asarray(tof) / 1.2
    return o


def drive(w: SafetyWrapper, o: np.ndarray, action=(1.0, 0.0, 0.0), n: int = 20) -> np.ndarray:
    out = None
    for _ in range(n):
        out = w.supervise(np.asarray(action, dtype=np.float32), o)
    return out


def test_clear_path_passes_full_docking_speed(base):
    w = make(base)
    a = drive(w, obs())
    assert a[0] == pytest.approx(1.0, abs=1e-6)  # 0.1 m/s = the docking cap: not limited
    assert not w.last["modified"] and not w.last["fired"]


def test_accel_ramp_limits_the_first_steps(base):
    w = make(base)
    a = w.supervise(np.array([1.0, 0.0, 0.0], np.float32), obs())
    assert 0 < a[0] < 1.0 and "jetson_ramp" in w.last["fired"]  # 0.5 m/s² · 50 ms = 0.025 m/s


def test_corner_tof_below_15cm_stops_immediately(base):
    w = make(base, docking_ignore="centre_only")
    drive(w, obs())
    a = w.supervise(np.array([1.0, 0.0, 0.0], np.float32), obs(tof=(0.10, 1.2, 1.2)))
    assert a[0] == 0.0 and w.last["stopped"]
    assert "jetson_tof_stop" in w.last["fired"]


def test_esp32_stops_on_its_own_without_the_jetson(base):
    w = make(base, jetson=False, docking_ignore="centre_only")
    drive(w, obs())
    a = w.supervise(np.array([1.0, 0.0, 0.0], np.float32), obs(tof=(1.2, 1.2, 0.12)))
    assert a[0] == 0.0 and "esp32_tof_stop" in w.last["fired"]


def test_corner_tof_below_40cm_slows_to_30_percent(base):
    for kw, rule in (({}, "jetson_tof_slow"), ({"jetson": False}, "esp32_tof_slow")):
        w = make(base, docking_ignore="centre_only", **kw)
        a = drive(w, obs(tof=(0.30, 1.2, 1.2)), n=40)
        assert a[0] == pytest.approx(0.9, abs=1e-4)  # 0.3 · 0.30 m/s = 0.09 m/s
        assert rule in w.last["fired"]


def test_docking_ignores_the_centre_sensor(base):
    """SF-03: the sensors that see the pallet itself are ignored while docking (firmware TOF_MASK_DOCKING_IGNORE)."""
    w = make(base)
    a = drive(w, obs(tof=(1.2, 0.03, 1.2)))
    assert a[0] == pytest.approx(1.0, abs=1e-6) and not w.last["fired"]


def test_front_all_ignores_the_corners_while_docking(base):
    w = make(base, docking_ignore="front_all")
    a = drive(w, obs(tof=(0.05, 0.03, 0.05)))
    assert a[0] == pytest.approx(1.0, abs=1e-6)


def test_reverse_is_not_blocked_by_front_sensors(base):
    w = make(base)
    a = drive(w, obs(tof=(0.05, 0.05, 0.05)), action=(-1.0, 0.0, 0.0), n=40)
    assert a[0] == pytest.approx(-1.0, abs=1e-6)


def test_docking_cap_holds_speed_at_01(base):
    """The action space already ends at 0.1 m/s, so check the cap on the layers directly with a faster command."""
    w = make(base)
    w.v_max = 0.3  # policy asks 0.3 m/s
    for _ in range(40):
        a = w.supervise(np.array([1.0, 0.0, 0.0], np.float32), obs())
    assert a[0] * 0.3 == pytest.approx(0.10, abs=1e-6)
    assert "jetson_docking_cap" in w.last["fired"]


def test_raised_fork_cap(base):
    w = make(base, jetson=False)
    w.v_max = 0.3
    w.fw_cfg.docking_max_mps = 0.3  # isolate SF-08 from SF-03
    for _ in range(40):
        a = w.supervise(np.array([1.0, 0.0, 0.0], np.float32), obs(lift_frac=0.5))  # 60 mm > 50 mm
    assert a[0] * 0.3 == pytest.approx(0.10, abs=1e-6)
    assert "esp32_fork_cap" in w.last["fired"]


def test_steering_and_lift_pass_through(base):
    w = make(base, docking_ignore="centre_only")
    a = drive(w, obs(tof=(0.10, 1.2, 1.2)), action=(1.0, -0.4, 0.7), n=3)
    assert a[0] == 0.0 and a[1] == pytest.approx(-0.4) and a[2] == pytest.approx(0.7)


def test_policy_sees_its_own_previous_action(base):
    w = make(base)
    o, *_ = w.step(np.array([1.0, 0.2, 0.0], np.float32))
    assert np.allclose(o[14:17], [1.0, 0.2, 0.0])  # not the ramped 0.25


def test_long_stop_aborts_the_episode(base):
    w = SafetyWrapper(base, abort_after_s=0.5, docking_ignore="centre_only")
    w.reset(seed=0, options={"distance": 0.5, "lateral": 0.0, "yaw_deg": 0, "elevation": 0.0, "lift": 0.0})
    truncated = False
    for _ in range(30):
        w._obs = obs(tof=(0.05, 1.2, 1.2))  # an obstacle 5 cm in front of the left corner, every step
        _, _, term, truncated, info = w.step(np.array([1.0, 0.0, 0.0], np.float32))
        if term or truncated:
            break
    assert truncated and info["safety_abort"]
    assert info["safety"]["rule_steps"]["jetson_tof_stop"] >= 10


def test_proposed_docking_stop_is_off_by_default_and_cuts_at_once_when_on(base):
    close = obs(tof=(0.015, 0.0, 0.03))  # forks fully in: corner 1.5 cm from the stringer face
    w = make(base, docking_ignore="front_all")
    drive(w, obs())
    assert drive(w, close, n=1)[0] == pytest.approx(1.0, abs=1e-6)  # the robot's rules let it continue
    w = make(base, docking_ignore="front_all", docking_stop_m=0.02)
    drive(w, obs())
    assert drive(w, close, n=1)[0] == 0.0 and "proposed_docking_stop" in w.last["fired"]
    assert drive(w, close, action=(-1.0, 0.0, 0.0), n=40)[0] == pytest.approx(-1.0, abs=1e-6)  # may back out
