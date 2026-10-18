"""The Python port of the ESP32 drive limiter must behave exactly like code/firmware/lib/forklift_core.

1. constants are parsed out of config.h;
2. the firmware's native ToF / docking / raised-fork tests (test/test_safety/test_safety.cpp) replayed on the port;
3. differential test against the compiled C++ (skipped when g++ is not installed).
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from forklift_rl import firmware_limiter as fw

FW = Path(__file__).resolve().parents[2] / "firmware" / "lib" / "forklift_core" / "src"
CONFIG_H = FW / "forklift" / "config.h"


# ---------------------------------------------------------------- 1. constants
def _config_h() -> dict[str, float]:
    vals = {}
    for m in re.finditer(r"constexpr\s+(?:float|uint32_t|bool|int)\s+(\w+)\s*=\s*([\w.]+?)f?;", CONFIG_H.read_text()):
        raw = m.group(2)
        vals[m.group(1)] = {"true": 1.0, "false": 0.0}.get(raw, None) if raw in ("true", "false") else float(raw)
    return vals


@pytest.mark.parametrize("name", ["MAX_SPEED_MPS", "TOF_SLOW_M", "TOF_STOP_M", "TOF_SLOW_FACTOR", "TOF_STALE_MS",
                                  "TOF_FAULT_STOPS", "TOF_NO_TARGET_M", "FORK_RAISED_MM", "RAISED_FORK_MAX_MPS",
                                  "DOCKING_MAX_MPS", "ACCEL_MPS2", "DECEL_MPS2", "CMD_EPS_MPS", "TURN_EPS_RADPS"])
def test_constants_match_config_h(name):
    assert float(getattr(fw, name)) == pytest.approx(_config_h()[name], abs=1e-9)


def test_docking_ignore_mask_matches_config_h():
    m = re.search(r"TOF_MASK_DOCKING_IGNORE\s*=\s*([^;]+);", CONFIG_H.read_text())
    bits = re.findall(r"1u\s*<<\s*(TOF_\w+)", m.group(1))
    assert sum(1 << getattr(fw, b) for b in bits) == fw.TOF_MASK_DOCKING_IGNORE


# ---------------------------------------------------------------- 2. firmware test vectors
class Rig:
    """test_safety.cpp's Rig: limiter ticked at 1 kHz, ToF readings kept fresh."""

    def __init__(self):
        self.lim = fw.DriveLimiter()
        self.tof = [fw.ToFReading(4.0, True, 0) for _ in range(fw.TOF_COUNT)]
        self.now = 0
        self.v = self.w = 0.0
        self.fork_mm, self.fork_valid, self.docking = 0.0, True, False
        self.frozen = -1
        self.out = None

    def cmd(self, v, w=0.0):
        self.v, self.w = v, w

    def run(self, ms):
        for _ in range(ms):
            for i, r in enumerate(self.tof):
                if r.valid and i != self.frozen:
                    r.stamp_ms = self.now
            self.out = self.lim.update(self.v, self.w, self.tof, self.now, self.fork_mm, self.fork_valid,
                                       self.docking, 0.001)
            self.now += 1


def test_fw_tof_front_slow_below_40cm():
    r = Rig(); r.cmd(0.3); r.run(1000)
    r.tof[fw.TOF_FRONT].distance_m = 0.35; r.run(500)
    assert (r.out.state, r.out.reason) == (fw.SLOW, fw.R_TOF_SLOW)
    assert r.out.v == pytest.approx(0.09, abs=1e-4)
    r.tof[fw.TOF_FRONT].distance_m = 0.45; r.run(1000)
    assert r.out.state == fw.ACTIVE


def test_fw_tof_front_corner_stop_below_15cm_is_immediate():
    r = Rig(); r.cmd(0.3); r.run(1000)
    r.tof[fw.TOF_FRONT_LEFT].distance_m = 0.12; r.run(1)
    assert (r.out.state, r.out.reason, r.out.v) == (fw.STOP, fw.R_TOF_STOP, 0.0)
    r.run(1000)
    assert r.out.state == fw.STOP
    r.tof[fw.TOF_FRONT_LEFT].distance_m = 1.0; r.run(1000)
    assert r.out.state == fw.ACTIVE


def test_fw_reversing_ignores_front_sensors():
    r = Rig()
    for i in (fw.TOF_FRONT, fw.TOF_FRONT_LEFT, fw.TOF_FRONT_RIGHT):
        r.tof[i].distance_m = 0.05
    r.cmd(-0.2); r.run(1000)
    assert r.out.state == fw.ACTIVE and r.out.v == pytest.approx(-0.2, abs=1e-4)
    r.cmd(0.2); r.run(1000)
    assert r.out.state == fw.STOP and r.out.v == 0.0


def test_fw_forward_ignores_rear_sensor_and_reverse_uses_it():
    r = Rig(); r.tof[fw.TOF_REAR].distance_m = 0.05
    r.cmd(0.2); r.run(1000)
    assert r.out.state == fw.ACTIVE
    r.cmd(-0.2); r.run(1000)
    assert r.out.state == fw.STOP and r.out.v == 0.0
    r.tof[fw.TOF_REAR].distance_m = 0.30; r.run(1000)
    assert r.out.state == fw.SLOW and r.out.v == pytest.approx(-0.09, abs=1e-4)


def test_fw_can_back_away_from_blocked_front():
    r = Rig(); r.cmd(0.3); r.run(1000)
    r.tof[fw.TOF_FRONT].distance_m = 0.10; r.run(10)
    assert r.out.state == fw.STOP
    r.cmd(-0.1); r.run(500)
    assert r.out.state == fw.ACTIVE and r.out.v == pytest.approx(-0.1, abs=1e-4)


def test_fw_side_sensors_only_slow_while_turning():
    r = Rig(); r.tof[fw.TOF_LEFT].distance_m = 0.10
    r.cmd(0.3, 0.0); r.run(1000)
    assert r.out.state == fw.ACTIVE
    r.cmd(0.3, 0.5); r.run(1000)
    assert r.out.state == fw.SLOW and r.out.v > 0


def test_fw_stale_leading_sensor_is_fail_safe_stop():
    r = Rig(); r.cmd(0.2); r.run(500)
    r.tof[fw.TOF_FRONT_RIGHT].valid = False; r.run(1)
    assert (r.out.state, r.out.reason) == (fw.STOP, fw.R_TOF_FAULT)
    r.cmd(-0.2); r.run(500)
    assert r.out.state == fw.ACTIVE


def test_fw_tof_stale_by_age():
    r = Rig(); r.cmd(0.2); r.run(200)
    r.frozen = fw.TOF_FRONT; r.run(fw.TOF_STALE_MS)
    assert r.out.state == fw.ACTIVE
    r.run(2)
    assert (r.out.state, r.out.reason) == (fw.STOP, fw.R_TOF_FAULT)


def test_fw_docking_ignores_front_sensors_and_caps_speed():
    """Mirror of firmware test_docking_ignores_front_sensors_and_caps_speed."""
    r = Rig(); r.docking = True
    r.tof[fw.TOF_FRONT].distance_m = 0.03
    r.tof[fw.TOF_FRONT_LEFT].distance_m = 0.04
    r.tof[fw.TOF_FRONT_RIGHT].distance_m = 0.04
    r.cmd(0.3); r.run(1000)
    assert r.out.state == fw.SLOW and r.out.v == pytest.approx(fw.DOCKING_MAX_MPS, abs=1e-4)
    r.docking = False; r.run(1)
    assert r.out.state == fw.STOP


def test_fw_raised_forks_cap_speed():
    r = Rig(); r.fork_mm = 60.0; r.cmd(0.3); r.run(1000)
    assert (r.out.state, r.out.reason) == (fw.SLOW, fw.R_FORK_RAISED)
    assert r.out.v == pytest.approx(fw.RAISED_FORK_MAX_MPS, abs=1e-4)
    r.fork_mm = 40.0; r.run(1000)
    assert r.out.state == fw.ACTIVE and r.out.v == pytest.approx(0.3, abs=1e-4)
    r.fork_mm = 80.0; r.run(300)
    assert r.out.v == pytest.approx(fw.RAISED_FORK_MAX_MPS, abs=1e-4)


def test_fw_unknown_fork_height_counts_as_raised():
    r = Rig(); r.fork_valid = False; r.cmd(-0.3); r.run(1000)
    assert r.out.v == pytest.approx(-fw.RAISED_FORK_MAX_MPS, abs=1e-4)


def test_fw_tightest_cap_wins():
    c = fw.SafetyConfig()
    assert fw.speed_cap(True, 80.0, True, False, c) == (pytest.approx(0.09, abs=1e-5), fw.R_TOF_SLOW)
    assert fw.speed_cap(False, 0.0, True, False, c) == (pytest.approx(c.max_speed_mps), fw.R_NONE)


# ---------------------------------------------------------------- 3. differential test against the C++
@pytest.fixture(scope="module")
def cpp(tmp_path_factory):
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ not installed: differential test against the firmware C++ skipped")
    out = tmp_path_factory.mktemp("fw") / ("fwshim.dll" if sys.platform == "win32" else "fwshim.so")
    src = [Path(__file__).with_name("firmware_shim.cpp"), FW / "forklift" / "safety_limiter.cpp",
           FW / "forklift" / "safety_state_machine.cpp", FW / "forklift" / "types.cpp"]
    cmd = [gxx, "-std=c++17", "-O1", "-shared", "-fPIC", "-I", str(FW), *map(str, src), "-o", str(out)]
    if sys.platform == "win32":
        cmd.append("-static")  # MinGW: no runtime DLLs to find next to the shim
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.skip(f"could not build the firmware shim: {res.stderr[-400:]}")
    try:
        lib = ctypes.CDLL(str(out))
    except OSError as e:
        pytest.skip(f"could not load the firmware shim: {e}")
    F, I, P = ctypes.c_float, ctypes.c_int, ctypes.c_void_p
    lib.shim_speed_cap.restype = F
    lib.shim_speed_cap.argtypes = [I, F, I, I, ctypes.POINTER(I)]
    lib.shim_ramp.restype = F
    lib.shim_ramp.argtypes = [F, F, F, F, F]
    lib.shim_evaluate_tof.argtypes = [I, F, ctypes.POINTER(F), ctypes.POINTER(I), ctypes.POINTER(ctypes.c_uint),
                                      ctypes.c_uint, I, I, ctypes.POINTER(I)]
    lib.shim_sm_new.restype = P
    lib.shim_sm_new.argtypes = [I]
    lib.shim_sm_free.argtypes = [P]
    lib.shim_sm_update.restype = I
    lib.shim_sm_update.argtypes = [P, ctypes.c_uint, F, F, ctypes.POINTER(F), F, I, I, F, ctypes.POINTER(F)]
    return lib


def _f32(x) -> float:
    return float(np.float32(x))


def test_cpp_evaluate_tof_and_speed_cap_match(cpp):
    rng = np.random.default_rng(1)
    for _ in range(5000):
        direction = int(rng.integers(-1, 2))
        w = _f32(rng.choice([0.0, 0.03, 0.2, -0.6]))
        dist = [_f32(rng.uniform(0.0, 0.6)) for _ in range(6)]
        valid = [int(rng.random() > 0.1) for _ in range(6)]
        now = int(rng.integers(200, 5000))
        stamp = [now - int(rng.choice([0, 50, 100, 101, 300])) for _ in range(6)]
        docking, mask = int(rng.integers(0, 2)), int(rng.choice([1, 7]))
        out = (ctypes.c_int * 7)()
        cpp.shim_evaluate_tof(direction, w, (ctypes.c_float * 6)(*dist), (ctypes.c_int * 6)(*valid),
                              (ctypes.c_uint * 6)(*stamp), now, docking, mask, out)
        e = fw.evaluate_tof(direction, w, [fw.ToFReading(d, bool(v), s) for d, v, s in zip(dist, valid, stamp)],
                            now, bool(docking), fw.SafetyConfig(mask_docking_ignore=mask))
        assert list(out) == [e.slow, e.stop, e.fault, e.considered_mask, e.slow_mask, e.stop_mask, e.fault_mask]

        fork = _f32(rng.uniform(0, 100))
        fvalid, slow = int(rng.random() > 0.1), int(rng.integers(0, 2))
        why = ctypes.c_int()
        cap = cpp.shim_speed_cap(slow, fork, fvalid, docking, ctypes.byref(why))
        assert (cap, why.value) == (pytest.approx(fw.speed_cap(bool(slow), fork, bool(fvalid), bool(docking),
                                                               fw.SafetyConfig())[0], abs=1e-6),
                                    fw.speed_cap(bool(slow), fork, bool(fvalid), bool(docking), fw.SafetyConfig())[1])


def test_cpp_ramp_matches(cpp):
    rng = np.random.default_rng(2)
    for _ in range(5000):
        cur, tgt = _f32(rng.uniform(-0.3, 0.3)), _f32(rng.uniform(-0.3, 0.3))
        dt = _f32(rng.choice([0.001, 0.01, 0.05]))
        assert cpp.shim_ramp(cur, tgt, 0.6, 1.2, dt) == pytest.approx(fw.ramp_speed(cur, tgt, 0.6, 1.2, dt), abs=1e-6)


@pytest.mark.parametrize("mask,dt_ms", [(1, 1), (7, 1), (1, 50), (7, 50)])
def test_cpp_state_machine_drive_section_matches(cpp, mask, dt_ms):
    """Random 4000-tick drives: speed, yaw rate and the safety_state byte agree with SafetyStateMachine::update."""
    rng = np.random.default_rng(mask * 100 + dt_ms)
    sm = cpp.shim_sm_new(mask)
    lim = fw.DriveLimiter(fw.SafetyConfig(mask_docking_ignore=mask))
    try:
        v = w = 0.0
        dist = [4.0] * 6
        fork, fvalid, docking = 0.0, 1, 1
        now = 0
        for k in range(4000):
            if rng.random() < 0.05:
                v = _f32(rng.choice([0.0, 0.1, 0.3, -0.1, -0.2, rng.uniform(-0.3, 0.3)]))
                w = _f32(rng.choice([0.0, 0.2, -0.4]))
            if rng.random() < 0.05:
                i = int(rng.integers(0, 6))
                dist[i] = _f32(rng.choice([4.0, 0.5, 0.3, 0.12, 0.03, rng.uniform(0, 1)]))
            if rng.random() < 0.01:
                fork, fvalid, docking = _f32(rng.uniform(0, 100)), int(rng.random() > 0.1), int(rng.integers(0, 2))
            out = (ctypes.c_float * 4)()
            byte = cpp.shim_sm_update(sm, now, v, w, (ctypes.c_float * 6)(*dist), fork, fvalid, docking,
                                      dt_ms / 1000.0, out)
            py = lim.update(v, w, [fw.ToFReading(d, True, now) for d in dist], now, fork, bool(fvalid),
                            bool(docking), _f32(dt_ms / 1000.0))
            assert out[0] == pytest.approx(py.v, abs=1e-4), k
            assert out[1] == pytest.approx(py.w, abs=1e-3), k
            assert out[2] == pytest.approx(py.v_target, abs=1e-6), k
            assert byte == py.byte, k
            lim.v_out = float(out[0])  # keep float32 rounding from accumulating over thousands of ramp steps
            now += dt_ms
    finally:
        cpp.shim_sm_free(sm)
