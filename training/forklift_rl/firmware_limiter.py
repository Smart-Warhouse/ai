"""Python port of the ESP32 drive limiter (code/firmware/lib/forklift_core: safety_limiter.cpp + the normal-driving
part of SafetyStateMachine::update), so the simulator can run the SAME last safety layer as the robot (AI-06, FW-07).

Ported 1:1, names kept: `direction_of`, `evaluate_tof`, `speed_cap`, `ramp_speed`, and `DriveLimiter.update` =
section "4. Normal driving: ToF ring + speed caps" of safety_state_machine.cpp. Not ported (always healthy in the
simulator): E-Stop latch, boot, watchdog / SAFE, battery.

Equivalence with the C++ is checked by tests/test_firmware_limiter.py three ways:
  1. the constants below are parsed out of config.h (fails if someone changes the firmware and not this file);
  2. the firmware's own native test vectors (test/test_safety/test_safety.cpp, ToF / docking / raised-fork / cap
     tests) are replayed against this port at 1 kHz with the same rig;
  3. when g++ is available, safety_limiter.cpp itself is compiled into a shared library with a tiny extern "C" shim
     and compared with this port on thousands of random inputs (differential test).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---- config.h (requirements §9) ------------------------------------------------------------------------------
MAX_SPEED_MPS = 0.30
TOF_SLOW_M = 0.40
TOF_STOP_M = 0.15
TOF_SLOW_FACTOR = 0.30
TOF_STALE_MS = 100
TOF_FAULT_STOPS = True
TOF_NO_TARGET_M = 4.0
FORK_RAISED_MM = 50.0
RAISED_FORK_MAX_MPS = 0.10
DOCKING_MAX_MPS = 0.10
ACCEL_MPS2 = 0.6
DECEL_MPS2 = 1.2
CMD_EPS_MPS = 0.005
TURN_EPS_RADPS = 0.05

TOF_FRONT, TOF_FRONT_LEFT, TOF_FRONT_RIGHT, TOF_LEFT, TOF_RIGHT, TOF_REAR = range(6)
TOF_COUNT = 6
TOF_MASK_FRONT = (1 << TOF_FRONT) | (1 << TOF_FRONT_LEFT) | (1 << TOF_FRONT_RIGHT)
TOF_MASK_REAR = 1 << TOF_REAR
TOF_MASK_SIDES = (1 << TOF_LEFT) | (1 << TOF_RIGHT)
TOF_MASK_DOCKING_IGNORE = (1 << TOF_FRONT) | (1 << TOF_FRONT_LEFT) | (1 << TOF_FRONT_RIGHT)  # = config.h

# SafetyState / SafetyReason (types.h): byte on `safety_state` = state << 4 | reason
BOOT, IDLE, ACTIVE, SLOW, STOP, SAFE, ESTOP = range(7)
R_NONE, R_TOF_SLOW, R_TOF_STOP, R_TOF_FAULT, R_FORK_RAISED, R_DOCKING = 0, 3, 4, 5, 6, 7


@dataclass
class SafetyConfig:
    max_speed_mps: float = MAX_SPEED_MPS
    tof_slow_m: float = TOF_SLOW_M
    tof_stop_m: float = TOF_STOP_M
    tof_slow_factor: float = TOF_SLOW_FACTOR
    tof_stale_ms: int = TOF_STALE_MS
    tof_fault_stops: bool = TOF_FAULT_STOPS
    mask_front: int = TOF_MASK_FRONT
    mask_rear: int = TOF_MASK_REAR
    mask_sides: int = TOF_MASK_SIDES
    mask_docking_ignore: int = TOF_MASK_DOCKING_IGNORE
    fork_raised_mm: float = FORK_RAISED_MM
    raised_fork_max_mps: float = RAISED_FORK_MAX_MPS
    docking_max_mps: float = DOCKING_MAX_MPS
    accel_mps2: float = ACCEL_MPS2
    decel_mps2: float = DECEL_MPS2
    cmd_eps_mps: float = CMD_EPS_MPS
    turn_eps_radps: float = TURN_EPS_RADPS


@dataclass
class ToFReading:
    distance_m: float = TOF_NO_TARGET_M
    valid: bool = False
    stamp_ms: int = 0


@dataclass
class TofEvaluation:
    slow: bool = False
    stop: bool = False
    fault: bool = False
    considered_mask: int = 0
    slow_mask: int = 0
    stop_mask: int = 0
    fault_mask: int = 0

    def merge(self, e: "TofEvaluation") -> None:
        self.slow |= e.slow
        self.stop |= e.stop
        self.fault |= e.fault
        self.considered_mask |= e.considered_mask
        self.slow_mask |= e.slow_mask
        self.stop_mask |= e.stop_mask
        self.fault_mask |= e.fault_mask


def direction_of(v: float, eps: float) -> int:
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def evaluate_tof(direction: int, cmd_w: float, tof: list[ToFReading], now_ms: int, docking: bool,
                 c: SafetyConfig) -> TofEvaluation:
    e = TofEvaluation()
    if direction == 0:
        return e
    leading = c.mask_front if direction > 0 else c.mask_rear
    if docking:
        leading &= ~c.mask_docking_ignore & 0xFF
    sides = c.mask_sides if abs(cmd_w) > c.turn_eps_radps else 0
    e.considered_mask = leading | sides
    for i in range(TOF_COUNT):
        bit = 1 << i
        if not e.considered_mask & bit:
            continue
        r = tof[i]
        fresh = r.valid and (now_ms - r.stamp_ms) <= c.tof_stale_ms  # uint32 arithmetic: now >= stamp here
        is_leading = (leading & bit) != 0
        if not fresh:
            if is_leading:
                e.fault = True
                e.fault_mask |= bit
            else:
                e.slow = True
                e.slow_mask |= bit
            continue
        if is_leading and r.distance_m < c.tof_stop_m:
            e.stop = True
            e.stop_mask |= bit
        elif r.distance_m < c.tof_slow_m:
            e.slow = True
            e.slow_mask |= bit
    if e.fault:
        if c.tof_fault_stops:
            e.stop = True
        else:
            e.slow = True
    return e


def speed_cap(tof_slow: bool, fork_height_mm: float, fork_height_valid: bool, docking: bool,
              c: SafetyConfig) -> tuple[float, int]:
    """(cap m/s, SafetyReason of the tightest cap; R_NONE = full speed)."""
    cap, reason = c.max_speed_mps, R_NONE

    def apply(limit: float, why: int) -> None:
        nonlocal cap, reason
        if limit < cap:
            cap, reason = limit, why

    if tof_slow:
        apply(c.max_speed_mps * c.tof_slow_factor, R_TOF_SLOW)
    if not fork_height_valid or fork_height_mm > c.fork_raised_mm:
        apply(c.raised_fork_max_mps, R_FORK_RAISED)
    if docking:
        apply(c.docking_max_mps, R_DOCKING)
    return cap, reason


def ramp_speed(current: float, target: float, accel: float, decel: float, dt: float) -> float:
    slowing = (target * current < 0.0) or abs(target) < abs(current)
    step = (decel if slowing else accel) * dt
    if target > current:
        return current + step if target - current > step else target
    return current - step if current - target > step else target


@dataclass
class DriveOutput:
    v: float
    w: float
    v_target: float
    speed_cap: float
    state: int
    reason: int
    tof: TofEvaluation = field(default_factory=TofEvaluation)
    cap_reason: int = R_NONE

    @property
    def byte(self) -> int:
        return (self.state & 0x0F) << 4 | (self.reason & 0x0F)


class DriveLimiter:
    """Normal-driving section of fl::SafetyStateMachine (link alive, fresh cmd_vel, no E-Stop / SAFE)."""

    def __init__(self, c: SafetyConfig | None = None) -> None:
        self.c = c or SafetyConfig()
        self.v_out = 0.0
        self.curvature = 0.0
        self.state = IDLE

    def update(self, cmd_v: float, cmd_w: float, tof: list[ToFReading], now_ms: int, fork_height_mm: float,
               fork_height_valid: bool, docking: bool, dt: float) -> DriveOutput:
        c = self.c
        if abs(cmd_v) > c.cmd_eps_mps:
            self.curvature = cmd_w / cmd_v
        dir_now = direction_of(self.v_out, c.cmd_eps_mps)
        dir_cmd = direction_of(cmd_v, c.cmd_eps_mps)
        eval_now = evaluate_tof(dir_now, cmd_w, tof, now_ms, docking, c)
        eval_cmd = eval_now if dir_cmd == dir_now else evaluate_tof(dir_cmd, cmd_w, tof, now_ms, docking, c)
        merged = TofEvaluation()
        merged.merge(eval_now)
        merged.merge(eval_cmd)

        cap, cap_reason = speed_cap(merged.slow, fork_height_mm, fork_height_valid, docking, c)
        target = min(max(cmd_v, -cap), cap)
        if eval_cmd.stop:
            target = 0.0
        if eval_now.stop:
            self.v_out = 0.0  # NFR-04: cut at once, no ramp
        self.v_out = ramp_speed(self.v_out, target, c.accel_mps2, c.decel_mps2, dt)

        moving = dir_cmd != 0 or self.v_out != 0.0
        stopped = eval_cmd.stop or eval_now.stop
        if not moving:
            self.state, reason = IDLE, R_NONE
        elif stopped:
            self.state, reason = STOP, (R_TOF_STOP if merged.stop_mask else R_TOF_FAULT)
        elif cap_reason != R_NONE:
            self.state, reason = SLOW, cap_reason
        else:
            self.state, reason = ACTIVE, R_NONE
        return DriveOutput(self.v_out, self.v_out * self.curvature, target, 0.0 if eval_cmd.stop else cap,
                           self.state, reason, merged, cap_reason)
