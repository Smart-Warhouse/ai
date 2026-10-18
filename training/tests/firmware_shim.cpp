// extern "C" access to the ESP32 safety limiter for tests/test_firmware_limiter.py (differential test against the
// Python port in forklift_rl/firmware_limiter.py). Built on the fly with g++; never part of the firmware.
#include "forklift/safety_limiter.h"
#include "forklift/safety_state_machine.h"

using namespace fl;

namespace {
SafetyConfig config(uint8_t ignore_mask) {
  SafetyConfig c;
  c.mask_docking_ignore = ignore_mask;
  return c;
}
}  // namespace

extern "C" {

// out: slow, stop, fault, considered, slow_mask, stop_mask, fault_mask
void shim_evaluate_tof(int direction, float cmd_w, const float* dist, const int* valid, const unsigned* stamp,
                       unsigned now_ms, int docking, int ignore_mask, int* out) {
  ToFReading tof[cfg::TOF_COUNT];
  for (int i = 0; i < cfg::TOF_COUNT; ++i) tof[i] = ToFReading{dist[i], valid[i] != 0, stamp[i]};
  const TofEvaluation e = evaluateTof(direction, cmd_w, tof, now_ms, docking != 0,
                                      config(static_cast<uint8_t>(ignore_mask)));
  out[0] = e.slow; out[1] = e.stop; out[2] = e.fault; out[3] = e.considered_mask;
  out[4] = e.slow_mask; out[5] = e.stop_mask; out[6] = e.fault_mask;
}

float shim_speed_cap(int tof_slow, float fork_mm, int fork_valid, int docking, int* reason) {
  const SpeedCap c = speedCap(tof_slow != 0, fork_mm, fork_valid != 0, docking != 0, SafetyConfig{});
  *reason = static_cast<int>(c.reason);
  return c.cap_mps;
}

float shim_ramp(float current, float target, float accel, float decel, float dt) {
  return rampSpeed(current, target, accel, decel, dt);
}

void* shim_sm_new(int ignore_mask) { return new SafetyStateMachine(config(static_cast<uint8_t>(ignore_mask))); }
void shim_sm_free(void* sm) { delete static_cast<SafetyStateMachine*>(sm); }

// One tick with a live link and a fresh cmd_vel; all ToF readings fresh. out: v, w, v_target, speed_cap.
int shim_sm_update(void* sm, unsigned now_ms, float cmd_v, float cmd_w, const float* dist, float fork_mm,
                   int fork_valid, int docking, float dt, float* out) {
  SafetyInputs in;
  in.now_ms = now_ms;
  in.boot_complete = true;
  in.heartbeat_seen = true;
  in.last_heartbeat_ms = now_ms;
  in.cmd_seen = true;
  in.last_cmd_ms = now_ms;
  in.cmd_v = cmd_v;
  in.cmd_w = cmd_w;
  for (int i = 0; i < cfg::TOF_COUNT; ++i) in.tof[i] = ToFReading{dist[i], true, now_ms};
  in.fork_height_mm = fork_mm;
  in.fork_height_valid = fork_valid != 0;
  in.docking = docking != 0;
  in.battery_pct = 80.0f;
  in.battery_valid = true;
  const SafetyOutput o = static_cast<SafetyStateMachine*>(sm)->update(in, dt);
  out[0] = o.v; out[1] = o.w; out[2] = o.v_target; out[3] = o.speed_cap;
  return o.wire();
}
}
