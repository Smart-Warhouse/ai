"""AI-06 in simulation: run the docking policy through the robot's safety layers and measure what they change.

    .venv/Scripts/python scripts/eval_safety.py            # ≈ 10 min on 10 processes
    .venv/Scripts/python scripts/eval_safety.py --episodes 100 --no-stress   # quick look

Same 1000 test seeds as models/docking_v3/meta.json (10000-10999). The policy runs from policy.onnx through
onnxruntime, as on the Jetson. Runs:
  nominal  policy alone | policy + supervisor | baseline alone | baseline + supervisor, the supervisor once with the
           firmware's docking mask of today (centre ToF ignored) and once with all three front sensors ignored;
  proposal the same plus a PROPOSED docking stop (front-corner ToF below a threshold chosen on the validation seeds
           20000-20399) — a candidate rule, not what the robot runs today;
  stress   beyond the training randomisation (doubled marker noise, 120 ms motor delay, ±3° steering bias, low
           friction + wheel slip), policy alone vs policy + supervisor (front sensors ignored, and with the proposal).
Writes models/docking_v3/safety_eval.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forklift_rl.baseline import BaselineController  # noqa: E402
from forklift_rl.env import ForkliftDockingEnv, load_config  # noqa: E402
from forklift_rl.safety_wrapper import RULES, SafetyWrapper  # noqa: E402

MODEL = ROOT / "models" / "docking_v3"
SEED0 = 10_000
VAL0 = 20_000
STRESS = ("marker_noise_x2", "motor_delay_120ms", "steer_bias_3deg", "low_friction")


def stress_config(name: str, seed: int) -> dict:
    cfg = load_config()
    rz = cfg["randomization"]
    if name == "marker_noise_x2":
        for k in ("aruco_noise_m", "aruco_noise_rad", "aruco_bias_m"):
            rz[k] = rz[k] * 2
    elif name == "motor_delay_120ms":
        rz["motor_delay_ms"] = [120, 120]
    elif name == "steer_bias_3deg":
        b = 3.0 if seed % 2 == 0 else -3.0  # half the episodes +3°, half −3°
        rz["steer_bias_deg"] = [b, b]
    elif name == "low_friction":
        # the base is driven by velocity actuators, so floor friction only acts on the pallet; traction loss of
        # the drive wheel is the wheel-slip factor (actual / commanded speed)
        rz["floor_friction"] = [0.3, 0.5]
        rz["wheel_slip"] = [0.80, 0.95]
    elif name != "nominal":
        raise ValueError(name)
    return cfg


def contact_cause(env: ForkliftDockingEnv) -> str:
    """Which obstacle the forklift touched (or 'pallet_pushed' when it moved the pallet without a robot contact)."""
    m, d = env.model, env.data
    hit = set()
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        for a, b in ((g1, g2), (g2, g1)):
            if a in env._robot_geoms and b in env._obstacle_geoms:
                hit.add(f"{m.geom(a).name}->{m.geom(b).name}")
    return ",".join(sorted(hit)) or "pallet_pushed"


class OnnxPolicy:
    def __init__(self, path: Path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {"obs": obs[None].astype(np.float32)})[0][0]


def make_env(condition: str, seed: int, supervisor: str | None):
    base = ForkliftDockingEnv(config=stress_config(condition, seed), randomize=True)
    if not supervisor:
        return base
    mask, _, stop = supervisor.partition("+stop@")  # e.g. "front_all+stop@0.03" (proposed rule)
    return SafetyWrapper(base, docking_ignore=mask, docking_stop_m=float(stop) if stop else None)


def run_chunk(job: tuple) -> list[dict]:
    controller, supervisor, condition, seeds = job
    envs: dict[int, object] = {}
    policy = OnnxPolicy(MODEL / "policy.onnx") if controller == "policy" else None
    out = []
    for seed in seeds:
        key = seed % 2 if condition == "steer_bias_3deg" else 0
        if key not in envs:
            envs[key] = make_env(condition, seed, supervisor)
        env = envs[key]
        base = env.unwrapped
        ctrl = policy if policy else BaselineController(base.cfg)
        ctrl.reset()
        obs, info = env.reset(seed=seed)
        done = term = False
        while not done:
            obs, _, term, trunc, info = env.step(ctrl.act(obs))
            done = term or trunc
        aborted = bool(info.get("safety_abort", False))
        rec = {"seed": seed, "success": bool(info["success"]), "collision": bool(info["collision"]),
               "aborted": aborted, "out_of_range": bool(term and not info["success"] and not info["collision"]),
               "timeout": bool(trunc and not aborted), "steps": int(info["steps"]),
               "lat_mm": abs(info["lat_tip"]) * 1000, "elevation": info["elevation"]}
        if info["collision"]:
            rec["cause"] = contact_cause(base)
        if supervisor:
            rec["rules"] = info["safety"]["rule_steps"]
            rec["modified_steps"] = info["safety"]["modified_steps"]
        out.append(rec)
    return out


def run_all(runs: list[tuple], seeds: list[int], workers: int) -> dict[tuple, list]:
    chunk = max(10, len(seeds) // (workers * 2))
    jobs, owners = [], []
    for run in runs:
        for i in range(0, len(seeds), chunk):
            jobs.append((*run, seeds[i:i + chunk]))
            owners.append(run)
    results: dict[tuple, list] = {r: [] for r in runs}
    with ProcessPoolExecutor(workers) as ex:
        for run, recs in zip(owners, ex.map(run_chunk, jobs)):
            results[run].extend(recs)
    return results


def summarise(recs: list[dict]) -> dict:
    n = len(recs)
    ok = [r for r in recs if r["success"]]
    s = {"episodes": n,
         "success_rate": len(ok) / n,
         "collision_rate": sum(r["collision"] for r in recs) / n,
         "timeout_rate": sum(r["timeout"] for r in recs) / n,
         "safety_abort_rate": sum(r["aborted"] for r in recs) / n,
         "out_of_range_rate": sum(r["out_of_range"] for r in recs) / n,
         "mean_time_s": float(np.mean([r["steps"] for r in ok])) / 20 if ok else None,
         "mean_lateral_mm": float(np.mean([r["lat_mm"] for r in ok])) if ok else None}
    if "rules" in recs[0]:
        steps = sum(r["steps"] for r in recs)
        s["interventions"] = {
            rule: {"episodes": sum(r["rules"][rule] > 0 for r in recs), "steps": sum(r["rules"][rule] for r in recs)}
            for rule in RULES}
        s["modified_step_share"] = sum(r["modified_steps"] for r in recs) / steps
    causes = Counter(r["cause"] for r in recs if r.get("cause"))
    if causes:
        s["collision_causes"] = dict(causes.most_common())
    return s


def outcome(r: dict) -> str:
    for k in ("success", "collision", "aborted", "timeout", "out_of_range"):
        if r[k]:
            return k
    return "other"


def paired(alone: list[dict], sup: list[dict]) -> dict:
    """What the supervisor did to the same seeds: collisions prevented, successes lost."""
    a = {r["seed"]: r for r in alone}
    b = {r["seed"]: r for r in sup}
    coll = [s for s in a if a[s]["collision"]]
    good = [s for s in a if a[s]["success"]]
    after_coll = Counter(outcome(b[s]) for s in coll)
    after_good = Counter(outcome(b[s]) for s in good)
    return {"alone_collisions": len(coll),
            "collisions_prevented": sum(not b[s]["collision"] for s in coll),
            "collisions_turned_into_success": after_coll.get("success", 0),
            "outcome_of_alone_collisions": dict(after_coll),
            "alone_successes": len(good),
            "successes_lost": len(good) - after_good.get("success", 0),
            "outcome_of_alone_successes": dict(after_good),
            "new_collisions": sum(b[s]["collision"] and not a[s]["collision"] for s in a),
            "new_timeouts_or_aborts": sum((b[s]["timeout"] or b[s]["aborted"])
                                          and not (a[s]["timeout"] or a[s]["aborted"]) for s in a),
            "collision_seeds_alone": sorted(coll)[:50]}


def tune_docking_stop(workers: int) -> tuple[float, dict]:
    """Pick the threshold of the PROPOSED docking stop on the validation seeds 20000-20399 (never the test seeds):
    max(success - 2*collision) summed over policy and baseline, the rule select_checkpoint.py uses."""
    seeds = list(range(VAL0, VAL0 + 400))
    cands = (0.015, 0.02, 0.025, 0.03, 0.04)
    runs = [(c, f"front_all+stop@{t}", "nominal") for t in cands for c in ("policy", "baseline")]
    res = run_all(runs, seeds, workers)
    table = {}
    for t in cands:
        table[str(t)] = {}
        for c in ("policy", "baseline"):
            s = summarise(res[(c, f"front_all+stop@{t}", "nominal")])
            table[str(t)][c] = {k: s[k] for k in ("success_rate", "collision_rate", "safety_abort_rate")}

    def score(t: float) -> float:
        return round(sum(table[str(t)][c]["success_rate"] - 2 * table[str(t)][c]["collision_rate"]
                         for c in ("policy", "baseline")), 6)

    best = max(reversed(cands), key=score)  # ties -> the larger (more protective) threshold
    return best, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--out", default=str(MODEL / "safety_eval.json"))
    args = ap.parse_args()

    t0 = time.time()
    stop_m, tuning = tune_docking_stop(args.workers)
    proposal = f"front_all+stop@{stop_m}"
    print(f"proposed docking stop: {stop_m} m (validation seeds)", json.dumps(tuning))
    runs = [(c, sup, "nominal") for c in ("policy", "baseline") for sup in (None, "centre_only", "firmware", proposal)]
    if not args.no_stress:
        runs += [("policy", sup, cond) for cond in STRESS for sup in (None, "front_all", proposal)]
    results = run_all(runs, list(range(SEED0, SEED0 + args.episodes)), args.workers)
    name = lambda r: r[0] + (f"+supervisor[{r[1]}]" if r[1] else "")  # noqa: E731

    nominal = {name(r): summarise(v) for r, v in results.items() if r[2] == "nominal"}
    comparisons = {f"{c}: supervisor[{m}] vs alone": paired(results[(c, None, "nominal")], results[(c, m, "nominal")])
                   for c in ("policy", "baseline") for m in ("centre_only", "firmware", proposal)}
    stress: dict = {}
    for r, v in results.items():
        if r[2] != "nominal":
            stress.setdefault(r[2], {})[name(r)] = summarise(v)
    for cond in stress:
        stress[cond]["paired"] = paired(results[("policy", None, cond)], results[("policy", "front_all", cond)])
        stress[cond]["paired_proposal"] = paired(results[("policy", None, cond)], results[("policy", proposal, cond)])

    res = {
        "requirement": "AI-06: every policy command passes the safety supervisor (0.1 m/s, ToF) before the ESP32",
        "model": "models/docking_v3/policy.onnx (onnxruntime, CPU)",
        "seeds": f"{SEED0}-{SEED0 + args.episodes - 1}",
        "layers": {"jetson": "robot repo ros2_ws/src/forklift_safety/forklift_safety/safety_core.py (imported as is)",
                   "esp32": "forklift_rl/firmware_limiter.py (port of lib/forklift_core, checked against the C++ in "
                            "tests/test_firmware_limiter.py)"},
        "docking_masks": {
            "firmware": "SF-03 ignores all three front ToF sensors while docking (config.h TOF_MASK_DOCKING_IGNORE, "
                        "fixed after this evaluation found the centre-only mask blocked every docking)",
            "centre_only": "HISTORY: the original mask, centre front ToF only — blocked 99.5 % of dockings",
            proposal: "front_all + PROPOSED rule (not on the robot): while docking, hard stop when a front-corner ToF "
                      f"reads < {stop_m} m (forks fully in / mast at the rack)"},
        "proposed_docking_stop": {"chosen_m": stop_m, "chosen_on": f"validation seeds {VAL0}-{VAL0 + 399}",
                                  "rule": "max(success - 2*collision), policy + baseline", "candidates": tuning},
        "abort_after_s": 3.0,
        "nominal": nominal,
        "paired": comparisons,
        "stress": stress,
        "wall_clock_min": round((time.time() - t0) / 60, 1),
    }
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in nominal.items():
        print(f"{k:42s} docked {v['success_rate']:.1%}  coll {v['collision_rate']:.1%}  "
              f"timeout {v['timeout_rate']:.1%}  abort {v['safety_abort_rate']:.1%}  t {v['mean_time_s']}")
    for k, v in comparisons.items():
        print(k, {x: v[x] for x in ("alone_collisions", "collisions_prevented", "successes_lost", "new_collisions")})
    for cond, d in stress.items():
        for k, v in d.items():
            if not k.startswith("paired"):
                print(f"{cond:18s} {k:42s} docked {v['success_rate']:.1%}  coll {v['collision_rate']:.1%}  "
                      f"abort {v['safety_abort_rate']:.1%}")
    print(f"wrote {args.out} in {res['wall_clock_min']} min")


if __name__ == "__main__":
    main()
