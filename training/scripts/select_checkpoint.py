"""Pick the best saved policy of a run on VALIDATION seeds, then report it once on the TEST seeds (AI-04, AI-09).

    .venv/Scripts/python scripts/select_checkpoint.py models/docking_v3

Candidates: policy_bc.zip (imitation only), every checkpoints/ppo_*_steps.zip, policy.zip (end of training).
Validation = seeds 20000-20399 (never used during training or for the test); test = seeds 10000-10999.
The winner is copied to policy_selected.zip and the choice is recorded in meta.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _eval_chunk(args: tuple[str, int, int]) -> tuple[int, int, list[float], list[float]]:
    from stable_baselines3 import PPO

    from forklift_rl.env import ForkliftDockingEnv

    path, first, n = args
    model = PPO.load(path, device="cpu")
    env = ForkliftDockingEnv(randomize=True)
    ok = coll = 0
    times, lat = [], []
    for s in range(first, first + n):
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        ok += info["success"]
        coll += info["collision"]
        if info["success"]:
            times.append(info["steps"] / 20)
            lat.append(abs(info["lat_tip"]) * 1000)
    return ok, coll, times, lat


def evaluate(path: Path, first_seed: int, episodes: int, pool) -> dict:
    workers = 16
    sizes = [episodes // workers + (1 if i < episodes % workers else 0) for i in range(workers)]
    starts = [first_seed + sum(sizes[:i]) for i in range(workers)]
    parts = pool.map(_eval_chunk, [(str(path), st, sz) for st, sz in zip(starts, sizes)])
    n = sum(sizes)
    ok = sum(p[0] for p in parts)
    coll = sum(p[1] for p in parts)
    times = [t for p in parts for t in p[2]]
    lat = [x for p in parts for x in p[3]]
    return {"episodes": n, "success_rate": ok / n, "collision_rate": coll / n,
            "mean_time_s": float(np.mean(times)) if times else None, "mean_lateral_mm": float(np.mean(lat)) if lat else None}


def main() -> None:
    run = Path(sys.argv[1]).resolve()
    candidates = []
    if (run / "policy_bc.zip").exists():
        candidates.append(("imitation only", run / "policy_bc.zip"))
    for ck in sorted((run / "checkpoints").glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1])):
        candidates.append((f"PPO {int(ck.stem.split('_')[1]):,} steps", ck))
    if (run / "policy.zip").exists():
        candidates.append(("PPO end of training", run / "policy.zip"))

    with mp.get_context("spawn").Pool(16) as pool:
        table = []
        for name, path in candidates:
            r = evaluate(path, 20_000, 400, pool)
            r.update(name=name, path=str(path.relative_to(ROOT)))
            table.append(r)
            print(f"[val] {name:<24s} success {r['success_rate']:.1%}  collision {r['collision_rate']:.1%}  "
                  f"time {r['mean_time_s'] or 0:.1f} s", flush=True)
        best = max(table, key=lambda r: (r["success_rate"] - 2 * r["collision_rate"], -(r["mean_time_s"] or 99)))
        test = evaluate(ROOT / best["path"], 10_000, 1000, pool)
    print(f"[test] selected {best['name']}: {test}", flush=True)

    shutil.copy(ROOT / best["path"], run / "policy_selected.zip")
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"version": run.name.split("_")[-1]}
    meta["validation"] = table
    meta["selected"] = {"name": best["name"], "path": best["path"], "rule": "max(success − 2·collision) on seeds 20000-20399"}
    meta["eval_1000"] = test
    meta["ai04_gate_passed"] = test["success_rate"] >= 0.95
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
