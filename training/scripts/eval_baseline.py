"""Score the classical controller on the same 1000 randomised episodes used for the RL gate (AI-07 reference).

    .venv/Scripts/python scripts/eval_baseline.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forklift_rl.baseline import BaselineController  # noqa: E402
from forklift_rl.env import ForkliftDockingEnv  # noqa: E402


def main(episodes: int = 1000, seed: int = 10_000) -> dict:
    env = ForkliftDockingEnv(randomize=True)
    ctrl = BaselineController(env.cfg)
    ok = coll = 0
    steps, lat = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ctrl.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(ctrl.act(obs))
            done = term or trunc
        ok += info["success"]
        coll += info["collision"]
        if info["success"]:
            steps.append(info["steps"])
            lat.append(abs(info["lat_tip"]))
    res = {"controller": "baseline", "episodes": episodes, "success_rate": ok / episodes, "collision_rate": coll / episodes,
           "mean_time_s": float(np.mean(steps)) / 20, "mean_lateral_mm": float(np.mean(lat)) * 1000}
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "baseline_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
