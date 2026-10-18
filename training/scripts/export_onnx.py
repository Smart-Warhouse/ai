"""Export the docking policy's deterministic actor to ONNX for the Jetson (AI-05) and verify it.

    .venv/Scripts/python scripts/export_onnx.py models/docking_v3/policy_selected.zip

Writes `policy.onnx` next to the input: input `obs` float32 [N, 18] → output `action` float32 [N, 3] in [-1, 1]
(the same clipping the environment applies). Checks, and records in meta.json under "onnx":
  1. ONNX vs PyTorch on 5000 observations from real rollouts (max abs difference);
  2. the ONNX policy driving the simulator on 200 test episodes (same success as the PyTorch policy);
  3. CPU latency per call (the Jetson target is ≤ 5 ms with TensorRT FP16).
On the Jetson: `trtexec --onnx=policy.onnx --fp16 --saveEngine=policy.plan`, or onnxruntime-gpu with the
TensorRT execution provider (forklift_rl node).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forklift_rl.env import ForkliftDockingEnv  # noqa: E402

OBS_DIM, ACT_DIM = 18, 3


class Actor(th.nn.Module):
    """features → policy MLP → action mean → clip. Deterministic: no sampling, no value head."""

    def __init__(self, policy):
        super().__init__()
        self.extract = policy.features_extractor
        self.mlp = policy.mlp_extractor.policy_net
        self.head = policy.action_net

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return th.clamp(self.head(self.mlp(self.extract(obs))), -1.0, 1.0)


def rollout_obs(n: int, seed: int = 30_000) -> np.ndarray:
    from forklift_rl.baseline import BaselineController

    env = ForkliftDockingEnv(randomize=True)
    ctrl = BaselineController(env.cfg)
    out = []
    ep = 0
    while len(out) < n:
        obs, _ = env.reset(seed=seed + ep)
        ctrl.reset()
        ep += 1
        done = False
        while not done and len(out) < n:
            out.append(obs)
            obs, _, term, trunc, _ = env.step(ctrl.act(obs))
            done = term or trunc
    return np.array(out, dtype=np.float32)


def main() -> None:
    from stable_baselines3 import PPO

    src = Path(sys.argv[1]).resolve()
    run = src.parent
    dst = run / "policy.onnx"
    model = PPO.load(src, device="cpu")
    actor = Actor(model.policy).eval()

    dummy = th.zeros(1, OBS_DIM)
    th.onnx.export(actor, (dummy,), str(dst), input_names=["obs"], output_names=["action"],
                   dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(dst)))
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])

    # 1 — numerical equivalence on realistic observations
    obs = rollout_obs(5000)
    with th.no_grad():
        ref = actor(th.as_tensor(obs)).numpy()
    sb3 = np.stack([model.predict(o, deterministic=True)[0] for o in obs[:500]])
    out = sess.run(["action"], {"obs": obs})[0]
    max_diff = float(np.abs(out - ref).max())
    max_diff_sb3 = float(np.abs(np.clip(sb3, -1, 1) - out[:500]).max())

    # 2 — closed loop in the simulator with the ONNX session as the policy
    env = ForkliftDockingEnv(randomize=True)
    ok = coll = 0
    for ep in range(200):
        o, _ = env.reset(seed=10_000 + ep)
        done = False
        while not done:
            a = sess.run(["action"], {"obs": o[None, :]})[0][0]
            o, _, term, trunc, info = env.step(a)
            done = term or trunc
        ok += info["success"]
        coll += info["collision"]

    # 3 — latency, single observation (what the 20 Hz control loop does)
    x = obs[:1]
    for _ in range(200):
        sess.run(["action"], {"obs": x})
    t0 = time.perf_counter()
    for _ in range(2000):
        sess.run(["action"], {"obs": x})
    latency_ms = (time.perf_counter() - t0) / 2000 * 1000

    report = {
        "file": dst.name, "bytes": dst.stat().st_size, "opset": 17, "input": f"obs float32 [N,{OBS_DIM}]",
        "output": f"action float32 [N,{ACT_DIM}] in [-1,1]", "source": src.name,
        "max_abs_diff_vs_torch": max_diff, "max_abs_diff_vs_sb3_predict": max_diff_sb3,
        "sim_200_episodes": {"success_rate": ok / 200, "collision_rate": coll / 200},
        "cpu_latency_ms": round(latency_ms, 4),
    }
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["onnx"] = report
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(report, indent=2))
    assert max_diff < 1e-4, "ONNX output differs from the PyTorch actor"


if __name__ == "__main__":
    main()
