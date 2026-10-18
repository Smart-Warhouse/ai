"""Train the docking policy with PPO (AI-02). Writes checkpoints, TensorBoard logs and a versioned model (AI-09).

    .venv/Scripts/python scripts/train_ppo.py --steps 2000000 --envs 16
    .venv/Scripts/tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forklift_rl.env import ForkliftDockingEnv  # noqa: E402


def make_env(rank: int, seed: int):
    def _init():
        env = ForkliftDockingEnv(randomize=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def evaluate(model, episodes: int, seed: int = 10_000, deterministic: bool = True) -> dict:
    env = ForkliftDockingEnv(randomize=True)
    ok = coll = 0
    steps, lat = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        ok += info["success"]
        coll += info["collision"]
        if info["success"]:
            steps.append(info["steps"])
            lat.append(abs(info["lat_tip"]))
    env.close()
    return {"episodes": episodes, "success_rate": ok / episodes, "collision_rate": coll / episodes,
            "mean_time_s": float(np.mean(steps)) / 20 if steps else None,
            "mean_lateral_mm": float(np.mean(lat)) * 1000 if lat else None}


class SuccessEval(BaseCallback):
    def __init__(self, every: int, episodes: int, out: Path):
        super().__init__()
        self.every, self.episodes, self.out = every, episodes, out
        self.next = every
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next:
            self.next += self.every
            r = evaluate(self.model, self.episodes)
            r["timesteps"] = self.num_timesteps
            self.history.append(r)
            self.logger.record("eval/success_rate", r["success_rate"])
            self.logger.record("eval/collision_rate", r["collision_rate"])
            print(f"[eval] {self.num_timesteps:>9,d} steps  success {r['success_rate']:.0%}  collision {r['collision_rate']:.0%}",
                  flush=True)
            (self.out / "eval_history.json").write_text(json.dumps(self.history, indent=2))
        return True


class Curriculum(BaseCallback):
    """Widen the start distribution whenever the rolling training success rate exceeds `threshold`."""

    def __init__(self, start: float = 0.15, step: float = 0.1, threshold: float = 0.6, window: int = 300):
        super().__init__()
        self.level, self.step_, self.threshold, self.window = start, step, threshold, window
        self.results: list[bool] = []

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_difficulty", self.level)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.results.append(bool(info.get("success")))
        if len(self.results) >= self.window:
            rate = sum(self.results[-self.window:]) / self.window
            self.logger.record("curriculum/train_success", rate)
            if rate >= self.threshold and self.level < 1.0:
                self.level = min(1.0, self.level + self.step_)
                self.training_env.env_method("set_difficulty", self.level)
                print(f"[curriculum] {self.num_timesteps:>9,d} steps  difficulty → {self.level:.2f} (train success {rate:.0%})",
                      flush=True)
                self.results.clear()
        self.logger.record("curriculum/difficulty", self.level)
        return True


def _demo_worker(args: tuple[int, int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Roll out the classical controller; returns obs, actions, discounted returns, successes."""
    from forklift_rl.baseline import BaselineController

    first_seed, n, gamma = args
    env = ForkliftDockingEnv(randomize=True)
    ctrl = BaselineController(env.cfg)
    O, A, G = [], [], []
    wins = 0
    for ep in range(n):
        obs, _ = env.reset(seed=first_seed + ep)
        ctrl.reset()
        ep_o, ep_a, ep_r = [], [], []
        done = False
        while not done:
            act = ctrl.act(obs)
            ep_o.append(obs)
            ep_a.append(act)
            obs, r, term, trunc, info = env.step(act)
            ep_r.append(r)
            done = term or trunc
        wins += info["success"]
        g, ret = 0.0, []
        for r in reversed(ep_r):
            g = r + gamma * g
            ret.append(g)
        O += ep_o
        A += ep_a
        G += ret[::-1]
    return np.array(O, np.float32), np.clip(np.array(A, np.float32), -1, 1), np.array(G, np.float32), wins


def pretrain_from_demos(model: PPO, episodes: int, workers: int, epochs: int = 60) -> dict:
    """Behaviour cloning of the classical controller + value-function regression on its returns.

    Gives PPO a policy that already docks, so it improves on the baseline instead of having to discover a
    millimetre-precise insertion by random exploration (which made it learn to stand still — runs v1/v2).
    """
    import multiprocessing as mp

    import torch as th

    per = max(1, episodes // workers)
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_demo_worker, [(500_000 + i * per, per, model.gamma) for i in range(workers)])
    obs = np.concatenate([p[0] for p in parts])
    act = np.concatenate([p[1] for p in parts])
    ret = np.concatenate([p[2] for p in parts])
    demo_success = sum(p[3] for p in parts) / (per * workers)
    pol = model.policy
    o_t, a_t, g_t = (th.as_tensor(x, device=pol.device) for x in (obs, act, ret))
    n = len(obs)

    def run(phase: str, params, n_epochs: int) -> None:
        opt = th.optim.Adam(params, lr=1e-3)
        sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
        for epoch in range(n_epochs):
            perm = th.randperm(n, device=pol.device)
            tot = 0.0
            for i in range(0, n, 2048):
                idx = perm[i:i + 2048]
                latent_pi, latent_vf = pol.mlp_extractor(pol.extract_features(o_t[idx]))
                if phase == "actor":
                    loss = ((pol.action_net(latent_pi) - a_t[idx]) ** 2).mean()
                else:
                    loss = ((pol.value_net(latent_vf).squeeze(-1) - g_t[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            sched.step()
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"[bc] {phase:5s} epoch {epoch:>3d}  mse {tot / n:.4f}", flush=True)

    actor_params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    critic_params = list(pol.mlp_extractor.value_net.parameters()) + list(pol.value_net.parameters())
    run("actor", actor_params, epochs)
    run("value", critic_params, max(10, epochs // 2))
    with th.no_grad():
        pol.log_std.fill_(-1.4)  # σ ≈ 0.25: explore around the demonstrated behaviour, not over it
    return {"demo_episodes": per * workers, "demo_steps": int(n), "demo_success": demo_success}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", default=None, help="model version tag, e.g. v1 (default: next free)")
    ap.add_argument("--eval-every", type=int, default=200_000)
    ap.add_argument("--curriculum", action="store_true", help="start with easy poses and widen on success")
    ap.add_argument("--bc-episodes", type=int, default=0,
                    help="warm-start from N classical-controller demonstrations (behaviour cloning) before PPO")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=None, help="stop an update early above this KL (fine-tuning)")
    ap.add_argument("--checkpoint-every", type=int, default=500_000)
    args = ap.parse_args()

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    version = args.version or f"v{len(list(models.glob('docking_v*'))) + 1}"
    out = models / f"docking_{version}"
    out.mkdir(exist_ok=True)

    venv = VecMonitor(SubprocVecEnv([make_env(i, args.seed * 100) for i in range(args.envs)]))
    model = PPO(
        "MlpPolicy", venv, seed=args.seed, verbose=0, tensorboard_log=str(ROOT / "runs"),
        n_steps=512, batch_size=4096, n_epochs=8, learning_rate=args.lr, gamma=0.99, gae_lambda=0.95,
        clip_range=args.clip, target_kl=args.target_kl, ent_coef=0.0 if args.bc_episodes else 0.003, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    )
    t0 = time.time()
    bc = pretrain_from_demos(model, args.bc_episodes, args.envs) if args.bc_episodes else None
    if bc:
        bc["bc_only_eval"] = evaluate(model, 200)
        print("[bc] policy after cloning:", bc["bc_only_eval"], flush=True)
        model.save(out / "policy_bc")
    model.learn(
        total_timesteps=args.steps, tb_log_name=f"docking_{version}",
        callback=[CheckpointCallback(args.checkpoint_every // args.envs, str(out / "checkpoints"), name_prefix="ppo"),
                  SuccessEval(args.eval_every, 100, out)] + ([Curriculum()] if args.curriculum else []),
    )
    model.save(out / "policy")
    final = evaluate(model, 1000)  # AI-04 gate: ≥ 95 % on 1000 random episodes
    meta = {
        "version": version, "trained_at": datetime.now(timezone.utc).isoformat(), "timesteps": args.steps,
        "envs": args.envs, "seed": args.seed, "curriculum": args.curriculum, "learning_rate": args.lr, "clip_range": args.clip, "target_kl": args.target_kl,
        "behaviour_cloning": bc, "wall_clock_min": round((time.time() - t0) / 60, 1),
        "eval_1000": final, "ai04_gate_passed": final["success_rate"] >= 0.95,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    venv.close()


if __name__ == "__main__":
    main()
