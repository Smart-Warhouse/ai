"""Render a demo video of the docking simulator: random actions → classical baseline → trained PPO policy.

    .venv/Scripts/python scripts/record_video.py --model models/docking_v1/policy.zip --out out/docking_demo.mp4

Each frame: main 3-D view, fork-camera inset (what the ArUco detector sees), top view inset, live HUD.
Episodes are fixed scenarios (listed in SCENES) so the video is reproducible; success rates on the end card come
from the 1000-episode evaluation in the model's meta.json, not from these clips.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forklift_rl.baseline import BaselineController  # noqa: E402
from forklift_rl.env import ForkliftDockingEnv  # noqa: E402

W, H = 1920, 1080
INSET_W, INSET_H = 480, 270
FPS = 40  # control runs at 20 Hz → 2× speed
YELLOW, DARK, WHITE, GREY = (245, 180, 0), (12, 22, 30), (255, 255, 255), (163, 180, 193)
GREEN, RED = (34, 197, 94), (239, 68, 68)

SCENES = [
    # (controller, title, reset options)
    ("random", "Before training — random actions", {"distance": 0.40, "lateral": 0.04, "yaw_deg": 8, "elevation": 0.0, "lift": 0.0}),
    ("baseline", "Classical controller (baseline)", {"distance": 0.50, "lateral": -0.06, "yaw_deg": -12, "elevation": 0.0, "lift": 0.03}),
    ("baseline", "Classical controller — rack level 2", {"distance": 0.45, "lateral": 0.05, "yaw_deg": 10, "elevation": 0.06, "lift": 0.0}),
    ("ppo", "Trained RL policy (PPO)", {"distance": 0.55, "lateral": 0.08, "yaw_deg": 15, "elevation": 0.0, "lift": 0.05}),
    ("ppo", "Trained RL policy — rack level 2", {"distance": 0.40, "lateral": -0.07, "yaw_deg": -18, "elevation": 0.06, "lift": 0.0}),
    ("ppo", "Trained RL policy — hard start", {"distance": 0.32, "lateral": 0.09, "yaw_deg": -20, "elevation": 0.0, "lift": 0.08}),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE, F_BIG, F_MED, F_SMALL = font(64, True), font(40, True), font(28), font(22)


class Writer:
    def __init__(self, path: Path):
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def add(self, img: Image.Image | np.ndarray, repeat: int = 1) -> None:
        arr = np.asarray(img, dtype=np.uint8)
        for _ in range(repeat):
            self.proc.stdin.write(arr.tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def card(lines: list[tuple[str, ImageFont.FreeTypeFont, tuple]], seconds: float, w: Writer) -> None:
    img = Image.new("RGB", (W, H), DARK)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 10], fill=YELLOW)
    total = sum(f.size * 1.5 for _, f, _ in lines)
    y = (H - total) / 2
    for text, f, color in lines:
        tw = d.textlength(text, font=f)
        d.text(((W - tw) / 2, y), text, font=f, fill=color)
        y += f.size * 1.5
    w.add(img, int(seconds * FPS))


class Views:
    def __init__(self, env: ForkliftDockingEnv):
        self.env = env
        self.main = mujoco.Renderer(env.model, H, W)
        self.small = mujoco.Renderer(env.model, INSET_H, INSET_W)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 1
        self.opt.sitegroup[0] = 0
        self.cam = mujoco.MjvCamera()

    def _cam(self, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
        pts = self.env._points()
        mid = (pts["base"] + np.array([0.12, 0.0])) / 2
        self.cam.lookat[:] = [mid[0], mid[1], 0.07 + self.env._elevation / 2]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        return self.cam

    def frame(self) -> Image.Image:
        d = self.env.data
        self.main.update_scene(d, camera=self._cam(0.95, -100, -22), scene_option=self.opt)
        img = Image.fromarray(self.main.render())
        self.small.update_scene(d, camera="fork_cam", scene_option=self.opt)
        fork = Image.fromarray(self.small.render())
        self.small.update_scene(d, camera=self._cam(1.0, -150, -55), scene_option=self.opt)
        top = Image.fromarray(self.small.render())
        x = W - INSET_W - 30
        draw = ImageDraw.Draw(img, "RGBA")
        for i, (inset, label) in enumerate(((fork, "Fork camera — what the ArUco detector sees"), (top, "Overview"))):
            y = 30 + i * (INSET_H + 60)
            draw.rectangle([x - 4, y - 4, x + INSET_W + 4, y + INSET_H + 40], fill=DARK)
            img.paste(inset, (x, y))
            draw.text((x + 10, y + INSET_H + 6), label, font=F_SMALL, fill=WHITE)
        return img


def hud(img: Image.Image, title: str, info: dict, obs: np.ndarray, t: float, result: str | None) -> Image.Image:
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle([30, 30, 760, 380], radius=18, fill=(12, 22, 30, 225))
    d.rectangle([30, 30, 760, 38], fill=YELLOW)
    d.text((56, 52), title, font=F_BIG, fill=WHITE)
    rows = [
        ("Insertion depth", f"{info['depth'] * 100:6.1f} cm   (goal ≥ 14.4)"),
        ("Lateral error", f"{info['lat_tip'] * 1000:6.1f} mm   (goal ≤ 10)"),
        ("Heading error", f"{math.degrees(info['heading']):6.1f} °"),
        ("Fork height error", f"{info['z_err'] * 1000:6.1f} mm"),
        ("Marker visible", "yes" if obs[5] > 0.5 else "no — odometry"),
        ("Time", f"{t:5.1f} s   (2× speed)"),
    ]
    y = 120
    for k, v in rows:
        d.text((56, y), k, font=F_SMALL, fill=GREY)
        d.text((300, y), v, font=F_SMALL, fill=WHITE)
        y += 38
    for i, on in enumerate(obs[12:14]):
        cx = 600 + i * 60
        d.ellipse([cx, 312, cx + 34, 346], fill=GREEN if on > 0.5 else (60, 72, 84))
    d.text((520, 318), "TCRT", font=F_SMALL, fill=GREY)
    if result:
        color = GREEN if result == "DOCKED" else RED
        tw = d.textlength(result, font=F_TITLE)
        d.rounded_rectangle([(W - tw) / 2 - 40, H - 170, (W + tw) / 2 + 40, H - 70], radius=20, fill=color + (235,))
        d.text(((W - tw) / 2, H - 162), result, font=F_TITLE, fill=WHITE)
    return img


def run_scene(env, views, writer, controller: str, title: str, opts: dict, policy, rng) -> str:
    obs, info = env.reset(seed=7, options=opts)
    base = BaselineController(env.cfg)
    result = None
    for step in range(env.cfg["control"]["max_steps"]):
        if controller == "random":
            act = rng.uniform(-1, 1, 3).astype(np.float32)
            act[0] = abs(act[0])
        elif controller == "baseline":
            act = base.act(obs)
        else:
            act, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(act)
        if term or trunc:
            result = "DOCKED" if info["success"] else "COLLISION" if info["collision"] else "TIMEOUT" if trunc else "LOST"
        writer.add(hud(views.frame(), title, info, obs, (step + 1) / 20, result if (term or trunc) else None))
        if term or trunc:
            writer.add(hud(views.frame(), title, info, obs, (step + 1) / 20, result), int(1.4 * FPS))
            break
    return result or "TIMEOUT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "docking_v1" / "policy.zip")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "docking_demo.mp4")
    args = ap.parse_args()
    from stable_baselines3 import PPO

    policy = PPO.load(args.model, device="cpu") if args.model.exists() else None
    meta_path = args.model.parent / "meta.json"  # the run folder's meta (selection results included)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    base_stats = json.loads((ROOT / "models" / "baseline_eval.json").read_text()) if (ROOT / "models" / "baseline_eval.json").exists() else {}

    env = ForkliftDockingEnv(randomize=True)
    views = Views(env)
    writer = Writer(args.out)
    rng = np.random.default_rng(3)
    card([("Forklift pallet-docking simulator", F_TITLE, WHITE),
          ("MuJoCo physics · Gymnasium env · PPO (Stable-Baselines3)", F_MED, YELLOW),
          ("1:14 forklift with rear-wheel steering · ArUco pallet pose · ToF ring · TCRT fork sensors", F_MED, GREY),
          ("Domain randomisation: friction, motor delay 20–80 ms, marker noise ±5 mm, wheel slip, load", F_MED, GREY)],
         4.0, writer)
    results = []
    for controller, title, opts in SCENES:
        if controller == "ppo" and policy is None:
            continue
        results.append((title, run_scene(env, views, writer, controller, title, opts, policy, rng)))

    lines = [("Results on 1000 random episodes", F_BIG, WHITE)]
    if base_stats:
        lines.append((f"Classical baseline: {base_stats['success_rate']:.1%} docked · "
                      f"{base_stats['collision_rate']:.1%} collisions · {base_stats['mean_time_s']:.1f} s", F_MED, GREY))
    if meta.get("behaviour_cloning"):
        b = meta["behaviour_cloning"]["bc_only_eval"]
        lines.append((f"After imitation only (200 episodes): {b['success_rate']:.1%} docked · "
                      f"{b['collision_rate']:.1%} collisions · {b['mean_time_s'] or 0:.1f} s", F_MED, GREY))
    if meta:
        e = meta["eval_1000"]
        chosen = meta.get("selected", {}).get("name", f"{meta['timesteps'] / 1e6:.0f} M steps")
        lines.append((f"Imitation + PPO {meta['version']} ({chosen}, {meta['wall_clock_min']:.0f} min on CPU): "
                      f"{e['success_rate']:.1%} docked · {e['collision_rate']:.1%} collisions · {e['mean_time_s'] or 0:.1f} s", F_MED, YELLOW))
        lines.append((f"AI-04 gate (≥ 95 % in simulation before real tests): {'PASSED' if meta['ai04_gate_passed'] else 'not yet'}",
                      F_MED, GREEN if meta["ai04_gate_passed"] else RED))
    lines.append(("Next: export ONNX → TensorRT FP16 on the Jetson (AI-05), 30 + 30 real trials (AI-07)", F_SMALL, GREY))
    card(lines, 5.0, writer)
    writer.close()
    print("wrote", args.out, results)


if __name__ == "__main__":
    main()
