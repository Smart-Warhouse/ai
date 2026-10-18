# Training — forklift pallet-docking simulator

MuJoCo + Gymnasium environment of the 1:14 forklift docking into a pallet, a classical baseline controller and
PPO training (requirements §8). Design, research and results: `../../docs/11-simulation-training.md`.

```bash
cd code/training
python -m venv --system-site-packages .venv          # reuses an installed torch if present
.venv/Scripts/pip install mujoco gymnasium stable-baselines3 imageio imageio-ffmpeg tensorboard pytest opencv-python-headless pillow pyyaml

.venv/Scripts/python -m pytest -q tests                               # env + baseline tests
.venv/Scripts/python scripts/eval_baseline.py                         # classical controller, 1000 episodes
.venv/Scripts/python scripts/train_ppo.py --steps 3000000 --curriculum # → models/docking_vN/
.venv/Scripts/tensorboard --logdir runs
.venv/Scripts/python scripts/record_video.py --model models/docking_v2/policy.zip --out out/docking_demo.mp4
```

| Path | What |
|---|---|
| `forklift_rl/config.yaml` | every dimension, limit, randomisation range and reward weight |
| `forklift_rl/model.py` | MJCF scene (forklift, pallet with ArUco, rack shelf) |
| `forklift_rl/env.py` | `ForkliftDocking-v0` Gymnasium env |
| `forklift_rl/baseline.py` | classical visual-servoing controller (AI-07 reference) |
| `scripts/` | train, evaluate, record video |
| `models/` | versioned policies + `meta.json` (AI-09), `baseline_eval.json` |
