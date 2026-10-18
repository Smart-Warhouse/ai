# ai — docking simulator, RL policy and vision

Part of the [Smart-Warhouse](https://github.com/Smart-Warhouse) project ([docs](https://github.com/Smart-Warhouse/docs)).

| Folder | What | Docs |
|---|---|---|
| [`training/`](training/) | MuJoCo + Gymnasium pallet-docking simulator, classical baseline, imitation + PPO policy (98.6 % docked on 1000 test episodes), ONNX export, safety evaluation | [11-simulation-training](https://github.com/Smart-Warhouse/docs/blob/main/11-simulation-training.md) |
| [`vision/`](vision/) | YOLO pipeline: capture, dataset checks, training, ONNX/TensorRT export, inference wrapper | [16-vision](https://github.com/Smart-Warhouse/docs/blob/main/16-vision.md) |

The training safety tests replay the firmware limiter and the Jetson supervisor from the
[robot](https://github.com/Smart-Warhouse/robot) repo: clone the repos side by side as described in the
[docs README](https://github.com/Smart-Warhouse/docs#workspace-layout), or set `SW_WORKSPACE`.
