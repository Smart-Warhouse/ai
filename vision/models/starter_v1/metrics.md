# Test-split metrics — starter_v1

STARTER MODEL on non-final data (COCO persons + MuJoCo synthetic) — NOT the PC-08 acceptance result.

Test split: 222 images (C:\Users\ASUS\Desktop\Grad. Project\code\vision\datasets\starter_v1\images\test).

| class | mAP@0.5 | mAP@0.5:0.95 | precision @0.25 | recall @0.25 |
|---|---|---|---|---|
| person | 0.770 | 0.567 | 0.797 | 0.680 |
| pallet | 0.850 | 0.751 | 0.789 | 0.797 |
| forklift | 0.883 | 0.754 | 0.774 | 0.801 |
| obstacle | 0.890 | 0.787 | 0.824 | 0.824 |
| **all** | **0.848** | **0.715** | | |

Person recall by image source (ONNX wrapper, conf 0.25):

- coco_person: 0.566 (90/159)
- synth_mujoco: 0.931 (67/72)

CPU latency, batch 1, onnxruntime (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel): total 24.6 ms mean / 27.9 ms p95 (pre 4.3, inference 19.7, post 0.5).
