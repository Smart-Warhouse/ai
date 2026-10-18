# Vision — YOLO perception layer (PC-06 … PC-08)

YOLO detector for the fork-carriage camera: classes `person`, `pallet`, `forklift`, `obstacle` (PC-06, in that
order = class ids 0–3), input 640×640, TensorRT FP16 on the Jetson Orin Nano Super. It runs next to the ArUco
layer (`code/ros2_ws/src/forklift_perception`): ArUco gives the pallet pose, YOLO feeds SF-06 (person within
1 m → slow / stop). Design, data-collection protocol and acceptance tests: [docs/16-vision.md](../../docs/16-vision.md).
Data provenance and licences: [DATA_SOURCES.md](DATA_SOURCES.md).

> `models/starter_v1` is a **starter model on non-final data** (COCO persons + MuJoCo renders). It proves the
> pipeline; it is **not** the PC-08 result. When the forklift camera is mounted: capture → label → add → retrain.

## Setup

```bash
cd code/vision
python -m venv .venv                                    # Python 3.11 (3.12 install on this laptop is broken)
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # RTX 4050
.venv/Scripts/pip install -r requirements-train.txt     # Ultralytics, MuJoCo, onnxruntime, pytest
# runtime / CI only (no torch):  pip install -r requirements.txt
.venv/Scripts/python -m pytest -q tests                 # CPU, no GPU, no dataset download
```

## Layout

| Path | What |
|---|---|
| `forklift_vision/detector.py` | **onnxruntime wrapper** — letterbox, NMS, `YoloOnnxDetector`, `Detection` (numpy + OpenCV + onnxruntime only) |
| `forklift_vision/geometry.py` | box → floor distance (`person_distance` for the safety supervisor) |
| `forklift_vision/dataset.py` | label parser, dataset checker, deterministic 70/20/10 split, `data.yaml` |
| `forklift_vision/capture.py` | frame grabber (webcam / USB / video / RTSP / ROS 2 topic) with per-session folders |
| `forklift_vision/import_labels.py` | CVAT / Roboflow YOLO export → labelled pool, class ids remapped by name |
| `forklift_vision/metrics.py` | precision / recall at a fixed confidence (the deployed operating point) |
| `scripts/` | `check_dataset`, `split_dataset`, `train`, `evaluate`, `export_onnx`, `bench_onnx`, `fetch_coco_persons`, `make_synthetic` |
| `jetson/build_trt_engine.sh` | `trtexec` FP16 engine + latency for T-11 (run on the Jetson) |
| `models/starter_v1/` | `starter_v1.pt`, `starter_v1.onnx`, `metrics.json/md`, `train_meta.json`, training curves |
| `data_manifests/` | per-image licence / attribution list of the COCO images used |
| `datasets/`, `runs/`, `weights/` | generated, gitignored |

## Real data workflow (plan.md B6)

```bash
# 1. capture on the forklift (one session per robot × lighting level × area), every 1 s, skip blur/duplicates
python -m forklift_vision.capture --ros-topic /fl1/image_raw --label fl1_lux300_aisleA --lux 300 --robot fl1 --max 300
python -m forklift_vision.capture --source 0 --label bench_lux150 --lux 150 --preview      # USB camera on a laptop
# 2. label in CVAT or Roboflow (docs/16-vision.md §3), export YOLO, import (ids remapped by class NAME)
python -m forklift_vision.import_labels --src exports/fl1_lux300_aisleA --images datasets/capture/<session>/images \
    --dst datasets/raw/fl1_lux300_aisleA
# 3. check every pool, then build the split (old images never change split when pools are added)
python scripts/check_dataset.py datasets/raw/fl1_lux300_aisleA
python scripts/split_dataset.py --out datasets/forklift_v1 --chunk 10 datasets/raw/fl1_* datasets/raw/fl2_* ...
python scripts/check_dataset.py datasets/forklift_v1 --strict          # exit 1 below 400 img/class or broken labels
# 4. train, export, evaluate (≈ 1 min/epoch per 2000 images on the RTX 4050)
python scripts/train.py --data datasets/forklift_v1/data.yaml --name forklift_v1 --epochs 100
python scripts/export_onnx.py models/forklift_v1/forklift_v1.pt --parity-images datasets/forklift_v1/images/test
python scripts/evaluate.py --weights models/forklift_v1/forklift_v1.pt --data datasets/forklift_v1/data.yaml \
    --label "forklift_v1 on the real test split (PC-08)"
# 5. on the Jetson: TensorRT FP16 + latency (PC-07)
scp models/forklift_v1/forklift_v1.onnx jetson:~/ && ssh jetson ./build_trt_engine.sh forklift_v1.onnx
```

Starter model reproduction (≈ 5 min data + ≈ 40 min training):

```bash
python scripts/fetch_coco_persons.py --direct 700 --cutout-images 400
python scripts/make_synthetic.py --n 1800 --person-cutouts datasets/raw/coco_person_cutouts --person-prob 0.35
python scripts/split_dataset.py --out datasets/starter_v1 --chunk 20 datasets/raw/synth_mujoco datasets/raw/coco_person
python scripts/train.py --data datasets/starter_v1/data.yaml --name starter_v1 --epochs 40 --workers 0
python scripts/export_onnx.py models/starter_v1/starter_v1.pt --parity-images datasets/starter_v1/images/test
python scripts/evaluate.py --weights models/starter_v1/starter_v1.pt --data datasets/starter_v1/data.yaml
python scripts/bench_onnx.py models/starter_v1/starter_v1.onnx
```

`--workers 0`: on this laptop (16 GB, several sessions open) Windows dataloader workers ran out of commit memory.

The scripts keep Ultralytics' settings in `code/vision/.ultralytics/` (weights, runs, datasets under
`code/vision`, usage sync off) instead of the global `%APPDATA%/Ultralytics` file, whose defaults point at the
git root (`scripts/_ultralytics_local.py`).

## Wrapper API (for the ROS 2 node)

```python
from forklift_vision.detector import YoloOnnxDetector          # numpy, cv2, onnxruntime — no rclpy, no torch
from forklift_vision.geometry import Camera, nearest_distance

det = YoloOnnxDetector("starter_v1.onnx", conf_thres=0.25, iou_thres=0.7,
                       providers=["CPUExecutionProvider"])        # Jetson: TensorRT/CUDA EPs of onnxruntime-gpu
dets = det(bgr)                    # list[Detection]: class_id, class_name, score, x1, y1, x2, y2 (original pixels)
batch = det.detect_batch([a, b])   # one session.run, dynamic batch
cam = Camera.from_k(info.k, info.width, info.height, height_m=0.054 + lift_m, pitch_rad=0.349)
person_m = nearest_distance(cam, dets, "person", min_score=0.25)   # inf when none -> publish Float32 person_distance
```

Lower-level pieces for a TensorRT-engine backend: `preprocess(images) -> (NCHW float32, [LetterboxInfo])`,
`postprocess(raw (B, 8, 8400), infos, conf, iou)`, `letterbox`, `nms`. Details in docs/16-vision.md §6.
