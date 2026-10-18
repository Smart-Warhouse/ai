"""Evaluate a trained model on the held-out test split; writes `<model folder>/metrics.json` + `metrics.md`.

Two views:
  1. Ultralytics `val` on the .pt weights: mAP@0.5 and mAP@0.5:0.95 per class (PC-08's mAP@0.5).
  2. The exported ONNX through `YoloOnnxDetector` at the deployed operating point (conf 0.25, NMS IoU 0.7),
     matched at IoU 0.5: per-class precision / recall — PC-08's person recall — split by image source
     (file prefix of the pool, e.g. coco_person / synth_mujoco / a camera session), and CPU latency.

    python scripts/evaluate.py --weights models/starter_v1/starter_v1.pt --data datasets/starter_v1/data.yaml
"""

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forklift_vision import CLASSES  # noqa: E402
from forklift_vision.dataset import label_path_for, parse_label_file  # noqa: E402
from forklift_vision.detector import YoloOnnxDetector, postprocess, preprocess  # noqa: E402
from forklift_vision.metrics import match_image, summarize, yolo_to_xyxy  # noqa: E402


import _ultralytics_local  # noqa: E402  (before any Ultralytics import)


def ultralytics_val(weights: Path, data: Path, split: str, device: str) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data), split=split, imgsz=640, batch=16, device=device, plots=False,
                               verbose=False, project=str(ROOT / "runs"), name="eval", exist_ok=True)
    names = m.names
    per = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per[names[int(c)]] = {"precision": float(p), "recall": float(r), "mAP50": float(ap50),
                              "mAP50_95": float(ap)}
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "per_class": per,
            "note": "precision/recall here are at the max-F1 confidence chosen by Ultralytics"}


def onnx_eval(onnx_path: Path, images: list[Path], conf: float, iou: float, threads: int) -> dict:
    det = YoloOnnxDetector(onnx_path, conf_thres=conf, iou_thres=iou, intra_op_threads=threads)
    nc = len(det.class_names)
    total = np.zeros((nc, 3), np.int64)
    by_source = defaultdict(lambda: np.zeros((nc, 3), np.int64))
    t_pre, t_inf, t_post = [], [], []
    for k, p in enumerate(images):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        x, infos = preprocess([img], det.input_size)
        t1 = time.perf_counter()
        out = det.raw(x)
        t2 = time.perf_counter()
        dets = postprocess(out, infos, conf, iou, 100, det.class_names)[0]
        t3 = time.perf_counter()
        if k >= 5:  # warm-up excluded
            t_pre.append(t1 - t0), t_inf.append(t2 - t1), t_post.append(t3 - t2)
        gts = yolo_to_xyxy(parse_label_file(label_path_for(p), nc).boxes, w, h)
        c = match_image(dets, gts, nc)
        total += c
        by_source[p.stem.split("__")[0]] += c
    ms = lambda v: {"mean": 1000 * float(np.mean(v)), "p95": 1000 * float(np.percentile(v, 95))}  # noqa: E731
    total_ms = [a + b + c for a, b, c in zip(t_pre, t_inf, t_post)]
    return {"conf_thres": conf, "nms_iou": iou, "match_iou": 0.5, "images": len(images),
            "per_class": summarize(total, det.class_names),
            "by_source": {s: summarize(c, det.class_names) for s, c in sorted(by_source.items())},
            "cpu_latency_ms": {"preprocess": ms(t_pre), "inference": ms(t_inf), "postprocess": ms(t_post),
                               "total": ms(total_ms), "batch": 1, "threads": threads or "onnxruntime default",
                               "cpu": platform.processor()}}


def fmt(v):
    return "–" if v is None else f"{v:.3f}"


def to_markdown(r: dict) -> str:
    u, o = r["ultralytics"], r["onnx"]
    lines = [f"# Test-split metrics — {r['model']}", "", r["label"], "",
             f"Test split: {o['images']} images ({r['test_split']}).", "",
             "| class | mAP@0.5 | mAP@0.5:0.95 | precision @0.25 | recall @0.25 |", "|---|---|---|---|---|"]
    for c in CLASSES:
        uc, oc = u["per_class"].get(c, {}), o["per_class"][c]
        lines.append(f"| {c} | {fmt(uc.get('mAP50'))} | {fmt(uc.get('mAP50_95'))} | {fmt(oc['precision'])} | "
                     f"{fmt(oc['recall'])} |")
    lines.append(f"| **all** | **{u['mAP50']:.3f}** | **{u['mAP50_95']:.3f}** | | |")
    lines += ["", "Person recall by image source (ONNX wrapper, conf 0.25):", ""]
    for s, v in o["by_source"].items():
        pr = v["person"]
        if pr["tp"] + pr["fn"]:
            lines.append(f"- {s}: {fmt(pr['recall'])} ({pr['tp']}/{pr['tp'] + pr['fn']})")
    lat = o["cpu_latency_ms"]
    lines += ["", f"CPU latency, batch 1, onnxruntime ({lat['cpu']}): total {lat['total']['mean']:.1f} ms mean / "
              f"{lat['total']['p95']:.1f} ms p95 (pre {lat['preprocess']['mean']:.1f}, inference "
              f"{lat['inference']['mean']:.1f}, post {lat['postprocess']['mean']:.1f})."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--onnx", type=Path, help="default: weights with .onnx")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--label", default="STARTER MODEL on non-final data (COCO persons + MuJoCo synthetic) — "
                                       "NOT the PC-08 acceptance result.")
    args = ap.parse_args()
    _ultralytics_local.apply()

    data = yaml.safe_load(args.data.read_text())
    img_dir = Path(data["path"]) / data[args.split]
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    onnx_path = args.onnx or args.weights.with_suffix(".onnx")
    report = {"model": args.weights.parent.name, "label": args.label, "test_split": str(img_dir),
              "ultralytics": ultralytics_val(args.weights, args.data, args.split, args.device),
              "onnx": onnx_eval(onnx_path, images, args.conf, args.iou, args.threads),
              "evaluated": time.strftime("%Y-%m-%d %H:%M")}
    out = args.weights.parent
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(report)
    (out / "metrics.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
