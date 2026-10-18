"""End-to-end latency of YoloOnnxDetector (preprocess + onnxruntime + postprocess) at batch 1.

Runs anywhere onnxruntime runs: the laptop CPU here, and the Jetson with onnxruntime-gpu
(`--providers TensorrtExecutionProvider CUDAExecutionProvider CPUExecutionProvider`).

    python scripts/bench_onnx.py models/starter_v1/starter_v1.onnx --n 200
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forklift_vision.detector import YoloOnnxDetector, postprocess, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--providers", nargs="+")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    det = YoloOnnxDetector(args.onnx, providers=args.providers, intra_op_threads=args.threads)
    rng = np.random.default_rng(0)
    img = (rng.random((args.height, args.width, 3)) * 255).astype(np.uint8)
    stages = {"preprocess": [], "inference": [], "postprocess": []}
    for i in range(args.warmup + args.n):
        t0 = time.perf_counter()
        x, infos = preprocess([img], det.input_size)
        t1 = time.perf_counter()
        out = det.raw(x)
        t2 = time.perf_counter()
        postprocess(out, infos, det.conf_thres, det.iou_thres, det.max_det, det.class_names)
        t3 = time.perf_counter()
        if i >= args.warmup:
            for k, v in zip(stages, (t1 - t0, t2 - t1, t3 - t2)):
                stages[k].append(v * 1000)
    total = np.sum([stages[k] for k in stages], axis=0)
    print(f"{args.onnx.name} on {platform.processor() or platform.machine()}, providers "
          f"{det.session.get_providers()}, {args.width}x{args.height} -> 640x640, batch 1, n={args.n}")
    for k, v in stages.items():
        print(f"  {k:<12} mean {np.mean(v):7.2f} ms   p95 {np.percentile(v, 95):7.2f} ms")
    print(f"  {'total':<12} mean {np.mean(total):7.2f} ms   p95 {np.percentile(total, 95):7.2f} ms   "
          f"-> {1000 / np.mean(total):.1f} fps")


if __name__ == "__main__":
    main()
