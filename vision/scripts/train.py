"""Fine-tune YOLO (v11n by default, COCO-pretrained) on a built dataset; best weights → models/<name>/.

    python scripts/train.py --data datasets/starter_v1/data.yaml --name starter_v1 --epochs 40
    python scripts/train.py --data datasets/forklift_v1/data.yaml --name forklift_v1 --epochs 100   # real data

Needs Ultralytics + PyTorch (requirements-train.txt). Ultralytics is AGPL-3.0 — see README.
"""

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt or yolov8n.pt (PC-06)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0 = first CUDA GPU, cpu = CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, help="wall-clock cap; overrides --epochs (Ultralytics `time`)")
    args = ap.parse_args()

    import _ultralytics_local

    _ultralytics_local.apply()
    import torch
    import ultralytics
    from ultralytics import YOLO

    out = ROOT / "models" / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = YOLO(str(ROOT / "weights" / args.model) if (ROOT / "weights" / args.model).exists() else args.model)
    model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, patience=args.patience, seed=args.seed,
                project=str(ROOT / "runs"), name=args.name, exist_ok=True, plots=True,
                # camera is fixed upright on the carriage: no vertical flips; mild HSV for the 3 light levels
                fliplr=0.5, flipud=0.0, hsv_v=0.5, mosaic=1.0, close_mosaic=5,
                **({"time": args.hours} if args.hours else {}))
    minutes = (time.time() - t0) / 60
    run_dir = ROOT / "runs" / args.name
    shutil.copy2(run_dir / "weights" / "best.pt", out / f"{args.name}.pt")
    for f in ("results.csv", "results.png", "confusion_matrix.png", "BoxPR_curve.png", "PR_curve.png"):
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, out / f)
    meta = {"name": args.name, "base_model": args.model, "data": str(args.data), "epochs": args.epochs,
            "imgsz": args.imgsz, "batch": args.batch, "train_minutes": round(minutes, 1),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device != "cpu" else "cpu",
            "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                         "python": platform.python_version()},
            "finished": time.strftime("%Y-%m-%d %H:%M")}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"trained in {minutes:.1f} min -> {out}")


if __name__ == "__main__":
    sys.exit(main())
