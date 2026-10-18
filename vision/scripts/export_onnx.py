"""Export trained weights to ONNX (dynamic batch, fixed 640×640) and check the wrapper against Ultralytics.

    python scripts/export_onnx.py models/starter_v1/starter_v1.pt [--parity-images datasets/starter_v1/images/test]

Output: `<same folder>/<name>.onnx`, opset 17 (TensorRT 8.6 / 10.x on JetPack 6 both parse it). The input is
`images` float32 (B, 3, 640, 640) RGB 0..1, the output `output0` (B, 4 + nc, 8400) — see forklift_vision.detector.
The parity check runs Ultralytics `predict` and `YoloOnnxDetector` on the same images and reports the box IoU
of matched detections, proving the pure-Python pre/post-processing matches.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import _ultralytics_local  # noqa: E402  (before any Ultralytics import)


def fix_input_hw(path: Path, size: int) -> None:
    """Ultralytics `dynamic=True` makes batch, height and width dynamic; pin H×W to `size` (only batch varies)."""
    import onnx

    m = onnx.load(str(path))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    dims[0].dim_param = "batch"
    for d in (dims[2], dims[3]):
        d.Clear()
        d.dim_value = size
    out = m.graph.output[0].type.tensor_type.shape.dim
    out[0].Clear()
    out[0].dim_param = "batch"
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def parity(pt: Path, onnx_path: Path, images: list[Path]) -> dict:
    import cv2
    from ultralytics import YOLO

    from forklift_vision.detector import YoloOnnxDetector, iou_matrix

    yolo = YOLO(str(pt))
    det = YoloOnnxDetector(onnx_path, conf_thres=0.25, iou_thres=0.7)
    ious, n_ul, n_ox = [], 0, 0
    for p in images:
        img = cv2.imread(str(p))
        # rect=False: square 640×640 letterbox like the exported model (default pt predict pads minimally)
        r = yolo.predict(img, imgsz=640, conf=0.25, iou=0.7, rect=False, verbose=False, device="cpu")[0]
        ub, uc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        od = det(img)
        n_ul, n_ox = n_ul + len(ub), n_ox + len(od)
        if len(ub) and len(od):
            ob = np.array([d.xyxy for d in od], np.float32)
            oc = np.array([d.class_id for d in od])
            m = iou_matrix(ub.astype(np.float32), ob) * (uc[:, None] == oc[None, :])
            ious += m.max(axis=1).tolist()
    return {"images": len(images), "ultralytics_boxes": n_ul, "onnx_wrapper_boxes": n_ox,
            "matched_iou_ge_0.9": int(np.sum(np.array(ious) >= 0.9)), "ultralytics_boxes_unmatched":
                int(np.sum(np.array(ious) < 0.5)) + max(0, n_ul - len(ious)),
            "mean_matched_iou": float(np.mean(ious)) if ious else None,
            "min_matched_iou": float(np.min(ious)) if ious else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--parity-images", type=Path, help="folder of images for the Ultralytics vs wrapper check")
    ap.add_argument("--parity-n", type=int, default=40)
    args = ap.parse_args()
    _ultralytics_local.apply()

    from ultralytics import YOLO

    exported = Path(YOLO(str(args.weights)).export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True,
                                                   opset=args.opset, half=False, device="cpu"))
    dst = args.weights.with_suffix(".onnx")
    if exported.resolve() != dst.resolve():
        shutil.move(str(exported), dst)
    fix_input_hw(dst, args.imgsz)
    print(f"ONNX -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")

    from forklift_vision.detector import YoloOnnxDetector

    det = YoloOnnxDetector(dst)
    print(f"onnxruntime OK: input {det.session.get_inputs()[0].shape}, output "
          f"{det.session.get_outputs()[0].shape}, classes {det.class_names}")
    if args.parity_images:
        imgs = sorted(args.parity_images.glob("*.jpg"))[:args.parity_n]
        print("parity Ultralytics vs YoloOnnxDetector:", parity(args.weights, dst, imgs))


if __name__ == "__main__":
    main()
