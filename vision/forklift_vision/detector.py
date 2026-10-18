"""onnxruntime wrapper for the exported YOLO model, with the same pre/post-processing as Ultralytics.

No rclpy, no torch, no Ultralytics: numpy + opencv + onnxruntime only (works with the Jetson's OpenCV 4.5 and
numpy < 2). A ROS 2 node does:

    from forklift_vision.detector import YoloOnnxDetector
    det = YoloOnnxDetector("yolo_starter.onnx", conf_thres=0.25)
    for d in det(bgr_image):              # list[Detection], pixel coords of the ORIGINAL image
        d.class_name, d.score, d.xyxy

`letterbox`, `preprocess` and `postprocess` are public so a pure TensorRT backend on the Jetson can reuse them
around its own engine call.

Model output (Ultralytics detect head, ONNX export): float32 `(B, 4 + nc, N)` — rows 0..3 are cx, cy, w, h in
letterboxed input pixels, rows 4.. are per-class scores already passed through a sigmoid. N = 8400 at 640×640.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from . import CLASSES

PAD_VALUE = 114  # Ultralytics LetterBox grey


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class LetterboxInfo:
    """How an original image was mapped into the network input: input = original * scale + (pad_x, pad_y)."""
    scale: float
    pad_x: float
    pad_y: float
    orig_shape: tuple[int, int]  # (h, w)


def letterbox(image: np.ndarray, new_shape: int | tuple[int, int] = 640,
              color: int = PAD_VALUE) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping the aspect ratio, pad to `new_shape` (h, w). Same rounding as Ultralytics LetterBox."""
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h, w = image.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))  # (w, h)
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return image, LetterboxInfo(scale=r, pad_x=left, pad_y=top, orig_shape=(h, w))


def preprocess(images: Sequence[np.ndarray], size: int = 640) -> tuple[np.ndarray, list[LetterboxInfo]]:
    """BGR uint8 HxWx3 images -> float32 NCHW RGB in [0, 1], plus the letterbox info of each image."""
    batch, infos = [], []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lb, info = letterbox(img, size)
        batch.append(lb[:, :, ::-1].transpose(2, 0, 1))
        infos.append(info)
    x = np.ascontiguousarray(np.stack(batch), dtype=np.float32) / 255.0
    return x, infos


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, a (N, 4) × b (M, 4) -> (N, M)."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS on xyxy boxes; returns kept indices, highest score first."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def postprocess(output: np.ndarray, infos: Sequence[LetterboxInfo], conf_thres: float = 0.25,
                iou_thres: float = 0.7, max_det: int = 100, class_names: Sequence[str] = CLASSES,
                agnostic: bool = False) -> list[list[Detection]]:
    """Raw `(B, 4 + nc, N)` output -> per-image detections in original-image pixels (class-aware NMS)."""
    if output.ndim != 3:
        raise ValueError(f"expected (B, 4+nc, N), got {output.shape}")
    results = []
    max_wh = 7680.0  # class offset for class-aware NMS in one pass (as Ultralytics)
    for pred, info in zip(output, infos):
        pred = pred.T  # (N, 4 + nc)
        scores_all = pred[:, 4:]
        cls = scores_all.argmax(axis=1)
        score = scores_all[np.arange(len(cls)), cls]
        m = score > conf_thres
        if not m.any():
            results.append([])
            continue
        boxes, score, cls = cxcywh_to_xyxy(pred[m, :4]), score[m], cls[m]
        offset = 0.0 if agnostic else cls[:, None].astype(np.float32) * max_wh
        keep = nms(boxes + offset, score, iou_thres)[:max_det]
        boxes, score, cls = boxes[keep], score[keep], cls[keep]
        # undo letterbox, clip to the original image
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_x) / info.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_y) / info.scale
        h, w = info.orig_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        dets = []
        for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), score.tolist(), cls.tolist()):
            name = class_names[c] if c < len(class_names) else str(c)
            dets.append(Detection(int(c), name, float(s), x1, y1, x2, y2))
        results.append(dets)
    return results


class YoloOnnxDetector:
    """Callable detector: `det(image)` -> list[Detection]; `det.detect_batch(images)` -> list per image.

    providers: onnxruntime execution providers, default CPU. On the Jetson with onnxruntime-gpu:
    `["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]`.
    """

    def __init__(self, model_path: str | Path, conf_thres: float = 0.25, iou_thres: float = 0.7,
                 max_det: int = 100, providers: Sequence | None = None, class_names: Sequence[str] | None = None,
                 intra_op_threads: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if intra_op_threads:
            so.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(str(model_path), sess_options=so,
                                            providers=list(providers or ["CPUExecutionProvider"]))
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape  # [batch, 3, H, W]; batch may be symbolic
        self.input_size = int(shape[2]) if isinstance(shape[2], int) else 640
        self.dynamic_batch = not isinstance(shape[0], int)
        self.class_names = tuple(class_names or self._names_from_metadata() or CLASSES)
        self.conf_thres, self.iou_thres, self.max_det = conf_thres, iou_thres, max_det

    def _names_from_metadata(self) -> tuple[str, ...] | None:
        meta = self.session.get_modelmeta().custom_metadata_map
        if "names" not in meta:
            return None
        names = ast.literal_eval(meta["names"])  # Ultralytics stores "{0: 'person', ...}"
        return tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else tuple(names)

    def raw(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not images:
            return []
        if self.dynamic_batch:
            x, infos = preprocess(images, self.input_size)
            out = self.raw(x)
        else:  # fixed-batch model: run one by one
            outs, infos = [], []
            for img in images:
                x, info = preprocess([img], self.input_size)
                outs.append(self.raw(x))
                infos += info
            out = np.concatenate(outs)
        return postprocess(out, infos, self.conf_thres, self.iou_thres, self.max_det, self.class_names)

    def __call__(self, image: np.ndarray) -> list[Detection]:
        return self.detect_batch([image])[0]
