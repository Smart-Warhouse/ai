"""Detection matching at a fixed confidence threshold (the deployed operating point, not the mAP sweep).

Used to report per-class precision/recall of the ONNX wrapper exactly as the robot will run it — PC-08 asks
for person recall ≥ 0.95, which only means something at the threshold the safety logic uses.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .detector import Detection, iou_matrix


def yolo_to_xyxy(labels: Iterable[Sequence[float]], w: int, h: int) -> list[tuple[int, np.ndarray]]:
    """YOLO label rows (cls, cx, cy, bw, bh normalised) → [(cls, xyxy pixels)]."""
    out = []
    for cls, cx, cy, bw, bh in labels:
        out.append((int(cls), np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                                        (cy + bh / 2) * h], np.float32)))
    return out


def match_image(dets: Sequence[Detection], gts: Sequence[tuple[int, np.ndarray]], nc: int,
                iou_thres: float = 0.5) -> np.ndarray:
    """Greedy per-class matching (highest score first). Returns counts (nc, 3) = [tp, fp, fn]."""
    counts = np.zeros((nc, 3), np.int64)
    for c in range(nc):
        d = sorted((x for x in dets if x.class_id == c), key=lambda x: -x.score)
        g = [b for k, b in gts if k == c]
        if not d:
            counts[c, 2] += len(g)
            continue
        if not g:
            counts[c, 1] += len(d)
            continue
        ious = iou_matrix(np.array([x.xyxy for x in d], np.float32), np.stack(g))
        used = np.zeros(len(g), bool)
        for i in range(len(d)):
            row = np.where(used, -1.0, ious[i])
            j = int(row.argmax())
            if row[j] >= iou_thres:
                used[j] = True
                counts[c, 0] += 1
            else:
                counts[c, 1] += 1
        counts[c, 2] += int((~used).sum())
    return counts


def summarize(counts: np.ndarray, class_names: Sequence[str]) -> dict:
    out = {}
    for c, name in enumerate(class_names):
        tp, fp, fn = (int(v) for v in counts[c])
        out[name] = {"tp": tp, "fp": fp, "fn": fn,
                     "precision": tp / (tp + fp) if tp + fp else None,
                     "recall": tp / (tp + fn) if tp + fn else None}
    return out
