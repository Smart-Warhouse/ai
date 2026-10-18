import numpy as np
import pytest

from forklift_vision.detector import Detection
from forklift_vision.metrics import match_image, summarize, yolo_to_xyxy


def det(c, s, x1, y1, x2, y2):
    return Detection(c, str(c), s, x1, y1, x2, y2)


def test_yolo_to_xyxy():
    (c, b), = yolo_to_xyxy([(2, 0.5, 0.5, 0.25, 0.5)], 640, 480)
    assert c == 2 and b.tolist() == pytest.approx([240, 120, 400, 360])


def test_match_image_counts():
    gts = [(0, np.array([0, 0, 10, 10], np.float32)), (0, np.array([50, 50, 60, 60], np.float32)),
           (1, np.array([0, 0, 10, 10], np.float32))]
    dets = [det(0, 0.9, 0, 0, 10, 10),        # tp
            det(0, 0.8, 1, 1, 10, 10),        # duplicate of the same gt -> fp
            det(1, 0.7, 30, 30, 40, 40),      # wrong place -> fp, gt of class 1 -> fn
            det(3, 0.6, 0, 0, 5, 5)]          # class without gt -> fp
    counts = match_image(dets, gts, nc=4)
    assert counts.tolist() == [[1, 1, 1], [0, 1, 1], [0, 0, 0], [0, 1, 0]]
    s = summarize(counts, ["person", "pallet", "forklift", "obstacle"])
    assert s["person"]["recall"] == 0.5 and s["person"]["precision"] == 0.5
    assert s["forklift"]["recall"] is None and s["obstacle"]["precision"] == 0.0
