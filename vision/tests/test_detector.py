import numpy as np
import pytest
from conftest import ROOT, tiny_output

from forklift_vision import CLASSES
from forklift_vision.detector import (PAD_VALUE, LetterboxInfo, YoloOnnxDetector, cxcywh_to_xyxy, iou_matrix,
                                      letterbox, nms, postprocess, preprocess)


# ------------------------------------------------------------------------------------------- letterbox
@pytest.mark.parametrize("h,w", [(480, 640), (640, 480), (640, 640), (300, 1000), (720, 1280), (17, 33)])
def test_letterbox_shape_and_padding(h, w):
    img = np.full((h, w, 3), 200, np.uint8)
    out, info = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    r = min(640 / h, 640 / w)
    assert info.scale == pytest.approx(r)
    nw, nh = round(w * r), round(h * r)
    # content block is where the original went, grey padding elsewhere
    assert (out[info.pad_y:info.pad_y + nh, info.pad_x:info.pad_x + nw] == 200).all()
    pad_total = out.size // 3 - nw * nh
    assert ((out == PAD_VALUE).all(axis=2)).sum() == pad_total


def test_letterbox_640x480_is_centered():
    _, info = letterbox(np.zeros((480, 640, 3), np.uint8), 640)
    assert (info.scale, info.pad_x, info.pad_y) == (1.0, 0, 80)


def test_preprocess_layout():
    img = np.zeros((480, 640, 3), np.uint8)
    img[..., 0] = 255  # pure blue in BGR
    x, infos = preprocess([img, img], 640)
    assert x.shape == (2, 3, 640, 640) and x.dtype == np.float32
    assert x[0, 2, 320, 320] == pytest.approx(1.0)  # blue lands in channel 2 (RGB order)
    assert x[0, 0, 320, 320] == 0.0
    assert x[0, 0, 10, 10] == pytest.approx(PAD_VALUE / 255)  # padding row
    assert len(infos) == 2


def test_preprocess_accepts_grayscale():
    x, _ = preprocess([np.zeros((480, 640), np.uint8)])
    assert x.shape == (1, 3, 640, 640)


# ------------------------------------------------------------------------------------------- NMS / IoU
def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 50 / 150, 0.0], rtol=1e-6)


def test_nms_keeps_best_and_disjoint():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 9]], np.float32)
    scores = np.array([0.5, 0.9, 0.7, 0.3], np.float32)
    keep = nms(boxes, scores, iou_thres=0.5)
    assert keep.tolist() == [1, 2]
    assert nms(boxes, scores, iou_thres=0.99).tolist() == [1, 2, 0, 3]
    assert nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5).tolist() == []


def test_cxcywh_to_xyxy():
    np.testing.assert_allclose(cxcywh_to_xyxy(np.array([[10, 20, 4, 6]], np.float32)), [[8, 17, 12, 23]])


# ------------------------------------------------------------------------------------------- postprocess
def test_postprocess_known_output():
    info = LetterboxInfo(scale=1.0, pad_x=0, pad_y=80, orig_shape=(480, 640))
    dets = postprocess(tiny_output(), [info], conf_thres=0.25, iou_thres=0.45)[0]
    assert [(d.class_name, round(d.score, 2)) for d in dets] == [("pallet", 0.9), ("person", 0.7),
                                                                 ("forklift", 0.6)]
    pallet = dets[0]
    assert pallet.xyxy == pytest.approx((270, 215, 370, 265))  # y shifted back by the 80 px padding
    assert pallet.center == pytest.approx((320, 240))


def test_postprocess_agnostic_merges_classes():
    info = LetterboxInfo(1.0, 0, 0, (640, 640))
    dets = postprocess(tiny_output(), [info], iou_thres=0.45, agnostic=True)[0]
    assert sorted(d.class_name for d in dets) == ["pallet", "person"]


def test_postprocess_scales_and_clips():
    out = np.zeros((1, 8, 3), np.float32)
    out[0, :4, 0] = (630, 320, 40, 40)  # runs past the right edge
    out[0, 4, 0] = 0.9
    info = LetterboxInfo(scale=0.5, pad_x=0, pad_y=140, orig_shape=(720, 1280))
    d = postprocess(out, [info])[0][0]
    assert d.x2 == 1280 and d.x1 == pytest.approx(1220)
    assert d.y1 == pytest.approx((300 - 140) / 0.5) and d.y2 == pytest.approx((340 - 140) / 0.5)


def test_postprocess_empty_and_bad_shape():
    assert postprocess(np.zeros((2, 8, 10), np.float32), [LetterboxInfo(1, 0, 0, (640, 640))] * 2) == [[], []]
    with pytest.raises(ValueError):
        postprocess(np.zeros((8, 10), np.float32), [])


def test_postprocess_max_det():
    out = np.zeros((1, 8, 50), np.float32)
    for k in range(50):
        out[0, :4, k] = (10 + k * 12, 300, 10, 10)
        out[0, 5, k] = 0.5 + k / 200
    dets = postprocess(out, [LetterboxInfo(1, 0, 0, (640, 640))], max_det=7)[0]
    assert len(dets) == 7 and dets[0].score == pytest.approx(0.5 + 49 / 200)


# ------------------------------------------------------------------------------------------- ONNX wrapper
def test_wrapper_with_tiny_model(tiny_onnx):
    det = YoloOnnxDetector(tiny_onnx, iou_thres=0.45)
    assert det.input_size == 640 and det.dynamic_batch
    assert det.class_names == CLASSES
    img = np.zeros((480, 640, 3), np.uint8)
    raw = det.raw(preprocess([img, img, img])[0])
    assert raw.shape == (3, 4 + len(CLASSES), 8400)
    dets = det(img)
    assert [d.class_name for d in dets] == ["pallet", "person", "forklift"]
    assert dets[0].xyxy == pytest.approx((270, 215, 370, 265))
    batch = det.detect_batch([img, np.zeros((640, 640, 3), np.uint8)])
    assert len(batch) == 2 and batch[1][0].xyxy == pytest.approx((270, 295, 370, 345))
    assert det.detect_batch([]) == []


REAL_MODEL = ROOT / "models" / "starter_v1" / "starter_v1.onnx"


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="starter model not exported")
def test_wrapper_with_exported_model():
    det = YoloOnnxDetector(REAL_MODEL)
    assert det.class_names == CLASSES
    img = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    raw = det.raw(preprocess([img, img])[0])
    assert raw.shape == (2, 8, 8400)
    for d in det(img):
        assert 0 <= d.x1 <= d.x2 <= 640 and 0 <= d.y1 <= d.y2 <= 480 and d.class_name in CLASSES
