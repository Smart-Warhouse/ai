import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forklift_vision import CLASSES  # noqa: E402

N_ANCHORS = 8400
# Fixed predictions baked into the tiny model: (cx, cy, w, h, class, score) in 640×640 input pixels
TINY_PREDICTIONS = [
    (320, 320, 100, 50, 1, 0.90),   # pallet — kept
    (325, 322, 100, 50, 1, 0.80),   # overlapping pallet — suppressed by NMS
    (320, 320, 100, 50, 2, 0.60),   # same box, other class — kept (class-aware NMS)
    (100, 200, 40, 120, 0, 0.70),   # person — kept
    (500, 500, 30, 30, 3, 0.10),    # obstacle below conf_thres — dropped
]


def tiny_output() -> np.ndarray:
    out = np.zeros((1, 4 + len(CLASSES), N_ANCHORS), np.float32)
    for k, (cx, cy, w, h, c, s) in enumerate(TINY_PREDICTIONS):
        out[0, :4, k] = (cx, cy, w, h)
        out[0, 4 + c, k] = s
    return out


@pytest.fixture(scope="session")
def tiny_onnx(tmp_path_factory) -> Path:
    """A model with the exported YOLO I/O signature (dynamic batch, 640×640, (B, 8, 8400)) whose output is
    TINY_PREDICTIONS for any input. Built with onnx.helper — no torch, no download."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    const = numpy_helper.from_array(tiny_output(), "preds")
    shape = numpy_helper.from_array(np.array([-1, 1, 1], np.int64), "shape")
    zero = numpy_helper.from_array(np.array(0.0, np.float32), "zero")
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Reshape", ["m", "shape"], ["m3"]),
        helper.make_node("Mul", ["m3", "zero"], ["z"]),
        helper.make_node("Add", ["z", "preds"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 640, 640])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", 4 + len(CLASSES), N_ANCHORS])],
        initializer=[const, shape, zero])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    names = {i: c for i, c in enumerate(CLASSES)}
    model.metadata_props.add(key="names", value=str(names))
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("model") / "tiny.onnx"
    onnx.save(model, str(path))
    return path
