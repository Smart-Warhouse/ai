"""import_labels (CVAT / Roboflow) and the capture helper's frame filter."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from forklift_vision.capture import FrameSaver, change, ros_image_to_bgr, sharpness
from forklift_vision.dataset import check_dataset
from forklift_vision.import_labels import import_export, remap_label_text


def _img(path, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.random.default_rng(seed).random((16, 16, 3)) * 255).astype(np.uint8))


def test_cvat_yolo11_export_is_remapped_by_name(tmp_path):
    src = tmp_path / "cvat"
    # annotator created the labels in another order
    (src / "obj_train_data").mkdir(parents=True)
    (src / "obj.names").write_text("forklift\nPerson\nobstacle\npallet\n")
    (src / "train.txt").write_text("data/obj_train_data/a.jpg\n")
    (src / "obj_train_data" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n3 0.7 0.7 0.1 0.1\n")
    (src / "obj_train_data" / "b.txt").write_text("")
    _img(tmp_path / "capture" / "images" / "a.jpg")
    _img(tmp_path / "capture" / "images" / "b.jpg")
    dst = tmp_path / "pool"
    s = import_export(src, dst, images=tmp_path / "capture" / "images")
    assert s["imported"] == 2 and s["mapping"] == {0: 2, 1: 0, 2: 3, 3: 1}
    assert (dst / "labels" / "a.txt").read_text().split("\n")[:3] == ["2 0.5 0.5 0.1 0.1", "0 0.2 0.2 0.1 0.1",
                                                                    "1 0.7 0.7 0.1 0.1"]
    assert (dst / "labels" / "b.txt").read_text() == ""
    r = check_dataset(dst, target=1)
    assert r["per_class"]["forklift"]["images"] == 1 and r["splits"]["all"]["empty"] == 1


def test_roboflow_export_with_alias_and_drop(tmp_path):
    src = tmp_path / "rf"
    (src).mkdir()
    (src / "data.yaml").write_text("names: ['human', 'pallet', 'shelf']\nnc: 3\n")
    for sub in ("train", "valid", "test"):
        _img(src / sub / "images" / f"{sub}_x.rf.abc.jpg")
        (src / sub / "labels").mkdir(parents=True)
        (src / sub / "labels" / f"{sub}_x.rf.abc.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.2 0.2\n")
    with pytest.raises(SystemExit):
        import_export(src, tmp_path / "p0")
    s = import_export(src, tmp_path / "p1", aliases={"human": "person"}, drop_unknown=True)
    assert s["imported"] == 3 and s["dropped_boxes"] == 3
    assert (tmp_path / "p1" / "labels" / "train_x.rf.abc.txt").read_text() == "0 0.5 0.5 0.1 0.1\n"


def test_remap_label_text():
    text, dropped = remap_label_text("0 1 1 1 1\n1 2 2 2 2\n", {0: 3, 1: None})
    assert text == "3 1 1 1 1\n" and dropped == 1


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_frame_saver_interval_blur_duplicate(tmp_path):
    clock = FakeClock()
    saver = FrameSaver(tmp_path, "fl1 lux300!", interval=1.0, max_frames=3, min_sharpness=30, min_change=2,
                       meta={"lux": 300}, clock=clock)
    rng = np.random.default_rng(0)
    sharp = lambda: (rng.random((480, 640, 3)) * 255).astype(np.uint8)  # noqa: E731
    flat = np.full((480, 640, 3), 128, np.uint8)
    a = sharp()
    assert saver.offer(a)
    clock.t = 0.5
    assert not saver.offer(sharp())          # inside the interval
    clock.t = 1.5
    assert not saver.offer(flat)             # blurred / featureless
    assert not saver.offer(a.copy())         # identical to the last saved
    assert saver.offer(sharp())
    clock.t = 3.0
    assert saver.offer(sharp()) and saver.done
    clock.t = 9.0
    assert not saver.offer(sharp())
    saver.close()
    files = sorted(p.name for p in (saver.dir / "images").iterdir())
    assert files == ["fl1_lux300_000001.jpg", "fl1_lux300_000002.jpg", "fl1_lux300_000003.jpg"]
    assert saver.skipped == {"blur": 1, "duplicate": 1}
    assert '"lux": 300' in (saver.dir / "session.json").read_text()
    assert (saver.dir / "labels").is_dir()


def test_sharpness_and_change():
    flat = np.zeros((64, 64, 3), np.uint8)
    noisy = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    assert sharpness(flat) == 0 and sharpness(noisy) > 100
    assert change(flat, flat) == 0 and change(flat, noisy) > 50


def test_ros_image_conversion():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 255  # red in RGB
    msg = SimpleNamespace(encoding="rgb8", height=2, width=3, step=9, data=rgb.tobytes())
    assert ros_image_to_bgr(msg)[0, 0].tolist() == [0, 0, 255]
    mono = SimpleNamespace(encoding="mono8", height=2, width=3, step=4, data=bytes(range(8)))  # padded rows
    assert ros_image_to_bgr(mono).shape == (2, 3, 3)
    jpg = cv2.imencode(".jpg", np.full((4, 4, 3), 90, np.uint8))[1].tobytes()
    comp = SimpleNamespace(format="jpeg", data=jpg)
    assert ros_image_to_bgr(comp).shape == (4, 4, 3)
    with pytest.raises(ValueError):
        ros_image_to_bgr(SimpleNamespace(encoding="32FC1", height=1, width=1, step=4, data=b"\0" * 4))
