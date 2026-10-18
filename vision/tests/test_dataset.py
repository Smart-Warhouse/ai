from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from forklift_vision import CLASSES
from forklift_vision.dataset import (assign_split, check_dataset, format_report, group_key, label_path_for,
                                     parse_label_file, split_dataset)


def make_pool(root: Path, n: int, labels: dict[int, str] | None = None, prefix: str = "img") -> Path:
    """n tiny images; labels[i] = label text (default: one pallet box); value None = no label file."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(root / "images" / f"{prefix}_{i:04d}.jpg"), np.full((8, 8, 3), i % 255, np.uint8))
        text = (labels or {}).get(i, "1 0.5 0.5 0.2 0.2\n")
        if text is not None:
            (root / "labels" / f"{prefix}_{i:04d}.txt").write_text(text)
    return root


def test_label_path_for():
    assert label_path_for(Path("d/images/train/a.jpg")) == Path("d/labels/train/a.txt")
    assert label_path_for(Path("images/x/images/b.png")) == Path("images/x/labels/b.txt")
    with pytest.raises(ValueError):
        label_path_for(Path("d/pics/a.jpg"))


def test_parse_label_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.1 0.1\n\n3 0.1 0.2 0.3 0.4\n9 0.5 0.5 0.1 0.1\n1 0.5 0.5 1.5 0.1\n"
                 "1 0.5 0.5\n2 a b c d\n2 0.5 0.5 0 0.1\n")
    lf = parse_label_file(p)
    assert [b[0] for b in lf.boxes] == [0, 3]
    assert len(lf.errors) == 5
    assert any("class id 9" in e for e in lf.errors)


def test_checker_counts_and_flags(tmp_path):
    labels = {
        0: "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n1 0.6 0.6 0.1 0.1\n",  # person + 2 pallets
        1: "",                                                         # empty (background)
        2: None,                                                       # missing
        3: "7 0.5 0.5 0.2 0.2\n",                                      # invalid class id
        4: "3 0.5 0.5 0.2 0.2\n",                                      # obstacle
    }
    make_pool(tmp_path, 5, labels)
    (tmp_path / "labels" / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    r = check_dataset(tmp_path, target=2)
    s = r["splits"]["all"]
    assert (s["images"], s["labelled"], s["empty"], s["missing"], s["invalid"]) == (5, 2, 1, 1, 1)
    assert r["per_class"]["person"] == {"images": 1, "instances": 1, "target": 2, "missing_to_target": 1}
    assert r["per_class"]["pallet"]["instances"] == 2 and r["per_class"]["pallet"]["images"] == 1
    assert r["per_class"]["forklift"]["images"] == 0
    assert r["problems"]["missing_label"] == [str(Path("images/img_0002.jpg"))]
    assert r["problems"]["empty_label"] == [str(Path("images/img_0001.jpg"))]
    assert r["problems"]["orphan_label"] == [str(Path("labels/ghost.txt"))]
    assert not r["meets_target"] and r["has_errors"]
    text = format_report(r)
    assert "[LOW] person" in text and "need 2 more" in text


def test_checker_meets_target(tmp_path):
    lines = "".join(f"{c} 0.5 0.5 0.2 0.2\n" for c in range(len(CLASSES)))
    make_pool(tmp_path, 3, {i: lines for i in range(3)})
    r = check_dataset(tmp_path, target=3)
    assert r["meets_target"] and not r["has_errors"]


def test_assign_split_deterministic_and_ratios():
    keys = [f"k{i}" for i in range(5000)]
    a = [assign_split(k) for k in keys]
    assert a == [assign_split(k) for k in keys]
    c = Counter(a)
    assert abs(c["train"] / 5000 - 0.7) < 0.03 and abs(c["val"] / 5000 - 0.2) < 0.03
    assert abs(c["test"] / 5000 - 0.1) < 0.02
    assert a != [assign_split(k, seed=1) for k in keys]
    with pytest.raises(ValueError):
        assign_split("x", (0.5, 0.2, 0.2))


def test_group_key_chunks_neighbouring_frames():
    assert group_key("fl1_lux300_000123", 10) == group_key("fl1_lux300_000129", 10) == "fl1_lux300_#12"
    assert group_key("fl1_lux300_000130", 10) != group_key("fl1_lux300_000129", 10)
    assert group_key("abc", 10) == "abc" and group_key("x_5", 1) == "x_5"


def test_split_dataset_layout_and_stability(tmp_path):
    pool_a = make_pool(tmp_path / "raw" / "poolA", 60, {7: None})
    out = tmp_path / "built"
    res = split_dataset([pool_a], out, seed=3)
    assert sum(res["counts"].values()) == 59 and len(res["skipped"]) == 1
    for split in ("train", "val", "test"):
        imgs = sorted(p.stem for p in (out / "images" / split).iterdir())
        lbls = sorted(p.stem for p in (out / "labels" / split).iterdir())
        assert imgs == lbls
        assert all(s.startswith("poolA__") for s in imgs)
    first = {p.name: p.parent.name for p in (out / "images").rglob("*.jpg")}
    data_yaml = (out / "data.yaml").read_text()
    assert "0: person" in data_yaml and "3: obstacle" in data_yaml and "test: images/test" in data_yaml
    r = check_dataset(out, target=1)
    assert set(r["splits"]) == {"train", "val", "test"} and not r["has_errors"]

    # adding a second pool never moves an image of the first one to another split
    pool_b = make_pool(tmp_path / "raw" / "poolB", 40, prefix="b")
    split_dataset([pool_a, pool_b], out, seed=3)
    second = {p.name: p.parent.name for p in (out / "images").rglob("*.jpg")}
    assert all(second[k] == v for k, v in first.items())
    assert len(second) == 99


def test_split_dataset_chunk_keeps_sequences_together(tmp_path):
    pool = make_pool(tmp_path / "raw" / "sess", 100)
    out = tmp_path / "built"
    split_dataset([pool], out, chunk=10)
    where = {p.stem: p.parent.name for p in (out / "images").rglob("*.jpg")}
    for block in range(10):
        assert len({where[f"sess__img_{block * 10 + i:04d}"] for i in range(10)}) == 1
