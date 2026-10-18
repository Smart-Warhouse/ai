"""YOLO-format dataset tooling: label parsing, the checker and the deterministic 70/20/10 split.

Layout (Ultralytics convention — every image `.../images/.../x.jpg` has its label at `.../labels/.../x.txt`):

    datasets/raw/<pool>/images/*.jpg          labelled pools (one per capture session, CVAT/Roboflow import, ...)
    datasets/raw/<pool>/labels/*.txt
    datasets/<name>/images/{train,val,test}/  built by `split_dataset`
    datasets/<name>/labels/{train,val,test}/
    datasets/<name>/data.yaml

Label line: `<class_id> <cx> <cy> <w> <h>`, all normalised to [0, 1], class ids in `CLASSES` order.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import CLASSES, TARGET_IMAGES_PER_CLASS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.7, 0.2, 0.1)


def label_path_for(image: Path) -> Path:
    """`.../images/a/b.jpg` -> `.../labels/a/b.txt` (last `images` component is swapped)."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"{image} is not under an 'images' folder")


def list_images(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and "images" in p.parts)


@dataclass
class LabelFile:
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_label_file(path: Path, nc: int = len(CLASSES)) -> LabelFile:
    out = LabelFile()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            out.errors.append(f"line {n}: expected 5 fields, got {len(fields)}")
            continue
        try:
            cls = int(fields[0])
            cx, cy, w, h = (float(v) for v in fields[1:])
        except ValueError:
            out.errors.append(f"line {n}: not numeric")
            continue
        if not 0 <= cls < nc:
            out.errors.append(f"line {n}: class id {cls} outside 0..{nc - 1}")
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            out.errors.append(f"line {n}: coordinates not normalised to [0, 1]")
            continue
        if w <= 0 or h <= 0:
            out.errors.append(f"line {n}: zero-size box")
            continue
        out.boxes.append((cls, cx, cy, w, h))
    return out


# ---------------------------------------------------------------------------------------------- checker
def check_dataset(root: Path, classes: tuple[str, ...] = CLASSES, target: int = TARGET_IMAGES_PER_CLASS) -> dict:
    """Images / instances per class and per split, plus every image whose label is missing, empty or broken.

    Split = the folder right under `images/` when it is train/val/test, otherwise `all`.
    """
    root = Path(root)
    nc = len(classes)
    splits: dict[str, dict] = {}
    problems = {"missing_label": [], "empty_label": [], "invalid_label": [], "orphan_label": []}
    totals_img = [0] * nc
    totals_inst = [0] * nc

    images = list_images(root)
    image_labels = set()
    for img in images:
        rel = img.relative_to(root)
        idx = rel.parts.index("images")
        split = rel.parts[idx + 1] if len(rel.parts) > idx + 2 and rel.parts[idx + 1] in SPLITS else "all"
        s = splits.setdefault(split, {"images": 0, "labelled": 0, "empty": 0, "missing": 0, "invalid": 0,
                                      "images_per_class": [0] * nc, "instances_per_class": [0] * nc})
        s["images"] += 1
        lp = label_path_for(img)
        image_labels.add(lp.resolve())
        if not lp.exists():
            s["missing"] += 1
            problems["missing_label"].append(str(rel))
            continue
        lf = parse_label_file(lp, nc)
        if lf.errors:
            s["invalid"] += 1
            problems["invalid_label"].append({"image": str(rel), "errors": lf.errors})
        if not lf.boxes:
            if not lf.errors:
                s["empty"] += 1
                problems["empty_label"].append(str(rel))
            continue
        s["labelled"] += 1
        present = set()
        for cls, *_ in lf.boxes:
            s["instances_per_class"][cls] += 1
            totals_inst[cls] += 1
            present.add(cls)
        for cls in present:
            s["images_per_class"][cls] += 1
            totals_img[cls] += 1

    for lp in sorted(p for p in root.rglob("*.txt") if "labels" in p.parts):
        if lp.resolve() not in image_labels:
            problems["orphan_label"].append(str(lp.relative_to(root)))

    per_class = {name: {"images": totals_img[i], "instances": totals_inst[i], "target": target,
                        "missing_to_target": max(0, target - totals_img[i])} for i, name in enumerate(classes)}
    return {
        "root": str(root),
        "classes": list(classes),
        "images": len(images),
        "splits": splits,
        "per_class": per_class,
        "meets_target": all(v["missing_to_target"] == 0 for v in per_class.values()),
        "has_errors": bool(problems["missing_label"] or problems["invalid_label"] or problems["orphan_label"]),
        "problems": problems,
    }


def format_report(report: dict, max_listed: int = 10) -> str:
    classes = report["classes"]
    lines = [f"Dataset {report['root']}: {report['images']} images", ""]
    head = f"{'split':<6} {'images':>7} {'labelled':>8} {'empty':>6} {'missing':>7} {'invalid':>7}  " + "  ".join(
        f"{c:>15}" for c in classes)
    lines += [head, "-" * len(head)]
    for split in sorted(report["splits"], key=lambda s: (SPLITS + ("all",)).index(s)):
        s = report["splits"][split]
        cells = "  ".join(f"{s['images_per_class'][i]:>6} img/{s['instances_per_class'][i]:>5}"
                          for i in range(len(classes)))
        lines.append(f"{split:<6} {s['images']:>7} {s['labelled']:>8} {s['empty']:>6} {s['missing']:>7} "
                     f"{s['invalid']:>7}  {cells}")
    lines += ["", "Images per class against the target:"]
    for name, v in report["per_class"].items():
        mark = "OK " if v["missing_to_target"] == 0 else "LOW"
        lines.append(f"  [{mark}] {name:<9} {v['images']:>6} / {v['target']} images, {v['instances']} boxes"
                     + (f"  (need {v['missing_to_target']} more)" if v["missing_to_target"] else ""))
    for kind, items in report["problems"].items():
        if items:
            lines.append(f"\n{kind}: {len(items)}")
            for it in items[:max_listed]:
                lines.append(f"  {it if isinstance(it, str) else json.dumps(it)}")
            if len(items) > max_listed:
                lines.append(f"  ... {len(items) - max_listed} more")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ split
_TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$")


def group_key(stem: str, chunk: int) -> str:
    """Frames `fl1_lux300_000123` … in blocks of `chunk` share a key, so near-identical neighbours of a
    capture session land in the same split (no train/test leakage). chunk ≤ 1 -> one key per image."""
    if chunk <= 1:
        return stem
    m = _TRAILING_NUMBER.match(stem)
    if not m:
        return stem
    return f"{m.group(1)}#{int(m.group(2)) // chunk}"


def assign_split(key: str, ratios: tuple[float, float, float] = DEFAULT_RATIOS, seed: int = 0) -> str:
    """Hash-based and therefore stable: adding images later never moves an existing image to another split."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    digest = hashlib.sha1(f"{seed}:{key}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2 ** 64
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "val"
    return "test"


def split_dataset(sources: list[Path], out: Path, ratios: tuple[float, float, float] = DEFAULT_RATIOS,
                  seed: int = 0, chunk: int = 1, classes: tuple[str, ...] = CLASSES,
                  include_unlabelled: bool = False) -> dict:
    """Copy the labelled pools in `sources` into `out/{images,labels}/{train,val,test}` and write data.yaml.

    Output file names are `<pool>__<original stem>` so pools never collide. Images without a label file are
    skipped (an empty label file = deliberate background image and is kept). Returns the counts per split.
    """
    out = Path(out)
    for kind in ("images", "labels"):
        for split in SPLITS:
            d = out / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    counts = {s: 0 for s in SPLITS}
    skipped = []
    assignment: dict[str, str] = {}
    for src in sources:
        src = Path(src)
        pool = src.name
        for img in list_images(src):
            lp = label_path_for(img)
            if not lp.exists() and not include_unlabelled:
                skipped.append(str(img))
                continue
            name = f"{pool}__{img.stem}"
            split = assign_split(group_key(f"{pool}__{img.stem}", chunk), ratios, seed)
            shutil.copy2(img, out / "images" / split / f"{name}{img.suffix.lower()}")
            dst_label = out / "labels" / split / f"{name}.txt"
            if lp.exists():
                shutil.copy2(lp, dst_label)
            else:
                dst_label.write_text("")
            counts[split] += 1
            assignment[name] = split
    write_data_yaml(out, classes)
    (out / "split.json").write_text(json.dumps({"ratios": ratios, "seed": seed, "chunk": chunk,
                                                "sources": [str(s) for s in sources], "counts": counts,
                                                "skipped_unlabelled": len(skipped), "assignment": assignment},
                                               indent=1))
    return {"counts": counts, "skipped": skipped}


def write_data_yaml(root: Path, classes: tuple[str, ...] = CLASSES) -> Path:
    root = Path(root).resolve()
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    text = (f"# generated by forklift_vision.dataset — class order is PC-06, do not reorder\n"
            f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    path = root / "data.yaml"
    path.write_text(text)
    return path
