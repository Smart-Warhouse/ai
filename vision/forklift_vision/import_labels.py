"""Import a CVAT or Roboflow YOLO export into a labelled pool `datasets/raw/<pool>/{images,labels}`.

Class ids are remapped BY NAME to the PC-06 order (person, pallet, forklift, obstacle), so the label order the
annotator happened to create in CVAT/Roboflow does not matter. Unknown names stop the import unless
`--drop-unknown` or `--map oldname=newname` is given.

Supported exports (auto-detected):
  * CVAT "YOLO 1.1"                     obj.names + obj_train_data/*.txt (+ images if "Save images" was ticked)
  * CVAT "Ultralytics YOLO Detection"   data.yaml + labels/<subset>/*.txt (+ images/<subset>/)
  * Roboflow "YOLOv8" / "YOLOv11"       data.yaml + {train,valid,test}/{images,labels}/

If the export has no images, pass `--images <capture session>/images` and images are matched by file stem.

    python -m forklift_vision.import_labels --src exports/cvat_fl1_lux300 --images datasets/capture/<session>/images \
        --dst datasets/raw/fl1_lux300
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from . import CLASSES
from .dataset import IMAGE_EXTS


def read_names(src: Path) -> list[str]:
    obj = next(src.rglob("obj.names"), None)
    if obj is not None:
        return [n.strip() for n in obj.read_text(encoding="utf-8").splitlines() if n.strip()]
    for y in sorted(src.rglob("*.yaml")):
        data = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=int)]
        if isinstance(names, list):
            return [str(n) for n in names]
    raise SystemExit(f"no obj.names or data.yaml with 'names' under {src}")


def build_mapping(names: list[str], aliases: dict[str, str], drop_unknown: bool,
                  classes: tuple[str, ...] = CLASSES) -> dict[int, int | None]:
    canon = {c.lower(): i for i, c in enumerate(classes)}
    mapping: dict[int, int | None] = {}
    unknown = []
    for i, n in enumerate(names):
        key = aliases.get(n.lower(), n.lower())
        if key in canon:
            mapping[i] = canon[key]
        elif drop_unknown or key == "":
            mapping[i] = None
        else:
            unknown.append(n)
    if unknown:
        raise SystemExit(f"unknown class names {unknown}; use --map name=<one of {list(classes)}> or "
                         f"--drop-unknown")
    return mapping


def remap_label_text(text: str, mapping: dict[int, int | None]) -> tuple[str, int]:
    """Rewrite class ids; returns (new text, dropped box count)."""
    out, dropped = [], 0
    for line in text.splitlines():
        f = line.split()
        if not f:
            continue
        new = mapping.get(int(f[0]))
        if new is None:
            dropped += 1
            continue
        out.append(" ".join([str(new)] + f[1:5]))
    return ("\n".join(out) + "\n") if out else "", dropped


def find_image(label: Path, src: Path, extra: Path | None, index: dict[str, Path]) -> Path | None:
    for ext in IMAGE_EXTS:  # next to the label (CVAT YOLO 1.1)
        p = label.with_suffix(ext)
        if p.exists():
            return p
    parts = list(label.parts)
    if "labels" in parts:  # images/ mirror (Ultralytics / Roboflow)
        i = len(parts) - 1 - parts[::-1].index("labels")
        parts[i] = "images"
        for ext in IMAGE_EXTS:
            p = Path(*parts).with_suffix(ext)
            if p.exists():
                return p
    return index.get(label.stem)


def import_export(src: Path, dst: Path, images: Path | None = None, aliases: dict[str, str] | None = None,
                  drop_unknown: bool = False, classes: tuple[str, ...] = CLASSES) -> dict:
    src, dst = Path(src), Path(dst)
    names = read_names(src)
    mapping = build_mapping(names, {k.lower(): v.lower() for k, v in (aliases or {}).items()}, drop_unknown,
                            classes)
    index = {}
    if images:
        index = {p.stem: p for p in Path(images).rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    stats = {"names": names, "mapping": mapping, "imported": 0, "no_image": [], "dropped_boxes": 0}
    for lp in sorted(src.rglob("*.txt")):
        if lp.name in ("train.txt", "valid.txt", "val.txt", "test.txt") or lp.parent == src:
            continue  # CVAT image lists / README
        img = find_image(lp, src, images, index)
        if img is None:
            stats["no_image"].append(str(lp.relative_to(src)))
            continue
        text, dropped = remap_label_text(lp.read_text(encoding="utf-8"), mapping)
        stats["dropped_boxes"] += dropped
        shutil.copy2(img, dst / "images" / img.name)
        (dst / "labels" / f"{img.stem}.txt").write_text(text)
        stats["imported"] += 1
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="unzipped CVAT / Roboflow export")
    ap.add_argument("--dst", required=True, type=Path, help="pool folder, e.g. datasets/raw/fl1_lux300")
    ap.add_argument("--images", type=Path, help="original images when the export has none")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW", help="rename a class, e.g. human=person")
    ap.add_argument("--drop-unknown", action="store_true", help="drop boxes of classes not in PC-06")
    args = ap.parse_args(argv)
    aliases = dict(m.split("=", 1) for m in args.map)
    s = import_export(args.src, args.dst, args.images, aliases, args.drop_unknown)
    print(f"classes in export: {s['names']} -> mapping {s['mapping']}")
    print(f"imported {s['imported']} labelled images into {args.dst}, dropped {s['dropped_boxes']} boxes")
    if s["no_image"]:
        print(f"WARNING: {len(s['no_image'])} label files without an image (pass --images), e.g. {s['no_image'][:3]}")


if __name__ == "__main__":
    main()
