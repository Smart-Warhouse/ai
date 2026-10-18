"""COCO 2017 `person` subset for the starter model — only images whose licence allows derivatives.

Keeps images with licence 4 (CC BY 2.0), 5 (CC BY-SA 2.0), 7 (no known copyright restrictions) or 8 (U.S.
Government work); NC and ND licences are excluded. Images with a crowd region (`iscrowd`, unlabelled people)
are skipped. Two disjoint image sets:
  * direct:  whole val2017 images, every person boxed          -> datasets/raw/coco_person/{images,labels}
  * cutout:  train2017 persons cut out with their mask  -> datasets/raw/coco_person_cutouts/*.png
             (pasted into the MuJoCo renders by make_synthetic.py)
Every used image is listed with its licence and Flickr URL in data_manifests/coco_person_manifest.csv
(attribution for CC BY / BY-SA).

    python scripts/fetch_coco_persons.py --annotations datasets/raw/coco/annotations_trainval2017.zip \
        --direct 700 --cutout-images 400
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OK_LICENSES = {4, 5, 7, 8}
PERSON = 1  # COCO category id


def person_index(zip_path: Path, split: str) -> tuple[dict, dict, dict, list]:
    """images, person annotations per image, licences, and the usable image ids of one COCO split."""
    with zipfile.ZipFile(zip_path) as z:
        coco = json.loads(z.read(f"annotations/instances_{split}.json"))
    licenses = {lic["id"]: lic for lic in coco["licenses"]}
    images = {im["id"]: dict(im, split=split) for im in coco["images"]}
    anns = defaultdict(list)
    crowd = set()
    for a in coco["annotations"]:
        if a["category_id"] != PERSON:
            continue
        if a["iscrowd"]:
            crowd.add(a["image_id"])
        anns[a["image_id"]].append(a)
    del coco
    candidates = sorted(i for i in anns if i not in crowd and images[i]["license"] in OK_LICENSES
                        and len(anns[i]) <= 8)
    print(f"{split}: {len(anns)} images with persons, {len(candidates)} with an allowed licence and no crowd")
    images = {i: images[i] for i in anns}
    return images, anns, licenses, candidates


def download(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    try:
        with urlopen(url, timeout=60) as r:
            data = r.read()
        if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
            return False
        dst.write_bytes(data)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=ROOT / "datasets/raw/coco/annotations_trainval2017.zip")
    ap.add_argument("--direct", type=int, default=700)
    ap.add_argument("--cutout-images", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=ROOT / "datasets/raw/coco")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    images, anns, licenses, cand_val = person_index(args.annotations, "val2017")
    rng.shuffle(cand_val)
    direct = cand_val[:args.direct]
    cutout = []
    if args.cutout_images:
        im_t, an_t, _, cand_train = person_index(args.annotations, "train2017")
        rng.shuffle(cand_train)
        cutout = cand_train[:args.cutout_images]
        images.update({i: im_t[i] for i in cutout})
        anns.update({i: an_t[i] for i in cutout})
        del im_t, an_t

    def fetch(i):
        d = args.cache / images[i]["split"]
        d.mkdir(parents=True, exist_ok=True)
        return download(images[i]["coco_url"], d / images[i]["file_name"])

    with ThreadPoolExecutor(8) as pool:
        ok = dict(zip(direct + cutout, pool.map(fetch, direct + cutout)))
    print(f"downloaded {sum(ok.values())}/{len(ok)}")

    out = ROOT / "datasets/raw/coco_person"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    cut_dir = ROOT / "datasets/raw/coco_person_cutouts"
    cut_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_boxes = n_cut = 0
    for use, ids in (("direct", direct), ("cutout", cutout)):
        for i in ids:
            if not ok[i]:
                continue
            im = images[i]
            W, H = im["width"], im["height"]
            src = args.cache / im["split"] / im["file_name"]
            if use == "direct":
                lines = []
                for a in anns[i]:
                    x, y, w, h = a["bbox"]
                    if w < 2 or h < 2:
                        continue
                    lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
                (out / "images" / f"coco_{i:012d}.jpg").write_bytes(src.read_bytes())
                (out / "labels" / f"coco_{i:012d}.txt").write_text("".join(lines))
                n_boxes += len(lines)
            else:
                img = cv2.imread(str(src))
                boxes = [a["bbox"] for a in anns[i]]
                made = 0
                for a in anns[i]:
                    x, y, w, h = a["bbox"]
                    if h < 120 or h < 1.3 * w or not isinstance(a["segmentation"], list):
                        continue  # small, or not upright (a hand, a lying person, a close-up torso)
                    if x < 2 or y < 2 or x + w > W - 2 or y + h > H - 2:
                        continue  # truncated by the image border
                    if any(b is not a["bbox"] and not (b[0] > x + w or b[0] + b[2] < x or b[1] > y + h
                                                       or b[1] + b[3] < y) for b in boxes):
                        continue  # overlaps another person -> cut-out would contain part of them
                    mask = np.zeros((H, W), np.uint8)
                    for poly in a["segmentation"]:
                        cv2.fillPoly(mask, [np.asarray(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 255)
                    x0, y0, x1, y1 = int(x), int(y), int(np.ceil(x + w)), int(np.ceil(y + h))
                    if mask[y0:y1, x0:x1].mean() < 255 * 0.25:
                        continue
                    rgba = np.dstack([img[y0:y1, x0:x1], mask[y0:y1, x0:x1]])
                    cv2.imwrite(str(cut_dir / f"coco_{i:012d}_{a['id']}.png"), rgba)
                    made += 1
                if not made:
                    continue
                n_cut += made
            lic = licenses[im["license"]]
            rows.append({"use": use, "coco_split": im["split"], "coco_image_id": i, "file_name": im["file_name"], "license_id": lic["id"],
                         "license": lic["name"], "license_url": lic["url"], "flickr_url": im.get("flickr_url", ""),
                         "coco_url": im["coco_url"]})
    man = ROOT / "data_manifests/coco_person_manifest.csv"
    man.parent.mkdir(exist_ok=True)
    with open(man, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    by_lic = defaultdict(int)
    for r in rows:
        by_lic[r["license"]] += 1
    print(f"direct: {sum(r['use'] == 'direct' for r in rows)} images, {n_boxes} person boxes -> {out}")
    print(f"cutout: {n_cut} person cut-outs from {sum(r['use'] == 'cutout' for r in rows)} images -> {cut_dir}")
    print(f"licences: {dict(by_lic)}; manifest -> {man}")


if __name__ == "__main__":
    main()
