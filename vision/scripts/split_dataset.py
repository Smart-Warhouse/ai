"""Build a train/val/test (70/20/10) dataset from labelled pools, deterministically.

    python scripts/split_dataset.py --out datasets/forklift_v1 datasets/raw/fl1_lux300 datasets/raw/fl2_lux150 ...
    python scripts/split_dataset.py --out datasets/forklift_v1 --chunk 10 datasets/raw/*      # camera sessions

The split of an image depends only on (seed, pool name, file name): re-running after adding new pools keeps
every old image where it was, so the test set never leaks into training. --chunk N keeps blocks of N
consecutive capture frames (`<label>_000123.jpg`) in the same split.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forklift_vision.dataset import check_dataset, format_report, split_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+", type=Path, help="folders with images/ and labels/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.2, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1)
    args = ap.parse_args()
    res = split_dataset(args.pools, args.out, tuple(args.ratios), args.seed, args.chunk)
    total = sum(res["counts"].values())
    print("split: " + ", ".join(f"{k} {v} ({v / max(total, 1):.1%})" for k, v in res["counts"].items()))
    if res["skipped"]:
        print(f"skipped {len(res['skipped'])} images without a label file (label them or delete them)")
    print(f"data.yaml -> {args.out / 'data.yaml'}\n")
    print(format_report(check_dataset(args.out)))


if __name__ == "__main__":
    main()
