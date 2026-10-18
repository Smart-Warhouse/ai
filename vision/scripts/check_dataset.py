"""Report images / boxes per class against the ≥ 400-images-per-class target and list broken labels.

    python scripts/check_dataset.py datasets/forklift_v1            # built dataset (train/val/test)
    python scripts/check_dataset.py datasets/raw/fl1_lux300         # one labelled pool
    python scripts/check_dataset.py datasets/forklift_v1 --json report.json --strict

--strict exits 1 when a class is below target or a label is missing/invalid/orphaned (use it as a CI/merge gate
for the real dataset).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forklift_vision import TARGET_IMAGES_PER_CLASS  # noqa: E402
from forklift_vision.dataset import check_dataset, format_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--target", type=int, default=TARGET_IMAGES_PER_CLASS)
    ap.add_argument("--json", type=Path, help="also write the full report as JSON")
    ap.add_argument("--max-listed", type=int, default=10)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    report = check_dataset(args.root, target=args.target)
    print(format_report(report, args.max_listed))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    if args.strict and (report["has_errors"] or not report["meets_target"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
