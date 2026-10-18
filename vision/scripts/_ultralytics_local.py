"""Import before Ultralytics: keeps its settings, downloads and runs inside code/vision.

Without this Ultralytics writes a global settings file whose weights/runs folders default to the git root
(it downloaded yolo26n.pt for its AMP check into the repo root) and enables anonymous usage sync.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def apply() -> None:
    import ultralytics.utils
    from ultralytics import settings

    settings.update({"weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs"),
                     "datasets_dir": str(ROOT / "datasets"), "sync": False})
    # computed at import from the (possibly just created, git-root based) settings; the AMP check reads it
    ultralytics.utils.WEIGHTS_DIR = ROOT / "weights"
