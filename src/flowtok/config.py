"""Reading and checking an experiment config (a JSON file in configs/)."""

import json
import sys
from pathlib import Path

from flowtok import models

SPLITS = ("val", "test")


def load(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("name", path.stem)
    cfg.setdefault("split", "val")
    cfg.setdefault("models", [])
    if cfg["split"] not in SPLITS:
        raise SystemExit(f"split must be one of {SPLITS}, not '{cfg['split']}'")
    for m in cfg["models"]:
        if m not in models.MODULES:
            raise SystemExit(f"Unknown model '{m}'. Available: {list(models.MODULES)}")
    return cfg


def config_path_from_argv(usage):
    """The only command-line argument: the path of the config file."""
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        raise SystemExit(usage)
    return sys.argv[1]
