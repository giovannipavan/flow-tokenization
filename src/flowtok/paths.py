"""Project folders and file locations, found from the position of this file."""

from pathlib import Path


def find_root():
    """Walk up from this file until the folder containing pyproject.toml."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("pyproject.toml not found above src/flowtok: is the repository complete?")


PROJECT_ROOT = find_root()

# Data (not committed)
DATA_DIR = PROJECT_ROOT / "data" / "student_dataset"
PREPARED = DATA_DIR / "flows_prepared.parquet"
ENC_DIR  = PROJECT_ROOT / "data" / "encoded"
DUCKDB_TMP = PROJECT_ROOT / "duckdb_tmp"

# Configurations
CONFIG_DIR = PROJECT_ROOT / "configs"

# Outputs
RESULTS_DIR     = PROJECT_ROOT / "results"
METRICS_DIR     = RESULTS_DIR / "metrics"       # committed
TUNING_DIR      = RESULTS_DIR / "tuning"        # committed
MODELS_DIR      = RESULTS_DIR / "models"        # not committed
PREDICTIONS_DIR = RESULTS_DIR / "predictions"   # not committed
SUMMARY_FILE    = RESULTS_DIR / "summary.csv"   # committed
