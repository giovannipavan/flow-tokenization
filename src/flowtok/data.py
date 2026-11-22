"""Loading of the encoded sets (data/encoded/) for the models that use them (B2, B4)."""

import json

import numpy as np
import pandas as pd

from flowtok import paths


def load_meta():
    """meta.json written by prepare: features, vocabulary sizes, row counts."""
    with open(paths.ENC_DIR / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def load_vocabularies():
    """vocabularies.json written by prepare: value -> code for each vocabulary."""
    with open(paths.ENC_DIR / "vocabularies.json", encoding="utf-8") as f:
        return json.load(f)


def load_encoded(name, meta=None):
    """Load one encoded set as a DataFrame with only the columns the models need.

    name: "train_enc", "tune_enc", "val_enc" or "test_enc".
    Columns: row_id, target, start_rtt and the integer codes id_* (in meta order).
    """
    meta = meta or load_meta()
    f = paths.ENC_DIR / f"{name}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found: run `python -m flowtok.prepare` first.")
    columns = ["row_id", meta["target"], *meta["num_features"], *meta["id_columns"]]
    return pd.read_parquet(f, columns=columns)


def to_arrays(df, meta):
    """Split a loaded set into numpy arrays: X (codes, int32), rtt (float32), y (float32)."""
    X   = df[meta["id_columns"]].to_numpy(np.int32)
    rtt = df["start_rtt"].to_numpy(np.float32)
    y   = df[meta["target"]].to_numpy(np.float32)
    return X, rtt, y


def n_codes_per_column(meta):
    """Size of the code table of each id_* column (4 reserved codes + frequent values)."""
    vocab = load_vocabularies()
    column_vocab = {c: name for name, cols in vocab["columns"].items() for c in cols}
    return [meta["n_codes"][column_vocab[c.removeprefix("id_")]] for c in meta["id_columns"]]
