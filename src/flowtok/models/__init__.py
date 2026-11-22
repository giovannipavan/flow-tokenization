"""Baseline models. Each module exposes the same function:

    run(con, cfg, split, model_dir) -> (pred_table, info, extras)

    con        DuckDB connection with the views `fp` and the encoded sets
    cfg        the block of the config for this model (a dict of hyperparameters)
    split      "val" or "test": where the predictions are made
    model_dir  folder where the trained model is saved

    pred_table name of a DuckDB table with one row per flow of the split: (flow_id, t_start, pred)
    info       dict saved with the metrics (parameters, training time, settings)
    extras     dict name -> DataFrame of additional tables to save (e.g. feature importance)

The final models are always trained on the training data only (train, or train + tune);
`split` only decides where they are evaluated.
"""

import importlib

MODULES = {
    "b0": "flowtok.models.b0_median",
    "b1": "flowtok.models.b1_pair_median",
    "b2": "flowtok.models.b2_linear",
    "b4": "flowtok.models.b4_lightgbm",
}


def get(name):
    """Import a model module only when it is used (torch and lightgbm are not needed for B0/B1)."""
    if name not in MODULES:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODULES)}")
    return importlib.import_module(MODULES[name])
