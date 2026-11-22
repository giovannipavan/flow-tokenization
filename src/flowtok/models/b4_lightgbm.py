"""B4 - Gradient boosting with regression trees (LightGBM), L1 objective.

Inputs: the 13 integer codes id_* (declared as categorical, so LightGBM can group several
categories in one split) and start_rtt. Unlike B2, trees capture interactions between fields.

The final model is trained on train_enc + tune_enc with a fixed number of trees
(`n_trees`). The settings were chosen on tune_enc with `tune` (early stopping on tune_enc).
"""

import time
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd

from flowtok import data, evaluation

DEFAULTS = {
    "n_trees": 411,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "min_data_per_group": 50,
    "cat_smooth": 10,
    "max_cat_threshold": 32,
    "seed": 0,
}
TUNING_DEFAULTS = {
    "grid": {"num_leaves": [31, 63, 127], "min_data_in_leaf": [50, 100, 1000]},
    "max_trees": 5000,
    "early_stopping": 100,
}


def lgb_params(cfg):
    """LightGBM parameters from a config block (everything except n_trees)."""
    params = {k: v for k, v in cfg.items() if k != "n_trees"}
    return {"objective": "l1", "metric": "l1", "verbose": -1, **params}


def features(meta):
    """Input columns, in the same order for every run: codes (meta order), then start_rtt."""
    cat = list(meta["id_columns"])
    return cat, cat + list(meta["num_features"])


def dataset(df, meta, reference=None):
    cat, feat = features(meta)
    return lgb.Dataset(df[feat], label=df[meta["target"]].to_numpy(), categorical_feature=cat,
                       reference=reference, params={"feature_pre_filter": False},
                       free_raw_data=False)


# =============================================================================
# Final model (used by run_baselines)
# =============================================================================

def run(con, cfg, split, model_dir):
    cfg = {**DEFAULTS, **cfg}
    meta = data.load_meta()
    _, feat = features(meta)

    train = pd.concat([data.load_encoded("train_enc", meta), data.load_encoded("tune_enc", meta)],
                      ignore_index=True)
    t0 = time.time()
    model = lgb.train(lgb_params(cfg), dataset(train, meta), num_boost_round=int(cfg["n_trees"]))
    train_time = time.time() - t0
    n_train = len(train)
    del train

    ev = data.load_encoded(f"{split}_enc", meta)
    pred = model.predict(ev[feat])
    table = evaluation.register_numpy_predictions(con, split, ev["row_id"].to_numpy(), pred, "pred_b4")

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "b4.txt"))

    importance = pd.DataFrame({
        "feature": model.feature_name(),
        "gain":    model.feature_importance(importance_type="gain"),
        "split":   model.feature_importance(importance_type="split"),
    })
    importance["gain_pct"]  = (100 * importance["gain"]  / importance["gain"].sum()).round(1)
    importance["split_pct"] = (100 * importance["split"] / importance["split"].sum()).round(1)
    importance = importance.sort_values("gain", ascending=False).reset_index(drop=True)

    n_leaves = sum(t["num_leaves"] for t in model.dump_model()["tree_info"])
    info = {
        "model": "B4 LightGBM (L1)",
        "n_params": int(n_leaves),
        "n_params_note": "total number of leaves; split structure not counted",
        "train_time_s": round(train_time, 1),
        "n_train_flows": int(n_train),
        **cfg,
    }
    return table, info, {"feature_importance": importance}


# =============================================================================
# Hyperparameter search (used by tune)
# =============================================================================

def tune(cfg, tuning):
    """Grid search: every combination of the values in tuning["grid"] (the other parameters
    come from the config) is trained on train_enc with early stopping on tune_enc.

    Returns one row per combination with the best number of trees and the tune MAE.
    Copy the chosen values, with best_trees as `n_trees`, into the config.
    """
    cfg = {**DEFAULTS, **cfg}
    tuning = {**TUNING_DEFAULTS, **tuning}
    meta = data.load_meta()
    train, tune_set = data.load_encoded("train_enc", meta), data.load_encoded("tune_enc", meta)
    d_train = dataset(train, meta)
    d_tune = dataset(tune_set, meta, reference=d_train)

    y_tr, y_tu = train[meta["target"]].to_numpy(), tune_set[meta["target"]].to_numpy()
    print(f"  reference B0 on tune: MAE = {np.abs(np.median(y_tr) - y_tu).mean():.4f}")

    names = list(tuning["grid"])
    rows = []
    for values in product(*tuning["grid"].values()):
        setting = dict(zip(names, values))
        t0 = time.time()
        model = lgb.train(lgb_params({**cfg, **setting}), d_train,
                          num_boost_round=int(tuning["max_trees"]),
                          valid_sets=[d_tune], valid_names=["tune"],
                          callbacks=[lgb.early_stopping(int(tuning["early_stopping"]), verbose=False)])
        row = {**setting, "best_trees": model.best_iteration,
               "tune_mae": round(model.best_score["tune"]["l1"], 4),
               "seconds": round(time.time() - t0)}
        print(f"    {row}")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("tune_mae").reset_index(drop=True)
