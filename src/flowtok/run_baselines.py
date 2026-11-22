"""Train, evaluate and save the models listed in a config.

Usage:  python -m flowtok.run_baselines configs/default.json

For each model in cfg["models"]:
    1. train the final model (run) and predict the split cfg["split"];
    2. evaluate the predictions with the shared metrics (overall and per subgroup);
    3. save metrics, model and predictions;
then update results/summary.csv with one row per (run, split, model).
"""

import json
import time
from datetime import datetime

import pandas as pd

from flowtok import config, data, db, evaluation, models, paths

USAGE = "Usage: python -m flowtok.run_baselines <config.json>   (e.g. configs/default.json)"


def update_summary(rows):
    """Add the new rows to results/summary.csv, replacing older rows of the same run/split/model."""
    new = pd.DataFrame(rows)
    if paths.SUMMARY_FILE.exists():
        old = pd.read_csv(paths.SUMMARY_FILE)
        keep = ~old.set_index(["run", "split", "model"]).index.isin(
            new.set_index(["run", "split", "model"]).index)
        new = pd.concat([old[keep], new], ignore_index=True)
    paths.SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(paths.SUMMARY_FILE, index=False)
    return new


def main():
    cfg = config.load(config.config_path_from_argv(USAGE))
    name, split = cfg["name"], cfg["split"]
    if not cfg["models"]:
        raise SystemExit('No models to run: add a "models" list to the config.')
    print(f"Run '{name}' | split: {split} | models: {cfg['models']}")
    if split == "test":
        print("\n" + "!" * 78)
        print("  WARNING: evaluating on the TEST split. Use it once, at the end of the project,")
        print("  with frozen configs: choices made after looking at it bias the final result.")
        print("!" * 78 + "\n")

    con = db.connect()
    db.register_prepared(con)
    found = db.register_encoded(con)
    if f"{split}_enc" not in found:
        raise SystemExit(f"data/encoded/{split}_enc.parquet not found: run `python -m flowtok.prepare`.")
    thresholds = evaluation.robust_thresholds(con, data.load_meta())

    metrics_dir     = paths.METRICS_DIR / name / split
    model_dir       = paths.MODELS_DIR / name
    predictions_dir = paths.PREDICTIONS_DIR / name / split
    for d in (metrics_dir, predictions_dir):
        d.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    summary_rows = []
    for m in cfg["models"]:
        print(f"\n=== {m} ===")
        t0 = time.time()
        pred_table, info, extras = models.get(m).run(con, cfg.get(m, {}), split, model_dir)
        result = evaluation.evaluate(con, pred_table, split, thresholds)
        overall = result.iloc[0]

        # metrics
        result.to_csv(metrics_dir / f"{m}_subgroups.csv", index=False)
        for extra_name, table in extras.items():
            table.to_csv(metrics_dir / f"{m}_{extra_name}.csv", index=False)
        summary = {
            **{k: float(overall[k]) for k in evaluation.METRICS},
            "n_flows": int(overall["n"]),
            "robust_rmse_thresholds": thresholds,
            **info,
        }
        with open(metrics_dir / f"{m}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # predictions (not committed; "save_predictions": false in the config to skip)
        if cfg.get("save_predictions", True):
            out = (predictions_dir / f"{m}.parquet").as_posix()
            con.execute(f"COPY (SELECT flow_id, t_start, pred FROM {pred_table}) TO '{out}' (FORMAT parquet)")

        print(result.round(3).to_string(index=False))
        print(f"({time.time() - t0:.0f} s)")
        summary_rows.append({
            "run": name, "split": split, "model": m, "description": info.get("model", m),
            **{k: round(float(overall[k]), 4) for k in evaluation.METRICS},
            "n_params": info.get("n_params"), "train_time_s": info.get("train_time_s"),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

    table = update_summary(summary_rows)
    print(f"\nMetrics in {metrics_dir}")
    print(f"Summary ({paths.SUMMARY_FILE}):")
    print(table[table["run"] == name].to_string(index=False))


if __name__ == "__main__":
    main()
