"""Hyperparameter search on the tune set (tune_enc), never on validation.

Usage:  python -m flowtok.tune configs/default.json

Runs the search of every model listed in cfg["tuning"] (B2: lambda grid; B4: grid over
LightGBM parameters) and writes one table per model in results/tuning/<name>/.
The chosen values are then copied by hand into the config.
"""

from flowtok import config, models, paths

USAGE = "Usage: python -m flowtok.tune <config.json>   (e.g. configs/default.json)"


def main():
    cfg = config.load(config.config_path_from_argv(USAGE))
    tuning = cfg.get("tuning", {})
    if not tuning:
        raise SystemExit('Nothing to tune: add a "tuning" block to the config (see configs/default.json).')

    out_dir = paths.TUNING_DIR / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for m, settings in tuning.items():
        module = models.get(m)
        if not hasattr(module, "tune"):
            raise SystemExit(f"Model '{m}' has no hyperparameter search.")
        print(f"\n=== tuning {m} ===")
        table = module.tune(cfg.get(m, {}), settings)
        table.to_csv(out_dir / f"{m}.csv", index=False)
        print(table.to_string(index=False))
    print(f"\nTables in {out_dir}")


if __name__ == "__main__":
    main()
