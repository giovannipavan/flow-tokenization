# Network flow tokenization

Project for the EPFL course **CS-433 Machine Learning** (AI4Science initiative, Fall 2026),
hosted by CERN IT. We study how to turn network flow telemetry from the Worldwide LHC
Computing Grid (WLCG) into discrete tokens, and whether those tokens keep the information
needed to predict how the round-trip time (RTT) of a flow changes over its lifetime.

> Status: work in progress. The project runs until 17 December 2026.

## Idea

Each flow is described by metadata known when it starts (endpoints, ports, sites,
application, ...) and by RTT observations at its start and at its end. The target is
`delta_rtt = end_rtt - start_rtt`, predicted from start-event information only, with a
chronological train / validation / test split. The work is organised in stages, so there
is always a complete result:

1. **Baselines.** Supervised models on non-tokenized inputs, from a constant prediction
   to gradient boosting. They set the floor and a practical ceiling for the comparison.
2. **Tokenization.** Represent each flow as a short sequence of discrete tokens
   (feature-level, flow-level, optionally learned with vector quantization).
3. **Comparison.** Same task, same split, same metrics as the baselines.

## Data

The data (a Firefly telemetry extract of about 67 million reconstructed flows over seven
days in October 2026) is **not stored in this repository**. Place it locally as:

```
data/
├── student_dataset/             # as shared by the supervisor
│   ├── flows.parquet
│   ├── manifest.json
│   ├── splits.json              # chronological split: 70 / 15 / 15 % by flow_start
│   └── flows_prepared.parquet   # cleaned and prepared flows, one row per flow
└── encoded/                     # written by `python -m flowtok.prepare`
```

See `AGENTS.md` for the fields, the leakage rules and the splits, and
`docs/project_summary.md` for the data analysis and the preparation decisions.

## Setup

```bash
git clone https://github.com/giovannipavan/flow-tokenization.git
cd flow-tokenization
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # makes the package `flowtok` (in src/) importable
```

## Running the pipeline

```bash
python -m flowtok.prepare                              # flows_prepared -> data/encoded/
python -m flowtok.run_baselines configs/default.json   # train, evaluate and save the baselines
```

Everything that defines an experiment is in a JSON file in `configs/`; the command line
only says which file to use. To run several experiments, run them one after the other:

```bash
for c in configs/default.json configs/quick.json; do python -m flowtok.run_baselines "$c"; done
```

Running experiments in parallel is not recommended on a laptop: each run uses up to 8 GB
of memory for DuckDB and loads the validation split (10 M flows) in memory.

### 1. Choosing the input parameters

There are two levels of parameters.

**Encoding (rarely changed)**: constants at the top of `src/flowtok/prepare.py`.

| Constant        | Default   | Meaning |
|-----------------|-----------|---------|
| `N_MIN`         | 1000      | minimum number of training flows for a value to get its own code (otherwise `RARE`) |
| `N_TRAIN`       | 3,000,000 | size of `train_enc`, sampled from the first 90 % of the training split |
| `N_TUNE`        | 1,000,000 | size of `tune_enc`, sampled from the last 10 % of the training split |
| `TUNE_FRACTION` | 0.9       | where the tune period starts inside the training split (in time order) |
| `VOCAB_COLUMNS` | 13 fields | categorical inputs and the vocabulary each one uses |

Changing them requires running `prepare` again, which overwrites `data/encoded/`: rename
the old folder first if you want to keep it. The values used are recorded in
`data/encoded/meta.json`.

**Experiment (changed often)**: a JSON file in `configs/`, for example `configs/default.json`:

```json
{
  "name": "default",
  "split": "val",
  "models": ["b0", "b1", "b2", "b4"],
  "b1": {"identifier": "L2", "orientation": "server_client", "min_flows": 1000},
  "b2": {"lambda": 1e-5, "epochs": 5, "lr": 0.01, "gamma": 0.5, "batch": 4096, "seed": 0},
  "b4": {"n_trees": 411, "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 50,
         "min_data_per_group": 50, "cat_smooth": 10, "max_cat_threshold": 32, "seed": 0}
}
```

- `name`: label of the run; results are saved under this name.
- `split`: where the models are evaluated: `"val"` (default) or `"test"`. Models are always
  trained on the training data only; `split` only changes where they are evaluated.
  The test split is meant to be used **once, at the end of the project**, with frozen
  configs and all models in the same run: choices made after looking at it would bias the
  final result. The program prints a warning when `split` is `"test"`.
- `models`: which baselines to run.
- one block per model with its hyperparameters (missing values take the defaults written
  at the top of each model file).
- `save_predictions` (optional, default `true`): save the predictions of each model.

To try a variant, copy `default.json`, change `name` and the values, and run the new file.
`configs/quick.json` runs only B0 and B1 (about a minute), useful to check the setup.

**Hyperparameter search.** The values in `default.json` were chosen on `tune_enc`, never on
validation. To repeat the search with the grids in the `tuning` block of the config:

```bash
python -m flowtok.tune configs/default.json
```

It writes one table per model to `results/tuning/<name>/`; the chosen values are then copied
by hand into the config (for B2 `best_epoch` becomes `epochs`, for B4 `best_trees` becomes
`n_trees`).

### 2. Choosing the model

All models predict `delta_rtt` from start-event information, are trained with an L1 loss
(consistent with the primary metric, the MAE) and are evaluated with the same metrics on
the same flows.

| Model | What it is | Data | Use it to |
|-------|------------|------|-----------|
| `b0` | global median of the training split | `flows_prepared` | get the floor: error with no information |
| `b1` | median per endpoint pair, with fallback pair -> ASN pair -> global median | `flows_prepared` | measure the value of the endpoint identity alone |
| `b2` | linear model: intercept + `log(start_rtt)` + one coefficient per category value | `data/encoded` | measure additive effects, without interactions |
| `b3` | MLP on the same inputs *(not implemented yet)* | `data/encoded` | reference for the token-based models |
| `b4` | LightGBM with categorical inputs | `data/encoded` | strong tabular reference; feature importance |

`b1` has six variants: `identifier` is `site` or `L2` (L2 = IPv4 /24 or IPv6 /48 prefix),
`orientation` is `server_client`, `src_dst` or `unordered`. The default is the best one on
validation.

Each model is a file in `src/flowtok/models/` with the same function
`run(con, cfg, split, model_dir)`, which returns the predictions as a table
`(flow_id, t_start, pred)`. Adding a model means adding a file with this function and one
line in `src/flowtok/models/__init__.py`.

### 3. Where results are saved

```
data/encoded/                          # written by prepare (not committed)
    train_enc.parquet, tune_enc.parquet, val_enc.parquet, test_enc.parquet
    value_map.parquet, vocabularies.json, meta.json, quality.csv

results/
    metrics/<name>/<split>/            # committed
        config.json                    # exact copy of the config used
        <model>.json                   # overall metrics + settings + training time
        <model>_subgroups.csv          # metrics per subgroup
        b1_levels.csv                  # share of flows predicted at each fallback level
        b4_feature_importance.csv
    summary.csv                        # one row per (run, split, model), committed
    tuning/<name>/<model>.csv          # hyperparameter search tables, committed
    models/<name>/                     # trained models (not committed)
    predictions/<name>/<split>/        # flow_id, t_start, pred per model (not committed)
```

Metrics are small and shared through git; models and predictions are large and can be
regenerated, so they stay local. Each run writes in its own `<name>/<split>/` folder, so a
run never overwrites another one; running the same config again replaces its own files and
its rows in `summary.csv`.

**Metrics** (in ms): MAE (primary), RMSE, robust RMSE (only flows whose `delta_rtt` lies
between the 0.05 % and 99.95 % training quantiles, [-184.23, 58.82] ms, the same flows for
every model: the full RMSE is dominated by about 0.1 % of extreme flows), and the 90th and
99th percentiles of the absolute error. They are reported overall and for the subgroups
FNAL, missing site, ambiguous orientation and flows longer than one minute.

## Results so far (validation split, 10.07 M flows)

| Model | MAE | RMSE | Robust RMSE | P90 \|err\| | P99 \|err\| |
|---|---|---|---|---|---|
| B0 global median | 3.298 | 47.29 | 5.812 | 4.510 | 23.55 |
| B1 median per (server, client) L2 prefix pair | 2.209 | 47.23 | 5.377 | 4.861 | 23.07 |
| B2 linear model | 1.966 | 47.22 | 5.264 | 4.599 | 23.03 |
| B4 LightGBM | **1.833** | 47.08 | **4.945** | **4.563** | **22.64** |

The up-to-date numbers are in `results/summary.csv`.

## Repository layout

```
.
├── README.md
├── AGENTS.md            # context for coding agents and collaborators
├── LICENSE
├── pyproject.toml       # makes src/flowtok installable (pip install -e .)
├── requirements.txt
├── configs/             # experiment configs and choices fitted on the training split
├── docs/                # project summary, notes, diagrams
├── src/flowtok/         # data preparation, models, evaluation
├── results/             # official outputs (see above)
├── scratch/             # working notebooks, not maintained, not part of the deliverable
└── data/                # not committed
```

## Team

Giovanni, Mario and Matteo (EPFL Master's students), with Carmen Misa (CERN IT) as host.

## License

See `LICENSE`.
