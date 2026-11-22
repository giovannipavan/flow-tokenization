"""
prepare.py - Encode the flows for the models (B2, B3, B4).

Reads flows_prepared.parquet and writes, in data/encoded/:
    train_enc.parquet      ~3 M flows from the first 90 % of train  (training)
    tune_enc.parquet       ~1 M flows from the last 10 % of train   (early stopping, tuning)
    val_enc.parquet        the whole validation split, 10.07 M flows (results)
    value_map.parquet      table vocabulary / value / code
    vocabularies.json      value -> code vocabularies, reserved codes, N_MIN
    meta.json              parameters, tune boundary, row counts, vocabulary sizes
    quality.csv            MISSING / RARE / UNK shares per column and per split

Design choices:
    - check that the columns used contain no NULL (a NULL would end up as UNK instead of MISSING);
    - codes with equal frequency are assigned in alphabetical order, so vocabularies are deterministic;
    - `row_id` is numbered in (t_start, flow_id) order, so it is stable across runs;
    - flow_id and t_start are kept in the output files, as a key to join predictions back to flows.

The test split is never read.
Usage:  python src/prepare.py   (from any folder inside the project)
"""

import json
import time
from pathlib import Path

import duckdb
import pandas as pd


# =============================================================================
# Parameters
# =============================================================================

N_MIN   = 1000          # minimum number of training flows for a value to get its own code
N_TRAIN = 3_000_000     # target size of train_enc
N_TUNE  = 1_000_000     # target size of tune_enc
TUNE_FRACTION = 0.9     # the tune set starts after the first 90 % of training flows (in time order)

PAD, MISSING, RARE, UNK = 0, 1, 2, 3

# Reproducible sampling: a flow is kept if its hash falls below a threshold.
# The sample is identical for a given DuckDB version. Once the files are written,
# the whole group uses those files, so the version no longer matters.
HASH = "hash(f.flow_id || CAST(f.t_start AS VARCHAR)) % 100000"

# vocabulary -> columns that use it (sites and L2 prefixes: vocabulary shared by server and client)
VOCAB_COLUMNS = {
    "ip_version":            ["ip_version"],
    "experiment":            ["experiment"],
    "exp_act":               ["exp_act"],
    "application":           ["application"],
    "hostname":              ["hostname"],
    "site":                  ["server_site", "client_site"],
    "L2":                    ["server_L2", "client_L2"],
    "service_port":          ["service_port"],
    "client_port_band":      ["client_port_band"],
    "orientation":           ["orientation"],
    "orientation_ambiguous": ["orientation_ambiguous"],
}
CAT_FEATURES = [c for columns in VOCAB_COLUMNS.values() for c in columns]
NUM_FEATURES = ["start_rtt"]
TARGET       = "delta_rtt"

# raw columns kept ONLY for the subgroup evaluation: they are NOT model inputs
EVAL_COLUMNS = ["server_site", "client_site", "orientation_ambiguous", "an_duration_s"]

# flows_prepared.parquet was written with some Italian column names.
# They are renamed when the file is read; once the file is regenerated with
# English names, this mapping is simply skipped.
LEGACY_COLUMN_NAMES = {
    "porta_servizio": "service_port",
    "porta_client":   "client_port_band",
    "verso":          "orientation",
    "verso_incerto":  "orientation_ambiguous",
    "an_durata_s":    "an_duration_s",
}


# =============================================================================
# Paths and connection
# =============================================================================

def find_root():
    """Walk up from the script folder until data/student_dataset is found."""
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "data" / "student_dataset").is_dir():
            return p
    raise FileNotFoundError("Folder data/student_dataset not found above the script.")


PROJECT_ROOT = find_root()
DATA_DIR = PROJECT_ROOT / "data" / "student_dataset"
PREPARED = DATA_DIR / "flows_prepared.parquet"
ENC_DIR  = PROJECT_ROOT / "data" / "encoded"


def connect():
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit = '8GB'")
    con.execute(f"SET temp_directory = '{(PROJECT_ROOT / 'duckdb_tmp').as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET TimeZone = 'UTC'")

    # View `fp` on flows_prepared.parquet, with legacy column names translated.
    source = f"read_parquet('{PREPARED.as_posix()}')"
    columns = set(con.execute(f"SELECT * FROM {source} LIMIT 0").df().columns)
    legacy = {old: new for old, new in LEGACY_COLUMN_NAMES.items() if old in columns}
    if legacy:
        excluded = ", ".join(legacy)
        renamed = ", ".join(f"{old} AS {new}" for old, new in legacy.items())
        select = f"SELECT * EXCLUDE ({excluded}), {renamed} FROM {source}"
    else:
        select = f"SELECT * FROM {source}"
    con.execute(f"CREATE OR REPLACE VIEW fp AS {select}")
    return con


# =============================================================================
# Steps
# =============================================================================

def check_nulls(con):
    """No NULL in the columns used (a NULL would never find its row in the value map)."""
    columns = CAT_FEATURES + NUM_FEATURES + [TARGET, "flow_id", "t_start"]
    counts = ", ".join(f"COUNT(*) FILTER (WHERE {c} IS NULL) AS {c}" for c in columns)
    nulls = con.execute(f"SELECT {counts} FROM fp WHERE split IN ('train', 'val')").df().iloc[0]
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        raise ValueError(f"NULL values found (replace them with 'MISSING' upstream):\n{nulls}")
    print("NULL check: no NULL in the columns used.")


def build_vocabularies(con):
    """For each vocabulary: values with >= N_MIN training flows -> codes from 4, by frequency."""
    vocab, seen = {}, {}
    for name, columns in VOCAB_COLUMNS.items():
        parts = [f"SELECT CAST({c} AS VARCHAR) AS value FROM fp WHERE split = 'train'"
                 for c in columns]
        counts = con.execute(f"""
            SELECT value, COUNT(*) AS n
            FROM ({" UNION ALL ".join(parts)})
            GROUP BY value
            ORDER BY n DESC, value          -- ties broken alphabetically: deterministic
        """).df()

        frequent = counts.loc[(counts["n"] >= N_MIN) & (counts["value"] != "MISSING"), "value"]
        vocab[name] = {v: i + 4 for i, v in enumerate(frequent)}
        seen[name] = set(counts["value"])
        print(f"  {name:22s} values in train: {len(counts):6d} | with >= {N_MIN} flows: {len(vocab[name]):5d}")
    return vocab, seen


def build_value_map(con, vocab, seen):
    """DuckDB table (vocab, value, code) for every value seen in train."""
    rows = []
    for name in VOCAB_COLUMNS:
        for value in sorted(seen[name]):
            if value == "MISSING":
                code = MISSING
            elif value in vocab[name]:
                code = vocab[name][value]
            else:
                code = RARE
            rows.append((name, value, code))
    map_df = pd.DataFrame(rows, columns=["vocab", "value", "code"])
    con.register("map_df", map_df)
    con.execute("CREATE OR REPLACE TABLE value_map AS SELECT * FROM map_df")
    print(f"  rows in the value map: {len(map_df):,}")
    return map_df


def encoded_query(condition):
    """Query returning the flows that satisfy `condition`, with one integer code per categorical."""
    id_columns, joins = [], []
    pairs = [(n, c) for n, cols in VOCAB_COLUMNS.items() for c in cols]
    for k, (name, c) in enumerate(pairs):
        id_columns.append(f"CAST(COALESCE(m{k}.code, {UNK}) AS INTEGER) AS id_{c}")
        joins.append(f"LEFT JOIN value_map AS m{k} "
                     f"ON m{k}.vocab = '{name}' AND m{k}.value = CAST(f.{c} AS VARCHAR)")
    raw = ", ".join(f"f.{c}" for c in EVAL_COLUMNS)
    return f"""
        SELECT ROW_NUMBER() OVER (ORDER BY f.t_start, f.flow_id) AS row_id,
               f.flow_id, f.t_start,
               f.{TARGET}, f.start_rtt,
               {", ".join(id_columns)},
               {raw}
        FROM fp AS f
        {" ".join(joins)}
        WHERE {condition}
    """


def split_conditions(con):
    """Tune boundary and SQL conditions of the three sets."""
    tune_start = con.execute(f"""
        SELECT quantile_disc(epoch(t_start), {TUNE_FRACTION}) FROM fp WHERE split = 'train'
    """).fetchone()[0]

    n_first, n_last = con.execute(f"""
        SELECT COUNT(*) FILTER (WHERE epoch(t_start) <  {tune_start}),
               COUNT(*) FILTER (WHERE epoch(t_start) >= {tune_start})
        FROM fp WHERE split = 'train'
    """).fetchone()

    threshold_train = round(N_TRAIN / n_first * 100_000)
    threshold_tune  = round(N_TUNE  / n_last  * 100_000)

    conditions = {
        "train_enc": f"f.split = 'train' AND epoch(f.t_start) <  {tune_start} AND {HASH} < {threshold_train}",
        "tune_enc":  f"f.split = 'train' AND epoch(f.t_start) >= {tune_start} AND {HASH} < {threshold_tune}",
        "val_enc":   "f.split = 'val'",
    }
    info = {
        "tune_start_epoch":     float(tune_start),
        "tune_start_utc":       str(pd.to_datetime(tune_start, unit="s", utc=True)),
        "n_train_first_90":     int(n_first),
        "n_train_last_10":      int(n_last),
        "hash_threshold_train": threshold_train,
        "hash_threshold_tune":  threshold_tune,
    }
    print(f"  tune start: {info['tune_start_utc']} | hash thresholds: "
          f"{threshold_train} / {threshold_tune} out of 100,000")
    return conditions, info


def write_sets(con, conditions):
    """Write the three sets straight from DuckDB to parquet (without going through Python)."""
    rows = {}
    for name, cond in conditions.items():
        out = (ENC_DIR / f"{name}.parquet").as_posix()
        t0 = time.time()
        con.execute(f"COPY ({encoded_query(cond)}) TO '{out}' (FORMAT parquet, COMPRESSION zstd)")
        rows[name] = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
        print(f"  {name:10s} {rows[name]:>12,} rows  ({time.time() - t0:.1f} s)  -> {out}")
    return rows


def quality_checks(con, vocab):
    """MISSING / RARE / UNK shares per column and set; frequent codes with no example in train_enc."""
    tables = []
    for name in ["train_enc", "tune_enc", "val_enc"]:
        f = (ENC_DIR / f"{name}.parquet").as_posix()
        parts = [f"""
            SELECT '{name}' AS split, '{c}' AS column_name,
                   100 * AVG(CASE WHEN id_{c} = {MISSING} THEN 1 ELSE 0 END) AS pct_missing,
                   100 * AVG(CASE WHEN id_{c} = {RARE}    THEN 1 ELSE 0 END) AS pct_rare,
                   100 * AVG(CASE WHEN id_{c} = {UNK}     THEN 1 ELSE 0 END) AS pct_unk
            FROM read_parquet('{f}')""" for c in CAT_FEATURES]
        tables.append(con.execute(" UNION ALL ".join(parts)).df())
    quality = pd.concat(tables, ignore_index=True)

    # "Frequent" codes (>= N_MIN in the full train split) that never appear in train_enc:
    # the model has no example to learn their coefficient / embedding.
    f_train = (ENC_DIR / "train_enc.parquet").as_posix()
    without_examples = {}
    for name, columns in VOCAB_COLUMNS.items():
        parts = [f"SELECT id_{c} AS id FROM read_parquet('{f_train}')" for c in columns]
        present = con.execute(f"""
            SELECT COUNT(DISTINCT id) FROM ({" UNION ALL ".join(parts)}) WHERE id >= 4
        """).fetchone()[0]
        without_examples[name] = len(vocab[name]) - int(present)
    return quality, without_examples


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.time()
    print(f"Project: {PROJECT_ROOT}")
    if not PREPARED.exists():
        raise FileNotFoundError(PREPARED)
    ENC_DIR.mkdir(parents=True, exist_ok=True)

    con = connect()
    print("DuckDB", duckdb.__version__)

    print("\n[1/6] NULL check")
    check_nulls(con)

    print("\n[2/6] Vocabularies (on the full train split)")
    vocab, seen = build_vocabularies(con)

    print("\n[3/6] Value map")
    map_df = build_value_map(con, vocab, seen)
    map_df.to_parquet(ENC_DIR / "value_map.parquet", index=False)

    print("\n[4/6] Tune boundary and samples")
    conditions, tune_info = split_conditions(con)

    print("\n[5/6] Writing the encoded sets")
    rows = write_sets(con, conditions)

    print("\n[6/6] Quality checks")
    quality, without_examples = quality_checks(con, vocab)
    quality.round(4).to_csv(ENC_DIR / "quality.csv", index=False)
    with pd.option_context("display.max_rows", 100, "display.width", 120):
        print(quality.round(3).to_string(index=False))
    print("Frequent codes with no example in train_enc:", without_examples)

    # --- Vocabularies and metadata as JSON ---
    with open(ENC_DIR / "vocabularies.json", "w", encoding="utf-8") as f:
        json.dump({
            "reserved": {"PAD": PAD, "MISSING": MISSING, "RARE": RARE, "UNK": UNK},
            "N_MIN": N_MIN,
            "columns": VOCAB_COLUMNS,
            "vocabularies": vocab,
        }, f, ensure_ascii=False, indent=2)

    meta = {
        "source": PREPARED.name,
        "duckdb": duckdb.__version__,
        "target": TARGET,
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "id_columns": ["id_" + c for c in CAT_FEATURES],
        "eval_only_columns": EVAL_COLUMNS,
        # embedding table sizes: 4 reserved codes + frequent values
        "n_codes": {name: 4 + len(v) for name, v in vocab.items()},
        "N_MIN": N_MIN, "N_TRAIN": N_TRAIN, "N_TUNE": N_TUNE, "TUNE_FRACTION": TUNE_FRACTION,
        "hash": HASH,
        **tune_info,
        "rows": {k: int(v) for k, v in rows.items()},
        "frequent_codes_without_examples_in_train_enc": without_examples,
    }
    with open(ENC_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone in {time.time() - t_start:.0f} s. Files in {ENC_DIR}")


if __name__ == "__main__":
    main()
