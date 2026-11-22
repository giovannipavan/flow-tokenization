"""DuckDB connection and views on the data files.

Views created by the functions below:
    fp                       flows_prepared.parquet (all splits, readable values)
    train_enc, tune_enc,     encoded sets written by prepare (only those that exist)
    val_enc, test_enc
DuckDB reads only the columns and rows that each query needs, so the 5 GB file is never
loaded in memory as a whole.
"""

import duckdb

from flowtok import paths

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

ENCODED_SETS = ["train_enc", "tune_enc", "val_enc", "test_enc"]


def connect(threads=8, memory_limit="8GB"):
    """New DuckDB connection with the settings used throughout the project."""
    con = duckdb.connect()
    con.execute(f"SET threads TO {threads}")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{paths.DUCKDB_TMP.as_posix()}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET TimeZone = 'UTC'")
    return con


def register_prepared(con):
    """View `fp` on flows_prepared.parquet, with legacy column names translated."""
    if not paths.PREPARED.exists():
        raise FileNotFoundError(f"{paths.PREPARED} not found (see README, section Data).")
    source = f"read_parquet('{paths.PREPARED.as_posix()}')"
    columns = set(con.execute(f"SELECT * FROM {source} LIMIT 0").df().columns)
    legacy = {old: new for old, new in LEGACY_COLUMN_NAMES.items() if old in columns}
    if legacy:
        excluded = ", ".join(legacy)
        renamed = ", ".join(f"{old} AS {new}" for old, new in legacy.items())
        select = f"SELECT * EXCLUDE ({excluded}), {renamed} FROM {source}"
    else:
        select = f"SELECT * FROM {source}"
    con.execute(f"CREATE OR REPLACE VIEW fp AS {select}")


def register_encoded(con):
    """One view per encoded set found in data/encoded/. Returns the names of the views."""
    found = []
    for name in ENCODED_SETS:
        f = paths.ENC_DIR / f"{name}.parquet"
        if f.exists():
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{f.as_posix()}')")
            found.append(name)
    return found
