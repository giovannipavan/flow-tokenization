"""Metrics shared by all models.

Every model produces a DuckDB table with one row per flow of the evaluated split and the
columns (flow_id, t_start, pred). `evaluate` joins it with the encoded set of that split
(`val_enc` or `test_enc`), which holds the target and the columns that define the subgroups,
and computes the same metrics for everyone:

    mae          mean absolute error (primary metric)
    rmse         root mean squared error
    rmse_robust  RMSE on flows with delta_rtt inside the robust thresholds only
                 (quantiles 0.05 % and 99.95 % of the training split, same flows for every model)
    p90_abs      90th percentile of the absolute error
    p99_abs      99th percentile of the absolute error

All values are in ms.
"""

from flowtok import data

# subgroup name -> SQL condition on the columns of the encoded set
SUBGROUPS = {
    "overall":               "TRUE",
    "FNAL":                  "server_site = 'USCMS-FNAL-WC1' OR client_site = 'USCMS-FNAL-WC1'",
    "missing site":          "server_site = 'MISSING' OR client_site = 'MISSING'",
    "ambiguous orientation": "orientation_ambiguous",
    "flow > 1 min":          "an_duration_s > 60",
}

METRICS = ["mae", "rmse", "rmse_robust", "p90_abs", "p99_abs"]


def robust_thresholds(con, meta=None):
    """[lo, hi] for the robust RMSE: from meta.json, or computed from the training split."""
    meta = meta or data.load_meta()
    if "robust_rmse_thresholds" in meta:
        return meta["robust_rmse_thresholds"]
    # encoded files written before the thresholds were stored in meta.json
    q_lo, q_hi = meta.get("robust_quantiles", [0.0005, 0.9995])
    lo, hi = con.execute(f"""
        SELECT quantile_cont(delta_rtt, [{q_lo}, {q_hi}]) FROM fp WHERE split = 'train'
    """).fetchone()[0]
    return [float(lo), float(hi)]


def check_predictions(con, pred_table, split):
    """One finite prediction for every flow of the split, no duplicates."""
    n_flows, n_pred, n_missing, n_bad = con.execute(f"""
        SELECT COUNT(*),
               COUNT(p.pred),
               COUNT(*) - COUNT(p.pred),
               COUNT(*) FILTER (WHERE NOT isfinite(p.pred))
        FROM {split}_enc AS e
        LEFT JOIN {pred_table} AS p ON e.flow_id = p.flow_id AND e.t_start = p.t_start
    """).fetchone()
    n_rows = con.execute(f"SELECT COUNT(*) FROM {pred_table}").fetchone()[0]
    if n_missing > 0 or n_bad > 0 or n_rows != n_flows:
        raise ValueError(f"Predictions do not match the {split} split: {n_flows:,} flows, "
                         f"{n_rows:,} predictions, {n_missing:,} flows without prediction, "
                         f"{n_bad:,} non-finite predictions.")


def evaluate(con, pred_table, split, thresholds):
    """Metrics per subgroup (one row each) for the predictions in `pred_table`."""
    check_predictions(con, pred_table, split)
    lo, hi = thresholds
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW eval_view AS
        SELECT e.delta_rtt, p.pred,
               e.server_site, e.client_site, e.orientation_ambiguous, e.an_duration_s
        FROM {split}_enc AS e
        JOIN {pred_table} AS p ON e.flow_id = p.flow_id AND e.t_start = p.t_start
    """)
    parts = []
    for order, (name, condition) in enumerate(SUBGROUPS.items()):
        parts.append(f"""
            SELECT {order} AS ord, '{name}' AS subgroup, COUNT(*) AS n,
                   AVG(ABS(delta_rtt - pred))                            AS mae,
                   SQRT(AVG(POWER(delta_rtt - pred, 2)))                 AS rmse,
                   SQRT(AVG(POWER(delta_rtt - pred, 2))
                        FILTER (WHERE delta_rtt BETWEEN {lo} AND {hi}))  AS rmse_robust,
                   quantile_cont(ABS(delta_rtt - pred), 0.90)            AS p90_abs,
                   quantile_cont(ABS(delta_rtt - pred), 0.99)            AS p99_abs
            FROM eval_view
            WHERE {condition}
        """)
    query = "SELECT * FROM (" + " UNION ALL ".join(parts) + ") ORDER BY ord"
    return con.execute(query).df().drop(columns="ord")


def register_numpy_predictions(con, split, row_id, pred, table):
    """Turn predictions computed in Python (aligned with row_id of the encoded set)
    into a DuckDB table with the common columns (flow_id, t_start, pred)."""
    import pandas as pd

    df = pd.DataFrame({"row_id": row_id, "pred": pred.astype("float64")})
    con.register("numpy_pred_df", df)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT e.flow_id, e.t_start, p.pred
        FROM {split}_enc AS e
        JOIN numpy_pred_df AS p ON e.row_id = p.row_id
    """)
    con.unregister("numpy_pred_df")
    return table
