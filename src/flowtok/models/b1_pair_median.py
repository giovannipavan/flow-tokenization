"""B1 - Median per endpoint pair, with hierarchical fallback.

For each pair of endpoints seen at least `min_flows` times in the training split, the
median of delta_rtt is stored. A flow gets the median of the first level available:
    1. pair of endpoints (CRIC sites or L2 prefixes, see `identifier`)
    2. pair of ASNs
    3. global median (B0)
It measures how much the identity of the two endpoints alone explains.

Settings (config block "b1"):
    identifier   "site" or "L2"            how an endpoint is identified
    orientation  "server_client"           (server, client): role in the connection
                 "src_dst"                 (source, destination): as recorded by Firefly
                 "unordered"               the pair without order
    min_flows    minimum training flows for a pair to get its own median
"""

import time

DEFAULTS = {"identifier": "L2", "orientation": "server_client", "min_flows": 1000}

# SQL expressions of the two keys of a pair; {p} = table prefix, {x} = identifier column suffix
ORIENTATIONS = {
    "unordered":     ("LEAST({p}server_{x}, {p}client_{x})", "GREATEST({p}server_{x}, {p}client_{x})"),
    "server_client": ("{p}server_{x}", "{p}client_{x}"),
    "src_dst":       ("{p}src_{x}",    "{p}dst_{x}"),
}


def keys(orientation, x, p=""):
    """The two SQL expressions of the pair, for an orientation and an identifier."""
    k1, k2 = ORIENTATIONS[orientation]
    return k1.format(p=p, x=x), k2.format(p=p, x=x)


def run(con, cfg, split, model_dir):
    cfg = {**DEFAULTS, **cfg}
    ident, orient, min_flows = cfg["identifier"], cfg["orientation"], int(cfg["min_flows"])
    if ident not in ("site", "L2"):
        raise ValueError(f"b1.identifier must be 'site' or 'L2', not '{ident}'")
    if orient not in ORIENTATIONS:
        raise ValueError(f"b1.orientation must be one of {list(ORIENTATIONS)}, not '{orient}'")

    # --- Training: medians on the full training split ---
    t0 = time.time()
    global_median = float(con.execute(
        "SELECT MEDIAN(delta_rtt) FROM fp WHERE split = 'train'").fetchone()[0])
    for level, x in [(1, ident), (2, "asn")]:
        k1, k2 = keys(orient, x)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE b1_level{level} AS
            SELECT {k1} AS k1, {k2} AS k2, MEDIAN(delta_rtt) AS med, COUNT(*) AS n
            FROM fp
            WHERE split = 'train'
            GROUP BY ALL
            HAVING COUNT(*) >= {min_flows}
        """)
    train_time = time.time() - t0

    # --- Prediction: the first level available wins ---
    a1, b1 = keys(orient, ident, p="f.")
    a2, b2 = keys(orient, "asn", p="f.")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pred_b1 AS
        SELECT f.flow_id, f.t_start,
               COALESCE(m1.med, m2.med, {global_median}) AS pred,
               CASE WHEN m1.med IS NOT NULL THEN 1
                    WHEN m2.med IS NOT NULL THEN 2
                    ELSE 3 END AS level
        FROM fp AS f
        LEFT JOIN b1_level1 AS m1 ON {a1} = m1.k1 AND {b1} = m1.k2
        LEFT JOIN b1_level2 AS m2 ON {a2} = m2.k1 AND {b2} = m2.k2
        WHERE f.split = '{split}'
    """)

    # share of flows predicted at each level
    levels = con.execute("""
        SELECT level, COUNT(*) AS n, 100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct
        FROM pred_b1 GROUP BY level ORDER BY level
    """).df()
    levels["level"] = levels["level"].map({1: f"{ident} pair", 2: "ASN pair", 3: "global median"})

    # --- Save the model: the two tables of medians ---
    model_dir.mkdir(parents=True, exist_ok=True)
    for level in (1, 2):
        out = (model_dir / f"b1_level{level}.parquet").as_posix()
        con.execute(f"COPY b1_level{level} TO '{out}' (FORMAT parquet)")

    n_medians = sum(con.execute(f"SELECT COUNT(*) FROM b1_level{l}").fetchone()[0] for l in (1, 2))
    info = {
        "model": f"B1 median per {ident} pair ({orient})",
        "n_params": int(n_medians) + 1,
        "train_time_s": round(train_time, 1),
        **cfg,
        "global_median": global_median,
    }
    return "pred_b1", info, {"levels": levels}
