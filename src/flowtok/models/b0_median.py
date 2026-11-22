"""B0 - Global median.

Predicts the median of delta_rtt over the whole training split for every flow.
It is the best constant prediction under the MAE: the floor, i.e. the error made
without using any information about the flow.
"""

import json
import time

DEFAULTS = {}


def train_median(con):
    """Median of delta_rtt over the full training split (46.7 M flows)."""
    return float(con.execute("SELECT MEDIAN(delta_rtt) FROM fp WHERE split = 'train'").fetchone()[0])


def run(con, cfg, split, model_dir):
    t0 = time.time()
    median = train_median(con)
    train_time = time.time() - t0

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pred_b0 AS
        SELECT flow_id, t_start, CAST({median} AS DOUBLE) AS pred
        FROM fp
        WHERE split = '{split}'
    """)

    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "b0.json", "w", encoding="utf-8") as f:
        json.dump({"train_median": median}, f, indent=2)

    info = {
        "model": "B0 global median",
        "n_params": 1,
        "train_time_s": round(train_time, 1),
        "train_median": median,
    }
    return "pred_b0", info, {}
