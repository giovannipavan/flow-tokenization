# Project context: network flow tokenization (CS-433, AI4Science)

This file explains the project to coding agents and collaborators. Read it first.
Where something is marked **TODO / unknown / unverified**, do not guess: inspect the
data or ask.

## What this is

Course project 2 of **CS-433 Machine Learning** (EPFL), under the AI4Science initiative.
Three Master's students work part-time from early November to **17 December 2026**
and deliver a **4-page report and the code**. The project counts for 30% of the
course grade. Code and results must be **open source**.

- **Team:** Giovanni (Computer Science), Mario and Matteo (Applied Mathematics).
- **Host / supervisor:** Carmen Misa (CERN), who works on networking data from the
  WLCG (the computing grid that moves data from CERN to research institutes).
- **Head of the host group:** Sofia Vallecorsa (CERN).

## Goal, in stages

The work is layered so that there is always a complete deliverable:

1. **Baseline (minimum deliverable).** Supervised prediction of the target
   `delta_rtt` (see Data) with a **chronological split**. Start from trivial baselines
   (global mean/median, median per site pair), then ridge regression, then gradient
   boosting (XGBoost / LightGBM), optionally a small MLP.
2. **Tokenization.** Turn each flow into a short sequence of discrete tokens built from
   start-event fields such as IPs, ports, site names, ASNs, protocol and application.
   Decide how to represent each field (examples: IP prefixes or site IDs; well-known
   ports as their own tokens and ephemeral ports as one generic token; continuous
   values binned into intervals). The first task agreed with the supervisor is a
   **literature review** and agreeing on the representation.
3. **Self-supervised pre-training.** A small transformer trained with masked
   prediction on token sequences, without labels.
4. **Comparison.** Fine-tune the pre-trained model on the same prediction task as
   stage 1 and compare it fairly against the baselines (same split, same metric).

If tokenization turns out to be too complex, stage 1 alone is still a complete project.

## Data

- **Source:** public Firefly network telemetry exported from the ESnet Stardust
  Elasticsearch service, covering US WLCG computing sites. Shared by the supervisor
  through CERNbox. More data may arrive in the following weeks.
- **Current release:** **67,244,519 reconstructed flows**; flow starts from
  2026-10-07T23:59:59 UTC to 2026-10-14T23:59:59 UTC.
- **Files in the data directory** (about 4-5 GB):
  - `flows.parquet`: the flows.
  - `manifest.json`: schema, version, creation time, validation results and the
    SHA-256 of `flows.parquet`.
  - `splits.json`: chronological split definitions.
  - the dataset README (full description; the summary below comes from it).
- Each row pairs exactly one start event and one end event, keyed by
  `(flow_id, flow_start)`. Ambiguous, duplicate and unmatched events were excluded.
  **No further filtering, clipping, imputation or unit conversion** was applied:
  negative RTTs and outliers are retained.
- **Provenance (unverified):** the column `meta.syslog.hostname` is the reporting host,
  which suggests many reporting hosts rather than a single monitoring station. Verify
  by counting distinct hosts and sites.

### Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `flow_id` | string | Original meta.id; identifier only (not a predictor). |
| `flow_start` | string | Start timestamp (string); actual flow initiation time. |
| `flow_end` | string | End timestamp from the end event. |
| `start_rtt` | double | RTT observed in the start event. |
| `end_rtt` | double | RTT observed in the end event. |
| `delta_rtt` | double | `end_rtt - start_rtt`; positive means RTT increased. **Target.** |
| `meta.src_ip`, `meta.dst_ip` | string | Source / destination IP (from the start event). |
| `meta.src_port`, `meta.dst_port` | int64 | Source / destination transport port. |
| `meta.cric.src.name`, `meta.cric.dst.name` | string | Site names (CRIC enrichment). |
| `meta.cric.src.asn`, `meta.cric.dst.asn` | int64 | AS numbers (CRIC enrichment). |
| `meta.application`, `meta.experiment_name` | string | Application / experiment labels. |
| `meta.activity_id` | int64 | Activity label. |
| `meta.protocol` | string | Transport protocol. |
| `meta.ip_version` | int64 | IP version. |
| `meta.syslog.hostname` | string | Reporting host. |
| `endpoints_match`, `endpoints_reversed`, `start_fields_match` | bool | Diagnostics that use end-event information. |

Notes:
- There is **no throughput column** in this release. The only RTT-related quantities
  are `start_rtt`, `end_rtt` and `delta_rtt`.
- `start_rtt` and `end_rtt` are endpoint observations, not averages over the flow.
  They come from `values.netlink.rtt`. **Unverified hypothesis:** this is the Linux
  kernel's per-connection TCP RTT estimate (a smoothed round-trip estimate measured by
  the reporting host), not a one-way delay. Confirm with the supervisor, and check the
  distribution of `meta.protocol` (QUIC/HTTP3 would not appear as TCP).
- RTT keeps the source telemetry units; the unit is **not calibrated**, and negative
  values exist in the data. Do not interpret physical magnitudes before checking.
- Nulls and empty strings keep their original meaning. Missing enrichment does not
  imply a zero or unknown category. Site labels and other context can be incomplete.
- The dataset has 67M rows; for exploration prefer lazy readers (DuckDB, Polars
  `scan_parquet`, PyArrow) and select only the columns needed.

### Leakage rules (important)

- Allowed inputs (from the start event): IPs, ports, protocol, IP version, reporting
  host, application, experiment, activity, and time features derived from
  `flow_start`.
- `start_rtt` is available only after the start-event RTT observation arrives. It is
  not suitable for predictions made before that measurement; decide explicitly whether
  to use it and document the choice.
- CRIC site names and ASNs are enrichment: check their availability at prediction time
  before using them.
- **Never use as inputs:** `end_rtt`, `delta_rtt`, `flow_end`, derived duration,
  `endpoints_match`, `endpoints_reversed`, `start_fields_match`.
- Fit tokenizers, vocabularies, imputers, scalers and feature selection on **training
  data only**. Use test data only for final evaluation.

### Chronological splits

`splits.json` defines train / validation / test (about 70% / 15% / 15%).
- Compare parsed start times in UTC, never timestamp strings or ingestion times.
  Equal-timestamp groups stay together.
- Training ends inclusively at `2026-10-12T17:12:05.477960+00:00`; validation ends
  inclusively at `2026-10-13T22:18:28.700870+00:00`; test starts strictly after.
- Counts: train 47,071,163; validation 10,086,678; test 10,086,678.
- Flows can cross split boundaries and hosts/IDs recur. Evaluate host/site
  generalization separately if needed. For a deployment-style simulation, exclude
  training labels whose end times were not yet known at the prediction cutoff.

Loading with DuckDB (run from the data directory, set `TimeZone` to UTC):

```python
import json, duckdb
con = duckdb.connect()
con.execute("SET TimeZone = 'UTC'")
con.execute("CREATE VIEW flows AS SELECT * FROM read_parquet('flows.parquet')")
splits = json.load(open('splits.json'))
```

## Environment

- Python virtual environment in `.venv/` (not committed). Install with
  `pip install -r requirements.txt`.
- `requirements.txt` lists the direct dependencies; `requirements-lock.txt` is a
  `pip freeze` snapshot of the exact versions used. Update the lock file whenever
  dependencies change.
- PyTorch is added later, once the GPU / CUDA setup of the target machine is known.
- Data lives outside the repo; its path is read from an environment variable
  (for example `FLOWS_DATA_DIR`) defined in a local `.env` file (not committed).

## Compute

- The supervisor recommends the **EPFL GPU cluster**. **TODO:** find out how students
  can get access and what the limits are. Using CERN resources would need an account
  and has long queues.
- Fallback for small experiments: Colab or Kaggle GPUs.
- Keep the model small (a few million parameters) so it trains on a single GPU.

## Repository notes

- Do **not** commit raw data to git (the data directory is several GB). Keep it in
  `.gitignore`; keep the code reproducible from a documented path.
- Suggested layout: `src/` (loading, preprocessing, tokenizer, models), `notebooks/`
  (exploration only), `results/` (metrics, figures, logs), `report/` (4-page report).
- Personal agent instructions (not shared) must not be committed here. If an agent
  needs its own entry file (for example `CLAUDE.md` for Claude Code), make it import
  this file instead of duplicating it.

## Working conventions

- **Language:** code, comments and documentation in English; talk to Giovanni in
  Italian unless asked otherwise.
- Use the provided chronological split; never shuffle randomly across time.
- Report **MAE and RMSE**. `delta_rtt` can be negative and heavy tailed, so inspect its
  distribution before choosing loss and metrics; do not apply a log transform blindly.
  Always compare against simple baselines (mean/median, median per site pair).
- Fix random seeds, log metrics and configs for every experiment.
- Prefer simple, readable code. Keep notebooks for exploration; put reusable logic in
  scripts or a package.
- Do not invent facts about the data or the literature. If you cite a paper, say what
  it is and flag anything unverified.

## Reading list to verify (may be incomplete or inaccurate)

- Masked-language-model pre-training (BERT-style).
- Network traffic pre-trained models: ET-BERT, NetFound.
- Embeddings for numerical features in tabular deep learning (Gorishniy et al.),
  FT-Transformer.
- Binning / quantization of continuous values and IP address tokenization schemes.
