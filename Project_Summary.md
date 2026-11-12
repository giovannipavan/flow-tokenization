# Network Flow Tokenization: Project Summary (Group 1)

*Status: 12 November 2026*

## 1. Objective

Assess whether CERN network flows (Scitags **Firefly** telemetry) can be represented as **sequences of discrete tokens** without losing information relevant to a downstream prediction task.

- **Motivation:** OmniJet-α tokenizes particle jets for transformer-style models; the same idea is tested here on network data.
- **Downstream task:** regression of `delta_rtt = end_rtt − start_rtt`, i.e. the RTT change over the lifetime of a flow (as specified in the dataset README; see open question 3).
- **Tokenization strategies compared:**
  - feature-level (one token per field);
  - flow-level (one token per field combination);
  - learned tokens via vector quantization (optional).
- **Shared with Group 2 (JEPA):** same data and same chronological split.

## 2. Dataset

- `flows.parquet`: **67,244,519 reconstructed flows**, 23 columns, covering flow starts from 2026-10-08 to 2026-10-14 (UTC).
- **One row = one flow:** start and end Firefly events paired on `(flow_id, flow_start)`.
- **Chronological split (70/15/15, on `flow_start`):**
  - train: up to 2026-10-12 17:12:05;
  - validation: up to 2026-10-13 22:18:28;
  - test: the remainder.
- **Excluded from model inputs** (end-event information): `end_rtt`, `delta_rtt`, `flow_end`, flow duration, `endpoints_match`, `endpoints_reversed`, `start_fields_match`. `flow_id` is used as an identifier only.
- **Allowed with conditions:**
  - `start_rtt`: predictions are made after the start event;
  - CRIC site and ASN: pending confirmation of availability at prediction time.
- **Fitted on the training split only:** all vocabularies, thresholds, encoders and imputation tables.

## 3. Exploratory findings

**Completeness.** IPs, ports, timestamps, RTTs, protocol, IP version, experiment, activity and reporting host contain no nulls.

| Field | Missing | Note |
|---|---|---|
| CRIC site / ASN (source) | 18.35% | always missing together |
| CRIC site / ASN (destination) | 4.03% | always missing together |
| `meta.application` | 0.71% null + 8.24% `unknown` | |
| `meta.experiment_name` | 0.43% `unknown` | |

Site and ASN are absent whenever the IP is not registered in CRIC. Sources (clients, worker nodes) are less covered than destinations (storage endpoints).

**RTT units.** Values are consistent with **milliseconds**: ~0.1 ms intra-site, ~11 ms FZK–CERN, ~236 ms TW-FTT–Nebraska. The minimum positive value is 0.002 ms.

**Target distribution.**

| | p01 | p25 | p50 | p75 | p99 |
|---|---|---|---|---|---|
| `delta_rtt` (ms) | −9.0 | 0.06 | 4.44 | 5.25 | 15.5 |

- 14.4% of values are negative. Extreme values reach −14,994 and +120,017 ms.
- The distribution is **bimodal**: a sharp mode at 0 and a mode at **~5 ms**.
- The ~5 ms mode is driven mainly by **long flows** (> 1 min) and appears for both values of `endpoints_reversed`.
- The apparent dependence on `endpoints_reversed` is largely **confounded by flow duration**: reversed flows are mostly long, matched flows mostly short.
- The ~5 ms value is roughly constant regardless of path length. This is consistent with a queueing target such as the fq_codel default (5 ms). Not verified.

**Zero RTTs are missing measurements.**
- **`end_rtt = 0` (357,462 flows):**
  - 99.99% have `start_rtt ≥ 1 ms`;
  - ~90% are flows towards **USCMS-FNAL-WC1**, reaching 89–100% of flows for some pairs (UCSDT2, Nebraska, MIT_CMS → FNAL), versus ≤ 0.02% on other high-volume pairs.
- **`start_rtt = 0` (773 flows):** spread across unrelated site pairs, with normal end RTTs.
- **The incidence drops over time:** 0.72% of train rows, 0.17% of validation rows, 0.003% of test rows.

## 4. Data cleaning

- **Rule:** remove flows with `start_rtt = 0` or `end_rtt = 0`. The target is undefined for these rows. The rule is applied to all splits and introduces no leakage.

| Split | Before | After | Removed |
|---|---|---|---|
| train | 47,071,163 | 46,730,686 | 340,477 (0.72%) |
| validation | 10,086,678 | 10,069,254 | 17,424 (0.17%) |
| test | 10,086,678 | 10,086,352 | 326 (0.003%) |
| **total** | **67,244,519** | **66,886,292** | **358,227 (0.53%)** |

- **Retained:**
  - negative `delta_rtt` values (genuine RTT decreases);
  - extreme values, for now (any clipping will use train-only quantiles).
- **Known limitation:** reduced and possibly unrepresentative coverage of flows towards FNAL. This subset is evaluated separately.

## 5. Representation of endpoint location

No imputation applied yet. Three representations will be compared:

| Option | Token | Property |
|---|---|---|
| A | CRIC site, with a `MISSING` token | interpretable; uninformative for 18% of sources |
| B | CRIC site reconstructed from the IP prefix where missing | prefix→site table built on train; used only if purity ≥ 95% and support ≥ 100; imputation flag added |
| C | IP prefix (`/24` IPv4, `/48` IPv6) | always available; larger vocabulary; independent of CRIC |

**Default:** prefix and site tokens together (C + A). B is retained if its coverage of missing sources is significant.

## 6. Plan

### 6.1 Data preparation (`02_preparation`)
1. **Cardinality** of categorical fields on train.
2. **Value concentration:** top-k coverage at 95% and 99%, to set frequency thresholds for `RARE`.
3. **IP prefixes** computed on unique IPs (Python `ipaddress`), with distinct prefix counts.
4. **Prefix→site table** on train; coverage of option B on missing sources.
5. **Train/validation/test overlap** per field, to estimate the expected `UNK` rate.
6. **Single prepared file** (`flows_prepared.parquet`) with columns for all representations (`site`, `pfx`, `site_b`, imputation flags, `split`). Each experiment selects columns via configuration. No duplicated datasets.

### 6.2 Baselines (`03_baselines`)
Trained on a random subsample of train (~3 M rows), evaluated on validation.

| # | Baseline | Purpose |
|---|---|---|
| 0 | Global train median | lower bound (optimal constant under MAE) |
| 1 | Median per site/prefix pair, with hierarchical fallback | value of endpoint identity |
| 2 | Linear regression | additive effects |
| 3 | MLP on non-tokenized inputs | direct comparison for tokenization: same network, different input encoding |
| 4 | LightGBM (native categoricals, L1 objective) | strong tabular reference; feature importance |

- **Diagnostic only:** LightGBM including flow duration, to estimate the error attributable to the unknown duration. Excluded from the official comparison.
- **Also covered:** hourly flow volume and `delta_rtt`, to decide on time-of-day features.

### 6.3 Tokenization (`04_tokenization`)
1. **Feature-level** tokens → embeddings → concatenation → same MLP as baseline 3.
2. **Flow-level** tokens.
3. **Vector quantization** (optional).
4. **Ablations:**
   - site vs prefix vs both;
   - prefix length (`/16`, `/24`, `/48`, `/64`);
   - with and without `start_rtt`;
   - leave-one-field-out.

### 6.4 Evaluation
- **Metrics:** MAE (primary) and RMSE on validation; vocabulary size; embedding memory; `UNK`/`RARE` rate.
- **Fair comparison:** all models share vocabularies, loss and data.
- **Test set:** used once, for the final comparison, including a separate breakdown for FNAL flows.

## 7. Open questions for CERN IT

1. Is the ~5 ms `delta_rtt` mode in long flows a real queueing effect (e.g. fq_codel) or a difference between start and end measurement sources?
2. Is `end_rtt = 0` towards USCMS-FNAL-WC1 a known producer issue, and was it fixed during the covered week?
3. Is the intended target `delta_rtt` (README) or the RTT itself (project slides)?
4. Are CRIC site and ASN available at prediction time in an operational setting?
5. Would a secondary task on telemetry quality (predicting failed end measurements) be of interest?