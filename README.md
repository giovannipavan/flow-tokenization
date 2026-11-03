# Network flow tokenization

Project for the EPFL course **CS-433 Machine Learning** (AI4Science initiative, Fall 2026).
We study how to turn network flow telemetry from the Worldwide LHC Computing Grid (WLCG)
into discrete tokens, and whether a model pre-trained on those tokens without labels
helps to predict how the round-trip time (RTT) of a flow changes over its lifetime.

> Status: work in progress. The project runs from November to 17 December 2026.

## Idea

Each flow is described by metadata known when it starts (endpoints, ports, protocol,
sites, application, ...) and by RTT observations at its start and at its end. The work
is organised in stages, so there is always a complete result:

1. **Baseline.** Supervised prediction of `delta_rtt = end_rtt - start_rtt` from
   start-event information, with a chronological train / validation / test split.
   Models: simple baselines, ridge regression, gradient boosting, optionally a small
   neural network.
2. **Tokenization.** Represent each flow as a short sequence of discrete tokens
   (for example IP prefixes or site IDs, port classes, binned numeric values).
3. **Self-supervised pre-training.** A small transformer trained with masked
   prediction on the token sequences.
4. **Comparison.** Fine-tuning the pre-trained model on the same prediction task and
   comparing it fairly against the baselines.

## Data

We use a public extract of Firefly network telemetry, exported from the ESnet Stardust
service, covering US computing sites of the WLCG. The extract used so far contains
about 67 million reconstructed flows over seven days in October 2026. Each row pairs
one flow start event with its end event.

The data is **not stored in this repository**. See `AGENTS.md` for a detailed
description of the fields, the leakage rules and the chronological splits.

## Setup

```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then tell the code where the data is, with a local `.env` file (not committed):

```
FLOWS_DATA_DIR=/path/to/the/data/folder
```

The folder is expected to contain `flows.parquet`, `manifest.json` and `splits.json`.

## Repository layout

```
.
├── AGENTS.md        # context for coding agents and collaborators
├── README.md        # this file
├── requirements.txt # direct dependencies
├── src/             # loading, preprocessing, tokenizer, models
├── notebooks/       # exploration only
├── results/         # metrics, figures, logs
└── report/          # sources of the 4-page report
```

## Team

Giovanni, Mario and Matteo (EPFL Master's students), with a research host at CERN.

## License

See `LICENSE`.
