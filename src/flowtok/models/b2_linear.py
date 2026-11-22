"""B2 - Linear model with additive effects, trained with an L1 loss.

    pred = b + w * x_rtt + sum_j theta_j[code_j]

b is the intercept, x_rtt the standardized log of start_rtt, and theta_j a table with one
coefficient per code of the categorical column j (nn.Embedding(n, 1), equivalent to one-hot
encoding followed by a dot product). Server and client columns have separate tables even
when they share a vocabulary, because the same site can have a different effect in the two
roles. An L2 penalty (weight decay, `lambda`) applies to the tables only.

The final model is trained on train_enc + tune_enc for a fixed number of epochs; the
learning rate is multiplied by `gamma` after each epoch. The settings were chosen on
tune_enc with `tune` (lambda grid, early stopping).
"""

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from flowtok import data, evaluation

DEFAULTS = {"lambda": 1e-5, "epochs": 5, "lr": 0.01, "gamma": 0.5, "batch": 4096, "seed": 0}
TUNING_DEFAULTS = {"lambda": [0.0, 1e-6, 1e-5, 1e-4, 1e-3], "max_epochs": 8, "patience": 3}


class Linear(nn.Module):
    def __init__(self, n_codes, initial_intercept):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n, 1) for n in n_codes])
        for e in self.emb:
            nn.init.zeros_(e.weight)
        self.w = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.tensor([float(initial_intercept)]))

    def forward(self, x_cat, x_rtt):
        out = self.b + self.w * x_rtt
        for j, e in enumerate(self.emb):
            out = out + e(x_cat[:, j]).squeeze(1)
        return out


# =============================================================================
# Data
# =============================================================================

def rtt_transform(rtt_train):
    """Standardized log(start_rtt), with mean and std estimated on the training data."""
    if rtt_train.min() <= 0:
        raise ValueError("start_rtt <= 0: the log is not defined")
    log_tr = np.log(rtt_train)
    mu, sigma = float(log_tr.mean()), float(log_tr.std())
    return mu, sigma


def apply_rtt(rtt, mu, sigma):
    return ((np.log(rtt) - mu) / sigma).astype(np.float32)


def load(name, meta):
    df = data.load_encoded(name, meta)
    X, rtt, y = data.to_arrays(df, meta)
    return df["row_id"].to_numpy(), X, rtt, y


# =============================================================================
# Training and prediction
# =============================================================================

@torch.no_grad()
def predict(model, X, x, block=500_000):
    """Predictions in blocks, to avoid huge tensors (validation has 10 M rows)."""
    model.eval()
    out = [model(torch.from_numpy(X[i:i + block]).long(), torch.from_numpy(x[i:i + block]))
           for i in range(0, len(X), block)]
    return torch.cat(out).numpy()


def fit(X, x, y, n_codes, lam, epochs, lr, gamma, batch, seed, initial_intercept,
        X_tune=None, x_tune=None, y_tune=None, patience=None):
    """Train with Adam and an L1 loss.

    Without a tune set: exactly `epochs` epochs, returns the final model.
    With a tune set: at most `epochs` epochs, stops when the tune MAE has not improved for
    `patience` epochs, and returns the best model. Returns (model, best_epoch, history).
    """
    torch.manual_seed(seed)
    model = Linear(n_codes, initial_intercept)
    opt = torch.optim.Adam([
        {"params": model.emb.parameters(), "weight_decay": lam},   # coefficients: penalized
        {"params": [model.w, model.b],     "weight_decay": 0.0},   # slope and intercept: not
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    Xt, xt, yt = torch.from_numpy(X).long(), torch.from_numpy(x), torch.from_numpy(y)
    n = len(yt)
    best_mae, best_state, history = float("inf"), None, []
    best_epoch = 0 if X_tune is not None else epochs

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            loss = (model(Xt[idx], xt[idx]) - yt[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train_mae": total / n}
        scheduler.step()

        if X_tune is not None:
            row["tune_mae"] = float(np.abs(predict(model, X_tune, x_tune) - y_tune).mean())
            if row["tune_mae"] < best_mae - 1e-4:
                best_mae, best_epoch = row["tune_mae"], epoch
                best_state = copy.deepcopy(model.state_dict())
        row["seconds"] = round(time.time() - t0, 1)
        history.append(row)
        print("    " + " | ".join(f"{k} = {v:.5g}" if isinstance(v, float) else f"{k} = {v}"
                                  for k, v in row.items()))

        if X_tune is not None and epoch - best_epoch >= patience:
            print(f"    early stopping: best epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, pd.DataFrame(history)


# =============================================================================
# Final model (used by run_baselines)
# =============================================================================

def run(con, cfg, split, model_dir):
    cfg = {**DEFAULTS, **cfg}
    meta = data.load_meta()
    n_codes = data.n_codes_per_column(meta)

    _, X_tr, rtt_tr, y_tr = load("train_enc", meta)
    _, X_tu, rtt_tu, y_tu = load("tune_enc", meta)
    row_ev, X_ev, rtt_ev, _ = load(f"{split}_enc", meta)

    # train + tune, one below the other
    X_tt   = np.concatenate([X_tr, X_tu])
    rtt_tt = np.concatenate([rtt_tr, rtt_tu])
    y_tt   = np.concatenate([y_tr, y_tu])
    mu, sigma = rtt_transform(rtt_tr)          # standardization estimated on train_enc, as in tuning
    x_tt, x_ev = apply_rtt(rtt_tt, mu, sigma), apply_rtt(rtt_ev, mu, sigma)

    # the untrained model predicts the median of the training data for every flow
    initial = float(torch.from_numpy(y_tt).median())

    t0 = time.time()
    model, _, _ = fit(X_tt, x_tt, y_tt, n_codes,
                      lam=cfg["lambda"], epochs=int(cfg["epochs"]), lr=cfg["lr"],
                      gamma=cfg["gamma"], batch=int(cfg["batch"]), seed=int(cfg["seed"]),
                      initial_intercept=initial)
    train_time = time.time() - t0

    pred = predict(model, X_ev, x_ev)
    table = evaluation.register_numpy_predictions(con, split, row_ev, pred, "pred_b2")

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_codes": n_codes, "mu": mu, "sigma": sigma,
                "id_columns": meta["id_columns"]}, model_dir / "b2.pt")

    info = {
        "model": "B2 linear (L1)",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "train_time_s": round(train_time, 1),
        "n_train_flows": int(len(y_tt)),
        **cfg,
    }
    return table, info, {}


# =============================================================================
# Hyperparameter search (used by tune)
# =============================================================================

def tune(cfg, tuning):
    """Grid over lambda: train on train_enc with early stopping on tune_enc.

    Returns one row per lambda with the best epoch and the tune MAE. Choose the
    largest lambda whose tune MAE is close to the best one, and copy it (with its
    best epoch as `epochs`) into the config.
    """
    cfg = {**DEFAULTS, **cfg}
    tuning = {**TUNING_DEFAULTS, **tuning}
    meta = data.load_meta()
    n_codes = data.n_codes_per_column(meta)

    _, X_tr, rtt_tr, y_tr = load("train_enc", meta)
    _, X_tu, rtt_tu, y_tu = load("tune_enc", meta)
    mu, sigma = rtt_transform(rtt_tr)
    x_tr, x_tu = apply_rtt(rtt_tr, mu, sigma), apply_rtt(rtt_tu, mu, sigma)
    initial = float(np.median(y_tr))
    print(f"  reference B0 on tune: MAE = {np.abs(initial - y_tu).mean():.4f}")

    rows = []
    for lam in tuning["lambda"]:
        print(f"  lambda = {lam:g}")
        _, best_epoch, history = fit(
            X_tr, x_tr, y_tr, n_codes, lam=lam, epochs=int(tuning["max_epochs"]), lr=cfg["lr"],
            gamma=cfg["gamma"], batch=int(cfg["batch"]), seed=int(cfg["seed"]),
            initial_intercept=initial, X_tune=X_tu, x_tune=x_tu, y_tune=y_tu,
            patience=int(tuning["patience"]))
        rows.append({"lambda": lam, "best_epoch": best_epoch,
                     "tune_mae": float(history["tune_mae"].min())})
    table = pd.DataFrame(rows)
    table["gap_to_best"] = table["tune_mae"] - table["tune_mae"].min()
    return table
