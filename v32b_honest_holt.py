"""
Honest evaluation of v32's Holt residual technique.
Leave-one-tslot-out: for each held-out morning tslot, fit Holt on the other 8 tslots' residuals
and forecast the held-out one. Reports true generalization R².
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
def parse_tslot(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = parse_tslot(train["timestamp"])

all_gh = sorted(set(train["geohash"]).union(set(test["geohash"])))
gh_to_idx = {g: i for i, g in enumerate(all_gh)}
N_GH = len(all_gh)
M48 = np.full((N_GH, 96), np.nan, dtype=np.float32)
d48 = train[train["day"] == 48]
M48[d48["geohash"].map(gh_to_idx).to_numpy(), d48["tslot"].to_numpy()] = d48["demand"].to_numpy(dtype=np.float32)
M49 = np.full((N_GH, 96), np.nan, dtype=np.float32)
d49 = train[train["day"] == 49]
M49[d49["geohash"].map(gh_to_idx).to_numpy(), d49["tslot"].to_numpy()] = d49["demand"].to_numpy(dtype=np.float32)
gh_mean48 = np.nanmean(M48, axis=1)
gh_mean48 = np.where(np.isnan(gh_mean48), np.nanmean(gh_mean48), gh_mean48)
M48_filled = M48.copy()
for i in range(N_GH):
    M48_filled[i, np.isnan(M48_filled[i])] = gh_mean48[i]


# leave-one-tslot-out on morning residuals
honest_preds_holt = []
honest_truths = []
honest_preds_naive = []

print(">> leave-one-tslot-out evaluation of Holt residual forecasts on morning slots 0..8")
for gh_i in range(N_GH):
    R_full = M49[gh_i, :9] - M48_filled[gh_i, :9]
    if np.isnan(R_full).any():
        continue
    for held in range(9):
        other = [t for t in range(9) if t != held]
        R_train = R_full[other]
        steps_ahead = held - max(other) if held > max(other) else 1
        try:
            if np.var(R_train) < 1e-10:
                pred = float(np.mean(R_train))
            else:
                # Fit on the prefix [t < held] for honest forecasting
                prefix = R_full[[t for t in other if t < held]]
                if len(prefix) >= 3:
                    model = ExponentialSmoothing(prefix, trend="add", damped_trend=True,
                                                  seasonal=None, initialization_method="estimated")
                    fit = model.fit(optimized=True, remove_bias=False)
                    pred = float(fit.forecast(steps=held - max(t for t in other if t < held))[-1])
                else:
                    pred = float(np.mean(R_train))
        except Exception:
            pred = float(np.mean(R_train))

        # naive baseline: residual stays at last known
        prefix = R_full[[t for t in other if t < held]]
        naive = float(prefix[-1]) if len(prefix) > 0 else 0.0

        honest_preds_holt.append(M48_filled[gh_i, held] + pred)
        honest_preds_naive.append(M48_filled[gh_i, held] + naive)
        honest_truths.append(M49[gh_i, held])

pred_holt = np.clip(np.array(honest_preds_holt), 0, 1)
pred_naive = np.clip(np.array(honest_preds_naive), 0, 1)
truth = np.array(honest_truths)
print(f"   evaluated on {len(truth)} (gh, tslot) cells")
print(f"   Holt honest R² = {r2_score(truth, pred_holt):.5f}")
print(f"   naive (last-residual) R² = {r2_score(truth, pred_naive):.5f}")
print(f"   d48-only R² = {r2_score(truth, np.clip([M48_filled[i//9, i%9] for i in range(len(truth))], 0, 1)):.5f}")
