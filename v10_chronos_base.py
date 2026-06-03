"""
v10: Upgrade Chronos to chronos-bolt-base + multi-quantile features.
Adds 3 separate stack members: q20, q50 (median), q80 forecasts.
Refits Ridge over CB, XGB, LGB, HGB, ET, KNN, ChrQ20, ChrQ50, ChrQ80.
"""
import sys, io, os, warnings, time
warnings.filterwarnings("ignore")

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            try: st.write(s); st.flush()
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_logfile = open(os.path.join(DATA_DIR, "run.log"), "w", encoding="utf-8")
sys.stdout = _Tee(io.TextIOWrapper(sys.__stdout__.buffer, encoding="utf-8", line_buffering=True), _logfile)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from chronos import BaseChronosPipeline

print(">> loading data + v8 artifacts (base ensemble)")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def to_tslot(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = to_tslot(train["timestamp"])
test["tslot"]  = to_tslot(test["timestamp"])

art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw    = art["y_raw"]
day49_idx = art["day49_idx"]
oof_cb   = art["oof_cb"];  pred_cb  = art["pred_cb"]
oof_xgb  = art["oof_xgb"]; pred_xgb = art["pred_xgb"]
oof_lgb  = art["oof_lgb"]; pred_lgb = art["pred_lgb"]
oof_hgb  = art["oof_hgb"]; pred_hgb = art["pred_hgb"]
oof_et   = art["oof_et"];  pred_et  = art["pred_et"]
oof_knn  = art["oof_knn"]; pred_knn = art["pred_knn"]
print(f"   loaded 6 base models  shape={oof_cb.shape}")


print(">> building per-geohash day-48 + day-49 known series")
all_gh = sorted(set(train["geohash"]).union(set(test["geohash"])))
gh_to_idx = {gh: i for i, gh in enumerate(all_gh)}
N_GH = len(all_gh)

d48_series = np.zeros((N_GH, 96), dtype=np.float32)
d48 = train[train["day"] == 48]
d48_series[d48["geohash"].map(gh_to_idx).to_numpy(), d48["tslot"].to_numpy()] = d48["demand"].to_numpy(dtype=np.float32)

d49_series = np.zeros((N_GH, 9), dtype=np.float32)
d49 = train[train["day"] == 49]
d49_series[d49["geohash"].map(gh_to_idx).to_numpy(), d49["tslot"].to_numpy()] = d49["demand"].to_numpy(dtype=np.float32)

context_full = np.concatenate([d48_series, d49_series], axis=1)  # (N_GH, 105) — for test forecast


print(">> loading Chronos-Bolt-base on GPU")
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda",
    dtype=torch.float32,
)
print(f"   device: {next(pipe.inner_model.parameters()).device}")


def chronos_quantile_forecast(context_np: np.ndarray, prediction_length: int, batch_size: int = 32, quantiles=(0.2, 0.5, 0.8)) -> np.ndarray:
    """Returns array shape (N, len(quantiles), prediction_length) of forecasts at requested quantiles."""
    n = context_np.shape[0]
    out = np.zeros((n, len(quantiles), prediction_length), dtype=np.float32)
    for i in range(0, n, batch_size):
        ctx = torch.tensor(context_np[i:i+batch_size])
        with torch.no_grad():
            q_pred, mean = pipe.predict_quantiles(
                ctx,
                prediction_length=prediction_length,
                quantile_levels=list(quantiles),
            )
        # q_pred shape: (B, T, len(quantiles))  -> transpose to (B, len(quantiles), T)
        q_pred = q_pred.permute(0, 2, 1).cpu().numpy()
        out[i:i+batch_size] = q_pred
        if (i // batch_size) % 4 == 0:
            print(f"     chronos batch {i//batch_size + 1}/{(n+batch_size-1)//batch_size}  q={q_pred.shape}")
    return out


# OOF: predict day-49 tslots 0..8 from day-48 only context
print(">> Chronos OOF forecasts (9 steps)")
t0 = time.time()
oof_q = chronos_quantile_forecast(d48_series, 9)        # (N_GH, 3, 9)
print(f"   OOF time: {time.time()-t0:.1f}s")

# TEST: predict day-49 tslots 9..95 (87 steps) from full context
print(">> Chronos TEST forecasts (87 steps)")
t0 = time.time()
test_q = chronos_quantile_forecast(context_full, 87)    # (N_GH, 3, 87)
print(f"   TEST time: {time.time()-t0:.1f}s")


# --------------------------------------------------------------------------- #
# Build per-row OOF & test arrays for the 3 quantile bands                    #
# --------------------------------------------------------------------------- #
print(">> mapping per-row OOF and test predictions for 3 quantiles")
oof_chr_q20 = np.full(len(train), np.nan, dtype=np.float32)
oof_chr_q50 = np.full(len(train), np.nan, dtype=np.float32)
oof_chr_q80 = np.full(len(train), np.nan, dtype=np.float32)
d49_rows = train[train["day"] == 49]
d49_gh   = d49_rows["geohash"].map(gh_to_idx).to_numpy()
d49_ts   = d49_rows["tslot"].to_numpy()
oof_chr_q20[d49_rows.index] = oof_q[d49_gh, 0, d49_ts]
oof_chr_q50[d49_rows.index] = oof_q[d49_gh, 1, d49_ts]
oof_chr_q80[d49_rows.index] = oof_q[d49_gh, 2, d49_ts]

test_gh = test["geohash"].map(gh_to_idx).to_numpy()
test_ts = test["tslot"].to_numpy()
pred_chr_q20 = test_q[test_gh, 0, test_ts - 9]
pred_chr_q50 = test_q[test_gh, 1, test_ts - 9]
pred_chr_q80 = test_q[test_gh, 2, test_ts - 9]


# --------------------------------------------------------------------------- #
# Report single-model R² and refit Ridge stack                                #
# --------------------------------------------------------------------------- #
y49 = y_raw[day49_idx]
single_r2 = {
    "cb":  r2_score(y49, oof_cb[day49_idx]),
    "xgb": r2_score(y49, oof_xgb[day49_idx]),
    "lgb": r2_score(y49, oof_lgb[day49_idx]),
    "hgb": r2_score(y49, oof_hgb[day49_idx]),
    "et":  r2_score(y49, oof_et[day49_idx]),
    "knn": r2_score(y49, oof_knn[day49_idx]),
    "chrQ20": r2_score(y49, oof_chr_q20[day49_idx]),
    "chrQ50": r2_score(y49, oof_chr_q50[day49_idx]),
    "chrQ80": r2_score(y49, oof_chr_q80[day49_idx]),
}
print(">> single-model R²:", {k: round(v,5) for k,v in single_r2.items()})


print(">> Ridge stacking with 9 base models")
oof_stack = np.column_stack([
    oof_cb[day49_idx], oof_xgb[day49_idx], oof_lgb[day49_idx], oof_hgb[day49_idx],
    oof_et[day49_idx], oof_knn[day49_idx],
    oof_chr_q20[day49_idx], oof_chr_q50[day49_idx], oof_chr_q80[day49_idx],
])
test_stack = np.column_stack([
    pred_cb, pred_xgb, pred_lgb, pred_hgb, pred_et, pred_knn,
    pred_chr_q20, pred_chr_q50, pred_chr_q80,
])

meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
labels = ['cb','xgb','lgb','hgb','et','knn','chrQ20','chrQ50','chrQ80']
print(">> Ridge meta weights =", dict(zip(labels, np.round(meta.coef_, 4))))
blend_r2 = r2_score(y49, meta.predict(oof_stack))
print(f">> Ridge-stacked OOF R² (9 models) = {blend_r2:.5f}  (score = {max(0,100*blend_r2):.3f})")

pred = meta.predict(test_stack)
pred = np.clip(pred, 0.0, 1.0)

sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())

np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    oof_cb=oof_cb, oof_xgb=oof_xgb, oof_lgb=oof_lgb, oof_hgb=oof_hgb,
    oof_et=oof_et, oof_knn=oof_knn,
    oof_chr_q20=oof_chr_q20, oof_chr_q50=oof_chr_q50, oof_chr_q80=oof_chr_q80,
    pred_cb=pred_cb, pred_xgb=pred_xgb, pred_lgb=pred_lgb, pred_hgb=pred_hgb,
    pred_et=pred_et, pred_knn=pred_knn,
    pred_chr_q20=pred_chr_q20, pred_chr_q50=pred_chr_q50, pred_chr_q80=pred_chr_q80,
    y_raw=y_raw, day49_idx=day49_idx,
    meta_coef=meta.coef_,
)
print(">> saved artifacts.npz")
