"""
v9: Add Chronos-Bolt time-series forecasts to the v8 ensemble.
- For each geohash, build day-48 demand series (96 tslots).
- Use Chronos-Bolt to forecast the next 96 values (day-49).
- Map Chronos output back to train day-49 rows (for OOF) and test rows.
- Reload v8 artifacts (oof_cb, oof_xgb, ...) and add Chronos as a 7th stack member.
- Refit Ridge meta-learner, write new submission.csv.
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

RNG = 42

print(">> loading data + v8 artifacts")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# parse timestamps
def to_tslot(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = to_tslot(train["timestamp"])
test["tslot"]  = to_tslot(test["timestamp"])

art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
print("   v8 artifact keys:", list(art.keys()))
y_raw    = art["y_raw"]
day49_idx = art["day49_idx"]
oof_cb   = art["oof_cb"];  pred_cb  = art["pred_cb"]
oof_xgb  = art["oof_xgb"]; pred_xgb = art["pred_xgb"]
oof_lgb  = art["oof_lgb"]; pred_lgb = art["pred_lgb"]
oof_hgb  = art["oof_hgb"]; pred_hgb = art["pred_hgb"]
oof_et   = art["oof_et"];  pred_et  = art["pred_et"]
oof_knn  = art["oof_knn"]; pred_knn = art["pred_knn"]
print(f"   loaded oof + pred arrays  shape={oof_cb.shape}  test_shape={pred_cb.shape}")


# --------------------------------------------------------------------------- #
# Build per-geohash day-48 demand series (96 tslots)                          #
# --------------------------------------------------------------------------- #
print(">> building per-geohash day-48 series")
all_gh = sorted(set(train["geohash"]).union(set(test["geohash"])))
gh_to_idx = {gh: i for i, gh in enumerate(all_gh)}
N_GH = len(all_gh)
print(f"   {N_GH} unique geohashes")

# fill array  shape (N_GH, 96) with NaN, then fill values from train day-48
d48_series = np.full((N_GH, 96), np.nan, dtype=np.float32)
d48 = train[train["day"] == 48]
gh_arr   = d48["geohash"].map(gh_to_idx).to_numpy()
tslot_arr = d48["tslot"].to_numpy()
dem_arr   = d48["demand"].to_numpy(dtype=np.float32)
d48_series[gh_arr, tslot_arr] = dem_arr
# also append day-49 train values (tslots 0..8) — these are valid context for forecast
d49_train = train[train["day"] == 49]
d49_series = np.full((N_GH, 9), np.nan, dtype=np.float32)
gh_arr49   = d49_train["geohash"].map(gh_to_idx).to_numpy()
tslot_arr49 = d49_train["tslot"].to_numpy()
dem_arr49   = d49_train["demand"].to_numpy(dtype=np.float32)
d49_series[gh_arr49, tslot_arr49] = dem_arr49

# For each gh, "full context" = day-48 (96) + day-49 known so far (9) = 105 max
# For day-49 forecasting at test tslots 9..95 we need to forecast 87 ahead from context of 105
n_nan_d48 = int(np.isnan(d48_series).sum())
n_nan_d49 = int(np.isnan(d49_series).sum())
print(f"   day48 series NaN count: {n_nan_d48}, day49 known NaN count: {n_nan_d49}")
# fill NaN with 0 (sparse demand → 0 is reasonable baseline)
d48_series = np.nan_to_num(d48_series, nan=0.0)
d49_series = np.nan_to_num(d49_series, nan=0.0)
context_full = np.concatenate([d48_series, d49_series], axis=1)  # shape (N_GH, 105)


# --------------------------------------------------------------------------- #
# Load Chronos and forecast                                                   #
# --------------------------------------------------------------------------- #
print(">> loading Chronos-Bolt-small on GPU")
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map="cuda",
    dtype=torch.float32,
)
print(f"   model on device: {next(pipe.inner_model.parameters()).device}")

# Forecast 87 steps ahead (test tslots 9..95) using full context of 105
# Chronos-Bolt internal limit is 64 steps per call — we'll do it in 2 calls
PRED_TEST = 87           # tslots 9..95 (inclusive)
PRED_OOF  = 9            # tslots 0..8 (for OOF, using d48-only context)


def chronos_forecast(context_np: np.ndarray, prediction_length: int, batch_size: int = 64) -> np.ndarray:
    """context_np: (N, L). Returns (N, prediction_length) with median forecast."""
    out = np.zeros((context_np.shape[0], prediction_length), dtype=np.float32)
    for i in range(0, context_np.shape[0], batch_size):
        ctx = torch.tensor(context_np[i:i+batch_size])
        with torch.no_grad():
            # Chronos-Bolt: limit_prediction_length=False allows > 64 by autoregressive extension
            preds = pipe.predict(ctx, prediction_length=prediction_length, limit_prediction_length=False)
        # preds: (B, num_quantiles, T) for ChronosBolt — take median (index 4 of 9 quantiles, or use predict_quantiles)
        # Actually predict() returns the raw forecast; for ChronosBolt it's (B, num_samples, T). Use median.
        med = preds.median(dim=1).values.cpu().numpy()
        out[i:i+batch_size] = med
        if (i // batch_size) % 4 == 0:
            print(f"     chronos batch {i//batch_size + 1}/{(context_np.shape[0]+batch_size-1)//batch_size}  shape={preds.shape}")
    return out


# OOF forecasts: predict day-49 tslots 0..8 from day-48 only context
print(">> Chronos OOF forecasts (day-49 tslots 0..8 from day-48 context)")
t0 = time.time()
oof_forecast_per_gh = chronos_forecast(d48_series, PRED_OOF)   # shape (N_GH, 9)
print(f"   OOF Chronos time: {time.time()-t0:.1f}s, shape: {oof_forecast_per_gh.shape}")

# Test forecasts: predict tslots 9..95 (87 steps) from day-48 + day-49-known context (105 timestamps)
print(">> Chronos TEST forecasts (day-49 tslots 9..95 from full context)")
t0 = time.time()
test_forecast_per_gh = chronos_forecast(context_full, PRED_TEST)   # shape (N_GH, 87)
print(f"   TEST Chronos time: {time.time()-t0:.1f}s, shape: {test_forecast_per_gh.shape}")


# --------------------------------------------------------------------------- #
# Map per-geohash forecasts back to OOF and test rows                          #
# --------------------------------------------------------------------------- #
print(">> mapping Chronos forecasts to row-level OOF and test predictions")
oof_chr = np.full(len(train), np.nan, dtype=np.float32)
# train day-49 rows are the OOF; for each, oof_chr = chronos_forecast at (gh, tslot)
d49_train_rows = train[train["day"] == 49].copy()
oof_chr[d49_train_rows.index] = oof_forecast_per_gh[
    d49_train_rows["geohash"].map(gh_to_idx).to_numpy(),
    d49_train_rows["tslot"].to_numpy(),
]
print(f"   OOF non-NaN: {int(np.isfinite(oof_chr).sum())}")

# Test rows are day-49 tslots 9..95 — map index = tslot - 9
pred_chr = np.zeros(len(test), dtype=np.float32)
test_gh_idx = test["geohash"].map(gh_to_idx).to_numpy()
test_tslot  = test["tslot"].to_numpy()
pred_chr = test_forecast_per_gh[test_gh_idx, test_tslot - 9]


# --------------------------------------------------------------------------- #
# Report individual Chronos OOF R² + refit Ridge stack                        #
# --------------------------------------------------------------------------- #
y49 = y_raw[day49_idx]
chronos_r2 = r2_score(y49, oof_chr[day49_idx])
print(f">> Chronos single-model OOF R² (day-49) = {chronos_r2:.5f}  (score = {max(0,100*chronos_r2):.3f})")

print(">> Ridge stacking with 7 base models")
oof_stack = np.column_stack([
    oof_cb[day49_idx], oof_xgb[day49_idx], oof_lgb[day49_idx], oof_hgb[day49_idx],
    oof_et[day49_idx], oof_knn[day49_idx], oof_chr[day49_idx],
])
test_stack = np.column_stack([pred_cb, pred_xgb, pred_lgb, pred_hgb, pred_et, pred_knn, pred_chr])

meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
oof_blend = meta.predict(oof_stack)
blend_r2 = r2_score(y49, oof_blend)
print(f">> Ridge meta weights = {dict(zip(['cb','xgb','lgb','hgb','et','knn','chr'], np.round(meta.coef_,4)))}")
print(f">> Ridge-stacked OOF R² (7 models) = {blend_r2:.5f}  (score = {max(0,100*blend_r2):.3f})")

pred = meta.predict(test_stack)
pred = np.clip(pred, 0.0, 1.0)

# write submission
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())

# save updated artifacts
np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    oof_cb=oof_cb, oof_xgb=oof_xgb, oof_lgb=oof_lgb, oof_hgb=oof_hgb,
    oof_et=oof_et, oof_knn=oof_knn, oof_chr=oof_chr,
    pred_cb=pred_cb, pred_xgb=pred_xgb, pred_lgb=pred_lgb, pred_hgb=pred_hgb,
    pred_et=pred_et, pred_knn=pred_knn, pred_chr=pred_chr,
    y_raw=y_raw, day49_idx=day49_idx,
    meta_coef=meta.coef_,
)
print(">> saved artifacts.npz (with Chronos)")
