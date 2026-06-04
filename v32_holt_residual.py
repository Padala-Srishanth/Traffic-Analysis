"""
v32 (Method 2): Per-geohash damped Holt's linear trend on day-49 residuals.

For each geohash:
  R(t) = y49(t) - y48_same_slot(t)  for t in 0..8
  Fit damped Holt's method on R(0..8)
  Forecast R_hat(9..95)
  Final: y49_pred(gh, t) = y48_same_slot(gh, t) + R_hat(gh, t)

Adds as a stack member, refits Ridge, applies smoothing.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing


print(">> loading data + day-48/49 matrices")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
def parse_tslot(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = parse_tslot(train["timestamp"])
test["tslot"]  = parse_tslot(test["timestamp"])

all_gh = sorted(set(train["geohash"]).union(set(test["geohash"])))
gh_to_idx = {g: i for i, g in enumerate(all_gh)}
N_GH = len(all_gh)

d48 = train[train["day"] == 48]
d49 = train[train["day"] == 49]

M48 = np.full((N_GH, 96), np.nan, dtype=np.float32)
M48[d48["geohash"].map(gh_to_idx).to_numpy(), d48["tslot"].to_numpy()] = d48["demand"].to_numpy(dtype=np.float32)
M49 = np.full((N_GH, 96), np.nan, dtype=np.float32)
M49[d49["geohash"].map(gh_to_idx).to_numpy(), d49["tslot"].to_numpy()] = d49["demand"].to_numpy(dtype=np.float32)

# fill M48 NaN with gh mean
gh_mean48 = np.nanmean(M48, axis=1)
gh_mean48 = np.where(np.isnan(gh_mean48), np.nanmean(gh_mean48), gh_mean48)
M48_filled = M48.copy()
for i in range(N_GH):
    M48_filled[i, np.isnan(M48_filled[i])] = gh_mean48[i]


# --------------------------------------------------------------------------- #
# Per-geohash: compute residuals R(0..8) = y49 - y48, fit Holt, forecast      #
# --------------------------------------------------------------------------- #
print(">> fitting damped Holt's per geohash on residuals R(0..8) -> forecast R(9..95)")
R_forecast = np.zeros((N_GH, 96), dtype=np.float32)  # full row, with [0..8] = known, [9..95] = forecast
fit_count = 0
fail_count = 0
t0 = time.time()

for gh_i in range(N_GH):
    R_known = M49[gh_i, :9] - M48_filled[gh_i, :9]
    # use M48 mean instead of NaN if M49 not available at a slot
    R_known_clean = np.where(np.isnan(R_known), 0.0, R_known)
    R_forecast[gh_i, :9] = R_known_clean

    if np.isfinite(R_known_clean).all() and np.var(R_known_clean) > 1e-10:
        try:
            # Damped Holt's method: trend + dampening, no seasonality (we only have 9 points)
            model = ExponentialSmoothing(R_known_clean, trend="add", damped_trend=True,
                                         seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True, remove_bias=False)
            forecast = fit.forecast(steps=87)
            # damp very large forecasts: cap at +-3 * morning std (or fallback bounds)
            cap = max(0.1, 3 * np.std(R_known_clean))
            forecast = np.clip(forecast, -cap, cap)
            R_forecast[gh_i, 9:] = forecast.astype(np.float32)
            fit_count += 1
        except Exception:
            # fallback: constant equal to mean residual
            R_forecast[gh_i, 9:] = float(np.mean(R_known_clean))
            fail_count += 1
    else:
        # no variation -> use mean residual (likely 0)
        R_forecast[gh_i, 9:] = float(np.mean(R_known_clean)) if np.isfinite(R_known_clean).all() else 0.0
        fail_count += 1

print(f"   Holt fits: succeeded {fit_count}, fallback {fail_count}  time={time.time()-t0:.1f}s")

# Build the forecasted day-49 demand matrix
M49_holt = M48_filled + R_forecast
M49_holt = np.clip(M49_holt, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Map matrix back to per-row predictions                                      #
# --------------------------------------------------------------------------- #
day49_idx = np.where(train["day"].values == 49)[0]
oof_holt = np.full(len(train), np.nan, dtype=np.float32)
oof_holt[day49_idx] = M49_holt[train.iloc[day49_idx]["geohash"].map(gh_to_idx).to_numpy(),
                                train.iloc[day49_idx]["tslot"].to_numpy()]
pred_holt = M49_holt[test["geohash"].map(gh_to_idx).to_numpy(), test["tslot"].to_numpy()]

art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw = art["y_raw"]
y49 = y_raw[day49_idx]
holt_r2 = r2_score(y49, oof_holt[day49_idx])
print(f">> Holt residual model OOF R² (day-49 morning) = {holt_r2:.5f}")


# --------------------------------------------------------------------------- #
# Refit Ridge stack with Holt as 11th model                                   #
# --------------------------------------------------------------------------- #
labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn","holt"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"], oof_holt]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"], pred_holt]

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
print(">> Ridge weights:", dict(zip(labels, np.round(meta.coef_, 4))))
v32_oof  = meta.predict(oof_stack)
v32_test = np.clip(meta.predict(test_stack), 0, 1)
print(f">> v32 raw OOF R² = {r2_score(y49, v32_oof):.5f}")


def smooth(values, gh, ts, w, b):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    df["out"] = b * df["sm"] + (1 - b) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()

oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)
best_r2 = r2_score(y49, v32_oof); best_test = v32_test; best_desc = "raw"
for w in (2, 3, 4):
    for b in (0.5, 0.75, 1.0):
        oof_s = smooth(v32_oof, oof_gh, oof_ts, w, b)
        test_s = np.clip(smooth(v32_test, test["geohash"], test["tslot"], w, b), 0, 1)
        r2 = r2_score(y49, oof_s)
        print(f"   w={w} b={b}: OOF R²={r2:.5f}")
        if r2 > best_r2:
            best_r2 = r2; best_test = test_s; best_desc = f"w{w}_b{b}"

print(f">> winner: {best_desc}  OOF R²={best_r2:.5f}")
sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())
