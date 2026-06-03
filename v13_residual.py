"""
v13: Two-stage residual model. baseline = d48_same_ts. A CatBoost trained
ONLY on day-49 train rows learns the day-over-day delta = y - baseline.
Final pred = baseline + delta. Adds this as an 8th stack member and refits
the Ridge meta over 8 base models.
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
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

RNG = 42

print(">> loading data + v9 artifacts")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def parse_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return h.astype(int), m.astype(int), (h * 60 + m).astype(int), (h * 4 + m // 15).astype(int)

train["hour"], train["minute"], train["tmin"], train["tslot"] = parse_ts(train["timestamp"])
test["hour"],  test["minute"],  test["tmin"],  test["tslot"]  = parse_ts(test["timestamp"])

# d48 lookup
d48 = train[train["day"] == 48][["geohash", "tslot", "demand"]]
d48_map = d48.set_index(["geohash", "tslot"])["demand"]

# per-gh d48 mean as fallback when (gh, tslot) missing from d48
gh_d48_mean = d48.groupby("geohash")["demand"].mean()
overall_d48_mean = float(d48["demand"].mean())

def fill_baseline(df):
    keys = list(zip(df["geohash"], df["tslot"]))
    base = pd.Series(keys, index=df.index).map(d48_map)
    base = base.fillna(df["geohash"].map(gh_d48_mean)).fillna(overall_d48_mean)
    return base.astype(np.float32).to_numpy()

train["baseline"] = fill_baseline(train)
test["baseline"]  = fill_baseline(test)
print(f"   train baseline NaN: {int(np.isnan(train['baseline']).sum())}")
print(f"   test baseline NaN:  {int(np.isnan(test['baseline']).sum())}")

# delta = y - baseline (only meaningful for day-49 train rows where day != baseline's day)
train["delta"] = train["demand"].astype(float) - train["baseline"]

# day-49 train rows = our training set for the delta model
d49 = train[train["day"] == 49].reset_index(drop=False).rename(columns={"index": "_orig_idx"})
print(f"   day-49 train rows: {len(d49)}")
print(f"   delta stats — mean: {d49['delta'].mean():.4f}, std: {d49['delta'].std():.4f}")


# --------------------------------------------------------------------------- #
# Build feature columns (subset that matters for cross-day delta)             #
# --------------------------------------------------------------------------- #
# Decode geohash → lat/lon (simple, used for the delta model)
GH = "0123456789bcdefghjkmnpqrstuvwxyz"
GH_MAP = {c: i for i, c in enumerate(GH)}
def decode(g):
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for ch in g:
        bits = GH_MAP[ch]
        for m in (16, 8, 4, 2, 1):
            b = (bits & m) > 0
            if even:
                mid = (lon_lo + lon_hi) / 2
                if b: lon_lo = mid
                else: lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if b: lat_lo = mid
                else: lat_hi = mid
            even = not even
    return (lat_lo + lat_hi)/2, (lon_lo + lon_hi)/2

all_gh = pd.unique(pd.concat([train["geohash"], test["geohash"]]))
gh_lat = {g: decode(g)[0] for g in all_gh}
gh_lon = {g: decode(g)[1] for g in all_gh}
train["lat"] = train["geohash"].map(gh_lat)
train["lon"] = train["geohash"].map(gh_lon)
test["lat"]  = test["geohash"].map(gh_lat)
test["lon"]  = test["geohash"].map(gh_lon)

for k in (3, 4, 5):
    train[f"gh{k}"] = train["geohash"].str.slice(0, k)
    test[f"gh{k}"]  = test["geohash"].str.slice(0, k)

# baseline-related features for the delta model
train["hour_sin"] = np.sin(2*np.pi*train["hour"]/24); train["hour_cos"] = np.cos(2*np.pi*train["hour"]/24)
test["hour_sin"]  = np.sin(2*np.pi*test["hour"]/24);  test["hour_cos"]  = np.cos(2*np.pi*test["hour"]/24)
train["tmin_sin"] = np.sin(2*np.pi*train["tmin"]/1440); train["tmin_cos"] = np.cos(2*np.pi*train["tmin"]/1440)
test["tmin_sin"]  = np.sin(2*np.pi*test["tmin"]/1440);  test["tmin_cos"]  = np.cos(2*np.pi*test["tmin"]/1440)

# d48 same-slot variability per gh (high std → unstable area → expect bigger delta)
gh_d48_stats = d48.groupby("geohash")["demand"].agg(["mean", "std", "max", "min"])
gh_d48_stats.columns = ["gh_d48_mean","gh_d48_std","gh_d48_max","gh_d48_min"]
train = train.merge(gh_d48_stats, left_on="geohash", right_index=True, how="left")
test  = test.merge(gh_d48_stats,  left_on="geohash", right_index=True, how="left")

CAT_FEATURES = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM_FEATURES = ["hour", "minute", "tmin", "tslot", "hour_sin", "hour_cos", "tmin_sin", "tmin_cos",
                "lat", "lon", "NumberofLanes", "Temperature",
                "baseline", "gh_d48_mean", "gh_d48_std", "gh_d48_max", "gh_d48_min"]
FEATURES = CAT_FEATURES + NUM_FEATURES
for c in CAT_FEATURES:
    train[c] = train[c].astype("string").fillna("NA")
    test[c]  = test[c].astype("string").fillna("NA")


# --------------------------------------------------------------------------- #
# Train delta CatBoost on day-49 train rows with 5-fold CV                    #
# --------------------------------------------------------------------------- #
print(">> training 5-fold delta CatBoost on day-49 only (GPU)")
day49_idx = np.where(train["day"].values == 49)[0]
X_train = train[FEATURES].copy()
X_test = test[FEATURES].copy()
y_delta = train["delta"].astype(float).values

cb_params = dict(
    iterations=3000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
    loss_function="RMSE", eval_metric="RMSE",
    random_seed=RNG, od_type="Iter", od_wait=200,
    verbose=300, task_type="GPU", devices="0",
)
cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]

oof_delta = np.full(len(train), np.nan)
pred_delta = np.zeros(len(test))
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
    tr_idx = day49_idx[tr_d49]
    va_idx = day49_idx[va_d49]
    model = CatBoostRegressor(**cb_params)
    model.fit(
        X_train.iloc[tr_idx], y_delta[tr_idx],
        eval_set=(X_train.iloc[va_idx], y_delta[va_idx]),
        cat_features=cat_idx, use_best_model=True,
    )
    oof_delta[va_idx] = model.predict(X_train.iloc[va_idx])
    pred_delta += model.predict(X_test) / kf.n_splits
    fold_r2 = r2_score(y_delta[va_idx], oof_delta[va_idx])
    print(f"  delta fold {fold}: best_iter={model.get_best_iteration()}  R²(delta)={fold_r2:.5f}")

print(f">> delta-target OOF R² on day-49 (delta scale, not raw): {r2_score(y_delta[day49_idx], oof_delta[day49_idx]):.5f}")

# Convert to demand-scale predictions: baseline + delta
oof_residual_model = train["baseline"].to_numpy() + oof_delta
pred_residual_model = test["baseline"].to_numpy() + pred_delta


# --------------------------------------------------------------------------- #
# Compare residual model to v9 base models, refit Ridge over 8 models         #
# --------------------------------------------------------------------------- #
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw, day49_idx_art = art["y_raw"], art["day49_idx"]
assert np.array_equal(day49_idx, day49_idx_art), "day-49 index mismatch!"
y49 = y_raw[day49_idx]

resid_r2 = r2_score(y49, oof_residual_model[day49_idx])
print(f">> Two-stage residual model OOF R² (day-49, demand scale) = {resid_r2:.5f}  (score = {max(0,100*resid_r2):.3f})")

labels = ["cb", "xgb", "lgb", "hgb", "et", "knn", "chrS_50", "residual"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"], oof_residual_model]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"], pred_residual_model]

single_r2 = {lbl: r2_score(y49, c[day49_idx]) for lbl, c in zip(labels, oof_cols)}
print(">> single-model R²:", {k: round(v, 5) for k, v in single_r2.items()})

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)

meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
blend_r2 = r2_score(y49, meta.predict(oof_stack))
print(">> Ridge meta weights =", dict(zip(labels, np.round(meta.coef_, 4))))
print(f">> Ridge-stacked OOF R² (8 models) = {blend_r2:.5f}  (score = {max(0,100*blend_r2):.3f})")

pred = np.clip(meta.predict(test_stack), 0.0, 1.0)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())

np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    **{k: art[k] for k in art.files},
    oof_residual=oof_residual_model, pred_residual=pred_residual_model,
    meta_coef_v13=meta.coef_,
)
print(">> saved artifacts.npz")
