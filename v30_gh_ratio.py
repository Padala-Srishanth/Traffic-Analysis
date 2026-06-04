"""
v30: Geohash-level ratio model.
Each geohash gets one row in a small training set:
  Inputs: gh-level static features (mean, std, percentiles, entropy, prefix, lat/lon, SVD emb, etc.)
  Target: log(mean(d49 morning) / mean(d48 morning) + epsilon)
Train CatBoost on 1259 rows with 5-fold CV.
At test time, lookup the predicted ratio per geohash and apply:
  pred = d48_same_ts * exp(predicted_log_ratio)
Adds the result as a stack member, refits Ridge, smooths.
"""
import sys, io, os, warnings
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
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostRegressor

RNG = 42

print(">> loading data")
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
print(f"   {N_GH} unique geohashes")

# Day-48 / day-49 matrices
d48 = train[train["day"] == 48]
d49 = train[train["day"] == 49]
M48 = np.full((N_GH, 96), np.nan, dtype=np.float32)
M48[d48["geohash"].map(gh_to_idx).to_numpy(), d48["tslot"].to_numpy()] = d48["demand"].to_numpy(dtype=np.float32)
M49 = np.full((N_GH, 96), np.nan, dtype=np.float32)
M49[d49["geohash"].map(gh_to_idx).to_numpy(), d49["tslot"].to_numpy()] = d49["demand"].to_numpy(dtype=np.float32)


# ---------- Build per-geohash feature table ----------
print(">> building per-geohash feature table")

# stats from day-48 full curve
def safe_mean(x): return np.nanmean(x)
def safe_std(x):  return np.nanstd(x)
def entropy(x):
    x = np.where(np.isnan(x), 0, x)
    s = x.sum()
    if s <= 0: return 0.0
    p = x / s
    return -np.nansum(np.where(p>0, p * np.log(p + 1e-9), 0))

gh_mean48     = np.nanmean(M48, axis=1)
gh_std48      = np.nanstd(M48, axis=1)
gh_median48   = np.nanmedian(M48, axis=1)
gh_max48      = np.nanmax(M48, axis=1)
gh_min48      = np.nanmin(M48, axis=1)
gh_q90_48     = np.nanpercentile(M48, 90, axis=1)
gh_q10_48     = np.nanpercentile(M48, 10, axis=1)
gh_count48    = (~np.isnan(M48)).sum(axis=1)
gh_morning48  = np.nanmean(M48[:, :9], axis=1)
gh_morning49  = np.nanmean(M49[:, :9], axis=1)
M48_for_peak = np.where(np.all(np.isnan(M48), axis=1, keepdims=True), 0.0, np.nan_to_num(M48, nan=-np.inf))
gh_peak_hour48 = np.argmax(M48_for_peak, axis=1)  # tslot of peak
gh_entropy48  = np.array([entropy(M48[i]) for i in range(N_GH)])

# fill NaN
def fill_with_overall(arr):
    overall = float(np.nanmean(arr))
    return np.where(np.isnan(arr), overall, arr).astype(np.float32)

gh_mean48     = fill_with_overall(gh_mean48)
gh_std48      = fill_with_overall(gh_std48)
gh_median48   = fill_with_overall(gh_median48)
gh_max48      = fill_with_overall(gh_max48)
gh_min48      = fill_with_overall(gh_min48)
gh_q90_48     = fill_with_overall(gh_q90_48)
gh_q10_48     = fill_with_overall(gh_q10_48)
gh_morning48  = fill_with_overall(gh_morning48)

# decode geohash for lat/lon
GH = "0123456789bcdefghjkmnpqrstuvwxyz"
GH_MAP = {c: i for i, c in enumerate(GH)}
def decode(g):
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for ch in g:
        bits = GH_MAP[ch]
        for m in (16, 8, 4, 2, 1):
            b = (bits & m) > 0
            if even:
                mid = (lon_lo + lon_hi)/2
                if b: lon_lo = mid
                else: lon_hi = mid
            else:
                mid = (lat_lo + lat_hi)/2
                if b: lat_lo = mid
                else: lat_hi = mid
            even = not even
    return (lat_lo + lat_hi)/2, (lon_lo + lon_hi)/2

gh_lat = np.array([decode(g)[0] for g in all_gh])
gh_lon = np.array([decode(g)[1] for g in all_gh])

# SVD embedding (16 dims) of day-48 normalised profile
M48_filled = M48.copy()
for i in range(N_GH):
    M48_filled[i, np.isnan(M48_filled[i])] = gh_mean48[i]
row_mean = M48_filled.mean(axis=1).reshape(-1, 1)
row_mean[row_mean < 1e-6] = 1e-6
M48_norm = M48_filled / row_mean
svd = TruncatedSVD(n_components=16, random_state=RNG)
gh_emb = svd.fit_transform(M48_norm)
print(f"   SVD explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.4f}")

# Per-geohash dominant categorical features (most frequent value)
def mode_or_na(s):
    s = s.dropna()
    if len(s) == 0: return "NA"
    return s.mode().iloc[0] if len(s.mode()) else "NA"

gh_static_cat = train.groupby("geohash").agg({
    "RoadType": mode_or_na,
    "LargeVehicles": mode_or_na,
    "Landmarks": mode_or_na,
    "Weather": mode_or_na,
    "NumberofLanes": "median",
    "Temperature": "mean",
})
gh_static_cat = gh_static_cat.reindex(all_gh)
gh_static_cat["NumberofLanes"] = gh_static_cat["NumberofLanes"].fillna(2).astype(int)
gh_static_cat["Temperature"]   = gh_static_cat["Temperature"].fillna(gh_static_cat["Temperature"].mean()).astype(np.float32)
for c in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
    gh_static_cat[c] = gh_static_cat[c].astype("string").fillna("NA")

# Build the per-geohash feature dataframe
df_gh = pd.DataFrame({
    "geohash": all_gh,
    "gh_mean48": gh_mean48, "gh_std48": gh_std48, "gh_median48": gh_median48,
    "gh_max48": gh_max48, "gh_min48": gh_min48,
    "gh_q90_48": gh_q90_48, "gh_q10_48": gh_q10_48,
    "gh_count48": gh_count48,
    "gh_morning48": gh_morning48,
    "gh_peak_hour48": gh_peak_hour48.astype(int),
    "gh_entropy48": gh_entropy48,
    "lat": gh_lat, "lon": gh_lon,
}).set_index("geohash")
for k in range(16):
    df_gh[f"emb_{k}"] = gh_emb[:, k]
df_gh = df_gh.join(gh_static_cat)
for k in (3, 4, 5):
    df_gh[f"gh{k}"] = df_gh.index.str.slice(0, k)
print(f"   per-geohash table shape: {df_gh.shape}")


# ---------- Target: log-ratio of day-49 morning / day-48 morning ----------
EPS = 1e-3
ratio = gh_morning49 / (gh_morning48 + EPS)
log_ratio = np.log(ratio + EPS)
has_target = ~np.isnan(gh_morning49) & (gh_morning48 > EPS)
n_target = int(has_target.sum())
print(f"   geohashes with target ratio: {n_target} / {N_GH}")


# ---------- Train CatBoost on per-geohash table ----------
print(">> training CatBoost on per-geohash log-ratio (5-fold CV)")
CAT_GH = ["RoadType", "LargeVehicles", "Landmarks", "Weather", "gh3", "gh4", "gh5"]
NUM_GH = ["gh_mean48", "gh_std48", "gh_median48", "gh_max48", "gh_min48",
          "gh_q90_48", "gh_q10_48", "gh_count48", "gh_morning48",
          "gh_peak_hour48", "gh_entropy48", "lat", "lon",
          "NumberofLanes", "Temperature"] + [f"emb_{k}" for k in range(16)]
FEAT_GH = CAT_GH + NUM_GH
X_gh = df_gh[FEAT_GH].copy()
for c in CAT_GH:
    X_gh[c] = X_gh[c].astype("string").fillna("NA")

cb_params = dict(
    iterations=2000, learning_rate=0.03, depth=5, l2_leaf_reg=5.0,
    loss_function="RMSE", eval_metric="RMSE",
    random_seed=RNG, od_type="Iter", od_wait=200,
    verbose=300, task_type="GPU", devices="0",
)
cat_idx_gh = [FEAT_GH.index(c) for c in CAT_GH]

oof_logratio = np.full(N_GH, np.nan, dtype=np.float32)
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
target_idx = np.where(has_target)[0]
for fold, (tr_pos, va_pos) in enumerate(kf.split(target_idx), 1):
    tr_idx = target_idx[tr_pos]; va_idx = target_idx[va_pos]
    m = CatBoostRegressor(**cb_params)
    m.fit(X_gh.iloc[tr_idx], log_ratio[tr_idx],
          eval_set=(X_gh.iloc[va_idx], log_ratio[va_idx]),
          cat_features=cat_idx_gh, use_best_model=True)
    oof_logratio[va_idx] = m.predict(X_gh.iloc[va_idx])
    fold_r2 = r2_score(log_ratio[va_idx], oof_logratio[va_idx])
    print(f"   gh-fold {fold}: best_iter={m.get_best_iteration()}  R2(logratio)={fold_r2:.5f}")

# Retrain on ALL labelled gh to score the unlabelled ones too
final_m = CatBoostRegressor(**cb_params)
final_m.fit(X_gh.iloc[target_idx], log_ratio[target_idx], cat_features=cat_idx_gh)
# Predict for ALL geohashes (those with no target use full-fit prediction)
all_pred_logratio = oof_logratio.copy()
nan_mask = np.isnan(all_pred_logratio)
all_pred_logratio[nan_mask] = final_m.predict(X_gh.iloc[nan_mask])
predicted_ratio = np.exp(all_pred_logratio) - EPS  # invert log(x + eps)
predicted_ratio = np.clip(predicted_ratio, 0.1, 10.0)
print(f">> predicted ratio: mean={predicted_ratio.mean():.3f}, std={predicted_ratio.std():.3f}")


# ---------- Build row-level predictions: d48_same_ts * predicted_ratio ----------
d48_lookup = d48.set_index(["geohash", "tslot"])["demand"]
def fill_baseline(df):
    keys = list(zip(df["geohash"], df["tslot"]))
    return pd.Series(keys, index=df.index).map(d48_lookup)\
            .fillna(df["geohash"].map(d48.groupby("geohash")["demand"].mean()))\
            .fillna(float(d48["demand"].mean())).astype(np.float32).to_numpy()
train["d48_same_ts"] = fill_baseline(train)
test["d48_same_ts"]  = fill_baseline(test)

train_gh_idx = train["geohash"].map(gh_to_idx).to_numpy()
test_gh_idx  = test["geohash"].map(gh_to_idx).to_numpy()

oof_ratio_pred  = train["d48_same_ts"].to_numpy() * predicted_ratio[train_gh_idx]
test_ratio_pred = test["d48_same_ts"].to_numpy() * predicted_ratio[test_gh_idx]
oof_ratio_pred  = np.clip(oof_ratio_pred, 0, 1)
test_ratio_pred = np.clip(test_ratio_pred, 0, 1)


# ---------- Refit Ridge stack ----------
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw = art["y_raw"]; day49_idx_art = art["day49_idx"]
day49_idx = np.where(train["day"].values == 49)[0]
y49 = y_raw[day49_idx]

print(f">> ratio-model OOF R2 (day-49) = {r2_score(y49, oof_ratio_pred[day49_idx]):.5f}")

labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn","ratio_gh"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"], oof_ratio_pred]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"], test_ratio_pred]

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
print(">> Ridge weights:", dict(zip(labels, np.round(meta.coef_, 4))))
v30_oof  = meta.predict(oof_stack)
v30_test = np.clip(meta.predict(test_stack), 0, 1)
print(f">> v30 raw OOF R2 = {r2_score(y49, v30_oof):.5f}")


def smooth(values, gh, ts, w, b):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    df["out"] = b * df["sm"] + (1 - b) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()

oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)
best_r2 = r2_score(y49, v30_oof); best_test = v30_test; best_desc = "raw"
for w in (2, 3, 4):
    for b in (0.5, 0.75, 1.0):
        oof_s  = smooth(v30_oof, oof_gh, oof_ts, w, b)
        test_s = np.clip(smooth(v30_test, test["geohash"], test["tslot"], w, b), 0, 1)
        r2 = r2_score(y49, oof_s)
        print(f"   w={w} b={b}: OOF R2={r2:.5f}")
        if r2 > best_r2:
            best_r2 = r2; best_test = test_s; best_desc = f"w{w}_b{b}"

print(f">> winner: {best_desc}  OOF R2={best_r2:.5f}")
sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())
