"""
v17: Pseudo-labeled XGBoost + LightGBM (different inductive bias from v15's CatBoost).
Uses v15 submission predictions as pseudo-labels on test rows, trains both XGBoost
and LightGBM on (real day-49 + pseudo-labeled test) with weight 1.0 for real, 0.5 for
pseudo. Adds both as stack members; Ridge refits over 12 base models.
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
import xgboost as xgb
import lightgbm as lgb

RNG = 42

print(">> loading data + current submission as pseudo-labels")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub_now = pd.read_csv(os.path.join(DATA_DIR, "submission.csv"))
test = test.merge(sub_now.rename(columns={"demand": "pseudo_demand"}), on="Index", how="left")
print(f"   pseudo-labels available: {int(test['pseudo_demand'].notna().sum())} / {len(test)}")


# --------------------------------------------------------------------------- #
# Feature engineering — same minimal subset as v15                            #
# --------------------------------------------------------------------------- #
def parse_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return h.astype(int), m.astype(int), (h * 60 + m).astype(int), (h * 4 + m // 15).astype(int)

train["hour"], train["minute"], train["tmin"], train["tslot"] = parse_ts(train["timestamp"])
test["hour"],  test["minute"],  test["tmin"],  test["tslot"]  = parse_ts(test["timestamp"])

d48 = train[train["day"] == 48][["geohash", "tslot", "demand"]]
d48_map = d48.set_index(["geohash", "tslot"])["demand"]
gh_d48_mean = d48.groupby("geohash")["demand"].mean()
gh_d48_std  = d48.groupby("geohash")["demand"].std()
overall_d48_mean = float(d48["demand"].mean())

def fill_baseline(df):
    keys = list(zip(df["geohash"], df["tslot"]))
    return pd.Series(keys, index=df.index).map(d48_map)\
            .fillna(df["geohash"].map(gh_d48_mean)).fillna(overall_d48_mean)\
            .astype(np.float32).to_numpy()

train["baseline"] = fill_baseline(train)
test["baseline"]  = fill_baseline(test)
train["gh_d48_mean"] = train["geohash"].map(gh_d48_mean).fillna(overall_d48_mean).astype(np.float32)
test["gh_d48_mean"]  = test["geohash"].map(gh_d48_mean).fillna(overall_d48_mean).astype(np.float32)
train["gh_d48_std"]  = train["geohash"].map(gh_d48_std).fillna(0.0).astype(np.float32)
test["gh_d48_std"]   = test["geohash"].map(gh_d48_std).fillna(0.0).astype(np.float32)

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

all_gh = pd.unique(pd.concat([train["geohash"], test["geohash"]]))
gh_lat = {g: decode(g)[0] for g in all_gh}
gh_lon = {g: decode(g)[1] for g in all_gh}
train["lat"] = train["geohash"].map(gh_lat); train["lon"] = train["geohash"].map(gh_lon)
test["lat"]  = test["geohash"].map(gh_lat);  test["lon"]  = test["geohash"].map(gh_lon)

train["hour_sin"] = np.sin(2*np.pi*train["hour"]/24); train["hour_cos"] = np.cos(2*np.pi*train["hour"]/24)
test["hour_sin"]  = np.sin(2*np.pi*test["hour"]/24);  test["hour_cos"]  = np.cos(2*np.pi*test["hour"]/24)

for k in (3, 4, 5):
    train[f"gh{k}"] = train["geohash"].str.slice(0, k)
    test[f"gh{k}"]  = test["geohash"].str.slice(0, k)


# label-encode categoricals for XGBoost/LightGBM
CAT_FEATURES = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM_FEATURES = ["day", "hour", "minute", "tmin", "tslot", "hour_sin", "hour_cos",
                "lat", "lon", "NumberofLanes", "Temperature",
                "baseline", "gh_d48_mean", "gh_d48_std"]
FEATURES = CAT_FEATURES + NUM_FEATURES

X_full = pd.concat([train[FEATURES].copy(), test[FEATURES].copy()], ignore_index=True)
for c in CAT_FEATURES:
    X_full[c] = X_full[c].astype("string").fillna("NA")
    codes, _ = pd.factorize(X_full[c], sort=True)
    X_full[c] = codes
X_full = X_full.astype(np.float32)
X_train = X_full.iloc[:len(train)].reset_index(drop=True)
X_test  = X_full.iloc[len(train):].reset_index(drop=True)
y_train = train["demand"].astype(float).to_numpy()
y_pseudo = test["pseudo_demand"].astype(float).to_numpy()


# --------------------------------------------------------------------------- #
# 5-fold CV for pseudo-labeled XGBoost                                        #
# --------------------------------------------------------------------------- #
day49_idx = np.where(train["day"].values == 49)[0]
X_d49 = X_train.iloc[day49_idx].reset_index(drop=True)
y_d49 = y_train[day49_idx]
print(f"   day-49 real rows: {len(X_d49)}  pseudo test rows: {len(X_test)}")

PSEUDO_W = 0.5
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)

print(">> pseudo-labeled XGBoost (5-fold, GPU)")
oof_pseudo_xgb = np.full(len(train), np.nan)
pred_pseudo_xgb = np.zeros(len(test))
xgb_params = dict(
    n_estimators=4000, learning_rate=0.05, max_depth=8,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=4, reg_lambda=1.0,
    tree_method="hist", device="cuda", objective="reg:squarederror",
    random_state=RNG, n_jobs=-1, early_stopping_rounds=200,
)
for fold, (tr_pos, va_pos) in enumerate(kf.split(np.arange(len(X_d49))), 1):
    X_real_tr = X_d49.iloc[tr_pos]; y_real_tr = y_d49[tr_pos]
    X_tr = pd.concat([X_real_tr, X_test], ignore_index=True)
    y_tr = np.concatenate([y_real_tr, y_pseudo])
    w_tr = np.concatenate([np.ones(len(X_real_tr)), np.full(len(X_test), PSEUDO_W)])
    X_va = X_d49.iloc[va_pos]; y_va = y_d49[va_pos]
    va_real_idx = day49_idx[va_pos]

    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_pseudo_xgb[va_real_idx] = m.predict(X_va)
    pred_pseudo_xgb += m.predict(X_test) / kf.n_splits
    print(f"   xgb fold {fold}: best_iter={m.best_iteration}  R²={r2_score(y_va, oof_pseudo_xgb[va_real_idx]):.5f}")

xgb_pseudo_r2 = r2_score(y_d49, oof_pseudo_xgb[day49_idx])
print(f">> Pseudo-XGBoost OOF R² (day-49 raw) = {xgb_pseudo_r2:.5f}  (score = {max(0,100*xgb_pseudo_r2):.3f})")


print(">> pseudo-labeled LightGBM (5-fold, GPU)")
oof_pseudo_lgb = np.full(len(train), np.nan)
pred_pseudo_lgb = np.zeros(len(test))
lgb_params = dict(
    n_estimators=4000, learning_rate=0.05, num_leaves=127,
    min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
    reg_lambda=1.0, device="gpu", objective="regression", metric="rmse",
    random_state=RNG, n_jobs=-1, verbose=-1,
)
for fold, (tr_pos, va_pos) in enumerate(kf.split(np.arange(len(X_d49))), 1):
    X_real_tr = X_d49.iloc[tr_pos]; y_real_tr = y_d49[tr_pos]
    X_tr = pd.concat([X_real_tr, X_test], ignore_index=True)
    y_tr = np.concatenate([y_real_tr, y_pseudo])
    w_tr = np.concatenate([np.ones(len(X_real_tr)), np.full(len(X_test), PSEUDO_W)])
    X_va = X_d49.iloc[va_pos]; y_va = y_d49[va_pos]
    va_real_idx = day49_idx[va_pos]

    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr, sample_weight=w_tr,
          eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_pseudo_lgb[va_real_idx] = m.predict(X_va)
    pred_pseudo_lgb += m.predict(X_test) / kf.n_splits
    print(f"   lgb fold {fold}: best_iter={m.best_iteration_}  R²={r2_score(y_va, oof_pseudo_lgb[va_real_idx]):.5f}")

lgb_pseudo_r2 = r2_score(y_d49, oof_pseudo_lgb[day49_idx])
print(f">> Pseudo-LightGBM OOF R² (day-49 raw) = {lgb_pseudo_r2:.5f}  (score = {max(0,100*lgb_pseudo_r2):.3f})")


# --------------------------------------------------------------------------- #
# Refit Ridge stack with 12 base models                                       #
# --------------------------------------------------------------------------- #
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw = art["y_raw"]; day49_idx_art = art["day49_idx"]
y49 = y_raw[day49_idx]

labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn","pseudoXGB","pseudoLGB"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"],
            oof_pseudo_xgb, oof_pseudo_lgb]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"],
             pred_pseudo_xgb, pred_pseudo_lgb]

single_r2 = {lbl: r2_score(y49, c[day49_idx]) for lbl, c in zip(labels, oof_cols)}
print(">> single-model R²:", {k: round(v, 5) for k, v in single_r2.items()})

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)

meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
blend_r2 = r2_score(y49, meta.predict(oof_stack))
print(">> Ridge meta weights =", dict(zip(labels, np.round(meta.coef_, 4))))
print(f">> Ridge-stacked OOF R² (12 models) = {blend_r2:.5f}  (score = {max(0,100*blend_r2):.3f})")

pred = np.clip(meta.predict(test_stack), 0.0, 1.0)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())

np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    **{k: art[k] for k in art.files},
    oof_pseudo_xgb=oof_pseudo_xgb, pred_pseudo_xgb=pred_pseudo_xgb,
    oof_pseudo_lgb=oof_pseudo_lgb, pred_pseudo_lgb=pred_pseudo_lgb,
    meta_coef_v17=meta.coef_,
)
print(">> saved artifacts.npz")
