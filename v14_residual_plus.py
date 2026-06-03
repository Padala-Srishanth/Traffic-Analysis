"""
v14: Strengthen the residual stack member.
- 3-seed CatBoost residual averaged
- + LightGBM residual (different inductive bias on same delta target)
Both trained ONLY on day-49 train rows. Each adds as a separate stack member,
giving the Ridge meta-learner 10 base models to combine.
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
import lightgbm as lgb

RNG = 42

print(">> loading data + v13 artifacts")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def parse_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return h.astype(int), m.astype(int), (h * 60 + m).astype(int), (h * 4 + m // 15).astype(int)

train["hour"], train["minute"], train["tmin"], train["tslot"] = parse_ts(train["timestamp"])
test["hour"],  test["minute"],  test["tmin"],  test["tslot"]  = parse_ts(test["timestamp"])

# baseline = d48_same_ts (with fallback)
d48 = train[train["day"] == 48][["geohash", "tslot", "demand"]]
d48_map = d48.set_index(["geohash", "tslot"])["demand"]
gh_d48_mean = d48.groupby("geohash")["demand"].mean()
overall_d48_mean = float(d48["demand"].mean())

def fill_baseline(df):
    keys = list(zip(df["geohash"], df["tslot"]))
    base = pd.Series(keys, index=df.index).map(d48_map)
    base = base.fillna(df["geohash"].map(gh_d48_mean)).fillna(overall_d48_mean)
    return base.astype(np.float32).to_numpy()

train["baseline"] = fill_baseline(train)
test["baseline"]  = fill_baseline(test)

train["delta"] = train["demand"].astype(float) - train["baseline"]

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
train["lat"] = train["geohash"].map(gh_lat); train["lon"] = train["geohash"].map(gh_lon)
test["lat"]  = test["geohash"].map(gh_lat);  test["lon"]  = test["geohash"].map(gh_lon)

for k in (3, 4, 5):
    train[f"gh{k}"] = train["geohash"].str.slice(0, k)
    test[f"gh{k}"]  = test["geohash"].str.slice(0, k)

train["hour_sin"] = np.sin(2*np.pi*train["hour"]/24); train["hour_cos"] = np.cos(2*np.pi*train["hour"]/24)
test["hour_sin"]  = np.sin(2*np.pi*test["hour"]/24);  test["hour_cos"]  = np.cos(2*np.pi*test["hour"]/24)
train["tmin_sin"] = np.sin(2*np.pi*train["tmin"]/1440); train["tmin_cos"] = np.cos(2*np.pi*train["tmin"]/1440)
test["tmin_sin"]  = np.sin(2*np.pi*test["tmin"]/1440);  test["tmin_cos"]  = np.cos(2*np.pi*test["tmin"]/1440)

gh_d48_stats = d48.groupby("geohash")["demand"].agg(["mean", "std", "max", "min"])
gh_d48_stats.columns = ["gh_d48_mean","gh_d48_std","gh_d48_max","gh_d48_min"]
train = train.merge(gh_d48_stats, left_on="geohash", right_index=True, how="left")
test  = test.merge(gh_d48_stats,  left_on="geohash", right_index=True, how="left")

# d48 same-tslot roll5 — smoothed
def add_rolling(df, value_col, window, out_name):
    grp = d48.sort_values(["geohash", "tslot"]).groupby("geohash")[value_col]
    smoothed = grp.transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    tmp = d48.assign(**{out_name: smoothed})[["geohash", "tslot", out_name]]
    return df.merge(tmp, on=["geohash", "tslot"], how="left")
train = add_rolling(train, "demand", 5, "d48_roll5")
test  = add_rolling(test,  "demand", 5, "d48_roll5")

CAT_FEATURES = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM_FEATURES = ["hour", "minute", "tmin", "tslot", "hour_sin", "hour_cos", "tmin_sin", "tmin_cos",
                "lat", "lon", "NumberofLanes", "Temperature",
                "baseline", "d48_roll5", "gh_d48_mean", "gh_d48_std", "gh_d48_max", "gh_d48_min"]
FEATURES = CAT_FEATURES + NUM_FEATURES
for c in CAT_FEATURES:
    train[c] = train[c].astype("string").fillna("NA")
    test[c]  = test[c].astype("string").fillna("NA")

X_train = train[FEATURES].copy()
X_test  = test[FEATURES].copy()
y_delta = train["delta"].astype(float).values

# label-encode categoricals for LightGBM
X_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for c in CAT_FEATURES:
    vals = pd.concat([X_lgb[c], X_test_lgb[c]], ignore_index=True)
    codes, _ = pd.factorize(vals, sort=True)
    X_lgb[c]      = codes[:len(X_lgb)]
    X_test_lgb[c] = codes[len(X_lgb):]
X_lgb = X_lgb.astype(np.float32); X_test_lgb = X_test_lgb.astype(np.float32)


# --------------------------------------------------------------------------- #
# Multi-seed CatBoost delta model on day-49 only                              #
# --------------------------------------------------------------------------- #
print(">> multi-seed delta CatBoost (3 seeds, 5-fold CV each, GPU)")
day49_idx = np.where(train["day"].values == 49)[0]
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]

cb_params_base = dict(
    iterations=3000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
    loss_function="RMSE", eval_metric="RMSE",
    od_type="Iter", od_wait=200,
    verbose=500, task_type="GPU", devices="0",
)

oof_delta_cb_seeds = []
pred_delta_cb_seeds = []
SEEDS = [42, 7, 123]
for s_i, seed in enumerate(SEEDS, 1):
    print(f">> CatBoost delta seed {s_i}/{len(SEEDS)} (random_seed={seed})")
    oof_seed = np.full(len(train), np.nan)
    pred_seed = np.zeros(len(test))
    p = dict(cb_params_base); p["random_seed"] = seed
    for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
        tr_idx = day49_idx[tr_d49]; va_idx = day49_idx[va_d49]
        m = CatBoostRegressor(**p)
        m.fit(X_train.iloc[tr_idx], y_delta[tr_idx],
              eval_set=(X_train.iloc[va_idx], y_delta[va_idx]),
              cat_features=cat_idx, use_best_model=True)
        oof_seed[va_idx] = m.predict(X_train.iloc[va_idx])
        pred_seed += m.predict(X_test) / kf.n_splits
        fold_r2 = r2_score(y_delta[va_idx], oof_seed[va_idx])
        print(f"   seed{seed} fold {fold}: best_iter={m.get_best_iteration()}  R²(delta)={fold_r2:.5f}")
    oof_delta_cb_seeds.append(oof_seed)
    pred_delta_cb_seeds.append(pred_seed)

oof_delta_cb = np.mean(oof_delta_cb_seeds, axis=0)
pred_delta_cb = np.mean(pred_delta_cb_seeds, axis=0)
oof_residual_cb = train["baseline"].to_numpy() + oof_delta_cb
pred_residual_cb = test["baseline"].to_numpy() + pred_delta_cb
print(f">> multi-seed CatBoost residual OOF R² (demand) = {r2_score(train['demand'].values[day49_idx], oof_residual_cb[day49_idx]):.5f}")


# --------------------------------------------------------------------------- #
# LightGBM delta model on day-49 only                                         #
# --------------------------------------------------------------------------- #
print(">> LightGBM delta model (5-fold CV, GPU)")
oof_delta_lgb = np.full(len(train), np.nan)
pred_delta_lgb = np.zeros(len(test))
lgb_params = dict(
    n_estimators=3000, learning_rate=0.05, num_leaves=63,
    min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
    reg_lambda=1.0, device="gpu", objective="regression", metric="rmse",
    random_state=RNG, n_jobs=-1, verbose=-1,
)
for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
    tr_idx = day49_idx[tr_d49]; va_idx = day49_idx[va_d49]
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X_lgb.iloc[tr_idx], y_delta[tr_idx],
          eval_set=[(X_lgb.iloc[va_idx], y_delta[va_idx])],
          callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_delta_lgb[va_idx] = m.predict(X_lgb.iloc[va_idx])
    pred_delta_lgb += m.predict(X_test_lgb) / kf.n_splits
    fold_r2 = r2_score(y_delta[va_idx], oof_delta_lgb[va_idx])
    print(f"   lgb fold {fold}: best_iter={m.best_iteration_}  R²(delta)={fold_r2:.5f}")

oof_residual_lgb = train["baseline"].to_numpy() + oof_delta_lgb
pred_residual_lgb = test["baseline"].to_numpy() + pred_delta_lgb
print(f">> LightGBM residual OOF R² (demand) = {r2_score(train['demand'].values[day49_idx], oof_residual_lgb[day49_idx]):.5f}")


# --------------------------------------------------------------------------- #
# Refit Ridge over 9 base models (replace v13 residual with v14 multi-seed,   #
# add LightGBM residual as 10th)                                              #
# --------------------------------------------------------------------------- #
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw, day49_idx_art = art["y_raw"], art["day49_idx"]
y49 = y_raw[day49_idx]

labels = ["cb", "xgb", "lgb", "hgb", "et", "knn", "chrS_50", "resCB", "resLGB"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            oof_residual_cb, oof_residual_lgb]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             pred_residual_cb, pred_residual_lgb]

single_r2 = {lbl: r2_score(y49, c[day49_idx]) for lbl, c in zip(labels, oof_cols)}
print(">> single-model R²:", {k: round(v, 5) for k, v in single_r2.items()})

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)

meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
blend_r2 = r2_score(y49, meta.predict(oof_stack))
print(">> Ridge meta weights =", dict(zip(labels, np.round(meta.coef_, 4))))
print(f">> Ridge-stacked OOF R² (9 models) = {blend_r2:.5f}  (score = {max(0,100*blend_r2):.3f})")

pred = np.clip(meta.predict(test_stack), 0.0, 1.0)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())

np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    **{k: art[k] for k in art.files},
    oof_residual_cb=oof_residual_cb, pred_residual_cb=pred_residual_cb,
    oof_residual_lgb=oof_residual_lgb, pred_residual_lgb=pred_residual_lgb,
    meta_coef_v14=meta.coef_,
)
print(">> saved artifacts.npz")
