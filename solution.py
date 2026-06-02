"""
Flipkart GRiD-Lock 2.0 — Online ML Challenge: Traffic Demand Prediction
Target: demand (regression), Metric: max(0, 100 * R2(actual, predicted)).
Model: CatBoost + XGBoost blend on engineered features.
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

_logfile = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.log"), "w", encoding="utf-8")
sys.stdout = _Tee(io.TextIOWrapper(sys.__stdout__.buffer, encoding="utf-8", line_buffering=True), _logfile)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.spatial import cKDTree
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

RNG = 42
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Geohash decoder (base-32 → lat, lon)                                        #
# --------------------------------------------------------------------------- #
_GH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_GH_MAP = {c: i for i, c in enumerate(_GH_BASE32)}

def decode_geohash(gh: str):
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for ch in gh:
        bits = _GH_MAP[ch]
        for mask in (16, 8, 4, 2, 1):
            b = (bits & mask) > 0
            if even:
                mid = (lon_lo + lon_hi) / 2.0
                if b: lon_lo = mid
                else: lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2.0
                if b: lat_lo = mid
                else: lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2.0, (lon_lo + lon_hi) / 2.0


# --------------------------------------------------------------------------- #
# Load                                                                        #
# --------------------------------------------------------------------------- #
print(">> loading data")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print("train", train.shape, "test", test.shape)

TARGET = "demand"
y_raw = train[TARGET].astype(float).values
# raw target — log1p was tried but hurt the leaderboard (compressed peaks)
y = y_raw


# --------------------------------------------------------------------------- #
# Feature engineering                                                         #
# --------------------------------------------------------------------------- #
def parse_ts(s: pd.Series):
    parts = s.str.split(":", expand=True).astype(int)
    h = parts[0]; m = parts[1]
    total = h * 60 + m
    return h, m, total

def add_basic_features(df: pd.DataFrame):
    df = df.copy()
    h, m, tm = parse_ts(df["timestamp"])
    df["hour"]   = h.astype(int)
    df["minute"] = m.astype(int)
    df["tmin"]   = tm.astype(int)          # minutes since midnight (0..1425)
    df["tslot"]  = (tm // 15).astype(int)  # 0..95 slot of the day
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["tmin_sin"] = np.sin(2 * np.pi * df["tmin"] / 1440.0)
    df["tmin_cos"] = np.cos(2 * np.pi * df["tmin"] / 1440.0)
    # geohash prefixes for hierarchical area
    for k in (1, 2, 3, 4, 5):
        df[f"gh{k}"] = df["geohash"].str.slice(0, k)
    return df

train = add_basic_features(train)
test  = add_basic_features(test)

# decode geohash → lat, lon (vectorize via cache)
print(">> decoding geohash → lat/lon")
all_gh = pd.unique(pd.concat([train["geohash"], test["geohash"]], ignore_index=True))
gh_coord = {g: decode_geohash(g) for g in all_gh}
gh_lat = {g: v[0] for g, v in gh_coord.items()}
gh_lon = {g: v[1] for g, v in gh_coord.items()}
train["lat"] = train["geohash"].map(gh_lat)
train["lon"] = train["geohash"].map(gh_lon)
test["lat"]  = test["geohash"].map(gh_lat)
test["lon"]  = test["geohash"].map(gh_lon)


# --------------------------------------------------------------------------- #
# Day-48 same-timestamp lookup (strongest cross-feature)                      #
# --------------------------------------------------------------------------- #
print(">> building day-48 lookups")
d48 = train[train["day"] == 48][["geohash", "tslot", "demand"]]
d48_map = d48.set_index(["geohash", "tslot"])["demand"]

# exact lookup
def map_pair(df, mapping, name):
    keys = list(zip(df["geohash"].values, df["tslot"].values))
    df[name] = pd.Series(keys, index=df.index).map(mapping)
    return df

train = map_pair(train, d48_map, "d48_same_ts")
test  = map_pair(test,  d48_map, "d48_same_ts")

# per-geohash aggregates from day 48 (full day curve)
gh_d48 = d48.groupby("geohash")["demand"]
gh_stats = pd.DataFrame({
    "gh_d48_mean":   gh_d48.mean(),
    "gh_d48_std":    gh_d48.std(),
    "gh_d48_median": gh_d48.median(),
    "gh_d48_max":    gh_d48.max(),
    "gh_d48_min":    gh_d48.min(),
    "gh_d48_count":  gh_d48.count(),
})
train = train.merge(gh_stats, left_on="geohash", right_index=True, how="left")
test  = test.merge(gh_stats,  left_on="geohash", right_index=True, how="left")

# per-tslot aggregate (citywide curve)
ts_d48 = d48.groupby("tslot")["demand"]
ts_stats = pd.DataFrame({
    "ts_d48_mean": ts_d48.mean(),
    "ts_d48_std":  ts_d48.std(),
})
train = train.merge(ts_stats, left_on="tslot", right_index=True, how="left")
test  = test.merge(ts_stats,  left_on="tslot", right_index=True, how="left")

# residual vs gh-mean (how this slot deviates from gh average)
train["d48_resid"] = train["d48_same_ts"] - train["gh_d48_mean"]
test["d48_resid"]  = test["d48_same_ts"]  - test["gh_d48_mean"]

# CRITICAL: for day-48 training rows, d48_same_ts IS the target itself (self lookup → leak).
# Mask all day-48-derived features on day-48 rows so the model can't trivially copy the target.
# (Test rows are all day-49, so these features remain valid at inference time.)
_d48_self_cols = ["d48_same_ts", "d48_resid"]  # roll3/roll5 set below also masked

# rolling neighbourhood of same-timestamp on day 48 (smooth the curve)
print(">> building rolling neighborhood features")
def add_rolling(df, key_cols, value_col, window, out_name):
    grp = d48.sort_values(["geohash", "tslot"]).groupby("geohash")[value_col]
    smoothed = grp.transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    tmp = d48.assign(**{out_name: smoothed})[["geohash", "tslot", out_name]]
    return df.merge(tmp, on=["geohash", "tslot"], how="left")

train = add_rolling(train, None, "demand", 3, "d48_roll3")
test  = add_rolling(test,  None, "demand", 3, "d48_roll3")
train = add_rolling(train, None, "demand", 5, "d48_roll5")
test  = add_rolling(test,  None, "demand", 5, "d48_roll5")

# v3 won at 91.15; v4 unmask hurt (90.16). Restored the mask: d48 features on
# day-48 training rows are leak (self-target). Masking them forces the model
# to learn meaningful corrections from other features.
d48_leak_cols = ["d48_same_ts", "d48_resid", "d48_roll3", "d48_roll5"]
train.loc[train["day"] == 48, d48_leak_cols] = np.nan


# --------------------------------------------------------------------------- #
# Day-49 lag features (last-known same-day demand)                            #
# --------------------------------------------------------------------------- #
# Train day-49 covers tslots 0..8 (00:00–02:00). For every (gh, query_tslot)
# we look up the d49 demand at the largest tslot strictly less than query_tslot.
# strictly-less avoids self-lookup leak for day-49 training rows.
print(">> building day-49 lag features")
d49 = train[train["day"] == 49][["geohash", "tslot", "demand"]].rename(
    columns={"demand": "d49_recent"}
).sort_values("tslot").reset_index(drop=True)

def attach_d49_recent(df: pd.DataFrame) -> pd.DataFrame:
    src = df[["geohash", "tslot"]].copy()
    src["_orig_order"] = np.arange(len(src))
    src = src.sort_values("tslot")
    merged = pd.merge_asof(
        src, d49, on="tslot", by="geohash",
        direction="backward", allow_exact_matches=False,
    )
    merged = merged.sort_values("_orig_order")
    df["d49_recent"] = merged["d49_recent"].to_numpy()
    return df

train = attach_d49_recent(train)
test  = attach_d49_recent(test)

# stable d49-only-aggregate (full known portion) per geohash — leakage-free
# because we compute mean of d49 demands per gh and then for day-49 rows we
# subtract self before normalising.
d49_full = train[train["day"] == 49].groupby("geohash")["demand"].agg(["mean", "max", "count"])
d49_full.columns = ["d49_known_mean", "d49_known_max", "d49_known_n"]
train = train.merge(d49_full, left_on="geohash", right_index=True, how="left")
test  = test.merge(d49_full,  left_on="geohash", right_index=True, how="left")

# leave-one-out adjustment for day-49 training rows
mask49 = train["day"] == 49
n = train.loc[mask49, "d49_known_n"]
mu = train.loc[mask49, "d49_known_mean"]
train.loc[mask49, "d49_known_mean"] = (mu * n - train.loc[mask49, "demand"]) / (n - 1).replace(0, np.nan)
# (max stays — small leak per row, but rare top-of-distribution effect)

# day-over-day delta at the latest known tslot
train["d49_minus_d48_recent"] = train["d49_recent"] - train["d48_same_ts"]
test["d49_minus_d48_recent"]  = test["d49_recent"]  - test["d48_same_ts"]

# how stale is the d49_recent lag — late tslots → high distance, less useful
train["tslot_minus_8"] = (train["tslot"] - 8).clip(lower=0)
test["tslot_minus_8"]  = (test["tslot"]  - 8).clip(lower=0)


# --------------------------------------------------------------------------- #
# Spatial neighbour feature                                                   #
# Mean demand of the k geographically nearest geohashes at the SAME tslot     #
# on day 48 (excluding the query geohash itself).                             #
# --------------------------------------------------------------------------- #
print(">> building spatial-neighbour features")
K_NEIGH = 5
d48_xy = train[train["day"] == 48][["geohash", "tslot", "lat", "lon", "demand"]].reset_index(drop=True)

def spatial_neighbour_stats(query: pd.DataFrame, source: pd.DataFrame, k: int = K_NEIGH):
    mean_arr = np.full(len(query), np.nan)
    std_arr = np.full(len(query), np.nan)
    max_arr = np.full(len(query), np.nan)
    wmean_arr = np.full(len(query), np.nan)
    q = query[["geohash", "tslot", "lat", "lon"]].reset_index(drop=False).rename(columns={"index": "_q_idx"})
    for ts, src in source.groupby("tslot"):
        if len(src) < 2: continue
        tree = cKDTree(src[["lat", "lon"]].values)
        ref_demand = src["demand"].values
        ref_gh = src["geohash"].values
        q_sub = q[q["tslot"] == ts]
        if len(q_sub) == 0: continue
        kk = min(k + 1, len(src))
        dists, nn_idx = tree.query(q_sub[["lat", "lon"]].values, k=kk)
        if kk == 1:
            nn_idx = nn_idx[:, None]; dists = dists[:, None]
        neigh_gh = ref_gh[nn_idx]
        neigh_dem = ref_demand[nn_idx]
        self_mask = neigh_gh == q_sub["geohash"].to_numpy()[:, None]
        neigh_dem_masked = np.where(self_mask, np.nan, neigh_dem)
        dists_masked = np.where(self_mask, np.nan, dists)
        idx_out = q_sub["_q_idx"].to_numpy()
        mean_arr[idx_out] = np.nanmean(neigh_dem_masked, axis=1)
        std_arr[idx_out]  = np.nanstd(neigh_dem_masked, axis=1)
        max_arr[idx_out]  = np.nanmax(neigh_dem_masked, axis=1)
        # inverse-distance weighted mean
        w = 1.0 / (dists_masked + 1e-6)
        w = np.where(np.isnan(neigh_dem_masked), 0.0, w)
        wsum = w.sum(axis=1)
        vsum = np.nansum(neigh_dem_masked * w, axis=1)
        wmean_arr[idx_out] = np.where(wsum > 0, vsum / wsum, np.nan)
    return mean_arr, std_arr, max_arr, wmean_arr

m, s, mx, wm = spatial_neighbour_stats(train, d48_xy)
train["spatial_nb_d48"] = m
train["spatial_nb_d48_std"] = s
train["spatial_nb_d48_max"] = mx
train["spatial_nb_d48_wmean"] = wm
m, s, mx, wm = spatial_neighbour_stats(test, d48_xy)
test["spatial_nb_d48"] = m
test["spatial_nb_d48_std"] = s
test["spatial_nb_d48_max"] = mx
test["spatial_nb_d48_wmean"] = wm
print("   non-null in train:", int(np.isfinite(train['spatial_nb_d48']).sum()),
      "test:", int(np.isfinite(test['spatial_nb_d48']).sum()))


# --------------------------------------------------------------------------- #
# gh × hour aggregates from day-48 — explicit per-(geohash, hour) pattern     #
# --------------------------------------------------------------------------- #
print(">> building gh×hour aggregates from day-48")
d48_full = train[train["day"] == 48].copy()
gh_hour = d48_full.groupby(["geohash", "hour"])["demand"].agg(["mean", "std"])
gh_hour.columns = ["gh_hour_d48_mean", "gh_hour_d48_std"]
train = train.merge(gh_hour, left_on=["geohash", "hour"], right_index=True, how="left")
test  = test.merge(gh_hour,  left_on=["geohash", "hour"], right_index=True, how="left")

# global per-hour mean — citywide hourly demand curve
hour_mean = d48_full.groupby("hour")["demand"].mean().rename("hour_d48_mean")
train = train.merge(hour_mean, left_on="hour", right_index=True, how="left")
test  = test.merge(hour_mean,  left_on="hour", right_index=True, how="left")


# --------------------------------------------------------------------------- #
# per-geohash day-over-day delta from the available day-49 training rows      #
# --------------------------------------------------------------------------- #
# For every (gh, tslot) where day-49 train exists, delta = d49(gh,tslot) - d48(gh,tslot).
# Average per gh → estimate of "how much higher/lower day 49 is for this neighbourhood".
print(">> computing per-geohash d49-d48 delta")
d49_tr = train[train["day"] == 49][["geohash", "tslot", "demand"]].rename(columns={"demand": "d49_d"})
d48_lookup_for_d49 = train[train["day"] == 48][["geohash", "tslot", "demand"]].rename(columns={"demand": "d48_d"})
joined = d49_tr.merge(d48_lookup_for_d49, on=["geohash", "tslot"], how="left")
joined["delta"] = joined["d49_d"] - joined["d48_d"]
gh_delta = joined.groupby("geohash")["delta"].agg(["mean", "std"])
gh_delta.columns = ["gh_d49_delta_mean", "gh_d49_delta_std"]
train = train.merge(gh_delta, left_on="geohash", right_index=True, how="left")
test  = test.merge(gh_delta,  left_on="geohash", right_index=True, how="left")

# leave-one-out for day-49 training rows: subtract this row's own delta contribution
mask49 = train["day"] == 49
if mask49.any():
    # recompute on the fly for each day-49 row using vectorized ops
    cur_d48 = train.loc[mask49].merge(
        d48_lookup_for_d49, on=["geohash", "tslot"], how="left"
    )["d48_d"].to_numpy()
    own_delta = train.loc[mask49, "demand"].to_numpy() - cur_d48
    # gh count of day-49 rows for that geohash
    cnt = train.loc[mask49, "geohash"].map(joined.groupby("geohash").size()).to_numpy()
    sum_delta = train.loc[mask49, "geohash"].map(joined.groupby("geohash")["delta"].sum()).to_numpy()
    loo_mean = np.where(cnt > 1, (sum_delta - own_delta) / (cnt - 1), np.nan)
    train.loc[mask49, "gh_d49_delta_mean"] = loo_mean

# geohash-prefix mean demand on day 48 (neighbourhood signal)
for k in (3, 4, 5):
    col = f"gh{k}"
    tmp = train[train["day"] == 48].groupby(col)["demand"].mean().rename(f"{col}_d48_mean")
    train = train.merge(tmp, left_on=col, right_index=True, how="left")
    test  = test.merge(tmp,  left_on=col, right_index=True, how="left")

# (prefix, tslot) mean — local-area diurnal pattern
for k in (4, 5):
    col = f"gh{k}"
    tmp = train[train["day"] == 48].groupby([col, "tslot"])["demand"].mean().rename(f"{col}_ts_mean")
    train = train.merge(tmp, left_on=[col, "tslot"], right_index=True, how="left")
    test  = test.merge(tmp,  left_on=[col, "tslot"], right_index=True, how="left")


# --------------------------------------------------------------------------- #
# Feature list                                                                #
# --------------------------------------------------------------------------- #
CAT_FEATURES = ["geohash", "gh1", "gh2", "gh3", "gh4", "gh5",
                "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM_FEATURES = [
    "day", "hour", "minute", "tmin", "tslot",
    "hour_sin", "hour_cos", "tmin_sin", "tmin_cos",
    "lat", "lon",
    "NumberofLanes", "Temperature",
    "d48_same_ts", "d48_resid", "d48_roll3", "d48_roll5",
    "gh_d48_mean", "gh_d48_std", "gh_d48_median", "gh_d48_max", "gh_d48_min", "gh_d48_count",
    "ts_d48_mean", "ts_d48_std",
    "gh3_d48_mean", "gh4_d48_mean", "gh5_d48_mean",
    "gh4_ts_mean", "gh5_ts_mean",
    "d49_recent", "d49_known_mean", "d49_known_max", "d49_known_n",
    "d49_minus_d48_recent",
    "gh_hour_d48_mean", "gh_hour_d48_std", "hour_d48_mean",
    "gh_d49_delta_mean", "gh_d49_delta_std",
    "tslot_minus_8", "spatial_nb_d48",
    "spatial_nb_d48_std", "spatial_nb_d48_max", "spatial_nb_d48_wmean",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

# CatBoost requires categorical columns to be string and not NaN
for c in CAT_FEATURES:
    train[c] = train[c].astype("string").fillna("NA")
    test[c]  = test[c].astype("string").fillna("NA")

X = train[FEATURES].copy()
X_test = test[FEATURES].copy()

print(">> feature matrix:", X.shape, "test:", X_test.shape)


# --------------------------------------------------------------------------- #
# Cross-validated CatBoost                                                    #
# --------------------------------------------------------------------------- #
print(">> 5-fold CatBoost training (val = day-49 train rows only)")
day49_idx = np.where(train["day"].values == 49)[0]
day48_idx = np.where(train["day"].values == 48)[0]
print(f"   day48 rows: {len(day48_idx)}  day49 rows: {len(day49_idx)}")

kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
oof_cb = np.full(len(X), np.nan)
pred_cb = np.zeros(len(X_test))

cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]
cb_params = dict(
    iterations=4000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=RNG,
    od_type="Iter",
    od_wait=200,
    verbose=200,
    task_type="GPU",
    devices="0",
)

# up-weight day-48 daytime rows (tslot >= 9) so training emphasises the test distribution
sample_weight = np.ones(len(X))
sample_weight[(train["day"].values == 48) & (train["tslot"].values >= 9)] = 1.5
print(f"   sample weights: {sample_weight.mean():.3f} mean, {(sample_weight>1).sum()} up-weighted rows")

SEEDS_CB = [42, 7, 123]
oof_cb_seeds = []
pred_cb_seeds = []

for s_i, seed in enumerate(SEEDS_CB, 1):
    print(f">> CatBoost seed {s_i}/{len(SEEDS_CB)} (random_seed={seed})")
    oof_seed = np.full(len(X), np.nan)
    pred_seed = np.zeros(len(X_test))
    cb_params_seed = dict(cb_params); cb_params_seed["random_seed"] = seed
    for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
        tr_idx = np.concatenate([day48_idx, day49_idx[tr_d49]])
        va_idx = day49_idx[va_d49]
        model = CatBoostRegressor(**cb_params_seed)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            sample_weight=sample_weight[tr_idx],
            eval_set=(X.iloc[va_idx], y[va_idx]),
            cat_features=cat_idx,
            use_best_model=True,
        )
        oof_seed[va_idx] = model.predict(X.iloc[va_idx])
        pred_seed += model.predict(X_test) / kf.n_splits
        print(f"   seed{seed} fold {fold}: best_iter={model.get_best_iteration()}  R²(raw)={r2_score(y_raw[va_idx], oof_seed[va_idx]):.5f}")
    print(f"   seed{seed} OOF R²={r2_score(y_raw[day49_idx], oof_seed[day49_idx]):.5f}")
    oof_cb_seeds.append(oof_seed); pred_cb_seeds.append(pred_seed)

oof_cb = np.mean(oof_cb_seeds, axis=0)
pred_cb = np.mean(pred_cb_seeds, axis=0)
cb_oof_r2 = r2_score(y_raw[day49_idx], oof_cb[day49_idx])
print(f">> CatBoost (multi-seed mean) OOF R² (day-49, raw) = {cb_oof_r2:.5f}  (score = {max(0, 100*cb_oof_r2):.3f})")

# final-fit disabled — hurt the leaderboard. CV-averaged predictions only.
pred_cb_full = pred_cb.copy()


# --------------------------------------------------------------------------- #
# Cross-validated XGBoost (one-hot for low-card cats, label-encode for high)  #
# --------------------------------------------------------------------------- #
print(">> 5-fold XGBoost training")
X_xgb = X.copy()
X_test_xgb = X_test.copy()

# label-encode all categoricals for XGBoost
for c in CAT_FEATURES:
    vals = pd.concat([X_xgb[c], X_test_xgb[c]], ignore_index=True)
    codes, _ = pd.factorize(vals, sort=True)
    X_xgb[c]      = codes[:len(X_xgb)]
    X_test_xgb[c] = codes[len(X_xgb):]

X_xgb = X_xgb.astype(np.float32)
X_test_xgb = X_test_xgb.astype(np.float32)

oof_xgb = np.full(len(X_xgb), np.nan)
pred_xgb = np.zeros(len(X_test_xgb))

xgb_params = dict(
    n_estimators=4000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=4,
    reg_lambda=1.0,
    tree_method="hist",
    device="cuda",
    objective="reg:squarederror",
    random_state=RNG,
    n_jobs=-1,
    early_stopping_rounds=200,
)

xgb_best_iters = []
for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
    tr_idx = np.concatenate([day48_idx, day49_idx[tr_d49]])
    va_idx = day49_idx[va_d49]
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_xgb.iloc[tr_idx], y[tr_idx],
        sample_weight=sample_weight[tr_idx],
        eval_set=[(X_xgb.iloc[va_idx], y[va_idx])],
        verbose=False,
    )
    oof_xgb[va_idx] = model.predict(X_xgb.iloc[va_idx])
    pred_xgb += model.predict(X_test_xgb) / kf.n_splits
    xgb_best_iters.append(int(model.best_iteration) + 1)
    print(f"  fold {fold}: best_iter={xgb_best_iters[-1]}  R²(raw)={r2_score(y_raw[va_idx], oof_xgb[va_idx]):.5f}")

xgb_oof_r2 = r2_score(y_raw[day49_idx], oof_xgb[day49_idx])
print(f">> XGBoost OOF R² (day-49, raw) = {xgb_oof_r2:.5f}  (score = {max(0, 100*xgb_oof_r2):.3f})")

# final-fit disabled — hurt the leaderboard. CV-averaged predictions only.
pred_xgb_full = pred_xgb.copy()


# --------------------------------------------------------------------------- #
# Cross-validated LightGBM (GPU, leaf-wise growth — different inductive bias) #
# --------------------------------------------------------------------------- #
print(">> 5-fold LightGBM training (GPU)")
oof_lgb = np.full(len(X_xgb), np.nan)
pred_lgb = np.zeros(len(X_test_xgb))

lgb_params = dict(
    n_estimators=4000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    device="gpu",
    objective="regression",
    metric="rmse",
    random_state=RNG,
    n_jobs=-1,
    verbose=-1,
)

for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
    tr_idx = np.concatenate([day48_idx, day49_idx[tr_d49]])
    va_idx = day49_idx[va_d49]
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_xgb.iloc[tr_idx], y[tr_idx],
        sample_weight=sample_weight[tr_idx],
        eval_set=[(X_xgb.iloc[va_idx], y[va_idx])],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    oof_lgb[va_idx] = model.predict(X_xgb.iloc[va_idx])
    pred_lgb += model.predict(X_test_xgb) / kf.n_splits
    print(f"  fold {fold}: best_iter={model.best_iteration_}  R²(raw)={r2_score(y_raw[va_idx], oof_lgb[va_idx]):.5f}")

lgb_oof_r2 = r2_score(y_raw[day49_idx], oof_lgb[day49_idx])
print(f">> LightGBM OOF R² (day-49, raw) = {lgb_oof_r2:.5f}  (score = {max(0, 100*lgb_oof_r2):.3f})")


# --------------------------------------------------------------------------- #
# Cross-validated HistGradientBoostingRegressor (sklearn, CPU)                #
# --------------------------------------------------------------------------- #
print(">> 5-fold HistGBM training (CPU)")
oof_hgb = np.full(len(X_xgb), np.nan)
pred_hgb = np.zeros(len(X_test_xgb))

hgb_params = dict(
    max_iter=1500,
    learning_rate=0.05,
    max_depth=None,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    n_iter_no_change=80,
    validation_fraction=None,   # we pass eval set manually via early_stopping=True; uses internal split
    random_state=RNG,
)

for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
    tr_idx = np.concatenate([day48_idx, day49_idx[tr_d49]])
    va_idx = day49_idx[va_d49]
    model = HistGradientBoostingRegressor(**hgb_params)
    model.fit(X_xgb.iloc[tr_idx], y[tr_idx], sample_weight=sample_weight[tr_idx])
    oof_hgb[va_idx] = model.predict(X_xgb.iloc[va_idx])
    pred_hgb += model.predict(X_test_xgb) / kf.n_splits
    print(f"  fold {fold}: n_iter={model.n_iter_}  R²(raw)={r2_score(y_raw[va_idx], oof_hgb[va_idx]):.5f}")

hgb_oof_r2 = r2_score(y_raw[day49_idx], oof_hgb[day49_idx])
print(f">> HistGBM OOF R² (day-49, raw) = {hgb_oof_r2:.5f}  (score = {max(0, 100*hgb_oof_r2):.3f})")


# --------------------------------------------------------------------------- #
# Blend                                                                       #
# --------------------------------------------------------------------------- #
# Ridge stacking: learn the optimal weights over the 4 base models on day-49 OOF
y49_raw = y_raw[day49_idx]
oof_stack = np.column_stack([oof_cb[day49_idx], oof_xgb[day49_idx],
                             oof_lgb[day49_idx], oof_hgb[day49_idx]])
test_stack = np.column_stack([pred_cb, pred_xgb, pred_lgb, pred_hgb])

# also report individual best-single
single_r2 = {
    "cb":  r2_score(y49_raw, oof_cb[day49_idx]),
    "xgb": r2_score(y49_raw, oof_xgb[day49_idx]),
    "lgb": r2_score(y49_raw, oof_lgb[day49_idx]),
    "hgb": r2_score(y49_raw, oof_hgb[day49_idx]),
}
print(">> single-model R²:", {k: round(v,5) for k,v in single_r2.items()})

meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49_raw)
oof_blend = meta.predict(oof_stack)
blend_r2 = r2_score(y49_raw, oof_blend)
print(f">> Ridge meta weights = {dict(zip(['cb','xgb','lgb','hgb'], np.round(meta.coef_,4)))}")
print(f">> Ridge-stacked OOF R² = {blend_r2:.5f}  (score = {max(0, 100*blend_r2):.3f})")

# persist artifacts for offline iteration
np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    oof_cb=oof_cb, oof_xgb=oof_xgb, oof_lgb=oof_lgb, oof_hgb=oof_hgb,
    pred_cb=pred_cb, pred_xgb=pred_xgb, pred_lgb=pred_lgb, pred_hgb=pred_hgb,
    y_raw=y_raw, day49_idx=day49_idx,
    meta_coef=meta.coef_,
)
print(">> saved artifacts.npz")

pred = meta.predict(test_stack)
pred = np.clip(pred, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Submission                                                                  #
# --------------------------------------------------------------------------- #
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())
