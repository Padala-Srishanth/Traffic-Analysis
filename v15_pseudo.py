"""
v15: Pseudo-labeled CatBoost focused on day-49 patterns.
Combines day-49 train rows (real labels) with test rows (v13 predictions as
pseudo-labels) so the model trains on ~50k day-49 examples including the
DAYTIME hours my OOF cannot see. Adds as a 9th stack member, refits Ridge.
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
from catboost import CatBoostRegressor

RNG = 42

print(">> loading data + artifacts + v13 submission as pseudo-labels")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
v13_sub = pd.read_csv(os.path.join(DATA_DIR, "submission.csv"))  # current = v13 restored
test = test.merge(v13_sub.rename(columns={"demand": "pseudo_demand"}), on="Index", how="left")
print(f"   test rows with pseudo-label: {int(test['pseudo_demand'].notna().sum())} / {len(test)}")

# Build same features as v13 residual model
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

# decode geohash
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

CAT_FEATURES = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM_FEATURES = ["day", "hour", "minute", "tmin", "tslot", "hour_sin", "hour_cos", "tmin_sin", "tmin_cos",
                "lat", "lon", "NumberofLanes", "Temperature",
                "baseline", "gh_d48_mean", "gh_d48_std", "gh_d48_max", "gh_d48_min"]
FEATURES = CAT_FEATURES + NUM_FEATURES
for c in CAT_FEATURES:
    train[c] = train[c].astype("string").fillna("NA")
    test[c]  = test[c].astype("string").fillna("NA")


# --------------------------------------------------------------------------- #
# Build the combined "day-49 + pseudo" training set                           #
# --------------------------------------------------------------------------- #
day49_idx = np.where(train["day"].values == 49)[0]
X_d49 = train.iloc[day49_idx][FEATURES].copy()
y_d49 = train.iloc[day49_idx]["demand"].astype(float).to_numpy()
X_test_full = test[FEATURES].copy()
# pseudo-set: day-49 (day=49 in feature set, since test is all day 49)
X_pseudo = test[FEATURES].copy()
y_pseudo = test["pseudo_demand"].astype(float).to_numpy()
print(f"   real day-49: {len(X_d49)}  pseudo (test): {len(X_pseudo)}  total: {len(X_d49)+len(X_pseudo)}")


# --------------------------------------------------------------------------- #
# 5-fold CV: hold out 20% of real day-49 as val, train on rest + all pseudo  #
# --------------------------------------------------------------------------- #
print(">> 5-fold pseudo-labeled CatBoost (real weight=1.0, pseudo weight=0.5, GPU)")
cb_params = dict(
    iterations=4000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
    loss_function="RMSE", eval_metric="RMSE",
    random_seed=RNG, od_type="Iter", od_wait=200,
    verbose=500, task_type="GPU", devices="0",
)
cat_idx = [FEATURES.index(c) for c in CAT_FEATURES]

oof_pseudo = np.full(len(train), np.nan)
pred_pseudo = np.zeros(len(test))
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
PSEUDO_W = 0.5
for fold, (tr_d49_pos, va_d49_pos) in enumerate(kf.split(np.arange(len(X_d49))), 1):
    # train portion = real-train (80% of day-49) + ALL pseudo
    X_real_tr = X_d49.iloc[tr_d49_pos]
    y_real_tr = y_d49[tr_d49_pos]
    X_tr = pd.concat([X_real_tr, X_pseudo], ignore_index=True)
    y_tr = np.concatenate([y_real_tr, y_pseudo])
    w_tr = np.concatenate([np.ones(len(X_real_tr)), np.full(len(X_pseudo), PSEUDO_W)])

    X_va = X_d49.iloc[va_d49_pos]
    y_va = y_d49[va_d49_pos]
    va_real_idx = day49_idx[va_d49_pos]

    m = CatBoostRegressor(**cb_params)
    m.fit(X_tr, y_tr,
          sample_weight=w_tr,
          eval_set=(X_va, y_va),
          cat_features=cat_idx, use_best_model=True)

    oof_pseudo[va_real_idx] = m.predict(X_va)
    pred_pseudo += m.predict(X_test_full) / kf.n_splits
    fold_r2 = r2_score(y_va, oof_pseudo[va_real_idx])
    print(f"   fold {fold}: best_iter={m.get_best_iteration()}  R²(raw)={fold_r2:.5f}")

pseudo_oof_r2 = r2_score(y_d49, oof_pseudo[day49_idx])
print(f">> Pseudo-CatBoost OOF R² (day-49 raw) = {pseudo_oof_r2:.5f}  (score = {max(0,100*pseudo_oof_r2):.3f})")


# --------------------------------------------------------------------------- #
# Refit Ridge over 8 base models (v13's 8) + pseudo as 9th                    #
# --------------------------------------------------------------------------- #
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw, day49_idx_art = art["y_raw"], art["day49_idx"]
y49 = y_raw[day49_idx]

labels = ["cb", "xgb", "lgb", "hgb", "et", "knn", "chrS_50", "residual", "pseudo"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"], art["oof_residual"], oof_pseudo]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"], art["pred_residual"], pred_pseudo]

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
    oof_pseudo=oof_pseudo, pred_pseudo=pred_pseudo,
    meta_coef_v15=meta.coef_,
)
print(">> saved artifacts.npz")
