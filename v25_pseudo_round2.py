"""
v25: Iterative pseudo-labeling round 2. Uses v19b's smoothed predictions
(currently in submission.csv at 91.676) as refined pseudo-labels. Trains
CatBoost on (real day-49 + test with pseudo-labels) — pseudo weight 0.5.
Adds the result to the Ridge stack with smoothing post-process.
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

print(">> loading data + v19b submission (91.676) as round-2 pseudo-labels")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub_now = pd.read_csv(os.path.join(DATA_DIR, "submission.csv"))   # v19b restored
test = test.merge(sub_now.rename(columns={"demand": "pseudo_demand"}), on="Index", how="left")
print(f"   pseudo-labels coverage: {int(test['pseudo_demand'].notna().sum())} / {len(test)}")

def parse_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return h.astype(int), m.astype(int), (h * 60 + m).astype(int), (h * 4 + m // 15).astype(int)
train["hour"], train["minute"], train["tmin"], train["tslot"] = parse_ts(train["timestamp"])
test["hour"],  test["minute"],  test["tmin"],  test["tslot"]  = parse_ts(test["timestamp"])

# minimum FE
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
gh_lat = {g: decode(g)[0] for g in all_gh}; gh_lon = {g: decode(g)[1] for g in all_gh}
train["lat"] = train["geohash"].map(gh_lat); train["lon"] = train["geohash"].map(gh_lon)
test["lat"]  = test["geohash"].map(gh_lat);  test["lon"]  = test["geohash"].map(gh_lon)

for k in (3, 4, 5):
    train[f"gh{k}"] = train["geohash"].str.slice(0, k)
    test[f"gh{k}"]  = test["geohash"].str.slice(0, k)

train["hour_sin"] = np.sin(2*np.pi*train["hour"]/24); train["hour_cos"] = np.cos(2*np.pi*train["hour"]/24)
test["hour_sin"]  = np.sin(2*np.pi*test["hour"]/24);  test["hour_cos"]  = np.cos(2*np.pi*test["hour"]/24)

CAT = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM = ["day", "hour", "minute", "tmin", "tslot", "hour_sin", "hour_cos",
       "lat", "lon", "NumberofLanes", "Temperature",
       "baseline", "gh_d48_mean", "gh_d48_std"]
FEAT = CAT + NUM
for c in CAT:
    train[c] = train[c].astype("string").fillna("NA")
    test[c]  = test[c].astype("string").fillna("NA")

day49_idx = np.where(train["day"].values == 49)[0]
X_d49 = train.iloc[day49_idx][FEAT].copy()
y_d49 = train.iloc[day49_idx]["demand"].astype(float).to_numpy()
X_test_full = test[FEAT].copy()
X_pseudo = test[FEAT].copy()
y_pseudo = test["pseudo_demand"].astype(float).to_numpy()

print(f"   real day-49: {len(X_d49)}  pseudo (round-2 from v19b): {len(X_pseudo)}")


# --------------------------------------------------------------------------- #
# 5-fold round-2 pseudo CatBoost                                              #
# --------------------------------------------------------------------------- #
print(">> training round-2 pseudo CatBoost (5-fold, GPU)")
cb_params = dict(
    iterations=4000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
    loss_function="RMSE", eval_metric="RMSE",
    random_seed=RNG, od_type="Iter", od_wait=200,
    verbose=500, task_type="GPU", devices="0",
)
cat_idx = [FEAT.index(c) for c in CAT]
oof_pseudo2 = np.full(len(train), np.nan)
pred_pseudo2 = np.zeros(len(test))
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
PSEUDO_W = 0.5

for fold, (tr_p, va_p) in enumerate(kf.split(np.arange(len(X_d49))), 1):
    X_real_tr = X_d49.iloc[tr_p]; y_real_tr = y_d49[tr_p]
    X_tr = pd.concat([X_real_tr, X_pseudo], ignore_index=True)
    y_tr = np.concatenate([y_real_tr, y_pseudo])
    w_tr = np.concatenate([np.ones(len(X_real_tr)), np.full(len(X_pseudo), PSEUDO_W)])
    X_va = X_d49.iloc[va_p]; y_va = y_d49[va_p]
    va_real_idx = day49_idx[va_p]

    m = CatBoostRegressor(**cb_params)
    m.fit(X_tr, y_tr, sample_weight=w_tr,
          eval_set=(X_va, y_va), cat_features=cat_idx, use_best_model=True)
    oof_pseudo2[va_real_idx] = m.predict(X_va)
    pred_pseudo2 += m.predict(X_test_full) / kf.n_splits
    print(f"   fold {fold}: best_iter={m.get_best_iteration()}  R²={r2_score(y_va, oof_pseudo2[va_real_idx]):.5f}")

pseudo2_r2 = r2_score(y_d49, oof_pseudo2[day49_idx])
print(f">> round-2 pseudo CatBoost OOF R² = {pseudo2_r2:.5f}")


# --------------------------------------------------------------------------- #
# Refit Ridge with the new pseudo round-2 model + smoothing                   #
# --------------------------------------------------------------------------- #
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw = art["y_raw"]; day49_idx_art = art["day49_idx"]
y49 = y_raw[day49_idx]

labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn","pseudo2"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"], oof_pseudo2]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"], pred_pseudo2]

single_r2 = {lbl: r2_score(y49, c[day49_idx]) for lbl, c in zip(labels, oof_cols)}
print(">> single-model R²:", {k: round(v, 5) for k, v in single_r2.items()})

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
print(">> Ridge meta weights =", dict(zip(labels, np.round(meta.coef_, 4))))
v25_oof  = meta.predict(oof_stack)
v25_test = np.clip(meta.predict(test_stack), 0, 1)
print(f">> v25 raw OOF R² = {r2_score(y49, v25_oof):.5f}")

# apply v19b winning smoothing
def smooth(values, gh, ts, w, b):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    df["out"] = b * df["sm"] + (1 - b) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()

oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)

# Search the smoothing grid again with the new stack
print(">> smoothing grid on v25")
best_r2 = r2_score(y49, v25_oof); best_test = v25_test; best_desc = "raw"
for w in (2, 3, 4):
    for b in (0.5, 0.75, 1.0):
        oof_s  = smooth(v25_oof, oof_gh, oof_ts, w, b)
        test_s = np.clip(smooth(v25_test, test["geohash"], test["tslot"], w, b), 0, 1)
        r2 = r2_score(y49, oof_s)
        print(f"   w={w} b={b}: OOF R²={r2:.5f}")
        if r2 > best_r2:
            best_r2 = r2; best_test = test_s; best_desc = f"w{w}_b{b}"

print(f">> winner: {best_desc}  OOF R²={best_r2:.5f}")

sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())

np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    **{k: art[k] for k in art.files},
    oof_pseudo2=oof_pseudo2, pred_pseudo2=pred_pseudo2,
    meta_coef_v25=meta.coef_,
)
print(">> saved artifacts.npz")
