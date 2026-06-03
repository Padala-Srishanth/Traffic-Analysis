"""
v23: Adversarial validation + importance weighting + final smoothed Ridge stack.
1. Train a LightGBM classifier to distinguish train rows from test rows.
2. For each train row, compute P(looks like test) via out-of-fold probability.
3. Use these probabilities as sample weights in a CatBoost regressor.
4. Add as a stack member; refit Ridge with smoothing.
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from catboost import CatBoostRegressor
import lightgbm as lgb

RNG = 42

print(">> loading data")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def parse_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return h.astype(int), m.astype(int), (h * 60 + m).astype(int), (h * 4 + m // 15).astype(int)
train["hour"], train["minute"], train["tmin"], train["tslot"] = parse_ts(train["timestamp"])
test["hour"],  test["minute"],  test["tmin"],  test["tslot"]  = parse_ts(test["timestamp"])

# feature engineering (minimal)
d48 = train[train["day"] == 48][["geohash", "tslot", "demand"]]
d48_map = d48.set_index(["geohash", "tslot"])["demand"]
gh_d48_mean = d48.groupby("geohash")["demand"].mean()
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


# --------------------------------------------------------------------------- #
# STEP 1: adversarial classifier — predict P(is_test | features)              #
# --------------------------------------------------------------------------- #
CAT = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM = ["day", "hour", "minute", "tmin", "tslot", "hour_sin", "hour_cos",
       "lat", "lon", "NumberofLanes", "Temperature", "baseline", "gh_d48_mean"]
FEAT = CAT + NUM

X_full = pd.concat([train[FEAT].copy(), test[FEAT].copy()], ignore_index=True)
for c in CAT:
    X_full[c] = X_full[c].astype("string").fillna("NA")
    codes, _ = pd.factorize(X_full[c], sort=True)
    X_full[c] = codes
X_full = X_full.astype(np.float32)
X_tr = X_full.iloc[:len(train)].reset_index(drop=True)
X_te = X_full.iloc[len(train):].reset_index(drop=True)

is_test = np.concatenate([np.zeros(len(train)), np.ones(len(test))]).astype(np.int64)
X_combined = pd.concat([X_tr, X_te], ignore_index=True)

print(">> training adversarial classifier (train vs test)")
adv_oof = np.zeros(len(X_combined))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
for fold, (tr_i, va_i) in enumerate(skf.split(X_combined, is_test), 1):
    m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.85, colsample_bytree=0.85,
        device="gpu", random_state=RNG, n_jobs=-1, verbose=-1,
    )
    m.fit(X_combined.iloc[tr_i], is_test[tr_i],
          eval_set=[(X_combined.iloc[va_i], is_test[va_i])],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    adv_oof[va_i] = m.predict_proba(X_combined.iloc[va_i])[:, 1]
    auc = roc_auc_score(is_test[va_i], adv_oof[va_i])
    print(f"   adv fold {fold}: AUC={auc:.4f}")
adv_auc = roc_auc_score(is_test, adv_oof)
print(f">> adversarial classifier OOF AUC = {adv_auc:.4f}  (>0.5 → distinguishable)")

# train rows that LOOK like test (high P(is_test)) get higher weight
prob_train_is_test = adv_oof[:len(train)]
print(f"   train P(is_test): mean={prob_train_is_test.mean():.4f}, "
      f"std={prob_train_is_test.std():.4f}, min={prob_train_is_test.min():.4f}, max={prob_train_is_test.max():.4f}")

# rescale to [0.2, 5.0]
w_low, w_high = 0.2, 5.0
p_norm = (prob_train_is_test - prob_train_is_test.min()) / (prob_train_is_test.max() - prob_train_is_test.min() + 1e-9)
adv_weight = w_low + (w_high - w_low) * p_norm
# also bake in original "day-49 heavy" weighting
adv_weight[train["day"].values == 49] = adv_weight[train["day"].values == 49] * 2.0
print(f"   adv_weight: mean={adv_weight.mean():.3f}, max={adv_weight.max():.3f}")


# --------------------------------------------------------------------------- #
# STEP 2: importance-weighted CatBoost regressor                              #
# --------------------------------------------------------------------------- #
print(">> training importance-weighted CatBoost (5-fold day-49 honest CV)")
day49_idx = np.where(train["day"].values == 49)[0]
day48_idx = np.where(train["day"].values == 48)[0]

# CatBoost needs the original (un-encoded) categoricals
X_train_cb = train[FEAT].copy(); X_test_cb = test[FEAT].copy()
for c in CAT:
    X_train_cb[c] = X_train_cb[c].astype("string").fillna("NA")
    X_test_cb[c]  = X_test_cb[c].astype("string").fillna("NA")
cat_idx = [FEAT.index(c) for c in CAT]

y_train = train["demand"].astype(float).values
oof_adv = np.full(len(train), np.nan)
pred_adv = np.zeros(len(test))

cb_params = dict(
    iterations=4000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
    loss_function="RMSE", eval_metric="RMSE",
    random_seed=RNG, od_type="Iter", od_wait=200,
    verbose=500, task_type="GPU", devices="0",
)
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
    tr_idx = np.concatenate([day48_idx, day49_idx[tr_d49]])
    va_idx = day49_idx[va_d49]
    m = CatBoostRegressor(**cb_params)
    m.fit(X_train_cb.iloc[tr_idx], y_train[tr_idx],
          sample_weight=adv_weight[tr_idx],
          eval_set=(X_train_cb.iloc[va_idx], y_train[va_idx]),
          cat_features=cat_idx, use_best_model=True)
    oof_adv[va_idx] = m.predict(X_train_cb.iloc[va_idx])
    pred_adv += m.predict(X_test_cb) / kf.n_splits
    print(f"   fold {fold}: best_iter={m.get_best_iteration()}  R²={r2_score(y_train[va_idx], oof_adv[va_idx]):.5f}")

adv_r2 = r2_score(y_train[day49_idx], oof_adv[day49_idx])
print(f">> adv-weighted CatBoost OOF R² = {adv_r2:.5f}")


# --------------------------------------------------------------------------- #
# STEP 3: refit Ridge over base + adv model, then apply temporal smoothing    #
# --------------------------------------------------------------------------- #
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw = art["y_raw"]; day49_idx_art = art["day49_idx"]
y49 = y_raw[day49_idx]

labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn","adv"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"], oof_adv]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"], pred_adv]

single_r2 = {lbl: r2_score(y49, c[day49_idx]) for lbl, c in zip(labels, oof_cols)}
print(">> single-model R²:", {k: round(v, 5) for k, v in single_r2.items()})

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
v23_oof  = meta.predict(oof_stack)
v23_test = np.clip(meta.predict(test_stack), 0, 1)
print(f">> Ridge meta weights = {dict(zip(labels, np.round(meta.coef_, 4)))}")
print(f">> v23 raw OOF R² = {r2_score(y49, v23_oof):.5f}")


def smooth(values, gh, ts, window, blend):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    df["out"] = blend * df["sm"] + (1 - blend) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()


oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)

# Apply the v19b winning smoothing (w=3, b=0.75)
v23_oof_sm  = smooth(v23_oof, oof_gh, oof_ts, 3, 0.75)
v23_test_sm = np.clip(smooth(v23_test, test["geohash"], test["tslot"], 3, 0.75), 0, 1)
v23_sm_r2 = r2_score(y49, v23_oof_sm)
print(f">> v23 + smooth(w=3,b=0.75) OOF R² = {v23_sm_r2:.5f}")

sub = pd.DataFrame({"Index": test["Index"].values, "demand": v23_test_sm})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())

np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    **{k: art[k] for k in art.files},
    oof_adv=oof_adv, pred_adv=pred_adv,
    meta_coef_v23=meta.coef_,
)
print(">> saved artifacts.npz")
