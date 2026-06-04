"""
v26: 8-order Fourier harmonics + autoregressive d49 rollout.
- Trains a CatBoost with K=1..8 Fourier features for hour and tslot.
- At inference, predicts test in TSLOT ORDER: each predicted demand becomes
  the new d49_recent for later tslots (autoregressive rollout per geohash).
- Adds as a stack member; refits Ridge; applies v19b winning smoothing.
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

print(">> loading data")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def parse_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return h.astype(int), m.astype(int), (h * 60 + m).astype(int), (h * 4 + m // 15).astype(int)

train["hour"], train["minute"], train["tmin"], train["tslot"] = parse_ts(train["timestamp"])
test["hour"],  test["minute"],  test["tmin"],  test["tslot"]  = parse_ts(test["timestamp"])

# ---------- minimal feature engineering ----------
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

train["d48_same_ts"] = fill_baseline(train)
test["d48_same_ts"]  = fill_baseline(test)
train["gh_d48_mean"] = train["geohash"].map(gh_d48_mean).fillna(overall_d48_mean).astype(np.float32)
test["gh_d48_mean"]  = test["geohash"].map(gh_d48_mean).fillna(overall_d48_mean).astype(np.float32)
train["gh_d48_std"]  = train["geohash"].map(gh_d48_std).fillna(0.0).astype(np.float32)
test["gh_d48_std"]   = test["geohash"].map(gh_d48_std).fillna(0.0).astype(np.float32)

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

for k in (3, 4, 5):
    train[f"gh{k}"] = train["geohash"].str.slice(0, k)
    test[f"gh{k}"]  = test["geohash"].str.slice(0, k)


# ---------- 1) 8-order Fourier features for hour and tslot ----------
print(">> adding 8-order Fourier features")
fourier_cols = []
for k in range(1, 9):
    train[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * train["hour"] / 24)
    train[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * train["hour"] / 24)
    train[f"tslot_sin_{k}"] = np.sin(2 * np.pi * k * train["tslot"] / 96)
    train[f"tslot_cos_{k}"] = np.cos(2 * np.pi * k * train["tslot"] / 96)
    test[f"hour_sin_{k}"]  = np.sin(2 * np.pi * k * test["hour"] / 24)
    test[f"hour_cos_{k}"]  = np.cos(2 * np.pi * k * test["hour"] / 24)
    test[f"tslot_sin_{k}"] = np.sin(2 * np.pi * k * test["tslot"] / 96)
    test[f"tslot_cos_{k}"] = np.cos(2 * np.pi * k * test["tslot"] / 96)
    fourier_cols += [f"hour_sin_{k}", f"hour_cos_{k}", f"tslot_sin_{k}", f"tslot_cos_{k}"]
print(f"   added {len(fourier_cols)} Fourier features")


# ---------- 2) build d49_recent (strict less, for training; AR for test) ----------
d49 = train[train["day"] == 49][["geohash", "tslot", "demand"]].rename(columns={"demand": "d49_recent"}).sort_values("tslot").reset_index(drop=True)
def attach_d49_recent(df):
    src = df[["geohash", "tslot"]].copy()
    src["_orig"] = np.arange(len(src))
    src = src.sort_values("tslot")
    merged = pd.merge_asof(src, d49, on="tslot", by="geohash", direction="backward", allow_exact_matches=False)
    merged = merged.sort_values("_orig")
    return merged["d49_recent"].to_numpy()

train["d49_recent"] = attach_d49_recent(train)
# we'll fill test["d49_recent"] DURING autoregressive inference; placeholder for now
test["d49_recent"]  = attach_d49_recent(test)
train["d49_minus_d48"] = train["d49_recent"] - train["d48_same_ts"]
test["d49_minus_d48"]  = test["d49_recent"]  - test["d48_same_ts"]


# ---------- feature columns ----------
CAT = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM_BASE = ["day", "hour", "minute", "tmin", "tslot",
            "lat", "lon", "NumberofLanes", "Temperature",
            "d48_same_ts", "gh_d48_mean", "gh_d48_std",
            "d49_recent", "d49_minus_d48"]
FEAT = CAT + NUM_BASE + fourier_cols
for c in CAT:
    train[c] = train[c].astype("string").fillna("NA")
    test[c]  = test[c].astype("string").fillna("NA")


# ---------- train CatBoost with 5-fold day-49 honest CV ----------
print(f">> training CatBoost on {len(FEAT)} features (5-fold day-49 CV, GPU)")
day49_idx = np.where(train["day"].values == 49)[0]
day48_idx = np.where(train["day"].values == 48)[0]
sample_weight = np.ones(len(train), dtype=np.float32)
sample_weight[(train["day"].values == 48) & (train["tslot"].values >= 9)] = 1.5
sample_weight[train["day"].values == 49] = 5.0

cb_params = dict(
    iterations=4000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
    loss_function="RMSE", eval_metric="RMSE",
    random_seed=RNG, od_type="Iter", od_wait=200,
    verbose=500, task_type="GPU", devices="0",
)
cat_idx = [FEAT.index(c) for c in CAT]
y_train = train["demand"].astype(float).to_numpy()

# Train one model per fold for OOF, then re-train on ALL train for AR inference
oof = np.full(len(train), np.nan)
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
    tr_idx = np.concatenate([day48_idx, day49_idx[tr_d49]])
    va_idx = day49_idx[va_d49]
    m = CatBoostRegressor(**cb_params)
    m.fit(train.iloc[tr_idx][FEAT], y_train[tr_idx],
          sample_weight=sample_weight[tr_idx],
          eval_set=(train.iloc[va_idx][FEAT], y_train[va_idx]),
          cat_features=cat_idx, use_best_model=True)
    oof[va_idx] = m.predict(train.iloc[va_idx][FEAT])
    print(f"   fold {fold}: best_iter={m.get_best_iteration()}  R²={r2_score(y_train[va_idx], oof[va_idx]):.5f}")

oof_r2 = r2_score(y_train[day49_idx], oof[day49_idx])
print(f">> Fourier CatBoost OOF R² (day-49) = {oof_r2:.5f}")


# ---------- re-train on ALL train, then AR-rollout for test ----------
print(">> retraining on full train for AR inference")
final_model = CatBoostRegressor(**{**cb_params, "iterations": 3500})  # use mean of fold best
final_model.fit(train[FEAT], y_train, sample_weight=sample_weight, cat_features=cat_idx)
print(">> autoregressive rollout for test predictions")

gh_to_idx = {g: i for i, g in enumerate(all_gh)}
N_GH = len(all_gh)

# known_d49[gh_idx, tslot] holds the demand value (real for tslots 0..8, predicted for 9..95)
known_d49 = np.full((N_GH, 96), np.nan, dtype=np.float32)
d49_train = train[train["day"] == 49]
known_d49[d49_train["geohash"].map(gh_to_idx).to_numpy(), d49_train["tslot"].to_numpy()] = \
    d49_train["demand"].to_numpy(dtype=np.float32)

test_pred = np.zeros(len(test), dtype=np.float32)
test_gh_idx = test["geohash"].map(gh_to_idx).to_numpy()

t0 = time.time()
for ts in range(9, 96):
    mask = test["tslot"].values == ts
    if mask.sum() == 0:
        continue
    sub = test[mask].copy().reset_index().rename(columns={"index": "_orig"})
    sub_gh_idx = test_gh_idx[mask]

    # recompute d49_recent for these rows from the current known_d49
    history = known_d49[sub_gh_idx, :ts]  # shape (n, ts)
    # last non-NaN value along axis=1
    valid = ~np.isnan(history)
    has_any = valid.any(axis=1)
    # index of last valid: argmax of valid[::-1] then flip
    rev_valid = valid[:, ::-1]
    rev_first = rev_valid.argmax(axis=1)
    last_idx = ts - 1 - rev_first
    d49_recent_vals = np.where(has_any, history[np.arange(len(history)), last_idx], np.nan)
    sub["d49_recent"] = d49_recent_vals.astype(np.float32)
    sub["d49_minus_d48"] = sub["d49_recent"] - sub["d48_same_ts"]

    preds = final_model.predict(sub[FEAT])
    preds = np.clip(preds, 0.0, 1.0).astype(np.float32)

    # write predictions and feed into known_d49 for later tslots
    orig_idx = sub["_orig"].to_numpy()
    test_pred[orig_idx] = preds
    known_d49[sub_gh_idx, ts] = preds

    if ts % 10 == 0 or ts == 95:
        print(f"   ts={ts}  rows={mask.sum()}  elapsed={time.time()-t0:.1f}s")

print(">> autoregressive rollout done")


# ---------- combine with prior 10-model stack ----------
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw = art["y_raw"]; day49_idx_art = art["day49_idx"]
y49 = y_raw[day49_idx]

labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn","ar_fourier"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"], oof]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"], test_pred]

single_r2 = {lbl: r2_score(y49, c[day49_idx]) for lbl, c in zip(labels, oof_cols)}
print(">> single-model R²:", {k: round(v, 5) for k, v in single_r2.items()})

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
print(">> Ridge meta weights =", dict(zip(labels, np.round(meta.coef_, 4))))
v26_oof  = meta.predict(oof_stack)
v26_test = np.clip(meta.predict(test_stack), 0, 1)
print(f">> v26 raw OOF R² = {r2_score(y49, v26_oof):.5f}")


# ---------- apply v19b-winning smoothing ----------
def smooth(values, gh, ts, w, b):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    df["out"] = b * df["sm"] + (1 - b) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()

oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)

best_r2 = r2_score(y49, v26_oof); best_test = v26_test; best_desc = "raw"
for w in (2, 3, 4):
    for b in (0.5, 0.75, 1.0):
        oof_s = smooth(v26_oof, oof_gh, oof_ts, w, b)
        test_s = np.clip(smooth(v26_test, test["geohash"], test["tslot"], w, b), 0, 1)
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
    oof_ar_fourier=oof, pred_ar_fourier=test_pred,
    meta_coef_v26=meta.coef_,
)
print(">> saved artifacts.npz")
