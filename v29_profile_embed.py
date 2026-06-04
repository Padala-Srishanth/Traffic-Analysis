"""
v29: Day-48 profile embeddings + behavioural-similarity features.
Each geohash gets:
  - A 32-dim SVD embedding of its normalised day-48 96-slot demand curve
  - Profile-similarity (cosine) 5 nearest neighbour geohashes
  - Neighbour-based ratio forecast features (their d49/d48 morning ratio)
Trains CatBoost with these added features; adds to the Ridge stack; smooths.
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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
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


# ------------------- Build day-48 profile matrix ------------------- #
all_gh = sorted(set(train["geohash"]).union(set(test["geohash"])))
gh_to_idx = {g: i for i, g in enumerate(all_gh)}
N_GH = len(all_gh)
print(f"   {N_GH} unique geohashes")

d48 = train[train["day"] == 48]
d49 = train[train["day"] == 49]

M48 = np.full((N_GH, 96), np.nan, dtype=np.float32)
M48[d48["geohash"].map(gh_to_idx).to_numpy(), d48["tslot"].to_numpy()] = d48["demand"].to_numpy(dtype=np.float32)
gh_mean48 = np.nanmean(M48, axis=1)
gh_mean48 = np.where(np.isnan(gh_mean48), np.nanmean(gh_mean48), gh_mean48)
M48_filled = M48.copy()
for i in range(N_GH):
    M48_filled[i, np.isnan(M48_filled[i])] = gh_mean48[i]

# normalised profile (divide row by row mean → "shape" matters)
row_mean = M48_filled.mean(axis=1).reshape(-1, 1)
row_mean[row_mean < 1e-6] = 1e-6
M48_norm = M48_filled / row_mean
print(f">> normalised day-48 profile matrix shape {M48_norm.shape}")


# ------------------- SVD embedding ------------------- #
print(">> computing 32-dim SVD embeddings of day-48 profiles")
svd = TruncatedSVD(n_components=32, random_state=RNG)
gh_emb = svd.fit_transform(M48_norm)             # (N_GH, 32)
print(f"   SVD explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.4f}")


# ------------------- Profile k-NN ------------------- #
print(">> finding profile-similarity 5 nearest neighbours (cosine)")
nn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
nn.fit(M48_norm)
_, neigh_idx = nn.kneighbors(M48_norm)          # (N_GH, 6) — first column is self
neigh_idx = neigh_idx[:, 1:]                     # exclude self → (N_GH, 5)


# ------------------- Per-geohash day-49 morning ratio ------------------- #
M49 = np.full((N_GH, 96), np.nan, dtype=np.float32)
M49[d49["geohash"].map(gh_to_idx).to_numpy(), d49["tslot"].to_numpy()] = d49["demand"].to_numpy(dtype=np.float32)
d48_morning = np.nanmean(M48_filled[:, :9], axis=1)
d49_morning = np.nanmean(M49[:, :9], axis=1)
d48_morning = np.where(d48_morning < 1e-6, 1e-6, d48_morning)
self_ratio = d49_morning / d48_morning  # NaN if no d49 known


# Neighbour ratio = mean of own-ratio across 5 profile neighbours (using their d49/d48 if known)
neigh_ratio_vals = self_ratio[neigh_idx]   # (N_GH, 5)
neigh_ratio_mean = np.nanmean(neigh_ratio_vals, axis=1)
neigh_ratio_std  = np.nanstd(neigh_ratio_vals, axis=1)
neigh_ratio_max  = np.nanmax(neigh_ratio_vals, axis=1)
neigh_ratio_min  = np.nanmin(neigh_ratio_vals, axis=1)
# fill NaN with global mean
global_ratio = np.nanmean(self_ratio)
neigh_ratio_mean = np.where(np.isnan(neigh_ratio_mean), global_ratio, neigh_ratio_mean)
neigh_ratio_std  = np.nan_to_num(neigh_ratio_std, nan=0.0)
neigh_ratio_max  = np.where(np.isnan(neigh_ratio_max), global_ratio, neigh_ratio_max)
neigh_ratio_min  = np.where(np.isnan(neigh_ratio_min), global_ratio, neigh_ratio_min)


# ------------------- Add features to train/test ------------------- #
gh_idx_train = train["geohash"].map(gh_to_idx).to_numpy()
gh_idx_test  = test["geohash"].map(gh_to_idx).to_numpy()

for k in range(32):
    train[f"emb_{k}"] = gh_emb[gh_idx_train, k].astype(np.float32)
    test[f"emb_{k}"]  = gh_emb[gh_idx_test, k].astype(np.float32)
emb_cols = [f"emb_{k}" for k in range(32)]

for name, arr in [("nb_ratio_mean", neigh_ratio_mean),
                  ("nb_ratio_std",  neigh_ratio_std),
                  ("nb_ratio_max",  neigh_ratio_max),
                  ("nb_ratio_min",  neigh_ratio_min)]:
    train[name] = arr[gh_idx_train].astype(np.float32)
    test[name]  = arr[gh_idx_test].astype(np.float32)
nb_cols = ["nb_ratio_mean", "nb_ratio_std", "nb_ratio_max", "nb_ratio_min"]


# Standard features
d48_lookup = d48.set_index(["geohash", "tslot"])["demand"]
def fill_baseline(df):
    keys = list(zip(df["geohash"], df["tslot"]))
    return pd.Series(keys, index=df.index).map(d48_lookup)\
            .fillna(df["geohash"].map(d48.groupby("geohash")["demand"].mean()))\
            .fillna(float(d48["demand"].mean()))\
            .astype(np.float32).to_numpy()
train["d48_same_ts"] = fill_baseline(train)
test["d48_same_ts"]  = fill_baseline(test)

gh_d48_stats = d48.groupby("geohash")["demand"].agg(["mean","std"])
gh_d48_stats.columns = ["gh_d48_mean","gh_d48_std"]
train = train.merge(gh_d48_stats, left_on="geohash", right_index=True, how="left")
test  = test.merge(gh_d48_stats,  left_on="geohash", right_index=True, how="left")
train["gh_d48_mean"] = train["gh_d48_mean"].fillna(d48["demand"].mean()).astype(np.float32)
test["gh_d48_mean"]  = test["gh_d48_mean"].fillna(d48["demand"].mean()).astype(np.float32)
train["gh_d48_std"]  = train["gh_d48_std"].fillna(0.0).astype(np.float32)
test["gh_d48_std"]   = test["gh_d48_std"].fillna(0.0).astype(np.float32)

# Profile-based forecast: d48_same_ts * nb_ratio_mean
train["nb_forecast"] = (train["d48_same_ts"] * train["nb_ratio_mean"]).astype(np.float32)
test["nb_forecast"]  = (test["d48_same_ts"]  * test["nb_ratio_mean"]).astype(np.float32)


# Geohash prefixes
for k in (3, 4, 5):
    train[f"gh{k}"] = train["geohash"].str.slice(0, k)
    test[f"gh{k}"]  = test["geohash"].str.slice(0, k)

train["hour_sin"] = np.sin(2*np.pi*train["hour"]/24); train["hour_cos"] = np.cos(2*np.pi*train["hour"]/24)
test["hour_sin"]  = np.sin(2*np.pi*test["hour"]/24);  test["hour_cos"]  = np.cos(2*np.pi*test["hour"]/24)

CAT = ["geohash", "gh3", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM = ["day","hour","minute","tmin","tslot","hour_sin","hour_cos",
       "NumberofLanes","Temperature",
       "d48_same_ts","gh_d48_mean","gh_d48_std",
       "nb_forecast"] + nb_cols + emb_cols
FEAT = CAT + NUM
for c in CAT:
    train[c] = train[c].astype("string").fillna("NA")
    test[c]  = test[c].astype("string").fillna("NA")
print(f">> feature count: {len(FEAT)}  (32 embeddings + 4 neighbour-ratio + 1 forecast)")


# ------------------- 5-fold CatBoost ------------------- #
print(">> training CatBoost on profile features (5-fold day-49 honest CV, GPU)")
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

oof = np.full(len(train), np.nan)
pred_test = np.zeros(len(test))
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
    pred_test += m.predict(test[FEAT]) / kf.n_splits
    fold_r2 = r2_score(y_train[va_idx], oof[va_idx])
    print(f"   fold {fold}: best_iter={m.get_best_iteration()}  R2={fold_r2:.5f}")

profile_oof_r2 = r2_score(y_train[day49_idx], oof[day49_idx])
print(f">> profile-CatBoost OOF R2 (day-49) = {profile_oof_r2:.5f}")


# ------------------- Refit Ridge over base + profile ------------------- #
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw = art["y_raw"]; day49_idx_art = art["day49_idx"]
y49 = y_raw[day49_idx]

labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn","profile"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"], oof]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"], pred_test]

single_r2 = {lbl: r2_score(y49, c[day49_idx]) for lbl, c in zip(labels, oof_cols)}
print(">> single-model R2:", {k: round(v, 5) for k, v in single_r2.items()})

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
print(">> Ridge weights:", dict(zip(labels, np.round(meta.coef_, 4))))
v29_oof  = meta.predict(oof_stack)
v29_test = np.clip(meta.predict(test_stack), 0, 1)
print(f">> v29 raw OOF R2 = {r2_score(y49, v29_oof):.5f}")


def smooth(values, gh, ts, w, b):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    df["out"] = b * df["sm"] + (1 - b) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()

oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)
best_r2 = r2_score(y49, v29_oof); best_test = v29_test; best_desc = "raw"
for w in (2,3,4):
    for b in (0.5,0.75,1.0):
        oof_s = smooth(v29_oof, oof_gh, oof_ts, w, b)
        test_s = np.clip(smooth(v29_test, test["geohash"], test["tslot"], w, b), 0, 1)
        r2 = r2_score(y49, oof_s)
        print(f"   w={w} b={b}: OOF R2={r2:.5f}")
        if r2 > best_r2:
            best_r2 = r2; best_test = test_s; best_desc = f"w{w}_b{b}"

print(f">> winner: {best_desc}  OOF R2={best_r2:.5f}")
sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())

np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    **{k: art[k] for k in art.files},
    oof_profile=oof, pred_profile=pred_test,
    meta_coef_v29=meta.coef_,
)
print(">> saved artifacts.npz")
