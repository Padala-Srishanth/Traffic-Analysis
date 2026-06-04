"""
v27: SVD/NMF matrix completion + KMeans clusters + multiplicative scaling.
- Build the (geohash × tslot) demand matrix for day 48
- Fit low-rank factorisation: M48 ≈ U48 · V
- For day 49: solve for U_49 using ONLY known morning tslots 0..8
- Reconstruct day-49 demand for tslots 9..95 as M_recon = U_49 · V
- Also: KMeans on (lat, lon) → cluster features; multiplicative scaling
- All added to Ridge stack; final smoothing applied.
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
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


print(">> loading data")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def parse_tslot(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = parse_tslot(train["timestamp"])
test["tslot"]  = parse_tslot(test["timestamp"])

# Decode geohash for clustering
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

all_gh = sorted(set(train["geohash"]).union(set(test["geohash"])))
gh_to_idx = {g: i for i, g in enumerate(all_gh)}
N_GH = len(all_gh)
gh_lat_arr = np.array([decode(g)[0] for g in all_gh])
gh_lon_arr = np.array([decode(g)[1] for g in all_gh])
print(f"   {N_GH} unique geohashes")

# Build day-48 demand matrix (1249 × 96). Fill missing with geohash mean.
print(">> building day-48 demand matrix")
M48 = np.full((N_GH, 96), np.nan, dtype=np.float32)
d48 = train[train["day"] == 48]
M48[d48["geohash"].map(gh_to_idx).to_numpy(), d48["tslot"].to_numpy()] = d48["demand"].to_numpy(dtype=np.float32)
gh_d48_mean = np.nanmean(M48, axis=1)
gh_d48_mean = np.where(np.isnan(gh_d48_mean), np.nanmean(gh_d48_mean), gh_d48_mean)
# fill missing entries with that geohash's mean
M48_filled = M48.copy()
for i in range(N_GH):
    mask = np.isnan(M48_filled[i])
    M48_filled[i, mask] = gh_d48_mean[i]
print(f"   M48 shape={M48_filled.shape}  missing pre-fill: {np.isnan(M48).sum()}")

# Build day-49 matrix (mostly empty; only tslots 0..8 filled)
M49_known = np.full((N_GH, 96), np.nan, dtype=np.float32)
d49 = train[train["day"] == 49]
M49_known[d49["geohash"].map(gh_to_idx).to_numpy(), d49["tslot"].to_numpy()] = d49["demand"].to_numpy(dtype=np.float32)
print(f"   M49 known entries: {(~np.isnan(M49_known)).sum()} / {N_GH * 96}")


# --------------------------------------------------------------------------- #
# SVD low-rank reconstruction                                                 #
# --------------------------------------------------------------------------- #
def svd_reconstruct(rank):
    print(f">> SVD rank={rank}")
    svd = TruncatedSVD(n_components=rank, random_state=42)
    U48 = svd.fit_transform(M48_filled)          # (N_GH, rank)
    V = svd.components_                          # (rank, 96)
    # solve for U_49 using only tslots 0..8 per gh: U_49 = M49_known[:, 0:9] @ pinv(V[:, 0:9])
    V_known = V[:, :9]                            # (rank, 9)
    M_known = M49_known[:, :9]                    # (N_GH, 9)
    # fill NaN in M_known with M48 morning average per geohash so the projection is stable
    M_known_filled = np.where(np.isnan(M_known), M48_filled[:, :9], M_known)
    # Solve U_49 @ V_known = M_known for U_49: U_49 = M_known @ V_known.T @ (V_known @ V_known.T)^-1
    VVt = V_known @ V_known.T + 1e-3 * np.eye(rank)    # (rank, rank)
    U_49 = M_known_filled @ V_known.T @ np.linalg.inv(VVt)   # (N_GH, rank)
    M49_recon = U_49 @ V                                # (N_GH, 96)
    return np.clip(M49_recon, 0.0, 1.0)

# Try several ranks; will pick the best on OOF later
svd_variants = {f"svd_r{r}": svd_reconstruct(r) for r in (3, 5, 10, 15, 25)}


# --------------------------------------------------------------------------- #
# NMF (non-negative matrix factorisation)                                     #
# --------------------------------------------------------------------------- #
print(">> NMF rank=10")
nmf = NMF(n_components=10, init="nndsvd", random_state=42, max_iter=400)
W48 = nmf.fit_transform(np.clip(M48_filled, 0, 1))      # (N_GH, 10)
H = nmf.components_                                       # (10, 96)
H_known = H[:, :9]
M_known_filled = np.where(np.isnan(M49_known[:, :9]), M48_filled[:, :9], M49_known[:, :9])
pinv_Hknown = np.linalg.pinv(H_known @ H_known.T + 1e-3 * np.eye(10)) @ H_known
W_49 = M_known_filled @ pinv_Hknown.T
nmf_recon = np.clip(W_49 @ H, 0, 1)
svd_variants["nmf_r10"] = nmf_recon


# --------------------------------------------------------------------------- #
# Multiplicative scaling forecast                                             #
# --------------------------------------------------------------------------- #
print(">> multiplicative scaling forecast")
d48_morning = np.nanmean(M48[:, :9], axis=1)
d49_morning = np.nanmean(M49_known[:, :9], axis=1)
# replace NaN/inf
d48_morning = np.where(np.isnan(d48_morning) | (d48_morning <= 1e-6), 1e-3, d48_morning)
d49_morning = np.where(np.isnan(d49_morning), d48_morning, d49_morning)
scale = d49_morning / d48_morning           # per geohash scalar
scale = np.clip(scale, 0.1, 10.0)
print(f"   scale: mean={scale.mean():.3f}, std={scale.std():.3f}")
# Multiplicative forecast for day-49 full matrix
M49_scale = M48_filled * scale[:, None]
M49_scale = np.clip(M49_scale, 0, 1)


# --------------------------------------------------------------------------- #
# KMeans cluster features on lat/lon                                          #
# --------------------------------------------------------------------------- #
print(">> KMeans clusters on lat/lon")
coords = np.column_stack([gh_lat_arr, gh_lon_arr])
gh_cluster = {}
for K in (50, 100):
    km = KMeans(n_clusters=K, random_state=42, n_init=5)
    labels_k = km.fit_predict(coords)
    gh_cluster[K] = labels_k


# --------------------------------------------------------------------------- #
# Build per-row OOF and test predictions for each variant                     #
# --------------------------------------------------------------------------- #
print(">> mapping variants to row-level OOF and test predictions")

day49_idx_train = np.where(train["day"].values == 49)[0]

def variant_to_rows(matrix: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    gh_idx = df["geohash"].map(gh_to_idx).to_numpy()
    ts = df["tslot"].to_numpy()
    return matrix[gh_idx, ts].astype(np.float32)

variant_oof  = {}
variant_test = {}
for key, mat in {**svd_variants, "mul_scale": M49_scale}.items():
    oof_arr = np.full(len(train), np.nan, dtype=np.float32)
    oof_arr[day49_idx_train] = variant_to_rows(mat, train.iloc[day49_idx_train])
    variant_oof[key]  = oof_arr
    variant_test[key] = variant_to_rows(mat, test)


# Honest R² for each variant on day-49 OOF
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw, day49_idx = art["y_raw"], art["day49_idx"]
y49 = y_raw[day49_idx]

print(">> single variant OOF R² on day-49:")
for key in variant_oof:
    print(f"   {key}: R²={r2_score(y49, variant_oof[key][day49_idx]):.5f}")


# --------------------------------------------------------------------------- #
# Refit Ridge over base models + these new variants                           #
# --------------------------------------------------------------------------- #
labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"]]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"]]

for key in variant_oof:
    labels.append(key)
    oof_cols.append(variant_oof[key])
    test_cols.append(variant_test[key])

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
v27_oof  = meta.predict(oof_stack)
v27_test = np.clip(meta.predict(test_stack), 0, 1)
print(">> Ridge meta weights =", dict(zip(labels, np.round(meta.coef_, 4))))
print(f">> v27 raw OOF R² = {r2_score(y49, v27_oof):.5f}")


# Apply v19b winning smoothing
def smooth(values, gh, ts, w, b):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    df["out"] = b * df["sm"] + (1 - b) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()

oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)

best_r2 = r2_score(y49, v27_oof); best_test = v27_test; best_desc = "raw"
for w in (2, 3, 4):
    for b in (0.5, 0.75, 1.0):
        oof_s = smooth(v27_oof, oof_gh, oof_ts, w, b)
        test_s = np.clip(smooth(v27_test, test["geohash"], test["tslot"], w, b), 0, 1)
        r2 = r2_score(y49, oof_s)
        print(f"   w={w} b={b}: OOF R²={r2:.5f}")
        if r2 > best_r2:
            best_r2 = r2; best_test = test_s; best_desc = f"w{w}_b{b}"

print(f">> winner: {best_desc}  OOF R²={best_r2:.5f}")
sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())
