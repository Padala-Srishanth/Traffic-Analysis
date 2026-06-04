"""
v27b: Honest SVD evaluation. Leave-one-tslot-out: for each known day-49 tslot,
solve U_49 using the OTHER 8 known tslots, predict the held-out tslot.
This eliminates the OOF leak in v27 and tells us if SVD reconstruction
generalizes legitimately.
"""
import sys, io, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
def parse_tslot(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = parse_tslot(train["timestamp"])

all_gh = sorted(set(train["geohash"]).union(set(test["geohash"])))
gh_to_idx = {g: i for i, g in enumerate(all_gh)}
N_GH = len(all_gh)

# day-48 matrix (filled with gh mean for missing)
M48 = np.full((N_GH, 96), np.nan, dtype=np.float32)
d48 = train[train["day"] == 48]
M48[d48["geohash"].map(gh_to_idx).to_numpy(), d48["tslot"].to_numpy()] = d48["demand"].to_numpy(dtype=np.float32)
gh_mean = np.nanmean(M48, axis=1)
gh_mean = np.where(np.isnan(gh_mean), np.nanmean(gh_mean), gh_mean)
M48_filled = M48.copy()
for i in range(N_GH):
    M48_filled[i, np.isnan(M48_filled[i])] = gh_mean[i]

# day-49 morning known matrix
M49_known = np.full((N_GH, 96), np.nan, dtype=np.float32)
d49 = train[train["day"] == 49]
M49_known[d49["geohash"].map(gh_to_idx).to_numpy(), d49["tslot"].to_numpy()] = d49["demand"].to_numpy(dtype=np.float32)


print("=" * 60)
print("HONEST SVD: leave-one-tslot-out CV across morning tslots 0..8")
print("=" * 60)
for rank in (3, 5, 10, 15, 25):
    svd = TruncatedSVD(n_components=rank, random_state=42)
    U48 = svd.fit_transform(M48_filled)
    V = svd.components_  # rank × 96

    # For each held-out morning tslot, solve U_49 from the other 8 and predict held-out
    honest_preds = []
    honest_truths = []
    for held in range(9):
        other = [t for t in range(9) if t != held]
        V_other = V[:, other]                  # rank × 8
        M_other = M49_known[:, other]          # N_GH × 8
        M_other_filled = np.where(np.isnan(M_other), M48_filled[:, other], M_other)
        VVt = V_other @ V_other.T + 1e-3 * np.eye(rank)
        U_49 = M_other_filled @ V_other.T @ np.linalg.inv(VVt)    # N_GH × rank
        pred_held = U_49 @ V[:, held]                            # N_GH
        truth_held = M49_known[:, held]
        valid = ~np.isnan(truth_held)
        honest_preds.append(pred_held[valid])
        honest_truths.append(truth_held[valid])

    pred_all = np.concatenate(honest_preds)
    truth_all = np.concatenate(honest_truths)
    pred_all_clipped = np.clip(pred_all, 0, 1)
    r2 = r2_score(truth_all, pred_all_clipped)
    print(f"   SVD rank={rank}: HONEST OOF R² = {r2:.5f}  (count={len(truth_all)})")


print()
print("Comparison — LEAKY OOF (what v27 reported):")
for rank in (3, 5, 10, 15, 25):
    svd = TruncatedSVD(n_components=rank, random_state=42)
    U48 = svd.fit_transform(M48_filled)
    V = svd.components_
    V_known = V[:, :9]
    M_known = np.where(np.isnan(M49_known[:, :9]), M48_filled[:, :9], M49_known[:, :9])
    VVt = V_known @ V_known.T + 1e-3 * np.eye(rank)
    U_49 = M_known @ V_known.T @ np.linalg.inv(VVt)
    recon = U_49 @ V
    valid_mask = ~np.isnan(M49_known[:, :9])
    pred_eval = recon[:, :9][valid_mask]
    truth_eval = M49_known[:, :9][valid_mask]
    r2 = r2_score(truth_eval, np.clip(pred_eval, 0, 1))
    print(f"   SVD rank={rank}: LEAKY OOF R² = {r2:.5f}")
