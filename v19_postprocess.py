"""
v19: Post-processing experiments on v16 (best leaderboard predictions).
Tries:
  (a) Restore v16 as baseline.
  (b) Smooth predictions per-geohash across nearby tslots (rolling mean).
  (c) Shrink each prediction toward the (gh, hour) mean (Bayesian regularization).
  (d) Calibrate output distribution to match d48 distribution shape.
Pick the variant with highest OOF R².
"""
import sys, io, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw, day49_idx = art["y_raw"], art["day49_idx"]
y49 = y_raw[day49_idx]
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def parse_tslot(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = parse_tslot(train["timestamp"])
test["tslot"]  = parse_tslot(test["timestamp"])
test["hour"]   = test["tslot"] // 4


# ---- (a) v16 baseline reconstruction ----
print(">> reconstructing v16 baseline")
labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"]]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"]]
oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
v16_oof_pred  = meta.predict(oof_stack)
v16_test_pred = np.clip(meta.predict(test_stack), 0.0, 1.0)
v16_r2 = r2_score(y49, v16_oof_pred)
print(f"   v16 OOF R² = {v16_r2:.5f}")


# Build full per-row OOF and test arrays for processing
oof_full = np.full(len(train), np.nan)
oof_full[day49_idx] = v16_oof_pred
test_full = v16_test_pred.copy()


# ---- (b) per-geohash temporal smoothing on test ----
print(">> (b) per-geohash temporal smoothing (rolling 3 / 5)")
test_w = test[["Index", "geohash", "tslot"]].copy()
test_w["pred"] = test_full

def smooth_test(window):
    tmp = test_w.sort_values(["geohash", "tslot"]).copy()
    tmp["smoothed"] = tmp.groupby("geohash")["pred"].transform(
        lambda s: s.rolling(window, center=True, min_periods=1).mean()
    )
    # blend with original 50/50
    return tmp.sort_index().set_index("Index").reindex(test_w["Index"])["smoothed"].to_numpy()

# We can ONLY evaluate smoothing impact through some held-out signal. Day-49 train is at
# tslots 0..8 (morning); we can simulate smoothing on the day-49 train OOF and check R².
train_w = train.loc[day49_idx, ["geohash", "tslot"]].copy()
train_w["pred"] = v16_oof_pred
train_w = train_w.reset_index().rename(columns={"index": "_orig"})

def smooth_oof(window):
    tmp = train_w.sort_values(["geohash", "tslot"]).copy()
    tmp["smoothed"] = tmp.groupby("geohash")["pred"].transform(
        lambda s: s.rolling(window, center=True, min_periods=1).mean()
    )
    return tmp.sort_values("_orig")["smoothed"].to_numpy()

for w in (3, 5):
    sm = smooth_oof(w)
    r2 = r2_score(y49, sm)
    print(f"   smooth w={w} (test-style, applied to OOF): OOF R²={r2:.5f}")


# ---- (c) shrinkage toward (gh, hour) mean ----
print(">> (c) shrink each prediction toward gh×hour mean (alpha grid)")
# Compute (gh, hour) mean from test predictions
test_w["hour"] = test["hour"].values
gh_hour_mean_pred = test_w.groupby(["geohash", "hour"])["pred"].transform("mean")
shrunk_grid = {}
for alpha in (0.0, 0.1, 0.2, 0.3, 0.5):
    shrunk = (1 - alpha) * test_full + alpha * gh_hour_mean_pred.values
    # For OOF, do the same: shrink v16_oof_pred toward gh×hour mean in train
    tw = train.loc[day49_idx, ["geohash"]].copy()
    tw["hour"] = train.loc[day49_idx, "tslot"].values // 4
    tw["pred"] = v16_oof_pred
    gh_hour_mean_oof = tw.groupby(["geohash", "hour"])["pred"].transform("mean")
    shrunk_oof = (1 - alpha) * v16_oof_pred + alpha * gh_hour_mean_oof.values
    r2 = r2_score(y49, shrunk_oof)
    shrunk_grid[alpha] = (r2, shrunk)
    print(f"   alpha={alpha}: OOF R²={r2:.5f}")


# ---- Pick best ----
candidates = {"v16_raw": (v16_r2, v16_test_pred)}
for alpha, (r2, pred) in shrunk_grid.items():
    candidates[f"shrink_a{alpha}"] = (r2, pred)

best = max(candidates, key=lambda k: candidates[k][0])
print(f">> winner: {best}  OOF R²={candidates[best][0]:.5f}")

pred_final = np.clip(candidates[best][1], 0.0, 1.0)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred_final})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())
