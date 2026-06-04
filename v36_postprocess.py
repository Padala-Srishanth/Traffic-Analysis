"""
v36: combined post-processing — wider smoothing + quantile mapping + hierarchical shrinkage.
1. Try wider rolling windows (w=7,9,11,15)
2. Quantile mapping: match output distribution to day-48 demand distribution
3. Hierarchical shrinkage toward (gh, hour), gh5, city-hour means
Combine any helpers; pick best by OOF; only submit if real gain.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import rankdata

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw, day49_idx = art["y_raw"], art["day49_idx"]
y49 = y_raw[day49_idx]
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
def to_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = to_ts(train["timestamp"]); test["tslot"] = to_ts(test["timestamp"])
train["hour"] = train["tslot"] // 4; test["hour"] = test["tslot"] // 4

# build prefixes
train["gh5"] = train["geohash"].str.slice(0, 5)
test["gh5"]  = test["geohash"].str.slice(0, 5)

# v16 baseline (positive Ridge, alpha=0.1 — the safe one)
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"]]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"]]
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(np.column_stack([c[day49_idx] for c in oof_cols]), y49)
v16_oof  = meta.predict(np.column_stack([c[day49_idx] for c in oof_cols]))
v16_test = np.clip(meta.predict(np.column_stack(test_cols)), 0, 1)


def smooth(values, gh, ts, w, b):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    df["out"] = b * df["sm"] + (1 - b) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()


oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)

v19b_oof  = smooth(v16_oof, oof_gh, oof_ts, 3, 0.75)
v19b_test = np.clip(smooth(v16_test, test["geohash"], test["tslot"], 3, 0.75), 0, 1)
v19b_r2 = r2_score(y49, v19b_oof)
print(f">> v19b baseline OOF R² = {v19b_r2:.6f}")


# ============================================================
# Step 1: Wider smoothing windows
# ============================================================
print("\n=== Wider smoothing ===")
for w in (3, 5, 7, 9, 11, 15, 21):
    for b in (0.5, 0.75, 0.85, 1.0):
        oof_s = smooth(v16_oof, oof_gh, oof_ts, w, b)
        r2 = r2_score(y49, oof_s)
        marker = " <-- baseline" if (w==3 and b==0.75) else ""
        print(f"   w={w:2d} b={b:.2f}: OOF R²={r2:.6f}  d={r2-v19b_r2:+.6f}{marker}")


# ============================================================
# Step 2: Quantile mapping
# ============================================================
print("\n=== Quantile mapping (remap v19b dist to day-48 dist) ===")
d48_demand = train.loc[train["day"] == 48, "demand"].to_numpy()
d48_sorted = np.sort(d48_demand)

def quantile_remap(values):
    ranks = rankdata(values, method="average") / (len(values) + 1)
    idx = np.clip((ranks * len(d48_sorted)).astype(int), 0, len(d48_sorted) - 1)
    return d48_sorted[idx]

v19b_oof_remap = quantile_remap(v19b_oof)
print(f"   pure quantile remap OOF R²={r2_score(y49, v19b_oof_remap):.6f}  d={r2_score(y49, v19b_oof_remap)-v19b_r2:+.6f}")
for alpha in (0.1, 0.2, 0.3, 0.5):
    blend = (1 - alpha) * v19b_oof + alpha * v19b_oof_remap
    r2 = r2_score(y49, blend)
    print(f"   alpha={alpha}: OOF R²={r2:.6f}  d={r2-v19b_r2:+.6f}")


# ============================================================
# Step 3: Hierarchical shrinkage
# ============================================================
print("\n=== Hierarchical shrinkage ===")
# precompute aggregates from day-48
d48 = train[train["day"] == 48]
gh_hour_mean = d48.groupby(["geohash", "hour"])["demand"].mean()
gh5_hour_mean = d48.groupby(["gh5", "hour"])["demand"].mean()
city_hour_mean = d48.groupby("hour")["demand"].mean()
city_overall_mean = float(d48["demand"].mean())

def get_gh_hour(df):
    return df.set_index(["geohash", "hour"]).index.map(gh_hour_mean).to_series(index=df.index).fillna(city_overall_mean).astype(np.float32).to_numpy()
def get_gh5_hour(df):
    return df.set_index(["gh5", "hour"]).index.map(gh5_hour_mean).to_series(index=df.index).fillna(city_overall_mean).astype(np.float32).to_numpy()
def get_city_hour(df):
    return df["hour"].map(city_hour_mean).fillna(city_overall_mean).astype(np.float32).to_numpy()

# Compute these for day-49 train rows
oof_df = train.iloc[day49_idx].copy()
oof_gh_hour = get_gh_hour(oof_df)
oof_gh5_hour = get_gh5_hour(oof_df)
oof_city_hour = get_city_hour(oof_df)

# Try shrinkage weights
for (a, b, c, d_) in [(0.70, 0.15, 0.10, 0.05),
                       (0.80, 0.10, 0.07, 0.03),
                       (0.85, 0.08, 0.05, 0.02),
                       (0.90, 0.05, 0.03, 0.02),
                       (0.95, 0.03, 0.01, 0.01)]:
    shrunk = a * v19b_oof + b * oof_gh_hour + c * oof_gh5_hour + d_ * oof_city_hour
    r2 = r2_score(y49, shrunk)
    print(f"   ({a:.2f}, {b:.2f}, {c:.2f}, {d_:.2f}): OOF R²={r2:.6f}  d={r2-v19b_r2:+.6f}")


# ============================================================
# Step 4: Combined best
# ============================================================
print("\n=== Combined: smoothing + quantile + shrinkage ===")
# Best smoothing variant found above (we'll find it programmatically below)
best_smooth = (3, 0.75, v19b_r2)
for w in (3, 5, 7, 9):
    for b in (0.5, 0.75, 0.85, 1.0):
        oof_s = smooth(v16_oof, oof_gh, oof_ts, w, b)
        r2 = r2_score(y49, oof_s)
        if r2 > best_smooth[2]:
            best_smooth = (w, b, r2)
print(f"   best smoothing: w={best_smooth[0]} b={best_smooth[1]} R²={best_smooth[2]:.6f}")

w_best, b_best, _ = best_smooth
base_oof = smooth(v16_oof, oof_gh, oof_ts, w_best, b_best)
base_test = np.clip(smooth(v16_test, test["geohash"], test["tslot"], w_best, b_best), 0, 1)

# Apply quantile mapping + shrinkage to base_oof
best_combined_r2 = best_smooth[2]
best_combo = None
for q_alpha in (0.0, 0.1, 0.2, 0.3):
    qoof = quantile_remap(base_oof)
    for sh_alpha in (0.0, 0.05, 0.10, 0.15):
        combined_oof = (1 - q_alpha - sh_alpha) * base_oof + q_alpha * qoof + sh_alpha * oof_gh_hour
        r2 = r2_score(y49, combined_oof)
        if r2 > best_combined_r2:
            best_combined_r2 = r2; best_combo = (q_alpha, sh_alpha)

if best_combo is not None:
    qa, sha = best_combo
    print(f"   best combo: smooth(w={w_best}, b={b_best}) + quantile {qa} + gh_hour {sha} = R² {best_combined_r2:.6f}  d={best_combined_r2-v19b_r2:+.6f}")
    # apply to test
    qtest = quantile_remap(base_test)
    test_gh_hour = get_gh_hour(test)
    final_test = (1 - qa - sha) * base_test + qa * qtest + sha * test_gh_hour
    final_test = np.clip(final_test, 0, 1)
else:
    print(f"   no combo beat smoothing alone")
    final_test = base_test


gain = max(best_combined_r2, best_smooth[2]) - v19b_r2
print(f"\n>> overall gain over v19b = {gain:+.6f}")

if gain > 0.0001:
    sub = pd.DataFrame({"Index": test["Index"].values, "demand": final_test})
    sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
    print(">> wrote submission.csv with combined post-process")
else:
    sub = pd.DataFrame({"Index": test["Index"].values, "demand": v19b_test})
    sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
    print(">> wrote submission.csv = v19b unchanged")
