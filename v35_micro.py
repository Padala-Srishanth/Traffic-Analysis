"""
v35: two micro-experiments.
1. Refit Ridge meta-learner with positive=False (allow negative weights)
2. Center-weighted rolling smoothing instead of equal-weighted
Compare OOF; submit the best.
"""
import os, warnings
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
def to_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = to_ts(train["timestamp"]); test["tslot"] = to_ts(test["timestamp"])

labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"]]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"]]


# Standard equal-weighted rolling
def smooth_eq(values, gh, ts, w, b):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(w, center=True, min_periods=1).mean())
    df["out"] = b * df["sm"] + (1 - b) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()


# Center-weighted rolling with custom weights
def smooth_centered(values, gh, ts, weights, b):
    """weights centered, e.g., [0.2, 0.6, 0.2] for win=3, [0.1, 0.2, 0.4, 0.2, 0.1] for win=5."""
    w = np.array(weights, dtype=float)
    w /= w.sum()
    width = (len(w) - 1) // 2
    def conv(s):
        a = s.to_numpy()
        if len(a) < len(w):
            return a  # too short, no convolution
        return np.convolve(a, w, mode="same")
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    out = []
    for _, grp in df.groupby("gh"):
        arr = grp["pred"].to_numpy()
        if len(arr) < len(w):
            sm = arr
        else:
            sm = np.convolve(arr, w, mode="same")
            # adjust edges by padding
        out_arr = b * sm + (1 - b) * arr
        out.append(pd.Series(out_arr, index=grp.index))
    out_series = pd.concat(out).reindex(df.index)
    df["out"] = out_series.values
    return df.sort_values("_o")["out"].to_numpy()


oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)


# ============================================================
# Experiment 1: Ridge variants
# ============================================================
print("=" * 60)
print("Experiment 1: Ridge with positive=False (allow negative)")
print("=" * 60)

X = np.column_stack([c[day49_idx] for c in oof_cols])
Xt = np.column_stack(test_cols)

# Baseline: positive=True
meta_pos = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta_pos.fit(X, y49)
oof_pos = meta_pos.predict(X)
test_pos = np.clip(meta_pos.predict(Xt), 0, 1)
oof_pos_sm = smooth_eq(oof_pos, oof_gh, oof_ts, 3, 0.75)
r2_pos = r2_score(y49, oof_pos_sm)
print(f"   positive=True, alpha=0.1, smooth(3, 0.75): OOF R² = {r2_pos:.6f}")

# Allow negative
for alpha in (0.01, 0.1, 1.0, 10.0):
    meta_neg = Ridge(alpha=alpha, fit_intercept=False, positive=False)
    meta_neg.fit(X, y49)
    oof_neg = meta_neg.predict(X)
    test_neg = np.clip(meta_neg.predict(Xt), 0, 1)
    oof_neg_sm = smooth_eq(oof_neg, oof_gh, oof_ts, 3, 0.75)
    r2_neg = r2_score(y49, oof_neg_sm)
    print(f"   positive=False, alpha={alpha}, smooth(3, 0.75): OOF R² = {r2_neg:.6f}")


# ============================================================
# Experiment 2: Center-weighted rolling
# ============================================================
print("\n" + "=" * 60)
print("Experiment 2: Center-weighted rolling smoothing")
print("=" * 60)

print(f"   baseline equal-weighted w=3 b=0.75: R² = {r2_pos:.6f}")

# Different weight schemes
weight_configs = [
    ("center_w3_2-1-2", [1, 2, 1]),
    ("center_w3_1-3-1", [1, 3, 1]),
    ("center_w3_1-4-1", [1, 4, 1]),
    ("center_w5_1-2-3-2-1", [1, 2, 3, 2, 1]),
    ("center_w5_1-2-4-2-1", [1, 2, 4, 2, 1]),
    ("center_w5_1-3-5-3-1", [1, 3, 5, 3, 1]),
]

best_alt = ("baseline", r2_pos, None)
for name, w in weight_configs:
    for b in (0.5, 0.75, 0.85, 1.0):
        sm_oof = smooth_centered(oof_pos, oof_gh, oof_ts, w, b)
        r2 = r2_score(y49, sm_oof)
        print(f"   {name} blend={b}: R² = {r2:.6f}  d={r2 - r2_pos:+.6f}")
        if r2 > best_alt[1]:
            best_alt = (f"{name}_b{b}", r2, (w, b))

print(f"\n>> best variant: {best_alt[0]} R² = {best_alt[1]:.6f}")
print(f">> baseline (v19b equiv): R² = {r2_pos:.6f}")
print(f">> gain = {best_alt[1] - r2_pos:+.6f}")

# Apply winner
if best_alt[2] is None:
    # baseline wins
    pred_final = np.clip(smooth_eq(test_pos, test["geohash"], test["tslot"], 3, 0.75), 0, 1)
    print(">> keeping v19b baseline")
else:
    w, b = best_alt[2]
    pred_final = np.clip(smooth_centered(test_pos, test["geohash"], test["tslot"], w, b), 0, 1)
    print(f">> applying {best_alt[0]}")

sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred_final})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())
