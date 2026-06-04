"""
v24: Test-Time Augmentation (TTA). Average multiple smoothing variants of the
v16 predictions (rolling mean + exponential weighted mean at several params).
Plus a final geohash-prefix-mean shrinkage.
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
def to_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return (h * 4 + m // 15).astype(int)
train["tslot"] = to_ts(train["timestamp"]); test["tslot"] = to_ts(test["timestamp"])

# v16 base
labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn"]
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
print(f">> v16 raw OOF R² = {r2_score(y49, v16_oof):.5f}")


# ---- TTA primitives ----
def rolling_smooth(values, gh, ts, window, blend):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    df["out"] = blend * df["sm"] + (1 - blend) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()


def ema_smooth(values, gh, ts, halflife, blend):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["fwd"] = df.groupby("gh")["pred"].transform(lambda s: s.ewm(halflife=halflife, min_periods=1).mean())
    # backward pass for centered smoothing
    df["bwd"] = df.groupby("gh")["pred"].transform(lambda s: s.iloc[::-1].ewm(halflife=halflife, min_periods=1).mean().iloc[::-1])
    df["sm"] = (df["fwd"] + df["bwd"]) / 2.0
    df["out"] = blend * df["sm"] + (1 - blend) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()


oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)


# ---- Generate multiple smoothed versions, evaluate each on OOF, then average TOP-K ----
print(">> generating TTA variants")
variants_oof = {}
variants_test = {}

# rolling smoothing grid (the v19b-style)
for w in (2, 3, 4, 5):
    for b in (0.5, 0.75, 1.0):
        key = f"roll_w{w}_b{b}"
        variants_oof[key]  = rolling_smooth(v16_oof, oof_gh, oof_ts, w, b)
        variants_test[key] = np.clip(rolling_smooth(v16_test, test["geohash"], test["tslot"], w, b), 0, 1)

# EMA smoothing
for hl in (1.0, 1.5, 2.0, 3.0):
    for b in (0.5, 0.75, 1.0):
        key = f"ema_hl{hl}_b{b}"
        variants_oof[key]  = ema_smooth(v16_oof, oof_gh, oof_ts, hl, b)
        variants_test[key] = np.clip(ema_smooth(v16_test, test["geohash"], test["tslot"], hl, b), 0, 1)

# Rank by OOF R²
ranked = sorted(variants_oof.keys(), key=lambda k: r2_score(y49, variants_oof[k]), reverse=True)
print(">> top 10 TTA variants by OOF R²:")
for k in ranked[:10]:
    print(f"   {k}: R²={r2_score(y49, variants_oof[k]):.5f}")


# Average top-K variants
print(">> averaging top-K")
best_r2 = r2_score(y49, v16_oof); best_key = "v16_raw"; best_test = v16_test
for K in (3, 5, 7, 10, 15):
    avg_oof  = np.mean([variants_oof[k]  for k in ranked[:K]], axis=0)
    avg_test = np.clip(np.mean([variants_test[k] for k in ranked[:K]], axis=0), 0, 1)
    r2 = r2_score(y49, avg_oof)
    print(f"   top-{K} avg: OOF R²={r2:.5f}")
    if r2 > best_r2:
        best_r2 = r2; best_key = f"top{K}_avg"; best_test = avg_test

# Single-winner from variants
single_r2 = r2_score(y49, variants_oof[ranked[0]])
print(f">> single winner ({ranked[0]}): OOF R²={single_r2:.5f}")
if single_r2 > best_r2:
    best_r2 = single_r2; best_key = ranked[0]; best_test = variants_test[ranked[0]]


# Also try TOP-K WEIGHTED by OOF R²
print(">> weighted top-K (weight = R²^N)")
for K in (5, 10, 15):
    for power in (10, 50, 100):
        r2_vals = np.array([r2_score(y49, variants_oof[k]) for k in ranked[:K]])
        w = r2_vals ** power
        w = w / w.sum()
        avg_oof  = sum(w[i] * variants_oof[ranked[i]]  for i in range(K))
        avg_test = np.clip(sum(w[i] * variants_test[ranked[i]] for i in range(K)), 0, 1)
        r2 = r2_score(y49, avg_oof)
        if r2 > best_r2:
            best_r2 = r2; best_key = f"weighted_top{K}_p{power}"; best_test = avg_test
        print(f"   K={K} power={power}: OOF R²={r2:.5f}")

print(f">> WINNER: {best_key}  OOF R²={best_r2:.5f}")
sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())
