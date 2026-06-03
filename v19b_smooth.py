"""
v19b: Apply per-geohash rolling smoothing to v16 test predictions.
Uses the variant that maximised OOF R² in v19 (smooth w=3 on OOF gave 0.96880).
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

# rebuild v16 Ridge stack
labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"]]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"]]
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(np.column_stack([c[day49_idx] for c in oof_cols]), y49)
v16_test_pred = np.clip(meta.predict(np.column_stack(test_cols)), 0.0, 1.0)
v16_oof_pred  = meta.predict(np.column_stack([c[day49_idx] for c in oof_cols]))
print(f">> v16 raw OOF R² = {r2_score(y49, v16_oof_pred):.5f}")


def smooth_predictions(values: np.ndarray, gh: pd.Series, ts: pd.Series, window: int, blend: float = 1.0) -> np.ndarray:
    """For each gh, sort by tslot, apply rolling mean. Then blend with original by `blend`."""
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_orig": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["smoothed"] = df.groupby("gh")["pred"].transform(
        lambda s: s.rolling(window, center=True, min_periods=1).mean()
    )
    df["blended"] = blend * df["smoothed"] + (1 - blend) * df["pred"]
    df = df.sort_values("_orig")
    return df["blended"].to_numpy()


# ---- Evaluate variants on OOF ----
print(">> evaluating smoothing variants on OOF")
oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)

variants = {"v16_raw": v16_oof_pred}
for w in (3, 5, 7):
    for blend in (0.25, 0.5, 0.75, 1.0):
        key = f"smooth_w{w}_b{blend}"
        v = smooth_predictions(v16_oof_pred, oof_gh, oof_ts, w, blend)
        variants[key] = v

# Also test shrinkage to (gh, hour) mean
def shrink(values, gh, ts, alpha):
    df = pd.DataFrame({"gh": gh.values, "hour": (ts.values // 4), "pred": values})
    df["mean"] = df.groupby(["gh", "hour"])["pred"].transform("mean")
    return ((1 - alpha) * df["pred"] + alpha * df["mean"]).to_numpy()

for alpha in (0.2, 0.3, 0.4, 0.5):
    variants[f"shrink_a{alpha}"] = shrink(v16_oof_pred, oof_gh, oof_ts, alpha)

# Combine smoothing + shrinkage
for w in (3, 5):
    for alpha in (0.2, 0.3):
        sm = smooth_predictions(v16_oof_pred, oof_gh, oof_ts, w, 1.0)
        combined = (1 - alpha) * sm + alpha * shrink(sm, oof_gh, oof_ts, 1.0)  # full shrink as base
        variants[f"smooth_w{w}+shrink_a{alpha}"] = combined

# Rank by OOF R²
ranked = sorted(variants.items(), key=lambda kv: r2_score(y49, kv[1]), reverse=True)
print(">> top 10 variants by OOF R²:")
for k, v in ranked[:10]:
    print(f"   {k}: OOF R² = {r2_score(y49, v):.5f}")

# Apply the WINNING variant to test predictions
winner_key, _ = ranked[0]
print(f">> applying winning variant '{winner_key}' to test predictions")

test_gh = test["geohash"]
test_ts = test["tslot"]

def reapply(name, pred_test):
    if name == "v16_raw":
        return pred_test
    if name.startswith("smooth_w") and "+shrink" not in name and "+" not in name:
        # smooth_wW_bB
        parts = name.split("_")
        w = int(parts[1][1:])
        blend = float(parts[2][1:])
        return smooth_predictions(pred_test, test_gh, test_ts, w, blend)
    if name.startswith("shrink_a"):
        alpha = float(name.split("a")[1])
        return shrink(pred_test, test_gh, test_ts, alpha)
    if "+shrink" in name:
        # smooth_wW+shrink_aA
        a_part, b_part = name.split("+")
        w = int(a_part.split("_")[1][1:])
        alpha = float(b_part.split("a")[1])
        sm = smooth_predictions(pred_test, test_gh, test_ts, w, 1.0)
        return (1 - alpha) * sm + alpha * shrink(sm, test_gh, test_ts, 1.0)
    return pred_test

pred_final = np.clip(reapply(winner_key, v16_test_pred), 0.0, 1.0)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred_final})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(f">> wrote submission.csv (winner='{winner_key}')", sub.shape)
print(sub.head())
