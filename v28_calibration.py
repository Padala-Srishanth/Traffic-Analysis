"""
v28: Group-specific calibration + targeted blending.
- Compute bias of v19b OOF predictions per group (RoadType, Weather, lane count, hour)
- Apply multiplicative/additive corrections that improve OOF R²
- Also try targeted blends with prior versions
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
train["hour"] = train["tslot"] // 4; test["hour"] = test["tslot"] // 4


# Rebuild v19b OOF and test predictions
labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"]]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"]]
meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(np.column_stack([c[day49_idx] for c in oof_cols]), y49)
v16_test = np.clip(meta.predict(np.column_stack(test_cols)), 0, 1)
v16_oof  = meta.predict(np.column_stack([c[day49_idx] for c in oof_cols]))


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
print(f">> v19b baseline OOF R² = {v19b_r2:.5f}")


# --------------------------------------------------------------------------- #
# Group-specific bias diagnosis on day-49 OOF                                 #
# --------------------------------------------------------------------------- #
print(">> diagnosing group-specific bias")
oof_df = train.iloc[day49_idx].copy().reset_index(drop=True)
oof_df["pred"] = v19b_oof
oof_df["truth"] = y49
oof_df["resid"] = oof_df["truth"] - oof_df["pred"]

for col in ["RoadType", "Weather", "NumberofLanes", "LargeVehicles", "Landmarks", "hour"]:
    if col not in oof_df.columns:
        continue
    g = oof_df.groupby(col, dropna=False).agg(n=("pred","size"), mean_pred=("pred","mean"),
                                              mean_truth=("truth","mean"), mean_resid=("resid","mean"))
    g["scale"]    = g["mean_truth"] / (g["mean_pred"] + 1e-9)
    g["abs_bias_pct"] = (g["mean_resid"] / (g["mean_truth"].abs() + 1e-9) * 100).round(2)
    print(f"\n   {col}:")
    print(g[["n", "mean_pred", "mean_truth", "scale", "abs_bias_pct"]].to_string())


# --------------------------------------------------------------------------- #
# Try group-specific multiplicative corrections                               #
# --------------------------------------------------------------------------- #
def apply_correction(oof_df, test_df, oof_preds, test_preds, group_col, kind="mul"):
    """kind: 'mul' = mean_truth/mean_pred ; 'add' = mean(truth-pred)"""
    g = oof_df.groupby(group_col).agg(mean_pred=("pred","mean"), mean_truth=("truth","mean"),
                                      mean_resid=("resid","mean"))
    if kind == "mul":
        factor = (g["mean_truth"] / (g["mean_pred"] + 1e-9)).clip(0.5, 2.0)
        new_oof  = oof_preds  * oof_df[group_col].map(factor).fillna(1.0).to_numpy()
        new_test = test_preds * test_df[group_col].map(factor).fillna(1.0).to_numpy()
    else:
        delta = g["mean_resid"].clip(-0.2, 0.2)
        new_oof  = oof_preds  + oof_df[group_col].map(delta).fillna(0.0).to_numpy()
        new_test = test_preds + test_df[group_col].map(delta).fillna(0.0).to_numpy()
    return new_oof, np.clip(new_test, 0, 1)


print("\n>> testing single-group corrections on OOF")
best_r2 = v19b_r2; best_test = v19b_test; best_desc = "v19b_raw"
candidates_to_combine = []
for group in ["RoadType", "Weather", "NumberofLanes", "LargeVehicles", "Landmarks", "hour"]:
    for kind in ("mul", "add"):
        new_oof, new_test = apply_correction(oof_df, test, v19b_oof, v19b_test, group, kind)
        r2 = r2_score(y49, new_oof)
        gain = r2 - v19b_r2
        print(f"   {group} ({kind}): OOF R²={r2:.5f}  delta={gain:+.5f}")
        if r2 > best_r2:
            best_r2 = r2; best_test = new_test; best_desc = f"corr_{group}_{kind}"
        if gain > 0.0001:
            candidates_to_combine.append((group, kind, gain))


print(f">> single-group winner: {best_desc}  OOF R²={best_r2:.5f}")


# --------------------------------------------------------------------------- #
# Try targeted blends with prior versions                                     #
# --------------------------------------------------------------------------- #
print("\n>> targeted blending tests")
# Reconstruct v13 and v15 blends to test mixing
labels_v13 = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual"]
oof_v13 = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
           art["oof_et"], art["oof_knn"], art["oof_chrS"], art["oof_residual"]]
test_v13 = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
            art["pred_et"], art["pred_knn"], art["pred_chrS"], art["pred_residual"]]
meta_v13 = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta_v13.fit(np.column_stack([c[day49_idx] for c in oof_v13]), y49)
v13_test = np.clip(meta_v13.predict(np.column_stack(test_v13)), 0, 1)
v13_oof  = meta_v13.predict(np.column_stack([c[day49_idx] for c in oof_v13]))

# Try various blend ratios
blends = [
    ("0.90 v19b + 0.10 v13", 0.90, v13_test, 0.10, v13_oof),
    ("0.85 v19b + 0.15 v13", 0.85, v13_test, 0.15, v13_oof),
    ("0.95 v19b + 0.05 v13", 0.95, v13_test, 0.05, v13_oof),
    ("0.80 v19b + 0.20 v13", 0.80, v13_test, 0.20, v13_oof),
    ("0.70 v19b + 0.30 v13", 0.70, v13_test, 0.30, v13_oof),
]
for desc, w_a, t_b, w_b, o_b in blends:
    oof_blend  = w_a * v19b_oof  + w_b * o_b
    test_blend = np.clip(w_a * v19b_test + w_b * t_b, 0, 1)
    r2 = r2_score(y49, oof_blend)
    corr = np.corrcoef(v19b_oof, o_b)[0,1]
    print(f"   {desc}: OOF R²={r2:.5f}  corr(v19b, v13)={corr:.4f}")
    if r2 > best_r2:
        best_r2 = r2; best_test = test_blend; best_desc = desc


print(f"\n>> overall winner: {best_desc}  OOF R²={best_r2:.5f}  (delta vs v19b = {best_r2-v19b_r2:+.5f})")
sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())
