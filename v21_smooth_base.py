"""
v21: Apply temporal smoothing to EACH base model's OOF + test predictions
BEFORE Ridge stacks them. Then optionally re-smooth the final Ridge output.
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

# All base models (10)
labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn"]
oof_cols_full = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
                 art["oof_et"], art["oof_knn"], art["oof_chrS"],
                 art["oof_residual"], art["oof_pseudo"], art["oof_nn"]]
test_cols_full = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
                  art["pred_et"], art["pred_knn"], art["pred_chrS"],
                  art["pred_residual"], art["pred_pseudo"], art["pred_nn"]]


def temporal_smooth(values, gh, ts, window, blend=1.0):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    df["out"] = blend * df["sm"] + (1 - blend) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()


oof_gh_all = train["geohash"]; oof_ts_all = train["tslot"]
test_gh = test["geohash"]; test_ts = test["tslot"]

# baseline v16
meta0 = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta0.fit(np.column_stack([c[day49_idx] for c in oof_cols_full]), y49)
v16_oof  = meta0.predict(np.column_stack([c[day49_idx] for c in oof_cols_full]))
v16_test = np.clip(meta0.predict(np.column_stack(test_cols_full)), 0, 1)
v16_r2 = r2_score(y49, v16_oof)
print(f">> baseline v16 OOF R² = {v16_r2:.5f}")

# v19b winning post-smooth applied to v16
oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)
v19b_oof  = temporal_smooth(v16_oof, oof_gh, oof_ts, 3, 0.75)
v19b_test = np.clip(temporal_smooth(v16_test, test_gh, test_ts, 3, 0.75), 0, 1)
v19b_r2 = r2_score(y49, v19b_oof)
print(f">> v19b (post-smooth) OOF R² = {v19b_r2:.5f}")

# Try smoothing each base model BEFORE Ridge
print(">> smoothing each base model OOF+test, then re-fit Ridge")
for w_base in (3, 5):
    for b_base in (0.5, 0.75, 1.0):
        sm_oof_cols  = []
        sm_test_cols = []
        for oof_col, test_col in zip(oof_cols_full, test_cols_full):
            sm_oof  = temporal_smooth(oof_col, oof_gh_all, oof_ts_all, w_base, b_base)
            sm_test = temporal_smooth(test_col, test_gh, test_ts, w_base, b_base)
            sm_oof_cols.append(sm_oof); sm_test_cols.append(sm_test)
        sm_oof_stack  = np.column_stack([c[day49_idx] for c in sm_oof_cols])
        sm_test_stack = np.column_stack(sm_test_cols)
        m = Ridge(alpha=0.1, fit_intercept=False, positive=True)
        m.fit(sm_oof_stack, y49)
        oof_pred = m.predict(sm_oof_stack)
        test_pred = np.clip(m.predict(sm_test_stack), 0, 1)
        r2 = r2_score(y49, oof_pred)
        # Also try post-smoothing after Ridge
        oof_post = temporal_smooth(oof_pred, oof_gh, oof_ts, 3, 0.75)
        test_post = np.clip(temporal_smooth(test_pred, test_gh, test_ts, 3, 0.75), 0, 1)
        r2_post = r2_score(y49, oof_post)
        print(f"   base_smooth w={w_base} b={b_base}:  pre-only R²={r2:.5f}  +post-smooth R²={r2_post:.5f}")


# Also try DOUBLE post-smoothing on v19b
print(">> double post-smoothing on v19b")
for w in (3, 5):
    for b in (0.25, 0.5, 0.75, 1.0):
        d = temporal_smooth(v19b_oof, oof_gh, oof_ts, w, b)
        d_test = temporal_smooth(v19b_test, test_gh, test_ts, w, b)
        r2 = r2_score(y49, d)
        print(f"   v19b + smooth w={w} b={b}: OOF R²={r2:.5f}")


# Also try a TIGHTER smoothing param search on v19b
print(">> dense post-smoothing grid")
best_r2 = -1; best_params = None; best_test = None
for w in (2, 3, 4):
    for b in np.linspace(0.5, 1.0, 11):
        sm_oof  = temporal_smooth(v16_oof, oof_gh, oof_ts, w, b)
        sm_test = np.clip(temporal_smooth(v16_test, test_gh, test_ts, w, b), 0, 1)
        r2 = r2_score(y49, sm_oof)
        if r2 > best_r2:
            best_r2 = r2; best_params = (w, b); best_test = sm_test
print(f"   best: w={best_params[0]} b={best_params[1]:.2f}  OOF R²={best_r2:.5f}")

# Save the winner
print(f">> using dense-grid winner w={best_params[0]} b={best_params[1]:.2f}")
sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())
