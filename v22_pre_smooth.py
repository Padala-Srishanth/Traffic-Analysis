"""
v22: Pre-smooth each base model (window=5, blend=1.0), then Ridge re-stacks
on the smoothed inputs. Best OOF in v21 grid (0.97003).
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

labels = ["cb","xgb","lgb","hgb","et","knn","chrS_50","residual","pseudo","nn"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], art["oof_nn"]]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], art["pred_nn"]]


def smooth(values, gh, ts, window, blend=1.0):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    df["out"] = blend * df["sm"] + (1 - blend) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()


# Pre-smooth each base model with the v21-winning params (w=5, b=1.0)
W, B = 5, 1.0
print(f">> pre-smoothing each base model (w={W}, b={B})")
sm_oof_cols  = []
sm_test_cols = []
for oof_col, test_col, lbl in zip(oof_cols, test_cols, labels):
    sm_oof = smooth(oof_col, train["geohash"], train["tslot"], W, B)
    sm_test = smooth(test_col, test["geohash"], test["tslot"], W, B)
    sm_oof_cols.append(sm_oof); sm_test_cols.append(sm_test)
sm_oof_stack  = np.column_stack([c[day49_idx] for c in sm_oof_cols])
sm_test_stack = np.column_stack(sm_test_cols)

m = Ridge(alpha=0.1, fit_intercept=False, positive=True)
m.fit(sm_oof_stack, y49)
oof_pred = m.predict(sm_oof_stack)
test_pred = m.predict(sm_test_stack)
r2_no_post = r2_score(y49, oof_pred)
print(f">> Ridge over pre-smoothed: OOF R²={r2_no_post:.5f}")
print(f">> Ridge weights = {dict(zip(labels, np.round(m.coef_, 4)))}")

# Try ALSO applying post-smoothing on top
oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)

best_r2 = r2_no_post; best_test = np.clip(test_pred, 0, 1); best_desc = "pre-smooth only"
for w_post in (2, 3, 4):
    for b_post in (0.0, 0.25, 0.5, 0.75, 1.0):
        post_oof = smooth(oof_pred, oof_gh, oof_ts, w_post, b_post)
        post_test = np.clip(smooth(test_pred, test["geohash"], test["tslot"], w_post, b_post), 0, 1)
        r2 = r2_score(y49, post_oof)
        if r2 > best_r2:
            best_r2 = r2; best_test = post_test; best_desc = f"pre w={W}b={B} + post w={w_post}b={b_post}"

print(f">> winner: {best_desc}  OOF R²={best_r2:.5f}")

sub = pd.DataFrame({"Index": test["Index"].values, "demand": best_test})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv", sub.shape)
print(sub.head())
