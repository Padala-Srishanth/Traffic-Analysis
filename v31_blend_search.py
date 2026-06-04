"""
v31: Exhaustive diversity-optimised blend search at the submission level.
Loads v19b (current best), the raw Chronos predictions, and the raw
per-base-model predictions. Searches blends that mix v19b with the
LEAST-correlated alternative predictions.
"""
import sys, io, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from itertools import product

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

# Rebuild v19b (smoothed Ridge of 10 base models)
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


# ---------- Build candidate "blend partners" ----------
candidates = {}

# Chronos (raw, not after Ridge)
candidates["chronos"] = (art["oof_chrS"], art["pred_chrS"])

# CatBoost alone
candidates["catboost"] = (art["oof_cb"], art["pred_cb"])
candidates["xgboost"]  = (art["oof_xgb"], art["pred_xgb"])
candidates["lightgbm"] = (art["oof_lgb"], art["pred_lgb"])
candidates["extratrees"] = (art["oof_et"], art["pred_et"])
candidates["histgbm"]  = (art["oof_hgb"], art["pred_hgb"])
candidates["pseudo"]   = (art["oof_pseudo"], art["pred_pseudo"])
candidates["residual"] = (art["oof_residual"], art["pred_residual"])
candidates["nn"]       = (art["oof_nn"], art["pred_nn"])

# Smoothed versions of each (apply v19b-winning smoothing to each candidate)
smoothed_candidates = {}
for name, (oof_c, test_c) in candidates.items():
    sm_oof  = smooth(oof_c, train["geohash"], train["tslot"], 3, 0.75)
    sm_test = np.clip(smooth(test_c, test["geohash"], test["tslot"], 3, 0.75), 0, 1)
    smoothed_candidates[f"{name}_sm"] = (sm_oof, sm_test)

# Combine raw + smoothed for the search
search_pool = {**candidates, **smoothed_candidates}

# Compute correlation with v19b for each candidate (on day-49 OOF)
print("\n>> correlation with v19b on day-49 OOF (lower = more diverse):")
print(f"{'partner':<22} {'corr':>8} {'r2':>8}")
corr_table = []
for name, (oof_c, _) in search_pool.items():
    c = np.corrcoef(v19b_oof, oof_c[day49_idx])[0, 1]
    r2_c = r2_score(y49, oof_c[day49_idx])
    corr_table.append((name, c, r2_c))
for name, c, r2_c in sorted(corr_table, key=lambda x: x[1]):
    print(f"{name:<22} {c:8.4f} {r2_c:8.4f}")


# ---------- Exhaustive blend search ----------
print("\n>> exhaustive 2-way blends: w * v19b + (1-w) * partner")
print(f"{'partner':<22} {'best_w':>7} {'best_r2':>10} {'delta':>10}")
results = []
for name, (oof_c, test_c) in search_pool.items():
    best_w, best_r2 = 1.0, v19b_r2
    for w in np.linspace(0.5, 1.0, 51):
        blend_oof = w * v19b_oof + (1 - w) * oof_c[day49_idx]
        r2 = r2_score(y49, blend_oof)
        if r2 > best_r2:
            best_r2 = r2; best_w = w
    delta = best_r2 - v19b_r2
    blend_test = np.clip(best_w * v19b_test + (1 - best_w) * test_c, 0, 1)
    results.append((name, best_w, best_r2, delta, blend_test))

for name, w, r2, delta, _ in sorted(results, key=lambda x: -x[2])[:10]:
    print(f"{name:<22} {w:7.3f} {r2:10.5f} {delta:+10.5f}")


# 3-way blends with the best 2 partners
print("\n>> 3-way blends w_v19b * v19b + w_p1 * p1 + w_p2 * p2")
results.sort(key=lambda x: -x[2])
top_partners = [r[0] for r in results[:6] if r[3] > 0]
if len(top_partners) >= 2:
    for p1, p2 in [(top_partners[i], top_partners[j]) for i in range(min(3, len(top_partners))) for j in range(i+1, min(4, len(top_partners)))]:
        oof_p1, _ = search_pool[p1]
        oof_p2, _ = search_pool[p2]
        best = (None, None, v19b_r2)
        for w1 in np.linspace(0, 0.3, 11):
            for w2 in np.linspace(0, 0.3, 11):
                w_v = 1 - w1 - w2
                if w_v < 0.4: continue
                blend = w_v * v19b_oof + w1 * oof_p1[day49_idx] + w2 * oof_p2[day49_idx]
                r2 = r2_score(y49, blend)
                if r2 > best[2]:
                    best = (w1, w2, r2)
        if best[0] is not None:
            print(f"   v19b + {p1}({best[0]:.3f}) + {p2}({best[1]:.3f}): OOF R²={best[2]:.5f}  delta={best[2]-v19b_r2:+.5f}")


# ---------- Pick winner ----------
top = sorted(results, key=lambda x: -x[2])[0]
name, w, r2, delta, blend_test = top
if delta > 0:
    print(f"\n>> WINNER: {name} (w_v19b={w:.3f})  OOF R²={r2:.5f}  delta={delta:+.5f}")
    sub = pd.DataFrame({"Index": test["Index"].values, "demand": blend_test})
    sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
    print(">> wrote submission.csv with new blend")
else:
    print(f"\n>> No blend beat v19b. Keeping v19b submission.")
    sub = pd.DataFrame({"Index": test["Index"].values, "demand": v19b_test})
    sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
    print(">> wrote submission.csv = v19b (unchanged)")
