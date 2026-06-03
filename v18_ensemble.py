"""
v18: Variance-reduction ensemble of the four best historical blends (v9, v13, v15, v16).
For each version, reconstruct its Ridge-blend test prediction from saved artifacts,
then average them. Picks the blend (best individual OR mean) by OOF R².
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


def fit_blend(labels, oof_list, test_list, name):
    oof_stack = np.column_stack([c[day49_idx] for c in oof_list])
    test_stack = np.column_stack(test_list)
    m = Ridge(alpha=0.1, fit_intercept=False, positive=True)
    m.fit(oof_stack, y49)
    oof_pred = m.predict(oof_stack)
    test_pred = m.predict(test_stack)
    r2 = r2_score(y49, oof_pred)
    print(f"   {name}: OOF R²={r2:.5f}  weights={dict(zip(labels, np.round(m.coef_,3)))}")
    return oof_pred, test_pred, r2

print(">> reconstructing each version's blend from saved artifacts")
print(">> v9 — 7 model stack")
v9_oof, v9_test, v9_r2 = fit_blend(
    ["cb","xgb","lgb","hgb","et","knn","chrS"],
    [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"], art["oof_et"], art["oof_knn"], art["oof_chrS"]],
    [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"], art["pred_et"], art["pred_knn"], art["pred_chrS"]],
    "v9"
)
print(">> v13 — 8 model stack (+ residual)")
v13_oof, v13_test, v13_r2 = fit_blend(
    ["cb","xgb","lgb","hgb","et","knn","chrS","res"],
    [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"], art["oof_et"], art["oof_knn"], art["oof_chrS"], art["oof_residual"]],
    [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"], art["pred_et"], art["pred_knn"], art["pred_chrS"], art["pred_residual"]],
    "v13"
)
print(">> v15 — 9 model stack (+ pseudoCB)")
v15_oof, v15_test, v15_r2 = fit_blend(
    ["cb","xgb","lgb","hgb","et","knn","chrS","res","pseudoCB"],
    [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"], art["oof_et"], art["oof_knn"], art["oof_chrS"], art["oof_residual"], art["oof_pseudo"]],
    [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"], art["pred_et"], art["pred_knn"], art["pred_chrS"], art["pred_residual"], art["pred_pseudo"]],
    "v15"
)
print(">> v16 — 10 model stack (+ NN)")
v16_oof, v16_test, v16_r2 = fit_blend(
    ["cb","xgb","lgb","hgb","et","knn","chrS","res","pseudoCB","nn"],
    [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"], art["oof_et"], art["oof_knn"], art["oof_chrS"], art["oof_residual"], art["oof_pseudo"], art["oof_nn"]],
    [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"], art["pred_et"], art["pred_knn"], art["pred_chrS"], art["pred_residual"], art["pred_pseudo"], art["pred_nn"]],
    "v16"
)

# 1. simple average of all four
print(">> simple average of v9, v13, v15, v16")
avg_oof  = (v9_oof  + v13_oof  + v15_oof  + v16_oof)  / 4.0
avg_test = (v9_test + v13_test + v15_test + v16_test) / 4.0
avg_r2 = r2_score(y49, avg_oof)
print(f"   avg OOF R²={avg_r2:.5f}")

# 2. equal blend of v15 + v16 (most similar)
mix_oof  = 0.5 * v15_oof  + 0.5 * v16_oof
mix_test = 0.5 * v15_test + 0.5 * v16_test
mix_r2 = r2_score(y49, mix_oof)
print(f"   v15+v16 (50/50) OOF R²={mix_r2:.5f}")

# 3. Ridge over the four blend predictions (meta-of-metas)
oof_meta_X = np.column_stack([v9_oof, v13_oof, v15_oof, v16_oof])
test_meta_X = np.column_stack([v9_test, v13_test, v15_test, v16_test])
mm = Ridge(alpha=0.1, fit_intercept=False, positive=True)
mm.fit(oof_meta_X, y49)
mm_oof = mm.predict(oof_meta_X)
mm_test = mm.predict(test_meta_X)
mm_r2 = r2_score(y49, mm_oof)
print(f"   Ridge-of-blends weights = {dict(zip(['v9','v13','v15','v16'], np.round(mm.coef_, 3)))}  OOF R²={mm_r2:.5f}")

# Pick winner
candidates = {
    "v9":  (v9_r2,  v9_test),
    "v13": (v13_r2, v13_test),
    "v15": (v15_r2, v15_test),
    "v16": (v16_r2, v16_test),
    "avg4": (avg_r2, avg_test),
    "mix15_16": (mix_r2, mix_test),
    "ridge_of_blends": (mm_r2, mm_test),
}
winner = max(candidates, key=lambda k: candidates[k][0])
print(f">> winner by OOF: {winner}  (OOF R²={candidates[winner][0]:.5f})")

pred = np.clip(candidates[winner][1], 0.0, 1.0)
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.DataFrame({"Index": test.Index.values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(f">> wrote submission.csv (using '{winner}'):", sub.shape)
print(sub.head())
