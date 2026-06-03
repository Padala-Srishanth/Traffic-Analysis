"""
v12: Try an XGBoost meta-learner over the 7 base models from v9 (best so far).
Compares against Ridge to see if a non-linear blend captures correlated errors.
Falls back to Ridge if XGBoost meta overfits the small OOF.
"""
import sys, io, os, warnings
warnings.filterwarnings("ignore")

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams:
            try: st.write(s); st.flush()
            except Exception: pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_logfile = open(os.path.join(DATA_DIR, "run.log"), "w", encoding="utf-8")
sys.stdout = _Tee(io.TextIOWrapper(sys.__stdout__.buffer, encoding="utf-8", line_buffering=True), _logfile)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import xgboost as xgb

print(">> loading artifacts")
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw, day49_idx = art["y_raw"], art["day49_idx"]
y49 = y_raw[day49_idx]

# 7-model v9 stack
labels = ["cb", "xgb", "lgb", "hgb", "et", "knn", "chrS_50"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"]]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"]]
oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)
print(f"   oof_stack shape: {oof_stack.shape}  test_stack shape: {test_stack.shape}")


# --- baseline: Ridge ---
print(">> baseline Ridge stack")
ridge = Ridge(alpha=0.1, fit_intercept=False, positive=True)
ridge.fit(oof_stack, y49)
ridge_oof_pred = ridge.predict(oof_stack)
ridge_r2 = r2_score(y49, ridge_oof_pred)
print(f"   Ridge OOF R² = {ridge_r2:.5f}  (score = {max(0,100*ridge_r2):.3f})")
print(f"   Ridge weights = {dict(zip(labels, np.round(ridge.coef_,4)))}")


# --- XGBoost meta-learner with low capacity to avoid overfit on small OOF ---
# Use INNER 5-fold CV on the OOF stack so we get honest meta-OOF predictions.
print(">> XGBoost meta-learner (inner 5-fold CV over OOF stack)")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_meta_oof = np.zeros_like(y49)
xgb_meta_test_preds = np.zeros((kf.n_splits, len(test_stack)))

for fold, (tr_i, va_i) in enumerate(kf.split(oof_stack), 1):
    m = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.85, colsample_bytree=1.0, min_child_weight=4, reg_lambda=2.0,
        tree_method="hist", device="cuda",
        objective="reg:squarederror", random_state=42, n_jobs=-1,
        early_stopping_rounds=30,
    )
    m.fit(oof_stack[tr_i], y49[tr_i],
          eval_set=[(oof_stack[va_i], y49[va_i])], verbose=False)
    xgb_meta_oof[va_i] = m.predict(oof_stack[va_i])
    xgb_meta_test_preds[fold-1] = m.predict(test_stack)
    fold_r2 = r2_score(y49[va_i], xgb_meta_oof[va_i])
    print(f"   meta-fold {fold}: best_iter={m.best_iteration}  R²={fold_r2:.5f}")

xgb_meta_r2 = r2_score(y49, xgb_meta_oof)
print(f">> XGBoost meta-OOF R² = {xgb_meta_r2:.5f}  (score = {max(0,100*xgb_meta_r2):.3f})")


# --- Blend Ridge and XGBoost meta predictions ---
print(">> Blending Ridge + XGB-meta on OOF")
best_w, best_r2 = 0.5, -1
for w in np.linspace(0, 1, 41):
    blend = w * ridge_oof_pred + (1 - w) * xgb_meta_oof
    r2 = r2_score(y49, blend)
    if r2 > best_r2: best_r2, best_w = r2, w
print(f"   best Ridge-XGBmeta mix: w_ridge={best_w:.3f}  OOF R²={best_r2:.5f}  (score = {max(0,100*best_r2):.3f})")


# --- Pick the winner among Ridge-only, XGB-only, mix ---
xgb_meta_test = xgb_meta_test_preds.mean(axis=0)
ridge_test = ridge.predict(test_stack)
candidates = {
    "ridge_only": (ridge_r2, ridge_test),
    "xgb_meta_only": (xgb_meta_r2, xgb_meta_test),
    "ridge_xgb_blend": (best_r2, best_w * ridge_test + (1 - best_w) * xgb_meta_test),
}
winner = max(candidates, key=lambda k: candidates[k][0])
print(f">> winner: {winner}  OOF R²={candidates[winner][0]:.5f}")

pred = np.clip(candidates[winner][1], 0.0, 1.0)
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.DataFrame({"Index": test.Index.values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())
