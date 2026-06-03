"""
v20: Combine temporal smoothing (across tslots per geohash) with spatial smoothing
(across geographically nearby geohashes at same tslot). Multiple parameter combos
evaluated on OOF, best applied to test.
"""
import sys, io, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree

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

# Geohash decoder for spatial smoothing
GH = "0123456789bcdefghjkmnpqrstuvwxyz"
GH_MAP = {c: i for i, c in enumerate(GH)}
def decode(g):
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for ch in g:
        bits = GH_MAP[ch]
        for m in (16, 8, 4, 2, 1):
            b = (bits & m) > 0
            if even:
                mid = (lon_lo + lon_hi)/2
                if b: lon_lo = mid
                else: lon_hi = mid
            else:
                mid = (lat_lo + lat_hi)/2
                if b: lat_lo = mid
                else: lat_hi = mid
            even = not even
    return (lat_lo + lat_hi)/2, (lon_lo + lon_hi)/2

all_gh = pd.unique(pd.concat([train["geohash"], test["geohash"]]))
gh_lat = {g: decode(g)[0] for g in all_gh}
gh_lon = {g: decode(g)[1] for g in all_gh}
train["lat"] = train["geohash"].map(gh_lat); train["lon"] = train["geohash"].map(gh_lon)
test["lat"]  = test["geohash"].map(gh_lat);  test["lon"]  = test["geohash"].map(gh_lon)

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


def temporal_smooth(values, gh, ts, window, blend=1.0):
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "pred": values, "_o": np.arange(len(values))})
    df = df.sort_values(["gh", "ts"])
    df["sm"] = df.groupby("gh")["pred"].transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    df["out"] = blend * df["sm"] + (1 - blend) * df["pred"]
    return df.sort_values("_o")["out"].to_numpy()


def spatial_smooth(values, gh, ts, lat, lon, k, blend=1.0):
    """At each (gh, ts), avg the k nearest gh's predictions at the same ts (excluding self)."""
    df = pd.DataFrame({"gh": gh.values, "ts": ts.values, "lat": lat.values, "lon": lon.values,
                       "pred": values, "_o": np.arange(len(values))})
    out = np.zeros(len(values), dtype=np.float32)
    for t, sub in df.groupby("ts"):
        if len(sub) < 2:
            out[sub["_o"].to_numpy()] = sub["pred"].to_numpy(); continue
        tree = cKDTree(sub[["lat", "lon"]].values)
        kk = min(k + 1, len(sub))  # +1 for self
        _, idx = tree.query(sub[["lat", "lon"]].values, k=kk)
        if kk == 1: idx = idx[:, None]
        neigh_pred = sub["pred"].to_numpy()[idx]   # shape (n, kk)
        neigh_gh   = sub["gh"].to_numpy()[idx]
        self_mask  = neigh_gh == sub["gh"].to_numpy()[:, None]
        neigh_pred_masked = np.where(self_mask, np.nan, neigh_pred)
        mean = np.nanmean(neigh_pred_masked, axis=1)
        out[sub["_o"].to_numpy()] = mean
    return blend * out + (1 - blend) * values


# ---- Evaluate variants on OOF ----
print(">> evaluating variants on OOF")
oof_gh = train.loc[day49_idx, "geohash"].reset_index(drop=True)
oof_ts = train.loc[day49_idx, "tslot"].reset_index(drop=True)
oof_lat = train.loc[day49_idx, "lat"].reset_index(drop=True)
oof_lon = train.loc[day49_idx, "lon"].reset_index(drop=True)

variants = {"v16_raw": v16_oof_pred}

# Temporal smoothing grid
for w in (3, 5, 7):
    for blend in (0.5, 0.75, 1.0):
        variants[f"temp_w{w}_b{blend}"] = temporal_smooth(v16_oof_pred, oof_gh, oof_ts, w, blend)

# Spatial smoothing
for k in (5, 10, 20):
    for blend in (0.25, 0.5, 0.75):
        variants[f"sp_k{k}_b{blend}"] = spatial_smooth(v16_oof_pred, oof_gh, oof_ts, oof_lat, oof_lon, k, blend)

# Temporal then spatial
for tw, tb in ((3, 0.75), (5, 0.5)):
    for k, sb in ((10, 0.25), (10, 0.5)):
        v = temporal_smooth(v16_oof_pred, oof_gh, oof_ts, tw, tb)
        v = spatial_smooth(v, oof_gh, oof_ts, oof_lat, oof_lon, k, sb)
        variants[f"temp_w{tw}_b{tb}+sp_k{k}_b{sb}"] = v

# Spatial then temporal
for k, sb in ((10, 0.5),):
    for tw, tb in ((3, 0.75), (5, 0.5)):
        v = spatial_smooth(v16_oof_pred, oof_gh, oof_ts, oof_lat, oof_lon, k, sb)
        v = temporal_smooth(v, oof_gh, oof_ts, tw, tb)
        variants[f"sp_k{k}_b{sb}+temp_w{tw}_b{tb}"] = v

# Rank
ranked = sorted(variants.items(), key=lambda kv: r2_score(y49, kv[1]), reverse=True)
print(">> top 15 by OOF R²:")
for k, v in ranked[:15]:
    print(f"   {k}: R²={r2_score(y49, v):.5f}")

winner_key, _ = ranked[0]
print(f">> winner: {winner_key}")

# Apply winner to test
test_gh = test["geohash"]; test_ts = test["tslot"]
test_lat = test["lat"]; test_lon = test["lon"]


def apply_name(name, pred_test):
    if name == "v16_raw":
        return pred_test
    cur = pred_test
    for op in name.split("+"):
        if op.startswith("temp"):
            # temp_wW_bB
            parts = op.split("_")
            w = int(parts[1][1:]); b = float(parts[2][1:])
            cur = temporal_smooth(cur, test_gh, test_ts, w, b)
        elif op.startswith("sp"):
            # sp_kK_bB
            parts = op.split("_")
            k = int(parts[1][1:]); b = float(parts[2][1:])
            cur = spatial_smooth(cur, test_gh, test_ts, test_lat, test_lon, k, b)
    return cur


pred_final = np.clip(apply_name(winner_key, v16_test_pred), 0.0, 1.0)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred_final})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(f">> wrote submission.csv (winner='{winner_key}')", sub.shape)
print(sub.head())
