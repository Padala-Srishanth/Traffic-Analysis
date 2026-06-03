"""
v16: Custom PyTorch NN with categorical embeddings + MLP head.
Genuinely different inductive bias from trees and Chronos.
Trained with the same 5-fold day-49 honest CV as the other base models;
added to the Ridge stack as the 10th member.
"""
import sys, io, os, warnings, time
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RNG = 42
torch.manual_seed(RNG); np.random.seed(RNG)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> torch device: {DEVICE}")


# --------------------------------------------------------------------------- #
# Load data + minimal features for NN                                         #
# --------------------------------------------------------------------------- #
print(">> loading data")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def parse_ts(s):
    h, m = s.str.split(":", expand=True).astype(int).T.values
    return h.astype(int), m.astype(int), (h * 60 + m).astype(int), (h * 4 + m // 15).astype(int)

train["hour"], train["minute"], train["tmin"], train["tslot"] = parse_ts(train["timestamp"])
test["hour"],  test["minute"],  test["tmin"],  test["tslot"]  = parse_ts(test["timestamp"])

# d48 lookup
d48 = train[train["day"] == 48][["geohash", "tslot", "demand"]]
d48_map = d48.set_index(["geohash", "tslot"])["demand"]
gh_d48_mean = d48.groupby("geohash")["demand"].mean()
gh_d48_std  = d48.groupby("geohash")["demand"].std()
overall_d48_mean = float(d48["demand"].mean())

def fill_baseline(df):
    keys = list(zip(df["geohash"], df["tslot"]))
    return pd.Series(keys, index=df.index).map(d48_map)\
            .fillna(df["geohash"].map(gh_d48_mean)).fillna(overall_d48_mean)\
            .astype(np.float32).to_numpy()

train["d48_same_ts"] = fill_baseline(train)
test["d48_same_ts"]  = fill_baseline(test)
train["gh_d48_mean"] = train["geohash"].map(gh_d48_mean).fillna(overall_d48_mean).astype(np.float32)
test["gh_d48_mean"]  = test["geohash"].map(gh_d48_mean).fillna(overall_d48_mean).astype(np.float32)
train["gh_d48_std"]  = train["geohash"].map(gh_d48_std).fillna(0.0).astype(np.float32)
test["gh_d48_std"]   = test["geohash"].map(gh_d48_std).fillna(0.0).astype(np.float32)

# gh × hour mean
gh_hour = d48.copy()
gh_hour["hour"] = gh_hour["tslot"] // 4
gh_hour_mean = gh_hour.groupby(["geohash", "hour"])["demand"].mean()
def gh_hour_lookup(df):
    return df.set_index(["geohash", "hour"]).index.map(gh_hour_mean).to_series(index=df.index).fillna(overall_d48_mean).astype(np.float32).to_numpy()
train["gh_hour_d48_mean"] = gh_hour_lookup(train)
test["gh_hour_d48_mean"]  = gh_hour_lookup(test)

# d49 recent (max-tslot day-49 demand for this geohash with tslot < query)
d49 = train[train["day"] == 49][["geohash", "tslot", "demand"]].rename(columns={"demand": "d49_recent"}).sort_values("tslot").reset_index(drop=True)
def attach_d49_recent(df):
    src = df[["geohash", "tslot"]].copy()
    src["_orig"] = np.arange(len(src))
    src = src.sort_values("tslot")
    merged = pd.merge_asof(src, d49, on="tslot", by="geohash", direction="backward", allow_exact_matches=False)
    merged = merged.sort_values("_orig")
    return merged["d49_recent"].to_numpy()
train["d49_recent"] = attach_d49_recent(train)
test["d49_recent"]  = attach_d49_recent(test)

# decode geohash
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

train["hour_sin"] = np.sin(2*np.pi*train["hour"]/24); train["hour_cos"] = np.cos(2*np.pi*train["hour"]/24)
test["hour_sin"]  = np.sin(2*np.pi*test["hour"]/24);  test["hour_cos"]  = np.cos(2*np.pi*test["hour"]/24)


# --------------------------------------------------------------------------- #
# Build NN inputs                                                             #
# --------------------------------------------------------------------------- #
CAT_COLS = ["geohash", "RoadType", "LargeVehicles", "Landmarks", "Weather", "hour"]
NUM_COLS = ["day", "tslot", "tmin", "hour_sin", "hour_cos",
            "lat", "lon", "NumberofLanes", "Temperature",
            "d48_same_ts", "gh_d48_mean", "gh_d48_std",
            "gh_hour_d48_mean", "d49_recent"]

# build vocabularies
cat_vocab = {}
for c in CAT_COLS:
    vals = pd.concat([train[c].astype("string").fillna("NA"),
                      test[c].astype("string").fillna("NA")], ignore_index=True)
    cat_vocab[c] = {v: i + 1 for i, v in enumerate(vals.unique())}  # +1 so 0 = padding/unknown
    cat_vocab[c]["__UNK__"] = 0
    print(f"   cat {c}: {len(cat_vocab[c])} unique")

def encode_cat(df):
    out = np.zeros((len(df), len(CAT_COLS)), dtype=np.int64)
    for i, c in enumerate(CAT_COLS):
        out[:, i] = df[c].astype("string").fillna("NA").map(cat_vocab[c]).fillna(0).astype(np.int64).to_numpy()
    return out

X_cat_train = encode_cat(train)
X_cat_test  = encode_cat(test)

# numeric: impute median + standardize
imp = SimpleImputer(strategy="median")
sc = StandardScaler()
X_num_train = sc.fit_transform(imp.fit_transform(train[NUM_COLS])).astype(np.float32)
X_num_test  = sc.transform(imp.transform(test[NUM_COLS])).astype(np.float32)
y_train = train["demand"].astype(np.float32).to_numpy()
print(f"   num_features: {X_num_train.shape[1]}, cat_features: {X_cat_train.shape[1]}")


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class TabNN(nn.Module):
    def __init__(self, cat_sizes, num_dim, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(sz, min(32, max(4, sz // 8))) for sz in cat_sizes])
        emb_dim = sum(e.embedding_dim for e in self.embeds)
        in_dim = emb_dim + num_dim
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.mlp(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# 5-fold CV on day-49 honest val                                              #
# --------------------------------------------------------------------------- #
day49_idx = np.where(train["day"].values == 49)[0]
day48_idx = np.where(train["day"].values == 48)[0]

# sample weights matching the GBDT pipeline (day-48 daytime: 1.5, day-49: 5.0)
sample_weight = np.ones(len(train), dtype=np.float32)
sample_weight[(train["day"].values == 48) & (train["tslot"].values >= 9)] = 1.5
sample_weight[train["day"].values == 49] = 5.0

cat_sizes = [len(cat_vocab[c]) + 1 for c in CAT_COLS]  # +1 safety for max idx
print(f"   cat embedding sizes: {cat_sizes}")

oof_nn = np.full(len(train), np.nan, dtype=np.float32)
pred_nn = np.zeros(len(test), dtype=np.float32)
kf = KFold(n_splits=5, shuffle=True, random_state=RNG)

BATCH = 4096
EPOCHS = 80
PATIENCE = 12

for fold, (tr_d49, va_d49) in enumerate(kf.split(day49_idx), 1):
    tr_idx = np.concatenate([day48_idx, day49_idx[tr_d49]])
    va_idx = day49_idx[va_d49]

    Xc_tr = torch.tensor(X_cat_train[tr_idx]); Xn_tr = torch.tensor(X_num_train[tr_idx])
    y_tr  = torch.tensor(y_train[tr_idx]); w_tr = torch.tensor(sample_weight[tr_idx])
    Xc_va = torch.tensor(X_cat_train[va_idx]).to(DEVICE)
    Xn_va = torch.tensor(X_num_train[va_idx]).to(DEVICE)
    y_va_np = y_train[va_idx]

    ds = TensorDataset(Xc_tr, Xn_tr, y_tr, w_tr)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)

    model = TabNN(cat_sizes, X_num_train.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_r2 = -1; best_pred = None; bad = 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0; n = 0
        for xc, xn, y, w in loader:
            xc = xc.to(DEVICE, non_blocking=True); xn = xn.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True); w = w.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            pred = model(xc, xn)
            loss = ((pred - y) ** 2 * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_sum += loss.item() * y.size(0); n += y.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            v_pred = model(Xc_va, Xn_va).cpu().numpy()
        v_r2 = r2_score(y_va_np, v_pred)
        if v_r2 > best_r2 + 1e-5:
            best_r2 = v_r2
            best_pred = v_pred
            # also stash test pred at this checkpoint
            with torch.no_grad():
                t_chunks = []
                for i in range(0, len(X_cat_test), 8192):
                    Xc_t = torch.tensor(X_cat_test[i:i+8192]).to(DEVICE)
                    Xn_t = torch.tensor(X_num_test[i:i+8192]).to(DEVICE)
                    t_chunks.append(model(Xc_t, Xn_t).cpu().numpy())
                best_test_pred = np.concatenate(t_chunks)
            bad = 0
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"   fold {fold} epoch {epoch:3d}  train_loss={loss_sum/n:.6f}  val_R²={v_r2:.5f}  best={best_r2:.5f}")
        if bad >= PATIENCE:
            print(f"   fold {fold} early stop at epoch {epoch}, best val R²={best_r2:.5f}")
            break

    oof_nn[va_idx] = best_pred
    pred_nn += best_test_pred / kf.n_splits
    print(f"  >> fold {fold}: val R²={best_r2:.5f}  time={time.time()-t0:.1f}s")

nn_oof_r2 = r2_score(y_train[day49_idx], oof_nn[day49_idx])
print(f">> NN OOF R² (day-49 raw) = {nn_oof_r2:.5f}  (score = {max(0,100*nn_oof_r2):.3f})")


# --------------------------------------------------------------------------- #
# Refit Ridge over base models + NN                                           #
# --------------------------------------------------------------------------- #
art = np.load(os.path.join(DATA_DIR, "artifacts.npz"))
y_raw, day49_idx_art = art["y_raw"], art["day49_idx"]
y49 = y_raw[day49_idx]

labels = ["cb", "xgb", "lgb", "hgb", "et", "knn", "chrS_50", "residual", "pseudo", "nn"]
oof_cols = [art["oof_cb"], art["oof_xgb"], art["oof_lgb"], art["oof_hgb"],
            art["oof_et"], art["oof_knn"], art["oof_chrS"],
            art["oof_residual"], art["oof_pseudo"], oof_nn]
test_cols = [art["pred_cb"], art["pred_xgb"], art["pred_lgb"], art["pred_hgb"],
             art["pred_et"], art["pred_knn"], art["pred_chrS"],
             art["pred_residual"], art["pred_pseudo"], pred_nn]

single_r2 = {lbl: r2_score(y49, c[day49_idx]) for lbl, c in zip(labels, oof_cols)}
print(">> single-model R²:", {k: round(v, 5) for k, v in single_r2.items()})

oof_stack = np.column_stack([c[day49_idx] for c in oof_cols])
test_stack = np.column_stack(test_cols)

meta = Ridge(alpha=0.1, fit_intercept=False, positive=True)
meta.fit(oof_stack, y49)
blend_r2 = r2_score(y49, meta.predict(oof_stack))
print(">> Ridge meta weights =", dict(zip(labels, np.round(meta.coef_, 4))))
print(f">> Ridge-stacked OOF R² (10 models) = {blend_r2:.5f}  (score = {max(0,100*blend_r2):.3f})")

pred = np.clip(meta.predict(test_stack), 0.0, 1.0)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": pred})
sub.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
print(">> wrote submission.csv shape =", sub.shape)
print(sub.head())

np.savez(
    os.path.join(DATA_DIR, "artifacts.npz"),
    **{k: art[k] for k in art.files},
    oof_nn=oof_nn, pred_nn=pred_nn,
    meta_coef_v16=meta.coef_,
)
print(">> saved artifacts.npz")
