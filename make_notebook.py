"""Generate solution.ipynb from solution.py (sectioned cells)."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "solution.py"), encoding="utf-8").read()
v9_path = os.path.join(HERE, "v9_chronos.py")
v9_src = open(v9_path, encoding="utf-8").read() if os.path.exists(v9_path) else None
v13_path = os.path.join(HERE, "v13_residual.py")
v13_src = open(v13_path, encoding="utf-8").read() if os.path.exists(v13_path) else None
v15_path = os.path.join(HERE, "v15_pseudo.py")
v15_src = open(v15_path, encoding="utf-8").read() if os.path.exists(v15_path) else None
v16_path = os.path.join(HERE, "v16_nn.py")
v16_src = open(v16_path, encoding="utf-8").read() if os.path.exists(v16_path) else None

# Split into logical sections on the banner comments
section_re = re.compile(
    r"# ---+ #\n# (.*?)\s*#\n# ---+ #\n", re.MULTILINE
)
chunks, prev_end, prev_title = [], 0, "Setup"
for m in section_re.finditer(src):
    chunks.append((prev_title, src[prev_end:m.start()].rstrip()))
    prev_title = m.group(1).strip()
    prev_end = m.end()
chunks.append((prev_title, src[prev_end:].rstrip()))

# Build markdown intro
intro_md = (
    "# Flipkart GRiD-Lock 2.0 — Traffic Demand Prediction\n\n"
    "**Problem.** Predict the `demand` value for each test record. "
    "Metric: `score = max(0, 100 * R²(actual, predicted))`.\n\n"
    "**Approach.**\n"
    "1. Feature engineering: decode 6-char geohash → (lat, lon); cyclic time features (hour/min sin–cos); "
    "geohash prefixes (1–5) for hierarchical area signal.\n"
    "2. Cross-day reference features: day-48 same-timestamp demand at each geohash, rolling smoothing, "
    "geohash and prefix aggregates (mean / std / median / etc.).\n"
    "3. **Leakage control:** the same-timestamp day-48 lookup is the target itself on day-48 rows, so it is "
    "masked to NaN on day-48 training rows. Validation uses **day-49 training rows only**, which mirrors "
    "the test scenario (predict day-49 from day-48 context).\n"
    "4. GPU-trained **CatBoost** and **XGBoost** regressors, 5-fold CV, weight-tuned blend.\n\n"
    "Reported honest OOF R² ≈ 0.953 (score ≈ 95.3)."
)

def make_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def make_code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

cells = [make_md(intro_md)]
for title, code in chunks:
    if not code.strip():
        continue
    cells.append(make_md(f"## {title}"))
    cells.append(make_code(code + "\n"))

if v9_src:
    cells.append(make_md("## v9 — Chronos-Bolt time-series forecast\n\n"
                         "Loads the artifacts produced above and adds a 7th stack member: "
                         "an Amazon Chronos-Bolt-small zero-shot forecast per geohash."))
    cells.append(make_code(v9_src + "\n"))

if v13_src:
    cells.append(make_md("## v13 — two-stage residual model\n\n"
                         "Hard-codes the dominant signal `baseline = d48_same_ts` and trains a "
                         "CatBoost on the cross-day delta `y - baseline`, using only day-49 train rows. "
                         "Final pred = `baseline + delta`. This becomes the 8th stack member and "
                         "the Ridge meta-learner refits over all 8 base models."))
    cells.append(make_code(v13_src + "\n"))

if v15_src:
    cells.append(make_md("## v15 — pseudo-labelled day-49 model\n\n"
                         "Uses v13 test predictions as pseudo-labels on the test rows, then trains a "
                         "CatBoost on ~50k mixed `(real day-49 train + pseudo-labelled test)` examples — "
                         "real rows get sample weight 1.0, pseudo rows 0.5. This lets the model see "
                         "DAYTIME hours that my honest CV cannot, while still being validated only on "
                         "real-labelled day-49 rows. Added as the 9th stack member; the Ridge meta refits."))
    cells.append(make_code(v15_src + "\n"))

if v16_src:
    cells.append(make_md("## v16 — PyTorch NN with categorical embeddings\n\n"
                         "A custom MLP with embeddings for `geohash`, `hour`, and other categoricals + "
                         "numeric features. Trained with the same 5-fold day-49 honest CV and sample "
                         "weighting as the GBDT base models. Added as the 10th stack member. The Ridge "
                         "meta-learner assigned it weight 0, indicating its predictions overlap too much "
                         "with the trees to contribute uniquely — but it's documented here as a "
                         "completeness check on alternative architectures."))
    cells.append(make_code(v16_src + "\n"))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(HERE, "solution.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("wrote", out, "with", len(cells), "cells")
