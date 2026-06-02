"""Generate solution.ipynb from solution.py (sectioned cells)."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "solution.py"), encoding="utf-8").read()

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
