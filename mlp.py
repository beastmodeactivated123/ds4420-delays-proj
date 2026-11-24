# MLP that predicts whether a flight arrives 15+ min late, using schedule info only
# Train on 2025, test on 2026
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

CATEGORICAL = ["carrier", "origin", "dest", "month", "day_of_week", "dep_hour", "arr_hour"]
NUMERIC = ["distance"]

df = pd.read_parquet("data/processed/flights.parquet")
for col in ["carrier", "origin", "dest"]:
    df[col] = df[col].astype(str)

# Random samples so training finishes in a few minutes
train = df[df["year"] == 2025].sample(400_000, random_state=42)
test = df[df["year"] == 2026].sample(100_000, random_state=42)
train, val = train_test_split(train, test_size=0.1, random_state=42)

prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])
X_train = prep.fit_transform(train)
X_val = prep.transform(val)
X_test = prep.transform(test)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024, max_iter=30,
                    early_stopping=True, random_state=42)
mlp.fit(X_train, train["late"])

# Only ~22% of flights are late, so a 0.5 cutoff almost never predicts "late".
# Pick the cutoff that gives the best F1 on the validation set instead.
val_prob = mlp.predict_proba(X_val)[:, 1]
cutoffs = np.arange(0.05, 0.6, 0.01)
best = cutoffs[np.argmax([f1_score(val["late"], val_prob >= c) for c in cutoffs])]

test_prob = mlp.predict_proba(X_test)[:, 1]
pred = test_prob >= best
y = test["late"]

results = f"""Train: {len(train):,} flights (2025) | Test: {len(test):,} flights (2026)
Share late in test: {y.mean():.1%}
Baseline (always predict on time): accuracy {1 - y.mean():.3f}, F1 0.000

MLP, hidden layers (64, 32), cutoff {best:.2f}
  AUC:       {roc_auc_score(y, test_prob):.3f}
  F1:        {f1_score(y, pred):.3f}
  Precision: {precision_score(y, pred):.3f}
  Recall:    {recall_score(y, pred):.3f}
  Accuracy:  {(pred == y).mean():.3f}
"""
print(results)
Path("results/mlp_results.txt").write_text(results)

# ---- Figures ----
RocCurveDisplay.from_predictions(y, test_prob, name="MLP")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title("ROC curve, 2026 test flights")
plt.savefig("figures/mlp_roc.png", dpi=200, bbox_inches="tight")

ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["On time", "Late"],
                                        cmap="Blues", values_format=",")
plt.title(f"MLP confusion matrix (cutoff {best:.2f})")
plt.savefig("figures/mlp_confusion.png", dpi=200, bbox_inches="tight")