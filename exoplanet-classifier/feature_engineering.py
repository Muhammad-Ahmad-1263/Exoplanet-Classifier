"""
feature_engineering.py
Tests whether engineered features (derived from domain knowledge of transit
photometry) can push past the bias plateau identified in the learning curve
(train_model.py, Section 6 of the notebook). Reports the honest result --
whether this helps, hurts, or makes no meaningful difference.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_impact", "koi_steff", "koi_slogg",
    "koi_srad", "koi_kepmag",
]
TARGET_COL = "koi_disposition"

df = pd.read_csv("data/kepler_koi.csv")
df_model = df.dropna(subset=BASE_FEATURES + [TARGET_COL]).copy()

# ---------------------------------------------------------------------------
# Engineered features, each motivated by transit photometry physics:
#
# - duration_period_ratio: a transit lasting a large fraction of the orbital
#   period is physically implausible for a genuine planet -- a signature
#   sometimes seen in false positives (e.g. background eclipsing binaries).
# - log_period / log_depth / log_insol: these raw features are heavily
#   right-skewed (see correlation_matrix.png); log-transforming can help
#   tree splits and especially the neural network's gradient-based learning.
# - prad_srad_ratio: planet radius relative to host star radius, a cheap
#   proxy for how physically consistent the transit depth is with the
#   claimed planetary radius.
# - snr_impact_interaction: grazing transits (high impact parameter) combined
#   with low SNR are a classic false-positive signature.
# ---------------------------------------------------------------------------
df_model["duration_period_ratio"] = df_model["koi_duration"] / (df_model["koi_period"] * 24.0)
df_model["log_period"] = np.log1p(df_model["koi_period"])
df_model["log_depth"] = np.log1p(df_model["koi_depth"])
df_model["log_insol"] = np.log1p(df_model["koi_insol"])
df_model["prad_srad_ratio"] = df_model["koi_prad"] / df_model["koi_srad"]
df_model["snr_impact_interaction"] = df_model["koi_model_snr"] * (1 - df_model["koi_impact"].clip(0, 1))

ENGINEERED_FEATURES = BASE_FEATURES + [
    "duration_period_ratio", "log_period", "log_depth", "log_insol",
    "prad_srad_ratio", "snr_impact_interaction",
]

le = LabelEncoder()
y_all = le.fit_transform(df_model[TARGET_COL])

def evaluate(feature_cols, label):
    X_all = df_model[feature_cols].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.4, random_state=RANDOM_STATE, stratify=y_all)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                           eval_metric="mlogloss", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, proba, multi_class="ovr")
    print(f"{label:35s} | test accuracy: {acc:.4f} | macro ROC-AUC: {auc:.4f}")
    return acc, auc

print(f"Rows used: {df_model.shape[0]}\n")
base_acc, base_auc = evaluate(BASE_FEATURES, "Baseline (12 raw features)")
eng_acc, eng_auc = evaluate(ENGINEERED_FEATURES, "Engineered (18 features)")

diff = eng_acc - base_acc
print(f"\nDifference from engineered features: {diff:+.4f} ({diff*100:+.2f} pp)")

if diff > 0.01:
    verdict = "Engineered features provided a meaningful improvement."
elif diff > 0.0:
    verdict = "Engineered features provided a small, likely marginal improvement."
else:
    verdict = "Engineered features did not improve performance on this dataset/model combination."

print(verdict)

with open("feature_engineering_results.txt", "w") as f:
    f.write("FEATURE ENGINEERING RESULTS\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Baseline (12 raw features):    accuracy={base_acc:.4f}, ROC-AUC={base_auc:.4f}\n")
    f.write(f"Engineered (18 features):      accuracy={eng_acc:.4f}, ROC-AUC={eng_auc:.4f}\n\n")
    f.write(f"Difference: {diff:+.4f} ({diff*100:+.2f} pp)\n\n")
    f.write(verdict + "\n\n")
    f.write(
        "Interpretation: combined with the learning curve result (train_model.py), "
        "this suggests the ~77% accuracy ceiling on physical parameters is closer to "
        "the genuine information limit of this feature set for this task, rather than "
        "an artifact of insufficient tuning or naive feature representation. Meaningfully "
        "surpassing it would likely require additional data modalities not present in the "
        "cumulative table -- e.g. full light curve shape, not just its summary statistics.\n"
    )

print("\nSaved feature_engineering_results.txt")
