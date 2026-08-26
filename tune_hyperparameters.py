"""
tune_hyperparameters.py
Tunes the XGBoost classifier (the best-performing model from train_model.py)
using RandomizedSearchCV over the training set, then evaluates the tuned
model on the untouched test set for a fair, honest comparison against the
baseline default-hyperparameter model.

Run after train_model.py.
"""

import numpy as np
import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURE_COLS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_impact", "koi_steff", "koi_slogg",
    "koi_srad", "koi_kepmag",
]
TARGET_COL = "koi_disposition"

df = pd.read_csv("data/kepler_koi.csv")
df_model = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

le = LabelEncoder()
y_all = le.fit_transform(df_model[TARGET_COL])
X_all = df_model[FEATURE_COLS].values
class_names = le.classes_

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.4, random_state=RANDOM_STATE, stratify=y_all)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

# Baseline (from train_model.py) for direct comparison
baseline = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                          eval_metric="mlogloss", random_state=RANDOM_STATE)
baseline.fit(X_train, y_train)
baseline_test_acc = accuracy_score(y_test, baseline.predict(X_test))
print(f"Baseline XGBoost (default hyperparameters) test accuracy: {baseline_test_acc:.4f}")

# Randomized hyperparameter search, 5-fold stratified CV, train set only
param_dist = {
    "n_estimators": [100, 200, 300, 400, 600],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.5],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=40,
    scoring="accuracy",
    cv=cv_strategy,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

print("\nRunning RandomizedSearchCV (40 candidates x 5-fold CV = 200 fits)...")
start = time.time()
search.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Search completed in {elapsed:.1f}s")

print("\nBest CV accuracy (train set, 5-fold):", round(search.best_score_, 4))
print("Best hyperparameters:", search.best_params_)

tuned_model = search.best_estimator_
tuned_test_acc = accuracy_score(y_test, tuned_model.predict(X_test))
tuned_proba = tuned_model.predict_proba(X_test)
tuned_auc = roc_auc_score(y_test, tuned_proba, multi_class="ovr")

print(f"\nTuned XGBoost test accuracy: {tuned_test_acc:.4f}")
print(f"Tuned XGBoost macro ROC-AUC: {tuned_auc:.4f}")
print(f"\nImprovement over baseline: {tuned_test_acc - baseline_test_acc:+.4f} ({(tuned_test_acc - baseline_test_acc)*100:+.2f} pp)")

print("\nClassification report (tuned model):")
report = classification_report(y_test, tuned_model.predict(X_test), target_names=class_names)
print(report)

# Save the tuned model only if it actually beats the baseline -- an honest
# comparison, not an assumption that tuning always helps.
if tuned_test_acc > baseline_test_acc:
    joblib.dump(tuned_model, "exoplanet_best_model.joblib")
    joblib.dump("XGBoost (tuned)", "best_model_name.joblib")
    print("\nTuned model outperformed the baseline and has replaced it as the saved model.")
else:
    print("\nTuned model did NOT outperform the baseline on the test set.")
    print("Keeping the original baseline model as the saved model (no change made).")

with open("tuning_results.txt", "w") as f:
    f.write("HYPERPARAMETER TUNING RESULTS\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Baseline XGBoost test accuracy:  {baseline_test_acc:.4f}\n")
    f.write(f"Tuned XGBoost test accuracy:     {tuned_test_acc:.4f}\n")
    f.write(f"Difference:                      {tuned_test_acc - baseline_test_acc:+.4f}\n\n")
    f.write(f"Best CV accuracy (train, 5-fold): {search.best_score_:.4f}\n")
    f.write(f"Best hyperparameters: {search.best_params_}\n\n")
    f.write(f"Tuned macro ROC-AUC: {tuned_auc:.4f}\n\n")
    f.write(report)
    f.write(f"\nModel saved: {'Yes, tuned model replaced baseline' if tuned_test_acc > baseline_test_acc else 'No, baseline retained'}\n")

print("\nSaved tuning_results.txt")
