"""
train_model.py
Trains and evaluates models to classify Kepler Objects of Interest (KOIs) as
CONFIRMED, CANDIDATE, or FALSE POSITIVE, using the real NASA Kepler cumulative
KOI table. Saves the final model, scaler, and diagnostic plots.

Data source: NASA Exoplanet Archive, Kepler Cumulative KOI Table
https://exoplanetarchive.ipac.caltech.edu/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras import layers, models

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
sns.set_style("darkgrid")

PLOTS_DIR = "plots"
DATA_PATH = "data/kepler_koi.csv"

# ---------------------------------------------------------------------------
# 1. Load and clean data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Raw data shape: {df.shape}")
print(df["koi_disposition"].value_counts())

# Physical / observational features. Automated vetting flags (koi_fpflag_*)
# are intentionally excluded from the primary model -- see the leakage check
# below for why.
FEATURE_COLS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_impact", "koi_steff", "koi_slogg",
    "koi_srad", "koi_kepmag",
]
TARGET_COL = "koi_disposition"

df_model = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
print(f"Rows after dropping missing values: {df_model.shape[0]}")

le = LabelEncoder()
y_all = le.fit_transform(df_model[TARGET_COL])
class_names = le.classes_
print("Classes:", list(class_names))

X_all = df_model[FEATURE_COLS].values

# ---------------------------------------------------------------------------
# 2. Leakage check: automated vetting flags vs. physical features
#    This dataset includes koi_fpflag_nt/ss/co/ec -- automated diagnostic
#    flags computed by the Kepler pipeline that are extremely predictive of
#    the final disposition almost by definition. Including them produces
#    inflated accuracy that does not reflect a genuine physical-parameter
#    classification task, so this is checked explicitly and reported before
#    being excluded from the final model.
# ---------------------------------------------------------------------------
FPFLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
df_leak = df.dropna(subset=FEATURE_COLS + FPFLAG_COLS + [TARGET_COL]).copy()
y_leak = le.transform(df_leak[TARGET_COL])
X_leak = df_leak[FEATURE_COLS + FPFLAG_COLS].values

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_leak, y_leak, test_size=0.2, random_state=RANDOM_STATE, stratify=y_leak)
leak_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)
leak_model.fit(Xl_train, yl_train)
leak_acc = accuracy_score(yl_test, leak_model.predict(Xl_test))
print(f"\n[Leakage check] Random Forest WITH vetting flags -> test accuracy: {leak_acc:.4f}")
print("These flags are excluded from the primary model below.\n")

# ---------------------------------------------------------------------------
# 3. Train / cross-validation / test split (physical features only)
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.4, random_state=RANDOM_STATE, stratify=y_all)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cv_s = scaler.transform(X_cv)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} | CV: {X_cv.shape[0]} | Test: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 4. Exploratory plots
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.countplot(x=df_model[TARGET_COL], order=df_model[TARGET_COL].value_counts().index,
              hue=df_model[TARGET_COL], palette="viridis", legend=False)
plt.title("Class Distribution: Kepler Objects of Interest")
plt.xlabel("")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/class_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(9, 7))
corr = df_model[FEATURE_COLS].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/correlation_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Neural network + bias/variance diagnosis via learning curve
# ---------------------------------------------------------------------------
def build_nn(input_dim, hidden_units=(64, 32), dropout=0.1):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units[0], activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(hidden_units[1], activation="relu"),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
train_errs, cv_errs = [], []
n_train = X_train_s.shape[0]

for frac in train_sizes:
    n_sub = int(n_train * frac)
    idx = np.random.RandomState(RANDOM_STATE).choice(n_train, n_sub, replace=False)
    m = build_nn(X_train_s.shape[1])
    m.fit(X_train_s[idx], y_train[idx], epochs=40, batch_size=32, verbose=0)
    train_acc = m.evaluate(X_train_s[idx], y_train[idx], verbose=0)[1]
    cv_acc = m.evaluate(X_cv_s, y_cv, verbose=0)[1]
    train_errs.append(1 - train_acc)
    cv_errs.append(1 - cv_acc)
    print(f"Train size {n_sub:5d} | train error: {1-train_acc:.3f} | cv error: {1-cv_acc:.3f}")

plt.figure(figsize=(8, 5))
sizes_actual = [int(n_train * f) for f in train_sizes]
plt.plot(sizes_actual, train_errs, marker="o", label="Train Error")
plt.plot(sizes_actual, cv_errs, marker="o", label="CV Error")
plt.xlabel("Training Set Size")
plt.ylabel("Error Rate")
plt.title("Learning Curve: Neural Network Bias/Variance Diagnosis")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/learning_curve.png", dpi=150)
plt.close()

# Final neural network trained on full training set
final_nn = build_nn(X_train_s.shape[1])
history = final_nn.fit(X_train_s, y_train, validation_data=(X_cv_s, y_cv),
                        epochs=60, batch_size=32, verbose=0)
nn_test_acc = final_nn.evaluate(X_test_s, y_test, verbose=0)[1]

# ---------------------------------------------------------------------------
# 6. Tree-based models
# ---------------------------------------------------------------------------
tree_models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                              eval_metric="mlogloss", random_state=RANDOM_STATE),
}

results = {"Neural Network": nn_test_acc}
fitted_models = {}
for name, model in tree_models.items():
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, test_pred)
    fitted_models[name] = model

results_df = pd.DataFrame(list(results.items()), columns=["Model", "Test Accuracy"]) \
               .sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
print("\nFinal test accuracy by model:")
print(results_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Model", y="Test Accuracy", hue="Model",
            palette="viridis", legend=False)
plt.ylim(0, 1)
plt.title("Model Comparison: Test Set Accuracy (Physical Features Only)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. Best model: full evaluation
# ---------------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
print(f"\nBest model: {best_name}")

if best_name == "Neural Network":
    y_pred_best = np.argmax(final_nn.predict(X_test_s, verbose=0), axis=1)
    y_proba_best = final_nn.predict(X_test_s, verbose=0)
else:
    y_pred_best = fitted_models[best_name].predict(X_test)
    y_proba_best = fitted_models[best_name].predict_proba(X_test)

report = classification_report(y_test, y_pred_best, target_names=class_names)
print(report)

try:
    auc = roc_auc_score(y_test, y_proba_best, multi_class="ovr")
    print(f"Macro ROC-AUC (OvR): {auc:.4f}")
except Exception as e:
    print("ROC-AUC could not be computed:", e)

plt.figure(figsize=(6.5, 5.5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix: {best_name}")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# Feature importance (from XGBoost, regardless of which model won overall)
importances = fitted_models["XGBoost"].feature_importances_
imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances}) \
           .sort_values("importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(imp_df["feature"], imp_df["importance"], color="teal")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. Save best model + scaler + label encoder for the Streamlit app
# ---------------------------------------------------------------------------
if best_name == "Neural Network":
    final_nn.save("exoplanet_nn_model.keras")
else:
    joblib.dump(fitted_models[best_name], "exoplanet_best_model.joblib")

joblib.dump(scaler, "scaler.joblib")
joblib.dump(le, "label_encoder.joblib")
joblib.dump(FEATURE_COLS, "feature_columns.joblib")
joblib.dump(best_name, "best_model_name.joblib")

with open("results_summary.txt", "w") as f:
    f.write("KEPLER EXOPLANET CLASSIFICATION - RESULTS SUMMARY\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Dataset: NASA Kepler Cumulative KOI Table\n")
    f.write(f"Rows used (after cleaning): {df_model.shape[0]}\n")
    f.write(f"Classes: {list(class_names)}\n\n")
    f.write(f"Leakage check (Random Forest WITH vetting flags): {leak_acc:.4f} test accuracy\n")
    f.write("These flags were excluded from the primary model as they are automated\n")
    f.write("pipeline outputs that near-determine disposition, not physical measurements.\n\n")
    f.write("Final test accuracy (physical features only):\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    f.write(f"Best model: {best_name}\n\n")
    f.write(report)

print("\nSaved model artifacts and plots. Done.")
