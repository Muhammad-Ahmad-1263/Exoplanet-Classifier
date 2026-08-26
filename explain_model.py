"""
explain_model.py
Generates SHAP (SHapley Additive exPlanations) plots for the trained XGBoost
model, showing which features drive individual predictions and overall
model behavior -- going beyond simple feature importance to explain the
*direction* and *magnitude* of each feature's effect per class.

Run after train_model.py.
"""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
X_all = df_model[FEATURE_COLS]
class_names = le.classes_

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.4, random_state=RANDOM_STATE, stratify=y_all)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp)

model = joblib.load("exoplanet_best_model.joblib")

print("Computing SHAP values on the test set (this may take a moment)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# shap_values shape depends on xgboost/shap version: either a list per class
# or a single (n_samples, n_features, n_classes) array. Handle both.
if isinstance(shap_values, list):
    sv_per_class = shap_values
else:
    sv_per_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

# Summary plot per class
for i, cname in enumerate(class_names):
    plt.figure()
    shap.summary_plot(sv_per_class[i], X_test, show=False, plot_size=(9, 6))
    plt.title(f"SHAP Summary: {cname}")
    plt.tight_layout()
    plt.savefig(f"plots/shap_summary_{cname.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plots/shap_summary_{cname.lower().replace(' ', '_')}.png")

# Overall mean absolute SHAP value per feature, averaged across classes
mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in sv_per_class], axis=0)
importance_df = pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs_shap}) \
                   .sort_values("mean_abs_shap", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance_df["feature"], importance_df["mean_abs_shap"], color="darkorange")
plt.title("Mean |SHAP value| Across All Classes")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("plots/shap_overall_importance.png", dpi=150)
plt.close()
print("Saved plots/shap_overall_importance.png")

print("\nTop 5 most influential features overall:")
print(importance_df.sort_values("mean_abs_shap", ascending=False).head(5).to_string(index=False))
