"""
app.py
Interactive Streamlit app for classifying a Kepler Object of Interest (KOI)
as CANDIDATE, CONFIRMED, or FALSE POSITIVE, using the trained model from
train_model.py.

Run with: streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap

st.set_page_config(page_title="Exoplanet Disposition Classifier", page_icon="🪐", layout="wide")

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("exoplanet_best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    le = joblib.load("label_encoder.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    best_name = joblib.load("best_model_name.joblib")
    return model, scaler, le, feature_cols, best_name

model, scaler, le, feature_cols, best_name = load_artifacts()

@st.cache_resource
def load_explainer(_model):
    if best_name == "Neural Network":
        return None
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

@st.cache_data
def load_reference_data():
    df = pd.read_csv("data/kepler_koi.csv")
    return df.dropna(subset=feature_cols + ["koi_disposition"])

ref_df = load_reference_data()

# Slider bounds derived from the 1st-99th percentile of the real dataset
SLIDER_CONFIG = {
    "koi_period":     ("Orbital Period (days)", 0.5, 500.0, 9.75, 0.1),
    "koi_duration":   ("Transit Duration (hours)", 0.5, 15.0, 3.79, 0.1),
    "koi_depth":      ("Transit Depth (ppm)", 10.0, 20000.0, 421.0, 10.0),
    "koi_prad":       ("Planetary Radius (Earth radii)", 0.3, 30.0, 2.39, 0.1),
    "koi_teq":        ("Equilibrium Temperature (K)", 150.0, 3000.0, 878.0, 10.0),
    "koi_insol":      ("Insolation Flux (Earth flux)", 0.0, 5000.0, 141.6, 1.0),
    "koi_model_snr":  ("Transit Signal-to-Noise", 1.0, 500.0, 23.0, 1.0),
    "koi_impact":     ("Impact Parameter", 0.0, 1.5, 0.54, 0.01),
    "koi_steff":      ("Stellar Effective Temperature (K)", 3500.0, 8000.0, 5767.0, 10.0),
    "koi_slogg":      ("Stellar Surface Gravity (log g)", 2.0, 5.0, 4.44, 0.01),
    "koi_srad":       ("Stellar Radius (solar radii)", 0.3, 5.0, 1.0, 0.01),
    "koi_kepmag":     ("Kepler Magnitude (brightness)", 8.0, 18.0, 14.52, 0.1),
}

st.title("🪐 Exoplanet Disposition Classifier")
st.caption(
    "Predicts whether a Kepler Object of Interest is a **Candidate**, "
    "**Confirmed Exoplanet**, or **False Positive**, using the trained "
    f"**{best_name}** model on real NASA Kepler mission data."
)

with st.expander("About this model", expanded=False):
    st.markdown(f"""
    - **Dataset:** NASA Kepler Cumulative KOI Table ({ref_df.shape[0]:,} objects after cleaning)
    - **Model:** {best_name}, trained on 12 physical/observational features
    - **Note:** automated Kepler pipeline vetting flags (`koi_fpflag_*`) are intentionally
      excluded from this model. Including them raises test accuracy to ~91%, but since those
      flags are themselves near-automated determinations of disposition, that number would
      overstate what a model can learn from genuine physical measurements. The ~77% accuracy
      reported for this model reflects classification from physical parameters alone.
    """)

col_inputs, col_results = st.columns([1, 1])

with col_inputs:
    st.subheader("Observed Parameters")
    preset = st.selectbox(
        "Quick preset",
        ["Custom", "Typical Confirmed Exoplanet", "Typical False Positive", "Dataset Median"],
    )

    preset_values = {}
    if preset == "Dataset Median":
        preset_values = ref_df[feature_cols].median().to_dict()
    elif preset == "Typical Confirmed Exoplanet":
        preset_values = ref_df[ref_df["koi_disposition"] == "CONFIRMED"][feature_cols].median().to_dict()
    elif preset == "Typical False Positive":
        preset_values = ref_df[ref_df["koi_disposition"] == "FALSE POSITIVE"][feature_cols].median().to_dict()

    input_values = {}
    for col in feature_cols:
        label, lo, hi, default, step = SLIDER_CONFIG[col]
        val = float(preset_values.get(col, default)) if preset_values else default
        val = min(max(val, lo), hi)
        input_values[col] = st.slider(label, min_value=lo, max_value=hi, value=val, step=step)

with col_results:
    st.subheader("Prediction")

    X_input = np.array([[input_values[c] for c in feature_cols]])

    if best_name == "Neural Network":
        import tensorflow as tf
        nn_model = tf.keras.models.load_model("exoplanet_nn_model.keras")
        X_scaled = scaler.transform(X_input)
        proba = nn_model.predict(X_scaled, verbose=0)[0]
    else:
        proba = model.predict_proba(X_input)[0]

    pred_idx = int(np.argmax(proba))
    pred_label = le.classes_[pred_idx]
    confidence = proba[pred_idx]

    label_colors = {
        "CONFIRMED": "#2ca02c",
        "CANDIDATE": "#ff7f0e",
        "FALSE POSITIVE": "#d62728",
    }

    st.markdown(
        f"### Predicted disposition: "
        f"<span style='color:{label_colors.get(pred_label, '#333')}'>{pred_label}</span>",
        unsafe_allow_html=True,
    )
    st.metric("Model confidence", f"{confidence:.1%}")

    fig = go.Figure(go.Bar(
        x=[le.classes_[i] for i in range(len(proba))],
        y=proba,
        marker_color=[label_colors.get(le.classes_[i], "#333") for i in range(len(proba))],
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 1], title="Probability"),
        xaxis_title="Class",
        title="Class Probabilities",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This tool is for educational purposes, built on real Kepler mission data. "
        "It does not represent an official NASA classification tool."
    )

if explainer is not None:
    st.subheader("Why this prediction? (SHAP explanation)")
    X_input_df = pd.DataFrame(X_input, columns=feature_cols)
    shap_vals = explainer.shap_values(X_input_df)

    if isinstance(shap_vals, list):
        class_shap = shap_vals[pred_idx][0]
    else:
        class_shap = shap_vals[0, :, pred_idx]

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": class_shap,
    }).sort_values("shap_value", key=abs, ascending=True)

    fig_shap = go.Figure(go.Bar(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        orientation="h",
        marker_color=["#2ca02c" if v > 0 else "#d62728" for v in shap_df["shap_value"]],
    ))
    fig_shap.update_layout(
        title=f"Feature contributions toward '{pred_label}'",
        xaxis_title="SHAP value (push toward / away from this class)",
        height=380,
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption(
        "Green bars pushed the prediction toward the shown class; red bars pushed against it. "
        "Values reflect this single prediction, not the model's global feature importance."
    )
