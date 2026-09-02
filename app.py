"""
app.py
Interactive Streamlit app for classifying a Kepler Object of Interest (KOI)
as CANDIDATE, CONFIRMED, or FALSE POSITIVE, using the trained model from
train_model.py.

Run with: streamlit run app.py
"""

import joblib
import hashlib
from urllib.parse import quote
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap

st.set_page_config(page_title="Exoplanet Disposition Classifier", page_icon="🪐", layout="wide")

# ---------------------------------------------------------------------------
# Theme toggle. Streamlit's officially supported theming lives in
# .streamlit/config.toml (set at deploy time), but most of Streamlit's own
# CSS is written against standard CSS custom properties (--background-color,
# --text-color, etc.), so re-declaring those at runtime lets a user-facing
# toggle re-skin the app without a restart. This is a best-effort override,
# not an official Streamlit API -- if a future Streamlit version changes
# these internals, the toggle may need updating, but it degrades gracefully
# (worst case: native widgets keep the config.toml light theme while the
# custom sections below still follow the toggle).
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

_theme_col, _ = st.columns([1, 5])
with _theme_col:
    st.session_state.theme = st.radio(
        "Theme", ["Light", "Dark"], horizontal=True,
        index=0 if st.session_state.theme == "Light" else 1,
        label_visibility="collapsed",
    )

if st.session_state.theme == "Dark":
    _bg_grad = "linear-gradient(135deg, #0e1117 0%, #1a1c27 50%, #10131a 100%)"
    _text_color, _bg_color, _sec_bg, _primary = "#fafafa", "#0e1117", "#262730", "#ff6b6b"
    _card_bg = "rgba(38, 39, 48, 0.75)"
else:
    _bg_grad = "linear-gradient(135deg, #eef3ff 0%, #ffffff 50%, #eaf6ff 100%)"
    _text_color, _bg_color, _sec_bg, _primary = "#262730", "#ffffff", "#f0f2f6", "#ff4b4b"
    _card_bg = "rgba(255, 255, 255, 0.75)"

st.markdown(
    f"""
    <style>
    :root {{
        --text-color: {_text_color};
        --background-color: {_bg_color};
        --secondary-background-color: {_sec_bg};
        --primary-color: {_primary};
    }}
    .stApp {{
        background: {_bg_grad};
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

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

# ---------------------------------------------------------------------------
# Solar System reference data, for a visual "which planet does this resemble?"
# comparison. Radius in Earth radii, orbital period in days, and equilibrium
# temperature in Kelvin (assuming a standard 0.3 Bond albedo, no greenhouse
# effect) -- the same physical convention Kepler's koi_teq uses, so these are
# directly comparable to a candidate's measured values.
# Images are real NASA-sourced photographs, public domain (US government
# work), hosted on Wikimedia Commons via the stable Special:FilePath redirect.
# ---------------------------------------------------------------------------
SOLAR_SYSTEM_PLANETS = {
    "Mercury": {"radius": 0.383, "period": 88.0,    "teq": 440, "image": "Mercury_in_color_-_Prockter07-edit1.jpg"},
    "Venus":   {"radius": 0.949, "period": 224.7,   "teq": 232, "image": "Venus-real_color.jpg"},
    "Earth":   {"radius": 1.000, "period": 365.25,  "teq": 255, "image": "The_Earth_seen_from_Apollo_17.jpg"},
    "Mars":    {"radius": 0.532, "period": 687.0,   "teq": 210, "image": "OSIRIS_Mars_true_color.jpg"},
    "Jupiter": {"radius": 11.21, "period": 4331.0,  "teq": 110, "image": "Jupiter_and_its_shrunken_Great_Red_Spot.jpg"},
    "Saturn":  {"radius": 9.45,  "period": 10747.0, "teq": 81,  "image": "Saturn_during_Equinox.jpg"},
    "Uranus":  {"radius": 4.01,  "period": 30589.0, "teq": 58,  "image": "Uranus2.jpg"},
    "Neptune": {"radius": 3.88,  "period": 59800.0, "teq": 47,  "image": "Neptune_Full.jpg"},
}
WIKIMEDIA_BASE = "https://commons.wikimedia.org/wiki/Special:FilePath/"


def wikimedia_direct_url(filename):
    """Build the actual upload.wikimedia.org CDN URL directly, rather than
    relying on the Special:FilePath redirect (which didn't reliably resolve
    as a hotlinked <img> source in testing). Wikimedia stores files under a
    path derived from the MD5 hash of the underscored filename -- this is
    the same deterministic scheme Wikimedia's own servers use, so it doesn't
    depend on any redirect or extra network round-trip."""
    fname = filename.replace(" ", "_")
    digest = hashlib.md5(fname.encode("utf-8")).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{digest[0]}/{digest[0:2]}/{quote(fname)}"


def find_closest_planet(prad, period, teq):
    """Nearest Solar System planet by log-scaled Euclidean distance across
    radius, orbital period, and equilibrium temperature. Log-scaling matters
    here since these quantities span several orders of magnitude (e.g.
    Mercury's 88-day orbit vs. Neptune's ~60,000-day orbit)."""
    best_name, best_dist = None, np.inf
    for name, p in SOLAR_SYSTEM_PLANETS.items():
        d = (
            (np.log10(prad) - np.log10(p["radius"])) ** 2
            + (np.log10(period) - np.log10(p["period"])) ** 2
            + (np.log10(teq) - np.log10(p["teq"])) ** 2
        )
        if d < best_dist:
            best_name, best_dist = name, d
    return best_name

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
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig_shap, width="stretch")
    st.caption(
        "Green bars pushed the prediction toward the shown class; red bars pushed against it. "
        "Values reflect this single prediction, not the model's global feature importance."
    )

st.divider()
st.subheader("🌍 How does this compare to our own Solar System?")
st.caption(
    "A candidate's radius, orbital period, and equilibrium temperature alone don't tell you "
    "much in the abstract -- but comparing them to planets we can actually picture does. "
    "Nearest match is by log-scaled distance across all three quantities."
)

closest = find_closest_planet(input_values["koi_prad"], input_values["koi_period"], input_values["koi_teq"])
closest_data = SOLAR_SYSTEM_PLANETS[closest]

col_match_img, col_match_info = st.columns([1, 2])
with col_match_img:
    try:
        st.image(wikimedia_direct_url(closest_data["image"]), caption=closest, width="stretch")
    except Exception:
        st.info(f"Closest match: **{closest}** (image unavailable)")
with col_match_info:
    st.markdown(f"### Closest Solar System analog: **{closest}**")
    st.markdown(
        f"""
        | | This candidate | {closest} |
        |---|---|---|
        | Radius (Earth radii) | {input_values['koi_prad']:.2f} | {closest_data['radius']:.2f} |
        | Orbital period (days) | {input_values['koi_period']:.1f} | {closest_data['period']:.1f} |
        | Equilibrium temp (K) | {input_values['koi_teq']:.0f} | {closest_data['teq']} |
        """
    )
    st.caption(
        "This is a rough size/temperature/orbit analogy for intuition only -- it says nothing "
        "about the candidate's actual composition, atmosphere, or habitability."
    )

# Radar chart: normalized (log-scaled) comparison across all three dimensions
radar_categories = ["Radius", "Orbital Period", "Equilibrium Temp"]

def normalize_for_radar(prad, period, teq):
    # Log-scale then min-max normalize against the full Mercury-to-Neptune range
    all_r = [p["radius"] for p in SOLAR_SYSTEM_PLANETS.values()]
    all_p = [p["period"] for p in SOLAR_SYSTEM_PLANETS.values()]
    all_t = [p["teq"] for p in SOLAR_SYSTEM_PLANETS.values()]
    def scale(val, ref_list):
        lo, hi = np.log10(min(ref_list)), np.log10(max(ref_list))
        return (np.log10(val) - lo) / (hi - lo)
    return [scale(prad, all_r), scale(period, all_p), scale(teq, all_t)]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=normalize_for_radar(input_values["koi_prad"], input_values["koi_period"], input_values["koi_teq"]) + [None],
    theta=radar_categories + [radar_categories[0]],
    fill="toself", name="This candidate", line_color="#1f77b4",
))
fig_radar.add_trace(go.Scatterpolar(
    r=normalize_for_radar(closest_data["radius"], closest_data["period"], closest_data["teq"]) + [None],
    theta=radar_categories + [radar_categories[0]],
    fill="toself", name=closest, line_color="#ff7f0e", opacity=0.6,
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
    showlegend=True,
    title="Normalized profile: this candidate vs. its closest Solar System match",
    height=420,
)
st.plotly_chart(fig_radar, width="stretch")

st.divider()
st.subheader("🪐 Explore the Solar System")
st.caption("Slide through our own planets to see how their real sizes, orbits, and temperatures compare.")

planet_order = list(SOLAR_SYSTEM_PLANETS.keys())
selected_planet = st.select_slider(
    "Drag to browse a planet",
    options=planet_order,
    value="Earth",
)
sp = SOLAR_SYSTEM_PLANETS[selected_planet]

col_slider_img, col_slider_info = st.columns([1, 2])
with col_slider_img:
    try:
        st.image(wikimedia_direct_url(sp["image"]), caption=selected_planet, width="stretch")
    except Exception:
        st.write(f"**{selected_planet}** (image unavailable)")
with col_slider_info:
    st.markdown(f"### {selected_planet}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Radius", f"{sp['radius']:.2f} R⊕")
    m2.metric("Orbital period", f"{sp['period']:,.0f} days")
    m3.metric("Equilibrium temp", f"{sp['teq']} K")
    if selected_planet == closest:
        st.success(f"This is the closest Solar System match to the candidate you configured above!")

# ---------------------------------------------------------------------------
# Project FAQ chatbot. This is a lightweight, rule-based assistant that
# answers questions about THIS project specifically (dataset, methodology,
# results, features) -- not a general-purpose LLM. That keeps the app fully
# self-contained and deployable with no API keys or external services.
# ---------------------------------------------------------------------------
st.divider()
st.subheader("💬 Ask About This Project")
st.caption(
    "A simple built-in FAQ assistant -- try asking about accuracy, the dataset, "
    "false positives, SHAP, or the leakage check."
)

FAQ_RESPONSES = [
    (["accuracy", "how good", "performance"],
     f"The final model ({best_name}) reaches about **77% test accuracy** and a **0.91 macro ROC-AUC** "
     "using physical parameters alone. Including the automated pipeline vetting flags would push this to "
     "~91%, but those flags are excluded here -- see the leakage check for why."),
    (["leak", "vetting flag", "fpflag", "cheat", "inflat"],
     "The raw Kepler table includes automated vetting flags computed by the mission's own pipeline. "
     "Including them as features inflates test accuracy to ~91%, but since those flags are themselves "
     "near-automated determinations of disposition, that would be leaking the answer into the input. "
     "This model excludes them and reports the more honest ~77% instead."),
    (["dataset", "data come from", "data source", "kepler koi", "nasa"],
     "This uses the real **NASA Exoplanet Archive Kepler Cumulative KOI Table** -- 9,564 actual Kepler "
     "Objects of Interest, cleaned down to about 9,200 usable rows with no missing values in the 12 "
     "features used here."),
    (["false positive", "false-positive"],
     "A **False Positive** is a detected signal that isn't a real planet -- often caused by an eclipsing "
     "binary star system or instrument noise mimicking a transit. About half of all Kepler detections "
     "turn out to be false positives."),
    (["candidate"],
     "A **Candidate** is a signal that looks planet-like but hasn't been fully vetted or confirmed yet. "
     "It's the hardest class to predict -- most of this model's confusion happens between Candidate and "
     "False Positive, which mirrors real Kepler vetting difficulty."),
    (["confirmed"],
     "**Confirmed Exoplanet** means the signal has been independently validated as a real planet, "
     "typically through follow-up observation or statistical validation beyond the transit signal alone."),
    (["shap", "explain", "why did", "why does"],
     "SHAP (SHapley Additive exPlanations) values show which input features pushed a specific prediction "
     "toward or away from each class. Adjust the sliders above and check the 'Why this prediction?' "
     "section -- green bars support the predicted class, red bars oppose it."),
    (["feature", "engineer"],
     "Adding 6 domain-motivated engineered features (transit duration/period ratio, log-transforms, "
     "planet-to-star radius ratio) only improved accuracy by about +0.11 percentage points -- suggesting "
     "the ~77% ceiling is close to the real information limit of this feature set, not a modeling gap."),
    (["tun", "hyperparameter"],
     "Hyperparameter tuning via RandomizedSearchCV (40 candidates x 5-fold CV) improved test accuracy by "
     "only about +0.05 percentage points over the default XGBoost settings -- another sign this model is "
     "near its ceiling on the current features."),
    (["model", "algorithm", "xgboost", "neural network", "random forest", "decision tree"],
     f"Four models were trained and compared: a TensorFlow neural network, a Decision Tree, a Random "
     f"Forest, and XGBoost. **{best_name}** performed best on the held-out test set."),
    (["solar system", "planet compar", "mercury", "venus", "jupiter", "saturn"],
     "The Solar System comparison finds which real planet (Mercury through Neptune) is the closest match "
     "to your candidate's radius, orbital period, and temperature -- purely for intuition, not a claim "
     "about composition or habitability."),
    (["andrew ng", "course", "deeplearning"],
     "This project was built after completing 'Advanced Learning Algorithms' by DeepLearning.AI and "
     "Stanford Online (via Coursera), taught by Andrew Ng, to apply neural networks, bias/variance "
     "diagnosis, and tree ensembles to a real dataset."),
    (["hello", "hi", "hey"],
     "Hi! Ask me about the model's accuracy, the dataset, SHAP explanations, or the leakage check."),
]

def answer_faq(question):
    q = question.lower()
    for keywords, answer in FAQ_RESPONSES:
        if any(kw in q for kw in keywords):
            return answer
    return (
        "I'm a simple built-in FAQ bot for this project, so I can only answer questions about it "
        "specifically. Try asking about: accuracy, the dataset, false positives vs. candidates, "
        "SHAP explanations, hyperparameter tuning, or the leakage check."
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi! Ask me anything about how this project works."}
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_question = st.chat_input("Ask about the model, dataset, or results...")
if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)
    reply = answer_faq(user_question)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
