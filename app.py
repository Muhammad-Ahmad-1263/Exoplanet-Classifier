"""
app.py
Interactive Streamlit app for classifying a Kepler Object of Interest (KOI)
as CANDIDATE, CONFIRMED, or FALSE POSITIVE, using the trained model from
train_model.py.

Run with: streamlit run app.py
"""

import joblib
import hashlib
import json
import os
from datetime import datetime
from urllib.parse import quote
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap

st.set_page_config(page_title="Exoplanet Disposition Classifier", page_icon="🪐", layout="wide")

# ---------------------------------------------------------------------------
# Simple file-based login + per-user history, with an admin view.
#
# HONEST LIMITATIONS (read before relying on this):
# - Passwords are SHA-256 hashed, not salted with a proper KDF like bcrypt/
#   argon2 -- adequate for a portfolio demo, not for real sensitive data.
# - Storage is a local JSON file. On Streamlit Community Cloud, the
#   filesystem is EPHEMERAL: it can reset on reboot or redeploy. This means
#   user accounts and history are not guaranteed to persist long-term unless
#   this is swapped for a real database (e.g. Supabase, SQLite on a mounted
#   volume, or Postgres). Fine for demos and active sessions; not a
#   production-grade user data store as-is.
# ---------------------------------------------------------------------------
USERS_FILE = "users_data.json"
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin123"  # change this immediately after first login


def _hash_pw(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    # First run: seed a default admin account
    users = {
        DEFAULT_ADMIN_USER: {
            "password_hash": _hash_pw(DEFAULT_ADMIN_PASS),
            "is_admin": True,
            "created": datetime.now().isoformat(timespec="seconds"),
            "history": [],
        }
    }
    _save_users(users)
    return users


def _save_users(users):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
    except IOError:
        st.warning("Could not save user data to disk (read-only or ephemeral filesystem).")


def _signup(username, password):
    users = _load_users()
    username = username.strip()
    if not username or not password:
        return False, "Username and password can't be empty."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if username in users:
        return False, "That username is already taken."
    users[username] = {
        "password_hash": _hash_pw(password),
        "is_admin": False,
        "created": datetime.now().isoformat(timespec="seconds"),
        "history": [],
    }
    _save_users(users)
    return True, "Account created! You can log in now."


def _login(username, password):
    users = _load_users()
    if username not in users:
        return False, "No account with that username."
    if users[username]["password_hash"] != _hash_pw(password):
        return False, "Incorrect password."
    return True, "Logged in."


def _log_prediction(username, record):
    users = _load_users()
    if username in users:
        users[username]["history"].append(record)
        _save_users(users)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if not st.session_state.logged_in:
    st.title("🪐 Exoplanet Disposition Classifier")
    st.caption("By Muhammad Ahmad — please log in or create an account to continue.")

    login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])
    with login_tab:
        with st.form("login_form"):
            li_user = st.text_input("Username")
            li_pass = st.text_input("Password", type="password")
            if st.form_submit_button("Log In"):
                ok, msg = _login(li_user, li_pass)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = li_user.strip()
                    st.rerun()
                else:
                    st.error(msg)
        st.caption(
            f"First time here? A default admin account exists for demo purposes: "
            f"username `{DEFAULT_ADMIN_USER}`, password `{DEFAULT_ADMIN_PASS}`. "
            f"This is a demo credential only -- don't rely on it for anything sensitive."
        )
    with signup_tab:
        with st.form("signup_form"):
            su_user = st.text_input("Choose a username")
            su_pass = st.text_input("Choose a password", type="password",
                                     help="Must be at least 8 characters long.")
            st.caption("Password must be at least 8 characters.")
            if st.form_submit_button("Sign Up"):
                ok, msg = _signup(su_user, su_pass)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
    st.stop()

# Logged in from here on -- show a small user bar
_userbar_l, _userbar_r = st.columns([5, 1])
with _userbar_l:
    st.caption(f"Logged in as **{st.session_state.username}**")
with _userbar_r:
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

# ---------------------------------------------------------------------------
# Theme + color customization. Streamlit's officially supported theming
# lives in .streamlit/config.toml (set at deploy time), but most of
# Streamlit's own CSS is written against standard CSS custom properties
# (--background-color, --text-color, etc.), so re-declaring those at
# runtime lets this on-page panel re-skin the app without a restart. This
# is a best-effort override, not an official Streamlit API -- it degrades
# gracefully if a future Streamlit version changes these internals.
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "Light": {"bg": "#ffffff", "text": "#262730", "accent": "#ff4b4b"},
    "Dark":  {"bg": "#0e1117", "text": "#fafafa", "accent": "#ff6b6b"},
}
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "custom_colors" not in st.session_state:
    st.session_state.custom_colors = dict(_DEFAULTS["Light"])

with st.expander("🎨 Customize appearance", expanded=False):
    c1, c2 = st.columns([1, 2])
    with c1:
        new_theme = st.radio("Quick theme", ["Light", "Dark"],
                              index=0 if st.session_state.theme == "Light" else 1)
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.session_state.custom_colors = dict(_DEFAULTS[new_theme])
    with c2:
        st.caption("Or pick your own colors:")
        p1, p2, p3 = st.columns(3)
        st.session_state.custom_colors["bg"] = p1.color_picker(
            "Background", st.session_state.custom_colors["bg"])
        st.session_state.custom_colors["text"] = p2.color_picker(
            "Text", st.session_state.custom_colors["text"])
        st.session_state.custom_colors["accent"] = p3.color_picker(
            "Accent", st.session_state.custom_colors["accent"])

_bg_color = st.session_state.custom_colors["bg"]
_text_color = st.session_state.custom_colors["text"]
_primary = st.session_state.custom_colors["accent"]
_sec_bg = _bg_color
_bg_grad = f"linear-gradient(135deg, {_bg_color} 0%, {_primary}22 50%, {_bg_color} 100%)"

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

# Real NASA-based surface texture maps (Solar System Scope, CC BY 4.0),
# mirrored on Wikimedia Commons, used for the 3D viewer below.
PLANET_TEXTURES = {
    "Mercury": "Solarsystemscope_texture_2k_mercury.jpg",
    "Venus":   "Solarsystemscope_texture_2k_venus_surface.jpg",
    "Earth":   "Solarsystemscope_texture_2k_earth_daymap.jpg",
    "Mars":    "Solarsystemscope_texture_2k_mars.jpg",
    "Jupiter": "Solarsystemscope_texture_2k_jupiter.jpg",
    "Saturn":  "Solarsystemscope_texture_2k_saturn.jpg",
    "Uranus":  "Solarsystemscope_texture_2k_uranus.jpg",
    "Neptune": "Solarsystemscope_texture_2k_neptune.jpg",
}
SATURN_RING_TEXTURE = "Solarsystemscope_texture_2k_saturn_ring_alpha.png"


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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="stExpander"], div[data-testid="stForm"] {
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {_primary}dd 0%, {_primary}88 100%);
        padding: 28px 32px;
        border-radius: 14px;
        margin-bottom: 20px;
    ">
        <h1 style="color: white; margin: 0; font-size: 2.1em;">🪐 Exoplanet Disposition Classifier</h1>
        <p style="color: rgba(255,255,255,0.92); margin-top: 8px; margin-bottom: 0; font-size: 1.05em;">
            Predicts whether a Kepler Object of Interest is a <b>Candidate</b>, <b>Confirmed Exoplanet</b>,
            or <b>False Positive</b>, using the trained <b>{best_name}</b> model on real NASA Kepler mission data.
        </p>
        <p style="color: rgba(255,255,255,0.75); margin-top: 10px; margin-bottom: 0; font-size: 0.85em;">
            Built by Muhammad Ahmad
        </p>
    </div>
    """,
    unsafe_allow_html=True,
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

    # Log this prediction to the user's history -- but only when the inputs
    # actually changed, so unrelated reruns (theme toggle, chat, etc.) don't
    # spam duplicate entries.
    _current_input_signature = tuple(round(v, 4) for v in input_values.values())
    if st.session_state.get("last_logged_input") != _current_input_signature:
        st.session_state.last_logged_input = _current_input_signature
        _log_prediction(st.session_state.username, {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inputs": {k: round(v, 4) for k, v in input_values.items()},
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
        })

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

st.divider()
st.subheader("🌐 3D Planet Explorer")
st.caption(
    "A real, textured 3D model -- drag to rotate, scroll (or pinch on mobile) to zoom in and out."
)

_planet_3d = st.selectbox("Choose a planet to view in 3D", list(PLANET_TEXTURES.keys()),
                           index=list(PLANET_TEXTURES.keys()).index(selected_planet))
_texture_url = wikimedia_direct_url(PLANET_TEXTURES[_planet_3d])
_ring_url = wikimedia_direct_url(SATURN_RING_TEXTURE) if _planet_3d == "Saturn" else ""
_3d_bg = "#0e1117" if st.session_state.theme == "Dark" else "#eef3ff"

st.components.v1.html(
    f"""
    <div id="viewer-container" style="width:100%; height:480px; background:{_3d_bg}; border-radius:12px; overflow:hidden; position:relative;">
        <div id="loading-msg" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#888; font-family:sans-serif; z-index:10;">
            Loading 3D model...
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const container = document.getElementById('viewer-container');
        const width = container.clientWidth;
        const height = 480;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.z = 3.2;

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);

        // Lighting: ambient fill + a directional "sun" light for real shading
        scene.add(new THREE.AmbientLight(0xffffff, 0.45));
        const sunLight = new THREE.DirectionalLight(0xffffff, 1.1);
        sunLight.position.set(5, 3, 5);
        scene.add(sunLight);

        // Starfield backdrop for depth
        const starGeo = new THREE.BufferGeometry();
        const starCount = 800;
        const starPositions = new Float32Array(starCount * 3);
        for (let i = 0; i < starCount * 3; i++) {{ starPositions[i] = (Math.random() - 0.5) * 60; }}
        starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
        const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({{ color: 0xffffff, size: 0.05 }}));
        scene.add(stars);

        const loader = new THREE.TextureLoader();
        const planetGroup = new THREE.Group();
        scene.add(planetGroup);

        loader.load(
            "{_texture_url}",
            function(texture) {{
                document.getElementById('loading-msg').style.display = 'none';
                const geometry = new THREE.SphereGeometry(1, 64, 64);
                const material = new THREE.MeshStandardMaterial({{ map: texture, roughness: 0.85, metalness: 0.05 }});
                const sphere = new THREE.Mesh(geometry, material);
                planetGroup.add(sphere);

                const ringUrl = "{_ring_url}";
                if (ringUrl.length > 0) {{
                    loader.load(ringUrl, function(ringTexture) {{
                        const ringGeo = new THREE.RingGeometry(1.3, 2.2, 64);
                        const pos = ringGeo.attributes.position;
                        const v3 = new THREE.Vector3();
                        for (let i = 0; i < pos.count; i++) {{
                            v3.fromBufferAttribute(pos, i);
                            ringGeo.attributes.uv.setXY(i, v3.length() < 1.75 ? 0 : 1, 1);
                        }}
                        const ringMat = new THREE.MeshStandardMaterial({{
                            map: ringTexture, side: THREE.DoubleSide, transparent: true
                        }});
                        const ring = new THREE.Mesh(ringGeo, ringMat);
                        ring.rotation.x = Math.PI / 2.3;
                        planetGroup.add(ring);
                    }});
                }}
            }},
            undefined,
            function(err) {{
                document.getElementById('loading-msg').textContent = 'Could not load texture -- showing placeholder.';
                const geometry = new THREE.SphereGeometry(1, 64, 64);
                const material = new THREE.MeshStandardMaterial({{ color: 0x6699cc, roughness: 0.8 }});
                planetGroup.add(new THREE.Mesh(geometry, material));
                setTimeout(() => {{ document.getElementById('loading-msg').style.display = 'none'; }}, 1500);
            }}
        );

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 1.6;
        controls.maxDistance = 10;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.6;

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """,
    height=500,
)
st.caption(
    "Real NASA-based surface imagery (Solar System Scope, CC BY 4.0), rendered live with Three.js. "
    "Drag to rotate · scroll or pinch to zoom · rotation auto-pauses while you're interacting."
)

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

with st.form("faq_chat_form", clear_on_submit=True):
    user_question = st.text_input("Ask about the model, dataset, or results...", label_visibility="collapsed")
    sent = st.form_submit_button("Send")

if sent and user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    reply = answer_faq(user_question)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()

# ---------------------------------------------------------------------------
# Personal prediction history for the logged-in user
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🕘 Your Prediction History")

_all_users = _load_users()
_my_history = _all_users.get(st.session_state.username, {}).get("history", [])

if not _my_history:
    st.caption("No predictions logged yet in this session -- adjust the sliders above to get started.")
else:
    _hist_df = pd.DataFrame([
        {
            "Time": h["timestamp"],
            "Prediction": h["prediction"],
            "Confidence": f"{h['confidence']:.1%}",
            **{k: v for k, v in h["inputs"].items()},
        }
        for h in reversed(_my_history[-50:])  # most recent 50, newest first
    ])
    st.dataframe(_hist_df, width="stretch", hide_index=True)
    st.caption(f"Showing your {min(len(_my_history), 50)} most recent predictions (of {len(_my_history)} total).")

# ---------------------------------------------------------------------------
# Admin panel -- visible only to accounts flagged is_admin
# ---------------------------------------------------------------------------
if _all_users.get(st.session_state.username, {}).get("is_admin", False):
    st.divider()
    st.subheader("🔐 Admin Panel")
    st.caption(
        "Visible only to admin accounts. Shows all registered users and every "
        "logged prediction across the whole app."
    )

    _user_rows = []
    _all_history_rows = []
    for uname, udata in _all_users.items():
        _user_rows.append({
            "Username": uname,
            "Is Admin": udata.get("is_admin", False),
            "Created": udata.get("created", "unknown"),
            "Predictions Made": len(udata.get("history", [])),
        })
        for h in udata.get("history", []):
            _all_history_rows.append({
                "User": uname,
                "Time": h["timestamp"],
                "Prediction": h["prediction"],
                "Confidence": f"{h['confidence']:.1%}",
            })

    admin_tab1, admin_tab2 = st.tabs(["Registered Users", "All Predictions"])
    with admin_tab1:
        st.dataframe(pd.DataFrame(_user_rows), width="stretch", hide_index=True)
    with admin_tab2:
        if _all_history_rows:
            _all_hist_df = pd.DataFrame(sorted(_all_history_rows, key=lambda r: r["Time"], reverse=True))
            st.dataframe(_all_hist_df, width="stretch", hide_index=True)
        else:
            st.caption("No predictions logged by any user yet.")

    st.caption(
        "⚠️ Reminder: this data is stored in a local JSON file, which is not guaranteed to "
        "persist across app reboots/redeploys on Streamlit Community Cloud. For a real "
        "production deployment, swap this for a proper database."
    )

st.divider()
st.markdown(
    f"""
    <div style="text-align: center; padding: 18px 0 8px 0; color: {_text_color}; opacity: 0.7; font-size: 0.85em;">
        Built by <b>Muhammad Ahmad</b> · Applying concepts from
        <i>Advanced Learning Algorithms</i> (DeepLearning.AI / Stanford Online, via Coursera)<br>
        Data: NASA Exoplanet Archive, Kepler Cumulative KOI Table
    </div>
    """,
    unsafe_allow_html=True,
)
