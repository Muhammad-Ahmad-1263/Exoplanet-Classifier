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
# Data is stored in two real, human-readable CSV files:
#   - users.csv   : one row per registered account
#   - history.csv : one row per prediction ever logged
#
# HONEST LIMITATIONS (read before relying on this):
# - Passwords are SHA-256 hashed, not salted with a proper KDF like bcrypt/
#   argon2 -- adequate for a portfolio demo, not for real sensitive data.
# - Storage is local CSV files. On Streamlit Community Cloud, the filesystem
#   is EPHEMERAL: it can reset on reboot or redeploy. This means accounts and
#   history are not guaranteed to persist long-term unless this is swapped
#   for a real database (e.g. Supabase, SQLite on a mounted volume, or
#   Postgres). Fine for demos and active sessions; not a production-grade
#   user data store as-is.
# ---------------------------------------------------------------------------
USERS_CSV = "users.csv"
HISTORY_CSV = "history.csv"
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin123"  # change this immediately after first login
_HISTORY_COLUMNS = ["username", "timestamp", "prediction", "confidence"] + [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_impact", "koi_steff", "koi_slogg",
    "koi_srad", "koi_kepmag",
]


def _hash_pw(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users_csv():
    if os.path.exists(USERS_CSV):
        try:
            return pd.read_csv(USERS_CSV, dtype={"username": str, "password_hash": str})
        except (pd.errors.EmptyDataError, IOError):
            pass
    # First run: seed a real users.csv with a default admin account
    df = pd.DataFrame([{
        "username": DEFAULT_ADMIN_USER,
        "password_hash": _hash_pw(DEFAULT_ADMIN_PASS),
        "is_admin": True,
        "created": datetime.now().isoformat(timespec="seconds"),
    }])
    df.to_csv(USERS_CSV, index=False)
    return df


def _save_users_csv(df):
    try:
        df.to_csv(USERS_CSV, index=False)
    except IOError:
        st.warning("Could not save user data to disk (read-only or ephemeral filesystem).")


def _load_history_csv():
    if os.path.exists(HISTORY_CSV):
        try:
            return pd.read_csv(HISTORY_CSV)
        except (pd.errors.EmptyDataError, IOError):
            pass
    df = pd.DataFrame(columns=_HISTORY_COLUMNS)
    df.to_csv(HISTORY_CSV, index=False)
    return df


def _signup(username, password):
    users = _load_users_csv()
    username = username.strip()
    if not username or not password:
        return False, "Username and password can't be empty."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if username in users["username"].astype(str).values:
        return False, "That username is already taken."
    new_row = pd.DataFrame([{
        "username": username,
        "password_hash": _hash_pw(password),
        "is_admin": False,
        "created": datetime.now().isoformat(timespec="seconds"),
    }])
    users = pd.concat([users, new_row], ignore_index=True)
    _save_users_csv(users)
    return True, "Account created! You can log in now."


def _login(username, password):
    users = _load_users_csv()
    match = users[users["username"].astype(str) == username]
    if match.empty:
        return False, "No account with that username."
    if str(match.iloc[0]["password_hash"]) != _hash_pw(password):
        return False, "Incorrect password."
    return True, "Logged in."


def _log_prediction(username, record):
    history = _load_history_csv()
    new_row = pd.DataFrame([{"username": username, **record}])
    history = pd.concat([history, new_row], ignore_index=True)
    try:
        history.to_csv(HISTORY_CSV, index=False)
    except IOError:
        st.warning("Could not save prediction history to disk (read-only or ephemeral filesystem).")


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
    "Light": {"bg": "#f7f9fb", "text": "#1e293b", "accent": "#2563eb"},
    "Dark":  {"bg": "#0f172a", "text": "#e2e8f0", "accent": "#3b82f6"},
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

# Flat, professional background -- deliberately NOT a gradient. A solid,
# muted surface reads as a serious data tool; a busy multi-color gradient
# behind dense tables and charts fights for attention against the content.
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
        background-color: {_bg_color};
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
MOON_TEXTURE = "Solarsystemscope_texture_2k_moon.jpg"

# Simplified moon data for the 3D viewer: distance/radius are in units of the
# parent planet's own radius (=1), not real-world scale, so everything stays
# visible in one frame. Mercury and Venus are correctly omitted -- neither
# has any moons. Only Earth's Moon uses a real texture; the rest use plain
# colored spheres since Solar System Scope doesn't publish free textures for
# them, but their presence and relative position/size are accurate.
MOON_DATA = {
    "Earth":   [{"name": "Moon", "distance": 1.8, "radius": 0.27, "color": "#aaaaaa", "textured": True}],
    "Mars":    [{"name": "Phobos", "distance": 1.4, "radius": 0.05, "color": "#8a7f6b", "textured": False},
                {"name": "Deimos", "distance": 1.8, "radius": 0.03, "color": "#9c9282", "textured": False}],
    "Jupiter": [{"name": "Io", "distance": 1.6, "radius": 0.11, "color": "#d9c26a", "textured": False},
                {"name": "Europa", "distance": 1.9, "radius": 0.10, "color": "#c9b998", "textured": False},
                {"name": "Ganymede", "distance": 2.3, "radius": 0.15, "color": "#8c8c8c", "textured": False},
                {"name": "Callisto", "distance": 2.7, "radius": 0.14, "color": "#5f5a52", "textured": False}],
    "Saturn":  [{"name": "Titan", "distance": 2.7, "radius": 0.16, "color": "#d9a441", "textured": False}],
    "Uranus":  [{"name": "Titania", "distance": 1.7, "radius": 0.09, "color": "#9fa8b0", "textured": False}],
    "Neptune": [{"name": "Triton", "distance": 1.7, "radius": 0.10, "color": "#a8c3d0", "textured": False}],
}

# Artistic (NOT real-scale) orbit/size layout for the full-system view, so
# every planet stays visible and clickable in a single frame -- real orbital
# distances range from 0.39 AU to 30 AU, which would make Mercury and
# Neptune impossible to show together at any usable zoom level. Sizes are
# ordered correctly (Jupiter largest, Mercury smallest) even though they
# aren't to true relative scale either, for the same reason.
ORBIT_SCENE_DATA = {
    "Mercury": {"orbit_radius": 3.2,  "visual_radius": 0.24, "period_days": 88},
    "Venus":   {"orbit_radius": 4.3,  "visual_radius": 0.42, "period_days": 225},
    "Earth":   {"orbit_radius": 5.6,  "visual_radius": 0.44, "period_days": 365},
    "Mars":    {"orbit_radius": 7.0,  "visual_radius": 0.30, "period_days": 687},
    "Jupiter": {"orbit_radius": 10.5, "visual_radius": 1.10, "period_days": 4331},
    "Saturn":  {"orbit_radius": 14.5, "visual_radius": 0.95, "period_days": 10747},
    "Uranus":  {"orbit_radius": 18.0, "visual_radius": 0.65, "period_days": 30589},
    "Neptune": {"orbit_radius": 21.0, "visual_radius": 0.63, "period_days": 59800},
}

# Brief, original one/two-sentence intros -- written in plain language, not
# copied from any source. Moon intro correctly notes no moons for Mercury
# and Venus.
PLANET_FACTS = {
    "Mercury": {
        "intro": "The smallest planet and the closest to the Sun, with wild temperature "
                  "swings between a scorching day side and a freezing night side.",
        "moon_intro": "Mercury has no moons.",
    },
    "Venus": {
        "intro": "Similar in size to Earth, but its thick atmosphere traps heat so "
                  "effectively that it's the hottest planet in the Solar System.",
        "moon_intro": "Venus has no moons.",
    },
    "Earth": {
        "intro": "The only planet known to host life, with liquid water covering most "
                  "of its surface and an atmosphere that supports a stable climate.",
        "moon_intro": "The Moon is Earth's only natural satellite -- large enough relative "
                        "to Earth that the two are sometimes considered a double-planet system.",
    },
    "Mars": {
        "intro": "Known as the Red Planet due to iron oxide (rust) covering its surface, "
                  "and a major target for future human exploration.",
        "moon_intro": "Mars has two small, irregularly shaped moons, Phobos and Deimos, "
                        "likely captured asteroids rather than moons formed alongside the planet.",
    },
    "Jupiter": {
        "intro": "The largest planet in the Solar System by far, a gas giant famous for "
                  "the Great Red Spot, a storm larger than Earth that has raged for centuries.",
        "moon_intro": "Jupiter's four largest moons -- Io, Europa, Ganymede, and Callisto -- "
                        "were discovered by Galileo in 1610. Ganymede is the largest moon in the "
                        "Solar System, and Europa is a leading candidate for a subsurface ocean.",
    },
    "Saturn": {
        "intro": "Famous for its spectacular ring system, made of countless particles of "
                  "ice and rock ranging from dust-sized to house-sized.",
        "moon_intro": "Titan, Saturn's largest moon, is the only moon known to have a dense "
                        "atmosphere and stable liquid lakes on its surface -- of methane, not water.",
    },
    "Uranus": {
        "intro": "An ice giant that rotates almost completely on its side, giving it the "
                  "most extreme seasons of any planet in the Solar System.",
        "moon_intro": "Titania, Uranus's largest moon, is named after a character from "
                        "Shakespeare's 'A Midsummer Night's Dream.'",
    },
    "Neptune": {
        "intro": "The most distant planet from the Sun, known for hosting the fastest "
                  "winds ever recorded in the Solar System.",
        "moon_intro": "Triton, Neptune's largest moon, orbits backwards relative to Neptune's "
                        "own rotation -- strong evidence it's a captured object, not one that "
                        "formed alongside the planet.",
    },
}


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
        background: {_primary};
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
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
            **{k: round(v, 4) for k, v in input_values.items()},
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
st.subheader("🌐 3D Solar System")
st.caption(
    "All eight planets orbiting live. Click any planet to zoom in and see its facts, moons, and radius. "
    "Drag to rotate the whole view, scroll or pinch to zoom, and use the button below to zoom back out."
)

_sun_texture_url = wikimedia_direct_url("Solarsystemscope_texture_2k_sun.jpg")
_moon_texture_url = wikimedia_direct_url(MOON_TEXTURE)
_ring_texture_url = wikimedia_direct_url(SATURN_RING_TEXTURE)
_planet_textures_json = json.dumps({p: wikimedia_direct_url(f) for p, f in PLANET_TEXTURES.items()})
_orbit_data_json = json.dumps(ORBIT_SCENE_DATA)
_moon_data_json = json.dumps(MOON_DATA)
_planet_facts_json = json.dumps(PLANET_FACTS)
_solar_system_stats_json = json.dumps(SOLAR_SYSTEM_PLANETS)
_3d_bg = "#0f172a" if st.session_state.theme == "Dark" else "#eef2f7"
_3d_panel_bg = "#1e293b" if st.session_state.theme == "Dark" else "#ffffff"
_3d_text = "#e2e8f0" if st.session_state.theme == "Dark" else "#1e293b"

st.components.v1.html(
    f"""
    <div id="viewer-container" style="width:100%; height:560px; background:{_3d_bg}; border-radius:12px; overflow:hidden; position:relative; font-family:'Inter',sans-serif;">
        <div id="loading-msg" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#888; z-index:10;">
            Loading Solar System...
        </div>
        <div id="info-panel" style="display:none; position:absolute; top:16px; right:16px; width:280px; max-width:42%;
             background:{_3d_panel_bg}; color:{_3d_text}; border-radius:12px; padding:16px 18px; box-shadow:0 4px 20px rgba(0,0,0,0.25); z-index:20;">
        </div>
        <button id="reset-view-btn" style="display:none; position:absolute; bottom:16px; left:16px; z-index:20;
             padding:8px 16px; border-radius:8px; border:none; cursor:pointer; background:{_3d_panel_bg}; color:{_3d_text}; box-shadow:0 2px 8px rgba(0,0,0,0.2);">
            🔭 View Whole Solar System
        </button>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const container = document.getElementById('viewer-container');
        const infoPanel = document.getElementById('info-panel');
        const resetBtn = document.getElementById('reset-view-btn');
        const width = container.clientWidth;
        const height = 560;

        const orbitData = {_orbit_data_json};
        const moonData = {_moon_data_json};
        const planetFacts = {_planet_facts_json};
        const planetStats = {_solar_system_stats_json};
        const planetTextureUrls = {_planet_textures_json};

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
        camera.position.set(0, 16, 26);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(width, height);
        container.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.35));
        const sunLight = new THREE.PointLight(0xffffff, 2.2, 200);
        scene.add(sunLight);

        // Starfield backdrop
        const starGeo = new THREE.BufferGeometry();
        const starCount = 1500;
        const starPositions = new Float32Array(starCount * 3);
        for (let i = 0; i < starCount * 3; i++) {{ starPositions[i] = (Math.random() - 0.5) * 200; }}
        starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
        scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({{ color: 0xffffff, size: 0.15 }})));

        const loader = new THREE.TextureLoader();

        // The Sun, at the center
        const sunGeo = new THREE.SphereGeometry(1.8, 48, 48);
        loader.load("{_sun_texture_url}",
            (tex) => {{ sunMesh.material.map = tex; sunMesh.material.needsUpdate = true; }});
        const sunMesh = new THREE.Mesh(sunGeo, new THREE.MeshBasicMaterial({{ color: 0xffdd88 }}));
        scene.add(sunMesh);

        // Faint orbit path rings, drawn once, purely visual
        Object.values(orbitData).forEach(function(od) {{
            const orbitGeo = new THREE.RingGeometry(od.orbit_radius - 0.02, od.orbit_radius + 0.02, 128);
            const orbitMat = new THREE.MeshBasicMaterial({{ color: 0x888888, side: THREE.DoubleSide, transparent: true, opacity: 0.25 }});
            const orbitRing = new THREE.Mesh(orbitGeo, orbitMat);
            orbitRing.rotation.x = Math.PI / 2;
            scene.add(orbitRing);
        }});

        const planetObjects = {{}};  // name -> {{ group, mesh, angle, speed, orbitRadius, visualRadius }}
        const clickableMeshes = [];
        const moonMeshes = [];

        function buildPlanet(name, data) {{
            const group = new THREE.Group();
            const geometry = new THREE.SphereGeometry(data.visual_radius, 48, 48);
            const material = new THREE.MeshStandardMaterial({{ color: 0x88aacc, roughness: 0.85 }});
            const mesh = new THREE.Mesh(geometry, material);
            mesh.userData.planetName = name;
            group.add(mesh);
            scene.add(group);
            clickableMeshes.push(mesh);

            loader.load(planetTextureUrls[name], (tex) => {{
                material.map = tex;
                material.color.set(0xffffff);
                material.needsUpdate = true;
            }});

            if (name === "Saturn") {{
                loader.load("{_ring_texture_url}", function(ringTexture) {{
                    const ringGeo = new THREE.RingGeometry(data.visual_radius * 1.3, data.visual_radius * 2.1, 64);
                    const pos = ringGeo.attributes.position;
                    const v3 = new THREE.Vector3();
                    for (let i = 0; i < pos.count; i++) {{
                        v3.fromBufferAttribute(pos, i);
                        ringGeo.attributes.uv.setXY(i, v3.length() < data.visual_radius * 1.7 ? 0 : 1, 1);
                    }}
                    const ring = new THREE.Mesh(ringGeo, new THREE.MeshStandardMaterial({{
                        map: ringTexture, side: THREE.DoubleSide, transparent: true
                    }}));
                    ring.rotation.x = Math.PI / 2.3;
                    group.add(ring);
                }});
            }}

            (moonData[name] || []).forEach(function(m, i) {{
                const moonGeo = new THREE.SphereGeometry(m.radius, 20, 20);
                const inclination = (i % 2 === 0 ? 1 : -1) * (0.1 + i * 0.05);
                function placeMoon(material) {{
                    const moon = new THREE.Mesh(moonGeo, material);
                    group.add(moon);
                    moonMeshes.push({{ mesh: moon, distance: data.visual_radius + m.distance, angle: Math.random() * Math.PI * 2, speed: 0.2 / Math.sqrt(m.distance), inclination: inclination }});
                }}
                if (m.textured) {{
                    loader.load("{_moon_texture_url}",
                        (tex) => placeMoon(new THREE.MeshStandardMaterial({{ map: tex, roughness: 0.9 }})),
                        undefined,
                        () => placeMoon(new THREE.MeshStandardMaterial({{ color: m.color, roughness: 0.9 }})));
                }} else {{
                    placeMoon(new THREE.MeshStandardMaterial({{ color: m.color, roughness: 0.9 }}));
                }}
            }});

            planetObjects[name] = {{
                group: group, mesh: mesh,
                angle: Math.random() * Math.PI * 2,
                speed: 1.2 / Math.sqrt(data.period_days),
                orbitRadius: data.orbit_radius,
            }};
        }}

        Object.entries(orbitData).forEach(([name, data]) => buildPlanet(name, data));
        document.getElementById('loading-msg').style.display = 'none';

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 2;
        controls.maxDistance = 60;

        // Click-to-zoom
        let focusedPlanet = null;
        let cameraTarget = null;   // THREE.Vector3 to lerp camera.position toward
        let lookAtTarget = null;   // THREE.Vector3 to lerp controls.target toward

        function showInfoPanel(name) {{
            const stats = planetStats[name];
            const facts = planetFacts[name];
            infoPanel.innerHTML =
                '<h3 style="margin:0 0 8px 0;">' + name + '</h3>' +
                '<p style="margin:0 0 10px 0; font-size:0.9em; line-height:1.4;">' + facts.intro + '</p>' +
                '<div style="font-size:0.85em; margin-bottom:10px;">' +
                '<b>Radius:</b> ' + stats.radius.toFixed(2) + ' Earth radii<br>' +
                '<b>Orbital period:</b> ' + stats.period.toLocaleString() + ' days<br>' +
                '<b>Equilibrium temp:</b> ' + stats.teq + ' K' +
                '</div>' +
                '<p style="margin:0; font-size:0.85em; line-height:1.4; opacity:0.85;"><b>Moons:</b> ' + facts.moon_intro + '</p>';
            infoPanel.style.display = 'block';
            resetBtn.style.display = 'block';
        }}

        function focusOnPlanet(name) {{
            focusedPlanet = name;
            showInfoPanel(name);
        }}

        function resetView() {{
            focusedPlanet = null;
            infoPanel.style.display = 'none';
            resetBtn.style.display = 'none';
            cameraTarget = new THREE.Vector3(0, 16, 26);
            lookAtTarget = new THREE.Vector3(0, 0, 0);
        }}
        resetBtn.addEventListener('click', resetView);

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        renderer.domElement.addEventListener('click', function(event) {{
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const hits = raycaster.intersectObjects(clickableMeshes);
            if (hits.length > 0) {{
                focusOnPlanet(hits[0].object.userData.planetName);
            }}
        }});

        function animate() {{
            requestAnimationFrame(animate);

            Object.values(planetObjects).forEach(function(p) {{
                p.angle += p.speed * 0.01;
                p.group.position.set(Math.cos(p.angle) * p.orbitRadius, 0, Math.sin(p.angle) * p.orbitRadius);
            }});
            moonMeshes.forEach(function(m) {{
                m.angle += m.speed * 0.02;
                m.mesh.position.set(Math.cos(m.angle) * m.distance, Math.sin(m.angle) * m.distance * m.inclination, Math.sin(m.angle) * m.distance);
            }});

            if (focusedPlanet && planetObjects[focusedPlanet]) {{
                const p = planetObjects[focusedPlanet];
                const planetWorldPos = p.group.position;
                const offsetDist = orbitData[focusedPlanet].visual_radius * 4 + 1.5;
                cameraTarget = new THREE.Vector3(planetWorldPos.x + offsetDist, planetWorldPos.y + offsetDist * 0.4, planetWorldPos.z + offsetDist);
                lookAtTarget = planetWorldPos.clone();
            }}
            if (cameraTarget) {{
                camera.position.lerp(cameraTarget, 0.06);
                controls.target.lerp(lookAtTarget, 0.06);
            }}

            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
    """,
    height=580,
)
st.caption(
    "Real NASA-based surface imagery (Solar System Scope, CC BY 4.0), rendered live with Three.js. "
    "Orbit distances, planet sizes, and speeds are artistically scaled (not real-world proportions) "
    "so all eight planets stay visible and clickable together -- real distances range from 0.39 to 30 "
    "astronomical units, which no single view could show at a usable zoom level."
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

_all_users_df = _load_users_csv()
_all_history_df = _load_history_csv()
_my_history_df = _all_history_df[_all_history_df["username"] == st.session_state.username]

if _my_history_df.empty:
    st.caption("No predictions logged yet -- adjust the sliders above to get started.")
else:
    _my_display = _my_history_df.sort_values("timestamp", ascending=False).head(50).copy()
    _my_display["confidence"] = _my_display["confidence"].apply(lambda c: f"{float(c):.1%}")
    _my_display = _my_display.drop(columns=["username"]).rename(columns={
        "timestamp": "Time", "prediction": "Prediction", "confidence": "Confidence"
    })
    st.dataframe(_my_display, width="stretch", hide_index=True)
    st.caption(
        f"Showing your {min(len(_my_history_df), 50)} most recent predictions "
        f"(of {len(_my_history_df)} total). Full data lives in history.csv."
    )

# ---------------------------------------------------------------------------
# Admin panel -- visible only to accounts flagged is_admin
# ---------------------------------------------------------------------------
_my_user_row = _all_users_df[_all_users_df["username"] == st.session_state.username]
_is_admin = bool(_my_user_row.iloc[0]["is_admin"]) if not _my_user_row.empty else False

if _is_admin:
    st.divider()
    st.subheader("🔐 Admin Panel")
    st.caption(
        "Visible only to admin accounts. Shows all registered users and every "
        "logged prediction across the whole app, read directly from users.csv and history.csv."
    )

    _pred_counts = _all_history_df["username"].value_counts() if not _all_history_df.empty else pd.Series(dtype=int)
    _user_display = _all_users_df.copy()
    _user_display["Predictions Made"] = _user_display["username"].map(_pred_counts).fillna(0).astype(int)
    _user_display = _user_display.rename(columns={
        "username": "Username", "is_admin": "Is Admin", "created": "Created"
    })[["Username", "Is Admin", "Created", "Predictions Made"]]

    admin_tab1, admin_tab2 = st.tabs(["Registered Users", "All Predictions"])
    with admin_tab1:
        st.dataframe(_user_display, width="stretch", hide_index=True)
    with admin_tab2:
        if not _all_history_df.empty:
            _hist_display = _all_history_df.sort_values("timestamp", ascending=False).copy()
            _hist_display["confidence"] = _hist_display["confidence"].apply(lambda c: f"{float(c):.1%}")
            _hist_display = _hist_display.rename(columns={
                "username": "User", "timestamp": "Time", "prediction": "Prediction", "confidence": "Confidence"
            })
            st.dataframe(_hist_display, width="stretch", hide_index=True)
        else:
            st.caption("No predictions logged by any user yet.")

    st.caption(
        "⚠️ Reminder: this data is stored in local CSV files (users.csv, history.csv), which are "
        "not guaranteed to persist across app reboots/redeploys on Streamlit Community Cloud. "
        "For a real production deployment, swap this for a proper database."
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
