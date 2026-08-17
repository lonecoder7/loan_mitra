import base64
import os
import textwrap
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# LOAN MITRA V2
# Custom HTML/CSS-style Banking Credit Intelligence UI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Branding assets:
#   logo.png -> full ऋण MITRA logo for the main header
#   icon.png -> emblem-only transparent icon for sidebar + browser tab
def first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

ICON_PATH = os.path.join(ASSETS_DIR, "icon.png")

page_icon = "₹"
try:
    from PIL import Image
    if os.path.isfile(ICON_PATH):
        page_icon = Image.open(ICON_PATH)
except Exception:
    page_icon = "₹"

st.set_page_config(
    page_title="ऋण MITRA | Credit Intelligence",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

APPROVAL_MODEL_PATH = os.path.join(MODEL_DIR, "loan_mitra_best_model.pkl")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "loan_mitra_risk_model.pkl")
APPROVAL_METRICS_PATH = os.path.join(MODEL_DIR, "model_comparison.csv")
RISK_METRICS_PATH = os.path.join(MODEL_DIR, "risk_model_comparison.csv")
DATA_PATH = os.path.join(DATA_DIR, "loan_applicants_5000.csv")

# -----------------------------
# Session state
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Load assets
# -----------------------------
@st.cache_resource
def load_models():
    approval = None
    risk = None
    approval_error = ""
    risk_error = ""

    if os.path.isfile(APPROVAL_MODEL_PATH):
        try:
            approval = joblib.load(APPROVAL_MODEL_PATH)
        except Exception as e:
            approval_error = str(e)
    else:
        approval_error = f"Missing: {APPROVAL_MODEL_PATH}"

    if os.path.isfile(RISK_MODEL_PATH):
        try:
            risk = joblib.load(RISK_MODEL_PATH)
        except Exception as e:
            risk_error = str(e)
    else:
        risk_error = f"Missing: {RISK_MODEL_PATH}"

    return approval, risk, approval_error, risk_error


def clean_dataset(df):
    df = df.copy()
    numeric_columns = [
        "Age", "Employment_Years", "Dependents", "Annual_Income",
        "Monthly_Expenses", "Existing_Loans", "Existing_EMI",
        "Savings_Balance", "Asset_Value", "Credit_Score",
        "Credit_History_Years", "Previous_Default",
        "Number_of_Credit_Accounts", "Late_Payments",
        "Loan_Amount_Requested", "Loan_Term_Months",
        "Monthly_Income", "DTI_Ratio", "Loan_to_Income",
        "Disposable_Income", "Default_Status"
    ]
    for col in numeric_columns:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (df[col].astype(str)
                    .str.replace("₹", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip())
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
    return df


@st.cache_data
def load_dataset():
    if os.path.isfile(DATA_PATH):
        try:
            return clean_dataset(pd.read_csv(DATA_PATH))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data

def load_metrics(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_image_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


approval_model, risk_model, approval_model_error, risk_model_error = load_models()
dataset = load_dataset()
approval_metrics = load_metrics(APPROVAL_METRICS_PATH)
risk_metrics = load_metrics(RISK_METRICS_PATH)

LOGO_B64 = load_image_b64(LOGO_PATH)
ICON_B64 = load_image_b64(ICON_PATH)

# Full logo is used only in the main header.
LOGO_IMG = (
    f'<img src="data:image/png;base64,{LOGO_B64}" class="brand-logo" alt="ऋण MITRA" />'
    if LOGO_B64 else
    '<div class="brand-fallback">ऋण <span>MITRA</span></div>'
)

# Emblem-only icon is used in the sidebar and browser tab.
SIDEBAR_ICON_IMG = (
    f'<img src="data:image/png;base64,{ICON_B64}" class="sidebar-icon" alt="ऋण MITRA" />'
    if ICON_B64 else
    '<div class="sidebar-icon-fallback">₹</div>'
)

# -----------------------------
# Global CSS
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #0b1f3a;
    --navy-2: #102a4c;
    --blue: #2563eb;
    --blue-soft: #eff6ff;
    --green: #16a34a;
    --green-soft: #ecfdf3;
    --amber: #d97706;
    --amber-soft: #fff7ed;
    --red: #dc2626;
    --red-soft: #fef2f2;
    --purple: #7c3aed;
    --purple-soft: #f3ecfe;
    --teal: #0891b2;
    --teal-soft: #e7f8fb;
    --gold: #d97706;
    --gold-2: #f59e0b;
    --bg: #f5f7fb;
    --card: #ffffff;
    --border: #e5eaf1;
    --muted: #64748b;
    --text: #172033;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: var(--bg);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    display: none;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
    box-shadow: 2px 0 18px rgba(15,23,42,.04);
}

section[data-testid="stSidebar"] > div {
    padding: 1rem .9rem;
}

section[data-testid="stSidebar"] * {
    color: var(--navy) !important;
}

section[data-testid="stSidebar"] .stRadio label {
    border-radius: 10px;
    padding: 10px 12px;
    margin: 3px 0;
    transition: all .18s ease;
    font-weight: 600;
    font-size: 14px;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--amber-soft);
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] {
    background: transparent;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: .35rem;
}

/* Hide native radio circles but preserve interaction */
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
    display: none;
}

/* Brand lockup (sidebar + topbar) */
.brand-logo {
    width: 58px;
    height: 58px;
    object-fit: contain;
    flex-shrink: 0;
    display: block;
}

.sidebar-icon {
    width: 92px;
    height: 92px;
    object-fit: contain;
    display: block;
    margin: 0 auto;
}

.sidebar-icon-fallback {
    width: 92px;
    height: 92px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 52px;
    font-weight: 900;
    color: var(--gold) !important;
}

.brand-fallback {
    font-size: 22px;
    font-weight: 900;
    color: var(--navy);
}

.brand-fallback span {
    color: var(--gold);
}

.sidebar-brand {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px 4px 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 10px;
}

.sidebar-brand .brand-title {
    font-size: 19px;
    font-weight: 900;
    line-height: 1.15;
    color: var(--navy) !important;
}

.sidebar-brand .brand-title .accent {
    color: var(--gold) !important;
}

.sidebar-brand .brand-sub {
    font-size: 9px;
    letter-spacing: .06em;
    color: var(--muted) !important;
    font-weight: 700;
    text-transform: uppercase;
    margin-top: 1px;
}

/* Top masthead bar shown on every page */
.top-masthead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 12px 22px;
    margin-bottom: 18px;
    box-shadow: 0 5px 18px rgba(15,23,42,.035);
}

.top-masthead .brand-title {
    font-size: 21px;
    font-weight: 900;
    color: var(--navy);
    line-height: 1.1;
}

.top-masthead .brand-title .accent {
    color: var(--gold);
}

.top-masthead .brand-sub {
    font-size: 10px;
    letter-spacing: .1em;
    color: var(--muted);
    font-weight: 700;
    text-transform: uppercase;
}

.officer-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 7px 14px 7px 7px;
    font-weight: 700;
    font-size: 13px;
    color: var(--navy);
}

.officer-avatar {
    width: 28px;
    height: 28px;
    border-radius: 999px;
    background: var(--navy);
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 800;
}

/* Bottom promo banner shown on every page */
.brand-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    background: linear-gradient(135deg, var(--navy) 0%, #173c6c 100%);
    border-radius: 16px;
    padding: 16px 26px;
    margin-top: 26px;
    color: #fff;
    box-shadow: 0 10px 28px rgba(11,31,58,.18);
}

.brand-banner .label {
    font-size: 11px;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #a9c4e6;
    font-weight: 700;
}

.brand-banner .tagline {
    font-size: 18px;
    font-weight: 800;
    color: var(--gold-2);
    margin-top: 3px;
}

.brand-banner .icon-ring {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: rgba(255,255,255,.1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

/* Inputs */
div[data-baseweb="input"], div[data-baseweb="select"],
div[data-baseweb="textarea"] {
    border-radius: 10px !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-color: #dbe2ea !important;
    background: #fff !important;
    border-radius: 10px !important;
    min-height: 42px;
}

input, textarea {
    color: var(--text) !important;
}

.stSelectbox label, .stNumberInput label, .stTextInput label {
    font-weight: 600 !important;
    color: #334155 !important;
}

.stButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
    border: 0;
    transition: transform .15s ease, box-shadow .15s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(37,99,235,.18);
}

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 6px 22px rgba(15, 23, 42, .045);
}

.card-tight {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 5px 18px rgba(15, 23, 42, .035);
}


.page-title {
    color: var(--navy);
    font-size: 29px;
    font-weight: 800;
    margin: 4px 0 2px;
}

.page-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 20px;
}

.section-head {
    display: flex;
    justify-content: space-between;
    align-items: end;
    margin: 22px 0 10px;
}

.section-head h3 {
    margin: 0;
    color: var(--navy);
    font-size: 18px;
}

.section-head span {
    color: var(--muted);
    font-size: 12px;
}

.metric-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px;
    min-height: 112px;
    box-shadow: 0 5px 18px rgba(15,23,42,.035);
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: var(--navy);
    font-size: 25px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 4px;
}

/* Colored metric icon chips, matching brand dashboard */
.metric-card-icon {
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.metric-icon-chip {
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
}

.chip-blue { background: var(--blue-soft); color: var(--blue); }
.chip-green { background: var(--green-soft); color: var(--green); }
.chip-red { background: var(--red-soft); color: var(--red); }
.chip-amber { background: var(--amber-soft); color: var(--amber); }
.chip-purple { background: var(--purple-soft); color: var(--purple); }
.chip-teal { background: var(--teal-soft); color: var(--teal); }

.metric-note-green { color: var(--green) !important; font-weight: 700; }
.metric-note-red { color: var(--red) !important; font-weight: 700; }
.metric-note-amber { color: var(--amber) !important; font-weight: 700; }

/* CSS donut chart */
.donut-wrap {
    display: flex;
    align-items: center;
    gap: 18px;
}

.donut {
    width: 118px;
    height: 118px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.donut-hole {
    width: 74px;
    height: 74px;
    border-radius: 50%;
    background: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.donut-hole .n {
    font-size: 17px;
    font-weight: 900;
    color: var(--navy);
    line-height: 1;
}

.donut-hole .l {
    font-size: 9px;
    color: var(--muted);
    margin-top: 2px;
}

.donut-legend {
    font-size: 13px;
    color: var(--text);
}

.donut-legend .row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 7px 0;
}

.donut-legend .dot {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    flex-shrink: 0;
}

.donut-legend .row b {
    display: block;
    font-size: 13px;
}

.donut-legend .row span {
    display: block;
    font-size: 11px;
    color: var(--muted);
}

/* Horizontal risk-level style bars */
.risk-bar-row {
    margin: 14px 0;
}

.risk-bar-row .top {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
}

.risk-bar-track {
    width: 100%;
    height: 9px;
    border-radius: 99px;
    background: #eef1f6;
    overflow: hidden;
}

.risk-bar-fill {
    height: 100%;
    border-radius: 99px;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .03em;
}

.badge-green { background: var(--green-soft); color: var(--green); }
.badge-amber { background: var(--amber-soft); color: var(--amber); }
.badge-red { background: var(--red-soft); color: var(--red); }
.badge-blue { background: var(--blue-soft); color: var(--blue); }

.info-box {
    padding: 15px 17px;
    border-radius: 12px;
    border: 1px solid #dbeafe;
    background: #f8fbff;
    color: #334155;
}

.factor {
    padding: 11px 13px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid var(--border);
    background: #fff;
    font-size: 13px;
}

.factor-positive { border-left: 4px solid var(--green); }
.factor-risk { border-left: 4px solid var(--amber); }

.decision-card {
    border-radius: 20px;
    padding: 30px;
    border: 1px solid var(--border);
    background: white;
    box-shadow: 0 12px 35px rgba(15,23,42,.07);
}

.decision-approved { border-top: 5px solid var(--green); }
.decision-rejected { border-top: 5px solid var(--red); }
.decision-review { border-top: 5px solid var(--amber); }

.decision-word {
    font-size: 34px;
    font-weight: 900;
    margin: 5px 0;
}

.risk-score {
    font-size: 54px;
    font-weight: 900;
    color: var(--navy);
    line-height: 1;
}

.progress-track {
    width: 100%;
    height: 10px;
    border-radius: 99px;
    background: #e9eef5;
    overflow: hidden;
    margin: 10px 0 5px;
}

.progress-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2563eb, #16a34a);
}

.footer-note {
    color: #94a3b8;
    font-size: 11px;
    text-align: center;
    padding: 25px 0 5px;
}

/* Streamlit form borders */
[data-testid="stForm"] {
    border: 0 !important;
    padding: 0 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

@media (max-width: 900px) {
    .hero h1 { font-size: 27px; }
    .page-title { font-size: 24px; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, str):
            value = value.replace("₹", "").replace(",", "").strip()
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def money(value):
    value = safe_float(value)
    if abs(value) >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"
    if abs(value) >= 1_00_000:
        return f"₹{value / 1_00_000:.2f} L"
    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.1f}K"
    return f"₹{value:,.0f}"


def badge(text, kind="blue"):
    return f'<span class="badge badge-{kind}">{text}</span>'


def risk_kind(risk):
    return {"LOW": "green", "MODERATE": "amber", "HIGH": "red"}.get(risk, "blue")


def decision_kind(decision):
    return {"APPROVED": "green", "REVIEW": "amber", "REJECTED": "red"}.get(decision, "blue")


def risk_score(default_probability):
    # 100 = strongest profile, 0 = highest modeled default risk.
    return int(np.clip(round(100 - default_probability * 100), 0, 100))


def calculate_factors(app):
    positive = []
    risks = []

    credit_score = app["Credit_Score"]
    dti = app["DTI_Ratio"]
    previous_default = app["Previous_Default"]
    late = app["Late_Payments"]
    employment = app["Employment_Years"]
    disposable = app["Disposable_Income"]
    monthly_income = app["Monthly_Income"]

    if credit_score >= 750:
        positive.append("Excellent credit score")
    elif credit_score >= 700:
        positive.append("Good credit score")
    elif credit_score < 600:
        risks.append("Low credit score")

    if dti <= 20:
        positive.append("Low debt-to-income ratio")
    elif dti > 40:
        risks.append("High debt-to-income ratio")

    if previous_default == 0:
        positive.append("No previous loan default")
    else:
        risks.append("Previous loan default recorded")

    if late == 0:
        positive.append("No recent late payments")
    elif late >= 3:
        risks.append("Multiple late payments")

    if employment >= 5:
        positive.append("Stable employment history")
    elif employment < 2:
        risks.append("Short employment history")

    if disposable > monthly_income * 0.25:
        positive.append("Healthy disposable income")
    elif disposable <= 0:
        risks.append("Very low or negative disposable income")

    return positive, risks


def prepare_model_input(application, model):
    X = pd.DataFrame([application])
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        expected = list(expected)
        for col in expected:
            if col not in X.columns:
                X[col] = 0
        X = X[expected]
    return X


def model_probability(model, X):
    if model is None:
        raise RuntimeError("Model is not loaded.")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not provide predict_proba().")
    probabilities = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is not None and 1 in list(classes):
        return float(probabilities[0][list(classes).index(1)])
    return float(probabilities[0][-1])


def run_prediction(application):
    if approval_model is None:
        raise RuntimeError(approval_model_error or "Approval model is unavailable.")
    if risk_model is None:
        raise RuntimeError(risk_model_error or "Risk model is unavailable.")

    approval_X = prepare_model_input(application, approval_model)
    risk_X = prepare_model_input(application, risk_model)

    approval_prediction = int(approval_model.predict(approval_X)[0])
    approval_probability = float(np.clip(model_probability(approval_model, approval_X), 0, 1))
    risk_prediction = int(risk_model.predict(risk_X)[0])
    default_probability = float(np.clip(model_probability(risk_model, risk_X), 0, 1))

    default_percentage = default_probability * 100
    if default_percentage < 20:
        risk_level = "LOW"
    elif default_percentage < 50:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    if approval_prediction == 1 and risk_level == "HIGH":
        decision = "REVIEW"
    elif approval_prediction == 1:
        decision = "APPROVED"
    else:
        decision = "REJECTED"

    positive, risks = calculate_factors(application)

    return {
        "loan_decision": decision,
        "approval_probability": round(approval_probability * 100, 2),
        "default_probability": round(default_percentage, 2),
        "risk_level": risk_level,
        "default_prediction": risk_prediction,
        "risk_score": risk_score(default_probability),
        "positive_factors": positive,
        "risk_factors": risks,
        "application": application,
    }

def save_history(result, name):
    app = result["application"]
    application_id = f"LM-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(st.session_state.history)+1:03d}"
    st.session_state.history.insert(
        0,
        {
            "Application ID": application_id,
            "Applicant": name,
            "Amount": app["Loan_Amount_Requested"],
            "Risk": result["risk_level"],
            "Decision": result["loan_decision"],
            "Approval %": result["approval_probability"],
            "Default %": result["default_probability"],
            "Date": datetime.now().strftime("%d %b %Y, %H:%M"),
        },
    )
    result["application_id"] = application_id


def metric_card(label, value, note="", icon=None, chip="blue", note_kind=None):
    note_class = f"metric-note metric-note-{note_kind}" if note_kind else "metric-note"
    if icon:
        st.markdown(
            f"""
            <div class="metric-card metric-card-icon">
                <div class="metric-icon-chip chip-{chip}">{icon}</div>
                <div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="{note_class}">{note}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="{note_class}">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def donut_chart(title, total, segments):
    """segments: list of (label, value, pct, color_hex)"""
    stops = []
    running = 0
    for _, _, pct, color in segments:
        start = running
        end = running + pct
        stops.append(f"{color} {start}% {end}%")
        running = end
    gradient = ", ".join(stops)
    legend_rows = "".join(
        f"""
        <div class="row">
            <span class="dot" style="background:{color};"></span>
            <div><b>{label}</b><span>{pct:.1f}% ({value:,})</span></div>
        </div>
        """
        for label, value, pct, color in segments
    )
    st.markdown(
        f"""
        <div class="card">
            <div style="font-weight:700;color:var(--navy);margin-bottom:10px;">{title}</div>
            <div class="donut-wrap">
                <div class="donut" style="background:conic-gradient({gradient});">
                    <div class="donut-hole"><div class="n">{total:,}</div><div class="l">Total</div></div>
                </div>
                <div class="donut-legend">{legend_rows}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_bar(rows):
    """Render risk distribution as actual HTML, not a Markdown code block."""
    body_parts = []
    for label, value, pct, color in rows:
        body_parts.append(
            f'<div class="risk-bar-row">'
            f'<div class="top"><span>{label}</span>'
            f'<span style="color:{color};">{value:,} ({pct:.1f}%)</span></div>'
            f'<div class="risk-bar-track">'
            f'<div class="risk-bar-fill" style="width:{max(0, min(float(pct), 100)):.2f}%;background:{color};"></div>'
            f'</div></div>'
        )

    body = "".join(body_parts)
    st.markdown(
        f'<div class="card">{body}</div>',
        unsafe_allow_html=True,
    )


def portfolio_risk_distribution(df):
    if df.empty:
        return [("Low Risk",0,0.0,"#16a34a"),("Moderate Risk",0,0.0,"#f59e0b"),("High Risk",0,0.0,"#dc2626")]

    low = moderate = high = 0
    if risk_model is not None:
        try:
            X = df.drop(columns=["Loan_Status", "Default_Status"], errors="ignore").copy()
            expected = getattr(risk_model, "feature_names_in_", None)
            if expected is not None:
                expected = list(expected)
                for col in expected:
                    if col not in X.columns:
                        X[col] = 0
                X = X[expected]
            probs = risk_model.predict_proba(X)
            classes = getattr(risk_model, "classes_", None)
            idx = list(classes).index(1) if classes is not None and 1 in list(classes) else probs.shape[1]-1
            probs = probs[:, idx] * 100
            low = int((probs < 20).sum())
            moderate = int(((probs >= 20) & (probs < 50)).sum())
            high = int((probs >= 50).sum())
        except Exception:
            pass

    if low + moderate + high == 0 and "Credit_Score" in df.columns:
        scores = pd.to_numeric(df["Credit_Score"], errors="coerce").fillna(0)
        low = int((scores >= 700).sum())
        moderate = int(((scores >= 600) & (scores < 700)).sum())
        high = int((scores < 600).sum())

    total = low + moderate + high
    return [
        ("Low Risk", low, low/total*100 if total else 0, "#16a34a"),
        ("Moderate Risk", moderate, moderate/total*100 if total else 0, "#f59e0b"),
        ("High Risk", high, high/total*100 if total else 0, "#dc2626"),
    ]

def section(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-head">
            <h3>{title}</h3>
            <span>{subtitle}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def top_masthead():
    st.markdown(
        f"""
        <div class="top-masthead">
            <div style="display:flex;align-items:center;gap:12px;">
                {LOGO_IMG}
                <div>
                    <div class="brand-title">ऋण <span class="accent">MITRA</span></div>
                    <div class="brand-sub">AI-Powered Loan Approval &amp; Credit Risk Assessment System</div>
                </div>
            </div>
            <div class="officer-chip">
                <span class="officer-avatar">CO</span> Credit Officer &#9662;
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def brand_banner():
    st.markdown(
        """
        <div class="brand-banner">
            <div>
                <div class="label">AI-Powered Credit Decision Support</div>
                <div class="tagline">वित का सही निर्णय, आपके साथ</div>
            </div>
            <div class="icon-ring">&#129504;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Sidebar
# -----------------------------
NAV_PAGES = [
    "Dashboard",
    "New Application",
    "Credit Analysis",
    "Applications",
    "Analytics",
    "Model Intelligence",
]
NAV_ICONS = {
    "Dashboard": "🏠",
    "New Application": "➕",
    "Credit Analysis": "🛡️",
    "Applications": "📁",
    "Analytics": "📊",
    "Model Intelligence": "🧠",
}

with st.sidebar:
    # Emblem only in the sidebar — no full logo text and no status panel.
    st.markdown(
        f"""
        <div class="sidebar-brand">
            {SIDEBAR_ICON_IMG}
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav = st.radio(
        "NAVIGATION",
        NAV_PAGES,
        index=NAV_PAGES.index(st.session_state.page),
        format_func=lambda p: f"{NAV_ICONS.get(p, '•')}   {p}",
        label_visibility="visible",
    )
    if nav != st.session_state.page:
        st.session_state.page = nav
        st.rerun()

top_masthead()

# -----------------------------
# Dashboard
# -----------------------------
if st.session_state.page == "Dashboard":
    hdr_l, hdr_r = st.columns([3, 2])
    with hdr_l:
        page_header("📊 Dashboard", "Overview of lending activity and credit risk")
    with hdr_r:
        st.markdown(
            """
            <div style="display:flex;justify-content:flex-end;gap:10px;margin-top:10px;">
                <div class="officer-chip">📅 Last 30 Days</div>
                <div class="officer-chip">⇩ Export</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not dataset.empty and "Loan_Status" in dataset.columns:
        total = len(dataset)
        approved = int(dataset["Loan_Status"].astype(str).str.strip().str.lower().isin(["approved","approve","1","yes","true"]).sum())
        rejected = total - approved
        approval_rate = approved / total * 100 if total else 0
        rejection_rate = rejected / total * 100 if total else 0
        if "Default_Status" in dataset.columns:
            defaults = pd.to_numeric(dataset["Default_Status"], errors="coerce").fillna(0)
            default_count = int((defaults > 0).sum())
            default_rate = default_count / total * 100 if total else 0
        else:
            default_count, default_rate = 0, 0
        avg_credit = dataset["Credit_Score"].mean() if "Credit_Score" in dataset.columns else 0
        avg_loan = dataset["Loan_Amount_Requested"].mean() if "Loan_Amount_Requested" in dataset.columns else 0
    else:
        total = approved = rejected = default_count = 0
        approval_rate = rejection_rate = default_rate = avg_credit = avg_loan = 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("TOTAL APPLICATIONS", f"{total:,}", "100% of total", icon="📄", chip="blue")
    with c2:
        metric_card("APPROVAL RATE", f"{approval_rate:.1f}%", f"{approved:,} Approved", icon="✔", chip="green", note_kind="green")
    with c3:
        metric_card("REJECTION RATE", f"{rejection_rate:.1f}%", f"{rejected:,} Rejected", icon="✖", chip="red", note_kind="red")
    with c4:
        metric_card("DEFAULT RATE", f"{default_rate:.1f}%", f"{default_count:,} Defaulted", icon="⚠", chip="amber", note_kind="amber")
    with c5:
        metric_card("AVG CREDIT SCORE", f"{avg_credit:.0f}", "Good" if avg_credit >= 700 else "Fair", icon="◎", chip="purple")
    with c6:
        metric_card("AVG LOAN AMOUNT", money(avg_loan), "Per Application", icon="₹", chip="teal")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns([2, 1, 1])
    with t1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:700;color:var(--navy);margin-bottom:6px;">Application Trend</div>', unsafe_allow_html=True)
        if not dataset.empty and "Credit_Score" in dataset.columns:
            trend = dataset["Credit_Score"].reset_index(drop=True)
            bucket = trend.groupby(trend.index // max(1, len(trend) // 40)).mean()
            bucket.index = [f"Batch {i+1}" for i in range(len(bucket))]
            st.line_chart(bucket)
        else:
            st.info("Reference dataset not found.")
        st.markdown('</div>', unsafe_allow_html=True)
    with t2:
        if total:
            donut_chart(
                "Approval vs Rejection",
                total,
                [
                    ("Approved", approved, approval_rate, "#16a34a"),
                    ("Rejected", rejected, rejection_rate, "#dc2626"),
                ],
            )
        else:
            st.markdown('<div class="card">Reference dataset not found.</div>', unsafe_allow_html=True)
    with t3:
        if total:
            no_default = total - default_count
            donut_chart(
                "Default vs No Default",
                total,
                [
                    ("Defaulted", default_count, default_rate, "#f59e0b"),
                    ("No Default", no_default, 100 - default_rate, "#2563eb"),
                ],
            )
        else:
            st.markdown('<div class="card">Reference dataset not found.</div>', unsafe_allow_html=True)

    b1, b2 = st.columns([2, 1])
    with b1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-weight:700;color:var(--navy);margin-bottom:6px;">Recent Applications</div>', unsafe_allow_html=True)
        if st.session_state.history:
            recent = pd.DataFrame(st.session_state.history[:5])[["Applicant", "Amount", "Decision", "Risk", "Date"]].copy()
            recent["Amount"] = recent["Amount"].map(money)
            st.dataframe(recent, use_container_width=True, hide_index=True)
        elif not dataset.empty:
            cols = [c for c in ["Applicant_Name", "Loan_Amount_Requested", "Loan_Status", "Credit_Score"] if c in dataset.columns]
            if cols:
                st.dataframe(dataset[cols].head(5), use_container_width=True, hide_index=True)
            else:
                st.caption("Completed applications from this session will appear here.")
        else:
            st.caption("Completed applications from this session will appear here.")
        st.markdown('</div>', unsafe_allow_html=True)
    with b2:
        st.markdown('<div style="font-weight:700;color:var(--navy);margin:0 0 6px 2px;">Risk Level Distribution</div>', unsafe_allow_html=True)
        if not dataset.empty:
            risk_rows = portfolio_risk_distribution(dataset)
            risk_bar(risk_rows)
            st.markdown(
                '<div class="info-box" style="margin-top:10px;">🛡️ Risk levels are based on modeled default probability: Low &lt; 20%, Moderate 20–49.9%, High ≥ 50%.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="card">Reference dataset not found.</div>', unsafe_allow_html=True)

    section("Quick Actions", "Start a workflow")
    a, b, c = st.columns(3)
    with a:
        if st.button("＋ New Loan Application", use_container_width=True):
            st.session_state.page = "New Application"
            st.rerun()
    with b:
        if st.button("⌕ Open Credit Analysis", use_container_width=True):
            st.session_state.page = "Credit Analysis"
            st.rerun()
    with c:
        if st.button("▣ View Applications", use_container_width=True):
            st.session_state.page = "Applications"
            st.rerun()

# -----------------------------
# New Application
# -----------------------------
elif st.session_state.page == "New Application":
    page_header("➕ New Loan Application", "Complete the applicant profile to generate a credit assessment.")

    with st.form("loan_application_form", clear_on_submit=False):
        section("01 • Applicant Profile", "Identity and employment")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_name = st.text_input("Applicant Name", placeholder="e.g. Rahul Kumar")
        with c2:
            age = st.number_input("Age", 21, 65, 30)
        with c3:
            education = st.selectbox("Education", ["High School", "Diploma", "Graduate", "Postgraduate"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business Owner", "Contract"])
        with c2:
            employment_years = st.number_input("Employment Years", 0, 40, 5)
        with c3:
            dependents = st.number_input("Dependents", 0, 10, 0)
        with c4:
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        st.markdown('</div>', unsafe_allow_html=True)

        section("02 • Financial Profile", "Income, obligations and assets")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            annual_income = st.number_input("Annual Income (₹)", 180000, 10000000, 750000, step=10000)
        with c2:
            monthly_expenses = st.number_input("Monthly Expenses (₹)", 0, 1000000, 25000, step=1000)
        with c3:
            savings_balance = st.number_input("Savings Balance (₹)", 0, 10000000, 200000, step=10000)

        c1, c2, c3 = st.columns(3)
        with c1:
            asset_value = st.number_input("Total Asset Value (₹)", 0, 50000000, 1500000, step=10000)
        with c2:
            existing_loans = st.number_input("Existing Loans", 0, 10, 1)
        with c3:
            existing_emi = st.number_input("Existing EMI (₹)", 0, 500000, 8000, step=500)
        st.markdown('</div>', unsafe_allow_html=True)

        section("03 • Credit Profile", "Repayment history and credit behaviour")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            credit_score = st.number_input("Credit Score", 300, 850, 740)
        with c2:
            credit_history_years = st.number_input("Credit History (Years)", 0, 40, 8)
        with c3:
            previous_default = st.selectbox("Previous Default", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with c4:
            number_of_credit_accounts = st.number_input("Credit Accounts", 0, 30, 4)
        late_payments = st.number_input("Late Payments", 0, 30, 0)
        st.markdown('</div>', unsafe_allow_html=True)

        section("04 • Loan Request", "Requested facility details")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amount = st.number_input("Loan Amount Requested (₹)", 50000, 10000000, 500000, step=10000)
        with c2:
            loan_term_months = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 84, 120], index=4)
        with c3:
            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Vehicle", "Personal", "Business", "Medical"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Calculations are shown before submit through the values used in the form.
        monthly_income = annual_income / 12
        dti_ratio = (existing_emi / monthly_income) * 100 if monthly_income else 0
        loan_to_income = loan_amount / annual_income if annual_income else 0
        disposable_income = monthly_income - monthly_expenses - existing_emi

        section("05 • Pre-Assessment", "Calculated financial indicators")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            metric_card("MONTHLY INCOME", money(monthly_income), "Annual income ÷ 12")
        with p2:
            metric_card("DTI RATIO", f"{dti_ratio:.2f}%", "Existing EMI ÷ monthly income")
        with p3:
            metric_card("LOAN / INCOME", f"{loan_to_income:.2f}", "Requested amount ÷ annual income")
        with p4:
            metric_card("DISPOSABLE INCOME", money(disposable_income), "Income − expenses − EMI")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Analyze Application →", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not applicant_name.strip():
            errors.append("Please enter applicant name.")
        if existing_emi > monthly_income:
            errors.append("Existing EMI cannot exceed monthly income.")
        if monthly_expenses > monthly_income:
            errors.append("Monthly expenses cannot exceed monthly income.")
        if loan_amount > annual_income * 3:
            errors.append("Requested loan amount is unusually high compared with annual income.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            application = {
                "Age": age,
                "Education": education,
                "Employment_Type": employment_type,
                "Employment_Years": employment_years,
                "Dependents": dependents,
                "Marital_Status": marital_status,
                "Annual_Income": annual_income,
                "Monthly_Expenses": monthly_expenses,
                "Existing_Loans": existing_loans,
                "Existing_EMI": existing_emi,
                "Savings_Balance": savings_balance,
                "Asset_Value": asset_value,
                "Credit_Score": credit_score,
                "Credit_History_Years": credit_history_years,
                "Previous_Default": previous_default,
                "Number_of_Credit_Accounts": number_of_credit_accounts,
                "Late_Payments": late_payments,
                "Loan_Amount_Requested": loan_amount,
                "Loan_Term_Months": loan_term_months,
                "Loan_Purpose": loan_purpose,
                "Monthly_Income": monthly_income,
                "DTI_Ratio": dti_ratio,
                "Loan_to_Income": loan_to_income,
                "Disposable_Income": disposable_income,
            }
            result = run_prediction(application)
            result["applicant_name"] = applicant_name.strip()
            save_history(result, applicant_name.strip())
            st.session_state.last_result = result
            st.session_state.page = "Credit Analysis"
            st.rerun()

# -----------------------------
# Credit Analysis
# -----------------------------
elif st.session_state.page == "Credit Analysis":
    result = st.session_state.last_result
    if not result:
        page_header("🛡️ Credit Analysis", "No application has been analyzed yet.")
        st.markdown(
            '<div class="card"><h3 style="color:#0b1f3a;">Ready for your first assessment?</h3><p style="color:#64748b;">Create a new loan application and the complete decision center will appear here.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Start New Application →", type="primary"):
            st.session_state.page = "New Application"
            st.rerun()
    else:
        app = result["application"]
        decision = result["loan_decision"]
        risk = result["risk_level"]
        dkind = decision_kind(decision)
        rkind = risk_kind(risk)
        decision_title = {"APPROVED": "LOAN APPROVED", "REJECTED": "LOAN REJECTED", "REVIEW": "MANUAL REVIEW RECOMMENDED"}[decision]
        decision_icon = {"APPROVED": "✓", "REJECTED": "×", "REVIEW": "!"}[decision]

        page_header("🛡️ Credit Decision Center", f"Application {result.get('application_id', '—')} • {result['applicant_name']}")

        st.markdown(
            f"""
            <div class="decision-card decision-{dkind}">
                <div style="color:#64748b;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;">MODEL DECISION</div>
                <div style="font-size:42px;margin-top:5px;">{decision_icon}</div>
                <div class="decision-word" style="color:{'#16a34a' if decision=='APPROVED' else '#dc2626' if decision=='REJECTED' else '#d97706'};">{decision_title}</div>
                <div style="color:#64748b;font-size:14px;">Applicant: <b>{result['applicant_name']}</b> • Requested: <b>{money(app['Loan_Amount_Requested'])}</b></div>
                <div style="margin-top:22px;">
                    <span class="badge badge-{rkind}">{risk} RISK</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("APPROVAL PROBABILITY", f"{result['approval_probability']:.2f}%", "Model confidence for approval")
        with c2:
            metric_card("DEFAULT PROBABILITY", f"{result['default_probability']:.2f}%", "Predicted probability of default")
        with c3:
            metric_card("CREDIT RISK SCORE", f"{result['risk_score']}/100", "100 = stronger modeled profile")
        with c4:
            metric_card("CREDIT SCORE", f"{app['Credit_Score']}", "Applicant credit score")

        section("Risk Score", "Modeled default-risk inverse")
        st.markdown(
            f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:end;">
                    <div><div class="risk-score">{result['risk_score']}</div><div style="color:#64748b;font-size:12px;">out of 100</div></div>
                    <div style="text-align:right;">{badge(risk + ' RISK', rkind)}<div style="color:#64748b;font-size:12px;margin-top:6px;">Default probability {result['default_probability']:.2f}%</div></div>
                </div>
                <div class="progress-track"><div class="progress-fill" style="width:{max(0, min(result['risk_score'], 100))}%;"></div></div>
                <div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:10px;"><span>HIGH RISK</span><span>LOW RISK</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section("Why this decision?", "Decision-support interpretation")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><b style="color:#0b1f3a;">Positive Factors</b>', unsafe_allow_html=True)
            if result["positive_factors"]:
                for item in result["positive_factors"]:
                    st.markdown(f'<div class="factor factor-positive">✓ {item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="factor">No significant positive factors identified.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><b style="color:#0b1f3a;">Risk Factors</b>', unsafe_allow_html=True)
            if result["risk_factors"]:
                for item in result["risk_factors"]:
                    st.markdown(f'<div class="factor factor-risk">⚠ {item}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="factor">No major rule-based risk factors identified.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        section("Financial Snapshot", "Key underwriting indicators")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("ANNUAL INCOME", money(app.get("Annual_Income", 0)))
        with c2:
            metric_card("MONTHLY EXPENSES", money(app.get("Monthly_Expenses", 0)))
        with c3:
            metric_card("DTI RATIO", f"{safe_float(app.get('DTI_Ratio')):.2f}%")
        with c4:
            metric_card("DISPOSABLE INCOME", money(app.get("Disposable_Income", 0)))

        section("Assessment", "Human-review recommendation")
        if risk == "LOW":
            msg = "The applicant demonstrates a relatively strong financial and credit profile. The modeled probability of default is low."
            st.markdown(f'<div class="info-box">🟢 <b>Low Credit Risk.</b> {msg}</div>', unsafe_allow_html=True)
        elif risk == "MODERATE":
            msg = "Some risk indicators are present. Additional verification and policy checks may be appropriate before final approval."
            st.markdown(f'<div style="background:#fffaf0;border:1px solid #fed7aa;padding:15px 17px;border-radius:12px;">🟠 <b>Moderate Credit Risk.</b> {msg}</div>', unsafe_allow_html=True)
        else:
            msg = "Significant risk indicators are present. The application should receive additional scrutiny and human underwriting review."
            st.markdown(f'<div style="background:#fff7f7;border:1px solid #fecaca;padding:15px 17px;border-radius:12px;">🔴 <b>High Credit Risk.</b> {msg}</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("＋ New Application", use_container_width=True):
                st.session_state.page = "New Application"
                st.rerun()
        with c2:
            st.download_button(
                "⇩ Download Assessment JSON",
                data=pd.Series({
                    "application_id": result.get("application_id", ""),
                    "applicant": result["applicant_name"],
                    "decision": decision,
                    "approval_probability": result["approval_probability"],
                    "default_probability": result["default_probability"],
                    "risk_level": risk,
                    "risk_score": result["risk_score"],
                }).to_json(indent=2),
                file_name=f"{result.get('application_id', 'loan_mitra_assessment')}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown('<div class="footer-note">ऋण MITRA is an educational decision-support system. Predictions must not replace official lending policy, regulatory requirements or human underwriting.</div>', unsafe_allow_html=True)

# -----------------------------
# Applications
# -----------------------------
elif st.session_state.page == "Applications":
    page_header("📁 Applications", "Review assessments generated during this session.")

    if not st.session_state.history:
        st.markdown('<div class="card"><h3 style="color:#0b1f3a;">No applications yet</h3><p style="color:#64748b;">Completed applications will appear here with their decision, risk and probability metrics.</p></div>', unsafe_allow_html=True)
        if st.button("Create First Application →", type="primary"):
            st.session_state.page = "New Application"
            st.rerun()
    else:
        dfh = pd.DataFrame(st.session_state.history)
        c1, c2, c3 = st.columns(3)
        with c1:
            search = st.text_input("Search Applicant", placeholder="Type a name...")
        with c2:
            risk_filter = st.selectbox("Risk", ["All", "LOW", "MODERATE", "HIGH"])
        with c3:
            decision_filter = st.selectbox("Decision", ["All", "APPROVED", "REVIEW", "REJECTED"])

        filtered = dfh.copy()
        if search:
            filtered = filtered[filtered["Applicant"].str.contains(search, case=False, na=False)]
        if risk_filter != "All":
            filtered = filtered[filtered["Risk"] == risk_filter]
        if decision_filter != "All":
            filtered = filtered[filtered["Decision"] == decision_filter]

        display = filtered.copy()
        display["Amount"] = display["Amount"].map(money)
        display["Approval %"] = display["Approval %"].map(lambda x: f"{x:.2f}%")
        display["Default %"] = display["Default %"].map(lambda x: f"{x:.2f}%")
        st.dataframe(display, use_container_width=True, hide_index=True)

# -----------------------------
# Analytics
# -----------------------------
elif st.session_state.page == "Analytics":
    page_header("📊 Portfolio Analytics", "Explore lending, credit and default patterns in the reference dataset.")

    if dataset.empty:
        st.warning("Reference dataset not found. Add data/loan_applicants_5000.csv to enable analytics.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            metric_card("RECORDS", f"{len(dataset):,}", "Reference portfolio")
        with c2:
            score = pd.to_numeric(dataset["Credit_Score"], errors="coerce").mean() if "Credit_Score" in dataset.columns else 0
            metric_card("AVG CREDIT SCORE", f"{safe_float(score):.0f}", "Portfolio average")
        with c3:
            income = pd.to_numeric(dataset["Annual_Income"], errors="coerce").mean() if "Annual_Income" in dataset.columns else 0
            metric_card("AVG INCOME", money(income), "Annual")
        with c4:
            loan = pd.to_numeric(dataset["Loan_Amount_Requested"], errors="coerce").mean() if "Loan_Amount_Requested" in dataset.columns else 0
            metric_card("AVG LOAN REQUEST", money(loan), "Requested")

        section("Loan Decisions", "Approved vs rejected")
        if "Loan_Status" in dataset.columns:
            status_counts = dataset["Loan_Status"].astype(str).str.strip().replace("", "Unknown").value_counts()
            if not status_counts.empty:
                st.bar_chart(status_counts)
        else:
            st.info("Loan_Status column not available.")

        section("Default Distribution", "Portfolio repayment outcome")
        if "Default_Status" in dataset.columns:
            d = pd.to_numeric(dataset["Default_Status"], errors="coerce").fillna(0)
            st.bar_chart(pd.Series({"No Default": int((d <= 0).sum()), "Default": int((d > 0).sum())}))
        else:
            st.info("Default_Status column not available.")

        section("Credit Score Distribution", "Applicant profile")
        if "Credit_Score" in dataset.columns:
            scores = pd.to_numeric(dataset["Credit_Score"], errors="coerce").dropna()
            if not scores.empty:
                st.bar_chart(scores.value_counts(bins=12).sort_index())
            else:
                st.info("No usable credit-score values found.")

        section("Income vs Requested Loan", "Portfolio relationship")
        if {"Annual_Income", "Loan_Amount_Requested"}.issubset(dataset.columns):
            rel = dataset[["Annual_Income", "Loan_Amount_Requested"]].copy()
            rel["Annual_Income"] = pd.to_numeric(rel["Annual_Income"], errors="coerce")
            rel["Loan_Amount_Requested"] = pd.to_numeric(rel["Loan_Amount_Requested"], errors="coerce")
            rel = rel.dropna().head(1000)
            if not rel.empty:
                st.line_chart(rel.set_index("Annual_Income"))
            else:
                st.info("No usable income/loan values found.")

        section("Risk Statistics", "Grouped reference averages")
        if "Loan_Status" in dataset.columns:
            cols = [c for c in ["Credit_Score","Annual_Income","Loan_Amount_Requested","DTI_Ratio","Existing_Loans","Late_Payments"] if c in dataset.columns]
            if cols:
                temp = dataset.copy()
                for col in cols:
                    temp[col] = pd.to_numeric(temp[col], errors="coerce")
                st.dataframe(temp.groupby("Loan_Status")[cols].mean(numeric_only=True).round(2), use_container_width=True)
            else:
                st.info("No numeric analytics columns found.")

        section("Dataset Information", "Reference data quality")
        a,b,c = st.columns(3)
        with a: metric_card("ROWS", f"{len(dataset):,}", "Records")
        with b: metric_card("COLUMNS", f"{len(dataset.columns):,}", "Features")
        with c: metric_card("MISSING VALUES", f"{int(dataset.isna().sum().sum()):,}", "Across dataset")

# -----------------------------
# Model Intelligence
# -----------------------------
elif st.session_state.page == "Model Intelligence":
    page_header("🧠 Model Intelligence", "Performance, model selection and deployment assets.")

    section("Runtime Model Status", "Current deployment state")
    c1,c2 = st.columns(2)
    with c1:
        if approval_model is not None:
            st.markdown(f'<div class="card">{badge("APPROVAL MODEL READY","green")}<div style="margin-top:12px;font-weight:700;color:#0b1f3a;">Approval Decision Engine</div><div style="margin-top:6px;color:#64748b;font-size:13px;">{os.path.basename(APPROVAL_MODEL_PATH)}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card">{badge("APPROVAL MODEL UNAVAILABLE","red")}<div style="margin-top:8px;color:#64748b;font-size:13px;">{approval_model_error}</div></div>', unsafe_allow_html=True)
    with c2:
        if risk_model is not None:
            st.markdown(f'<div class="card">{badge("RISK MODEL READY","green")}<div style="margin-top:12px;font-weight:700;color:#0b1f3a;">Default Risk Engine</div><div style="margin-top:6px;color:#64748b;font-size:13px;">{os.path.basename(RISK_MODEL_PATH)}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card">{badge("RISK MODEL UNAVAILABLE","red")}<div style="margin-top:8px;color:#64748b;font-size:13px;">{risk_model_error}</div></div>', unsafe_allow_html=True)

    def show_model_metrics(title, df):
        section(title, "Stored model evaluation results")
        if df.empty:
            st.info("No model comparison CSV found. Runtime model status is shown above.")
            return
        clean = df.copy()
        clean.columns = [str(c).strip() for c in clean.columns]
        st.dataframe(clean, use_container_width=True, hide_index=True)
        numeric = []
        for col in clean.columns:
            if str(col).lower() == "model":
                continue
            vals = pd.to_numeric(clean[col], errors="coerce")
            if vals.notna().any():
                numeric.append(col)
        if not numeric:
            return
        rank_col = next((c for c in ["ROC_AUC","ROC-AUC","ROC AUC","AUC","Accuracy"] if c in clean.columns), None)
        if rank_col is not None:
            vals = pd.to_numeric(clean[rank_col], errors="coerce")
            if vals.notna().any():
                best = clean.loc[vals.idxmax()]
                st.markdown(f'<div class="info-box">🏆 <b>Best stored model:</b> {best.get("Model", "Selected model")}</div>', unsafe_allow_html=True)
                cards = []
                for label, col in [("Accuracy","Accuracy"),("Precision","Precision"),("Recall","Recall"),("F1","F1_Score"),("ROC-AUC","ROC_AUC")]:
                    if col in clean.columns:
                        cards.append((label, safe_float(best.get(col))))
                if cards:
                    cols = st.columns(len(cards))
                    for c,(label,val) in zip(cols,cards):
                        with c: metric_card(label.upper(), f"{val:.3f}", "Best stored result")

    show_model_metrics("Approval Model", approval_metrics)
    show_model_metrics("Credit Risk Model", risk_metrics)

    section("Decision Engine", "How ऋण MITRA uses the models")
    a,b,c = st.columns(3)
    with a:
        st.markdown('<div class="card"><div style="font-size:28px;">📥</div><h4 style="color:#0b1f3a;">1. Applicant Data</h4><p style="color:#64748b;font-size:13px;">Financial, employment, credit and requested-loan information is collected.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><div style="font-size:28px;">🤖</div><h4 style="color:#0b1f3a;">2. ML Prediction</h4><p style="color:#64748b;font-size:13px;">The approval model estimates approval probability while the risk model estimates default probability.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><div style="font-size:28px;">🛡️</div><h4 style="color:#0b1f3a;">3. Decision Support</h4><p style="color:#64748b;font-size:13px;">Probabilities are translated into approval, review and risk indicators for human underwriting.</p></div>', unsafe_allow_html=True)

    section("Risk Classification", "Default probability thresholds")
    a,b,c = st.columns(3)
    with a: st.markdown('<div class="card"><span class="badge badge-green">LOW</span><h4 style="color:#0b1f3a;">Below 20%</h4><p style="color:#64748b;font-size:13px;">Low modeled default probability.</p></div>', unsafe_allow_html=True)
    with b: st.markdown('<div class="card"><span class="badge badge-amber">MODERATE</span><h4 style="color:#0b1f3a;">20% – 49.9%</h4><p style="color:#64748b;font-size:13px;">Additional review may be appropriate.</p></div>', unsafe_allow_html=True)
    with c: st.markdown('<div class="card"><span class="badge badge-red">HIGH</span><h4 style="color:#0b1f3a;">50% or higher</h4><p style="color:#64748b;font-size:13px;">Additional underwriting scrutiny is recommended.</p></div>', unsafe_allow_html=True)

brand_banner()
st.markdown('<div class="footer-note">ऋण MITRA • AI Banking Decision Support • For educational / demonstration use</div>', unsafe_allow_html=True)