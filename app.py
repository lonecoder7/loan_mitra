import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ============================================================
# ऋण MITRA V2
# Custom HTML/CSS-style Banking Credit Intelligence UI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

ICON_PATH = os.path.join(BASE_DIR, "icon.png")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")


st.set_page_config(
    page_title="ऋण MITRA | Credit Intelligence",
    page_icon=Image.open(ICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)


# Explicit browser favicon using the same transparent emblem.
# Streamlit's page_icon handles the native tab icon; this metadata reinforces it in the page head.
try:
    import base64
    _favicon_b64 = base64.b64encode(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png"), "rb").read()).decode()
    st.markdown(
        f'<link rel="icon" type="image/png" href="data:image/png;base64,{_favicon_b64}">',
        unsafe_allow_html=True,
    )
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

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
    approval = joblib.load(APPROVAL_MODEL_PATH)
    risk = joblib.load(RISK_MODEL_PATH)
    return approval, risk


@st.cache_data

def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data

def load_metrics(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


approval_model, risk_model = load_models()
dataset = load_dataset()
approval_metrics = load_metrics(APPROVAL_METRICS_PATH)
risk_metrics = load_metrics(RISK_METRICS_PATH)

# -----------------------------
# Global CSS
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--navy:#071f42;--navy2:#0d2f5b;--blue:#2563eb;--bg:#f7f9fc;--border:#e4e9f1;--muted:#64748b;--text:#14213d;--green:#16a34a;--red:#ef4444;--amber:#f59e0b;}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
.stApp{background:var(--bg)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stToolbar"]{display:none}
.block-container{max-width:1500px;padding:0 1.25rem 1.5rem}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#061d3e 0%,#06244d 100%);border:0}
section[data-testid="stSidebar"]>div{padding:1rem .65rem}
section[data-testid="stSidebar"] *{color:#fff!important}
.sidebar-logo{height:112px;display:flex;align-items:center;justify-content:center;margin-bottom:12px}
.sidebar-logo img{width:100px;height:100px;object-fit:contain;filter:drop-shadow(0 8px 18px rgba(0,0,0,.22))}
section[data-testid="stSidebar"] .stRadio label{border-radius:10px;padding:11px 12px;margin:3px 0;transition:.15s;background:transparent}
section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(255,255,255,.08)}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"]>label>div:first-child{display:none}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"]>label[data-checked="true"]{background:#155aa5}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{margin:0}
.topbar{height:82px;background:#fff;border-bottom:1px solid #e8edf4;display:flex;align-items:center;justify-content:space-between;padding:0 18px 0 18px;margin:0 -1.25rem 18px}
.brand{display:flex;align-items:center;gap:10px}.brand img{width:48px;height:48px;object-fit:contain}.brand-name{font-size:29px;font-weight:800;color:#18345c;line-height:1}.brand-name span{color:#c98a19}.brand-sub{font-size:12px;color:#344563;margin-top:4px}
.profile{display:flex;align-items:center;gap:13px;color:#14213d}.avatar{width:42px;height:42px;border-radius:50%;background:#0b315f;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800}.role{font-size:14px}.chev{font-size:18px;color:#334155}
.page-head{display:flex;align-items:center;justify-content:space-between;margin:4px 0 16px}.page-head-left{display:flex;align-items:center;gap:14px}.page-icon{font-size:40px;line-height:1;color:#102e56}.page-title{font-size:27px;font-weight:800;color:#10234a;margin:0}.page-sub{font-size:13px;color:#51627d;margin-top:5px}.head-actions{display:flex;gap:10px}.head-pill{background:#fff;border:1px solid #dce3ec;border-radius:10px;padding:11px 15px;color:#10234a;font-size:13px}
.kpi{background:#fff;border:1px solid var(--border);border-radius:16px;padding:18px 16px;min-height:120px;box-shadow:0 4px 16px rgba(15,23,42,.035)}
.kpi-top{display:flex;align-items:center;gap:11px}.kpi-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:22px;font-weight:800}.kpi-label{font-size:12px;color:#42526b}.kpi-value{font-size:28px;font-weight:800;color:#10234a;margin-top:6px}.kpi-note{font-size:11px;margin-top:5px}.green{color:#0c9b4b}.red{color:#ef4444}.orange{color:#d97706}.purple{color:#7c3aed}.cyan{color:#0891b2}
.panel{background:#fff;border:1px solid var(--border);border-radius:16px;padding:18px;box-shadow:0 4px 16px rgba(15,23,42,.035)}.panel-title{font-size:17px;font-weight:800;color:#10234a;margin-bottom:12px}.panel-head{display:flex;justify-content:space-between;align-items:center}.select-pill{border:1px solid #dce3ec;background:#fff;border-radius:9px;padding:8px 12px;font-size:12px;color:#10234a}
.donut-wrap{display:flex;align-items:center;justify-content:center;gap:22px;min-height:185px}.donut{width:150px;height:150px;border-radius:50%;position:relative;display:flex;align-items:center;justify-content:center}.donut:after{content:"";position:absolute;width:84px;height:84px;border-radius:50%;background:#fff}.donut-center{position:relative;z-index:2;text-align:center;color:#10234a}.donut-total{font-size:18px;font-weight:800}.donut-small{font-size:11px;color:#64748b}.legend{display:flex;flex-direction:column;gap:14px}.legend-row{display:flex;align-items:flex-start;gap:8px;font-size:13px;color:#10234a}.dot{width:11px;height:11px;border-radius:50%;margin-top:3px}.legend-val{font-weight:700;margin-top:3px}
.risk-row{display:grid;grid-template-columns:110px 1fr 100px;gap:12px;align-items:center;margin:18px 0}.risk-label{font-weight:700;font-size:13px;color:#10234a}.track{height:8px;background:#edf1f5;border-radius:99px;overflow:hidden}.fill{height:100%;border-radius:99px}.risk-val{text-align:right;color:#334155;font-size:13px}.alert{margin-top:16px;background:#f1f6ff;border-radius:12px;padding:13px 16px;color:#163b70;font-size:12px}
.table{width:100%;border-collapse:collapse;font-size:12px}.table th{background:#f8fafc;color:#334155;text-align:left;padding:10px;border-bottom:1px solid #e8edf3}.table td{padding:10px;border-bottom:1px solid #eef2f6;color:#24324b}.badge{padding:4px 9px;border-radius:6px;font-size:11px;font-weight:700;display:inline-block}.b-green{background:#dcfce7;color:#15803d}.b-red{background:#fee2e2;color:#dc2626}.b-amber{background:#ffedd5;color:#c2410c}.b-blue{background:#dbeafe;color:#1d4ed8}
.ai-banner{height:112px;border-radius:17px;background:linear-gradient(100deg,#06234a,#062f63);margin-top:16px;padding:23px 26px;position:relative;overflow:hidden;color:#fff}.ai-eyebrow{font-size:11px;letter-spacing:.16em;font-weight:700;color:#a8c9f6}.ai-title{font-size:23px;font-weight:800;margin-top:10px}.ai-chip{position:absolute;right:70px;top:22px;width:72px;height:72px;border:2px solid #1bb4ff;border-radius:20px;display:flex;align-items:center;justify-content:center;color:#42c8ff;font-size:25px;font-weight:800;box-shadow:0 0 30px rgba(35,182,255,.18)}
.footer-note{font-size:10px;color:#94a3b8;text-align:center;padding:14px 0}
.stButton>button{border-radius:10px!important}.stDownloadButton>button{border-radius:10px!important}
@media(max-width:1100px){.kpi-value{font-size:23px}.brand-name{font-size:24px}.risk-row{grid-template-columns:90px 1fr 85px}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def money(value):
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


def run_prediction(application):
    input_df = pd.DataFrame([application])

    approval_prediction = int(approval_model.predict(input_df)[0])
    approval_probability = float(approval_model.predict_proba(input_df)[0][1])

    risk_prediction = int(risk_model.predict(input_df)[0])
    default_probability = float(risk_model.predict_proba(input_df)[0][1])

    default_percentage = default_probability * 100
    if default_percentage < 20:
        risk_level = "LOW"
    elif default_percentage < 50:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    # Keep the model's original approval output, but expose a realistic
    # manual-review state for moderate/high risk approved predictions.
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


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

# -----------------------------
# Helpers
# -----------------------------
def money(value):
    return f"₹{value:,.0f}"

def risk_kind(risk):
    return {"LOW":"green","MODERATE":"amber","HIGH":"red"}.get(risk,"blue")

def decision_kind(decision):
    return {"APPROVED":"green","REVIEW":"amber","REJECTED":"red"}.get(decision,"blue")

def risk_score(default_probability):
    return int(np.clip(round(100-default_probability*100),0,100))

def calculate_factors(app):
    positive=[]; risks=[]
    if app["Credit_Score"]>=750: positive.append("Excellent credit score")
    elif app["Credit_Score"]>=700: positive.append("Good credit score")
    elif app["Credit_Score"]<600: risks.append("Low credit score")
    if app["DTI_Ratio"]<=20: positive.append("Low debt-to-income ratio")
    elif app["DTI_Ratio"]>40: risks.append("High debt-to-income ratio")
    if app["Previous_Default"]==0: positive.append("No previous loan default")
    else: risks.append("Previous loan default recorded")
    if app["Late_Payments"]==0: positive.append("No recent late payments")
    elif app["Late_Payments"]>=3: risks.append("Multiple late payments")
    if app["Employment_Years"]>=5: positive.append("Stable employment history")
    elif app["Employment_Years"]<2: risks.append("Short employment history")
    if app["Disposable_Income"]>app["Monthly_Income"]*.25: positive.append("Healthy disposable income")
    elif app["Disposable_Income"]<=0: risks.append("Very low or negative disposable income")
    return positive, risks

def run_prediction(application):
    input_df=pd.DataFrame([application])
    approval_prediction=int(approval_model.predict(input_df)[0])
    approval_probability=float(approval_model.predict_proba(input_df)[0][1])
    risk_prediction=int(risk_model.predict(input_df)[0])
    default_probability=float(risk_model.predict_proba(input_df)[0][1])
    dp=default_probability*100
    risk_level="LOW" if dp<20 else "MODERATE" if dp<50 else "HIGH"
    decision="REVIEW" if approval_prediction==1 and risk_level=="HIGH" else "APPROVED" if approval_prediction==1 else "REJECTED"
    positive,risks=calculate_factors(application)
    return {"loan_decision":decision,"approval_probability":round(approval_probability*100,2),"default_probability":round(dp,2),"risk_level":risk_level,"default_prediction":risk_prediction,"risk_score":risk_score(default_probability),"positive_factors":positive,"risk_factors":risks,"application":application}

def save_history(result,name):
    app=result["application"]
    application_id=f"RM-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(st.session_state.history)+1:03d}"
    st.session_state.history.insert(0,{"Application ID":application_id,"Applicant":name,"Amount":app["Loan_Amount_Requested"],"Risk":result["risk_level"],"Decision":result["loan_decision"],"Approval %":result["approval_probability"],"Default %":result["default_probability"],"Date":datetime.now().strftime('%d %b %Y, %H:%M')})
    result["application_id"]=application_id

def metric_card(label,value,note,icon="•",icon_bg="#2563eb",note_class=""):
    st.markdown(f'''<div class="kpi"><div class="kpi-top"><div class="kpi-icon" style="background:{icon_bg};">{icon}</div><div class="kpi-label">{label}</div></div><div class="kpi-value">{value}</div><div class="kpi-note {note_class}">{note}</div></div>''',unsafe_allow_html=True)

def section(title,subtitle=""):
    st.markdown(f'<div class="panel-head" style="margin:16px 0 10px"><div class="panel-title" style="margin:0">{title}</div><div style="font-size:12px;color:#64748b">{subtitle}</div></div>',unsafe_allow_html=True)

def page_header(title,subtitle):
    st.markdown(f'<div class="page-head"><div class="page-head-left"><div class="page-icon">▦</div><div><div class="page-title">{title}</div><div class="page-sub">{subtitle}</div></div></div><div class="head-actions"><div class="head-pill">▣ &nbsp; May 13 – Jun 12, 2025 &nbsp;⌄</div><div class="head-pill">⇧ &nbsp; Export</div></div></div>',unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    import base64 as _b64
    _icon64=_b64.b64encode(open(ICON_PATH,"rb").read()).decode()
    st.markdown(f'<div class="sidebar-logo"><img src="data:image/png;base64,{_icon64}" alt="ऋण MITRA"></div>',unsafe_allow_html=True)
    nav_options=["Dashboard","New Application","Credit Analysis","Applications","Analytics","Model Intelligence","Settings","Help & Support"]
    nav=st.radio("NAVIGATION",nav_options,index=nav_options.index(st.session_state.page),label_visibility="visible")
    if nav!=st.session_state.page:
        st.session_state.page=nav; st.rerun()

# -----------------------------
# Main Header
# -----------------------------
_brand64=_b64.b64encode(open(ICON_PATH,"rb").read()).decode()
st.markdown(f'''<div class="topbar"><div class="brand"><img src="data:image/png;base64,{_brand64}" alt="ऋण MITRA"><div><div class="brand-name">ऋण <span>MITRA</span></div><div class="brand-sub">AI-Powered Credit Decision Support</div></div></div><div class="profile"><div style="font-size:22px">☼</div><div class="avatar">CO</div><div class="role">Credit Officer</div><div class="chev">⌄</div></div></div>''',unsafe_allow_html=True)

# -----------------------------
# Dashboard
# -----------------------------
if st.session_state.page == "Dashboard":
    page_header("Dashboard","Overview of lending activity and credit risk")
    if not dataset.empty:
        total=len(dataset)
        approved=int((dataset["Loan_Status"]=="Approved").sum()) if "Loan_Status" in dataset else 0
        rejected=int((dataset["Loan_Status"]=="Rejected").sum()) if "Loan_Status" in dataset else 0
        approval_rate=approved/total*100 if total else 0
        rejection_rate=rejected/total*100 if total else 0
        defaulted=int(dataset["Default_Status"].sum()) if "Default_Status" in dataset else 0
        default_rate=defaulted/total*100 if total else 0
        no_default=total-defaulted
        avg_credit=float(dataset["Credit_Score"].mean()) if "Credit_Score" in dataset else 0
        avg_loan=float(dataset["Loan_Amount_Requested"].mean()) if "Loan_Amount_Requested" in dataset else 0
    else:
        total=approved=rejected=defaulted=no_default=0; approval_rate=rejection_rate=default_rate=avg_credit=avg_loan=0

    k=st.columns(6)
    with k[0]: metric_card("TOTAL APPLICATIONS",f"{total:,}","100% of total","▤","#2563eb")
    with k[1]: metric_card("APPROVAL RATE",f"{approval_rate:.1f}%",f"{approved:,} Approved","✓","#16a34a","green")
    with k[2]: metric_card("REJECTION RATE",f"{rejection_rate:.1f}%",f"{rejected:,} Rejected","×","#ef4444","red")
    with k[3]: metric_card("DEFAULT RATE",f"{default_rate:.1f}%",f"{defaulted:,} Defaulted","!","#f59e0b","orange")
    with k[4]: metric_card("AVG. CREDIT SCORE",f"{avg_credit:.0f}","Good","◔","#7c3aed","purple")
    with k[5]: metric_card("AVG. LOAN AMOUNT",money(avg_loan),"Per Application","₹","#0891b2","cyan")

    st.markdown('<div style="height:1px"></div>',unsafe_allow_html=True)
    left,right=st.columns([1.35,1.35])
    with left:
        st.markdown('<div class="panel"><div class="panel-head"><div class="panel-title">Application Trend</div><div class="select-pill">Daily　⌄</div></div>',unsafe_allow_html=True)
        if total:
            bins=20
            counts=np.array_split(dataset,bins)
            trend=pd.DataFrame({"Applications":[len(x) for x in counts]},index=["May 13","May 15","May 17","May 19","May 20","May 22","May 24","May 26","May 27","May 29","May 31","Jun 02","Jun 03","Jun 05","Jun 07","Jun 09","Jun 10","Jun 11","Jun 12","Jun 12"])
            st.line_chart(trend,height=190,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        c1,c2=st.columns(2)
        with c1:
            approved_pct=approval_rate
            st.markdown(f'''<div class="panel"><div class="panel-title">Approval vs Rejection</div><div class="donut-wrap"><div class="donut" style="background:conic-gradient(#16a34a 0 {approved_pct:.2f}%,#ef4444 {approved_pct:.2f}% 100%);"><div class="donut-center"><div class="donut-total">{total:,}</div><div class="donut-small">Total</div></div></div><div class="legend"><div class="legend-row"><span class="dot" style="background:#16a34a"></span><div>Approved<div class="legend-val">{approval_rate:.1f}% ({approved:,})</div></div></div><div class="legend-row"><span class="dot" style="background:#ef4444"></span><div>Rejected<div class="legend-val">{rejection_rate:.1f}% ({rejected:,})</div></div></div></div></div>''',unsafe_allow_html=True)
        with c2:
            default_pct=default_rate
            st.markdown(f'''<div class="panel"><div class="panel-title">Default vs No Default</div><div class="donut-wrap"><div class="donut" style="background:conic-gradient(#f59e0b 0 {default_pct:.2f}%,#2563eb {default_pct:.2f}% 100%);"><div class="donut-center"><div class="donut-total">{total:,}</div><div class="donut-small">Total</div></div></div><div class="legend"><div class="legend-row"><span class="dot" style="background:#f59e0b"></span><div>Defaulted<div class="legend-val">{default_rate:.1f}% ({defaulted:,})</div></div></div><div class="legend-row"><span class="dot" style="background:#2563eb"></span><div>No Default<div class="legend-val">{(100-default_rate):.1f}% ({no_default:,})</div></div></div></div></div>''',unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>',unsafe_allow_html=True)
    left,right=st.columns([1.25,1])
    with left:
        st.markdown('<div class="panel"><div class="panel-head"><div class="panel-title">Recent Applications</div><div class="select-pill">View all</div></div>',unsafe_allow_html=True)
        if not dataset.empty:
            recent=dataset.head(5).copy()
            rows=[]
            for i,(_,r) in enumerate(recent.iterrows()):
                name=str(r.get("Name",f"Applicant {i+1}"))
                amount=money(float(r.get("Loan_Amount_Requested",0)))
                decision=str(r.get("Loan_Status","Review")).upper()
                if decision=="APPROVED": dec='<span class="badge b-green">Approved</span>'
                elif decision=="REJECTED": dec='<span class="badge b-red">Rejected</span>'
                else: dec='<span class="badge b-amber">Review</span>'
                score=float(r.get("Credit_Score",0))
                risk="Low" if score>=700 else "Moderate" if score>=600 else "High"
                rb="b-green" if risk=="Low" else "b-amber" if risk=="Moderate" else "b-red"
                rows.append(f'<tr><td>{name}</td><td>{amount}</td><td>{dec}</td><td><span class="badge {rb}">{risk}</span></td><td>Jun 12, 2025</td></tr>')
            html='<table class="table"><thead><tr><th>Applicant Name</th><th>Loan Amount</th><th>Decision</th><th>Risk Level</th><th>Date</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
            st.markdown(html,unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Risk Level Distribution</div>',unsafe_allow_html=True)
        low=int((dataset["Credit_Score"]>=700).sum()) if not dataset.empty and "Credit_Score" in dataset else 0
        high=int((dataset["Credit_Score"]<600).sum()) if not dataset.empty and "Credit_Score" in dataset else 0
        moderate=max(total-low-high,0)
        for label,count,color in [("Low Risk",low,"#16a34a"),("Moderate Risk",moderate,"#f59e0b"),("High Risk",high,"#ef4444")]:
            pct=count/total*100 if total else 0
            st.markdown(f'<div class="risk-row"><div class="risk-label">{label}</div><div class="track"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div><div class="risk-val">{count:,} ({pct:.1f}%)</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="alert">◉ &nbsp; High risk applications require additional review and documentation.</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="ai-banner"><div class="ai-eyebrow">AI-POWERED CREDIT DECISION SUPPORT</div><div class="ai-title">ऋण MITRA Credit Intelligence</div><div class="ai-chip">AI</div></div>',unsafe_allow_html=True)

# -----------------------------
# New Application
# -----------------------------
elif st.session_state.page == "New Application":
    page_header("New Loan Application", "Complete the applicant profile to generate a credit assessment.")

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
        page_header("Credit Analysis", "No application has been analyzed yet.")
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

        page_header("Credit Decision Center", f"Application {result.get('application_id', '—')} • {result['applicant_name']}")

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
                <div class="progress-track"><div class="progress-fill" style="width:{result['risk_score']}%;"></div></div>
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
            metric_card("ANNUAL INCOME", money(app["Annual_Income"]))
        with c2:
            metric_card("MONTHLY EXPENSES", money(app["Monthly_Expenses"]))
        with c3:
            metric_card("DTI RATIO", f"{app['DTI_Ratio']:.2f}%")
        with c4:
            metric_card("DISPOSABLE INCOME", money(app["Disposable_Income"]))

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
                file_name=f"{result.get('application_id', 'rṇ_mitra_assessment')}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown('<div class="footer-note">ऋण MITRA is an educational decision-support system. Predictions must not replace official lending policy, regulatory requirements or human underwriting.</div>', unsafe_allow_html=True)

# -----------------------------
# Applications
# -----------------------------
elif st.session_state.page == "Applications":
    page_header("Applications", "Review assessments generated during this session.")

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
    page_header("Portfolio Analytics", "Explore the reference dataset used by the credit models.")
    if dataset.empty:
        st.warning("Reference dataset not found. Add data/loan_applicants_5000.csv to enable analytics.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("RECORDS", f"{len(dataset):,}")
        with c2:
            metric_card("AVG CREDIT SCORE", f"{dataset['Credit_Score'].mean():.0f}")
        with c3:
            metric_card("AVG INCOME", money(dataset['Annual_Income'].mean()))
        with c4:
            metric_card("AVG LOAN REQUEST", money(dataset['Loan_Amount_Requested'].mean()))

        section("Loan Decisions", "Approved vs rejected")
        if "Loan_Status" in dataset.columns:
            st.bar_chart(dataset["Loan_Status"].value_counts())

        section("Credit Score Distribution", "Applicant profile")
        if "Credit_Score" in dataset.columns:
            st.bar_chart(dataset["Credit_Score"].value_counts(bins=12).sort_index())

        section("Income vs Requested Loan", "Sampled portfolio relationship")
        if {"Annual_Income", "Loan_Amount_Requested"}.issubset(dataset.columns):
            sample = dataset[["Annual_Income", "Loan_Amount_Requested"]].copy().head(1000)
            sample = sample.set_index("Annual_Income")
            st.line_chart(sample)

        section("Risk Statistics", "Grouped reference averages")
        if "Loan_Status" in dataset.columns:
            cols = [c for c in ["Credit_Score", "Annual_Income", "Loan_Amount_Requested", "DTI_Ratio", "Existing_Loans", "Late_Payments"] if c in dataset.columns]
            if cols:
                st.dataframe(dataset.groupby("Loan_Status")[cols].mean().round(2), use_container_width=True)

# -----------------------------
# Model Intelligence
# -----------------------------
elif st.session_state.page == "Model Intelligence":
    page_header("Model Intelligence", "Performance, model selection and deployment assets.")

    section("Approval Model", "Best model selected by ROC-AUC")
    if not approval_metrics.empty:
        st.dataframe(approval_metrics.style.format({c: "{:.3f}" for c in approval_metrics.columns if c != "Model"}), use_container_width=True, hide_index=True)
        best = approval_metrics.sort_values("ROC_AUC", ascending=False).iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, key in zip([c1,c2,c3,c4,c5], ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]):
            with col:
                metric_card(label.upper(), f"{best[key]:.3f}")

    section("Credit Risk Model", "Default prediction model")
    if not risk_metrics.empty:
        st.dataframe(risk_metrics.style.format({c: "{:.3f}" for c in risk_metrics.columns if c != "Model"}), use_container_width=True, hide_index=True)
        best = risk_metrics.sort_values("ROC_AUC", ascending=False).iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, key in zip([c1,c2,c3,c4,c5], ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], ["Accuracy","Precision","Recall","F1_Score","ROC_AUC"]):
            with col:
                metric_card(label.upper(), f"{best[key]:.3f}")

    section("Model Assets", "Configured credit decision pipelines")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div style=\"font-size:18px;font-weight:800;color:#0b1f3a;\">Approval Model</div>", unsafe_allow_html=True)
        st.write("Preprocessing + selected approval classifier")
        st.caption(os.path.basename(APPROVAL_MODEL_PATH))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div style=\"font-size:18px;font-weight:800;color:#0b1f3a;\">Credit Risk Model</div>", unsafe_allow_html=True)
        st.write("Preprocessing + selected default-risk classifier")
        st.caption(os.path.basename(RISK_MODEL_PATH))
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Settings / Help
# -----------------------------
elif st.session_state.page == "Settings":
    page_header("Settings","Application preferences")
    st.markdown('<div class="panel"><div class="panel-title">ऋण MITRA Preferences</div><p style="color:#64748b">UI theme, dashboard defaults and decision-support settings can be configured here.</p></div>',unsafe_allow_html=True)
elif st.session_state.page == "Help & Support":
    page_header("Help & Support","Guidance for using the credit decision workspace")
    st.markdown('<div class="panel"><div class="panel-title">How to use ऋण MITRA</div><p style="color:#64748b">Start a New Application, complete the financial and credit profile, then review the model decision and risk factors in Credit Analysis.</p></div>',unsafe_allow_html=True)

# Minimal application disclaimer only. No system-status panel or demo footer.
st.markdown('<div class="footer-note">ऋण MITRA is an educational decision-support system. Predictions must not replace official lending policy, regulatory requirements or human underwriting.</div>', unsafe_allow_html=True)
