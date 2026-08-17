# ऋण MITRA — AI-Powered Credit Decision Support

A Streamlit-based banking dashboard engineered for loan approval support, default-risk assessment, portfolio analytics, and model intelligence.

> **Disclaimer:** This project is intended strictly for academic, portfolio, and demonstration purposes. It is not financial advice or an automated lending authority. Real lending decisions require appropriate validation, governance, regulatory compliance, explainability, human oversight, fairness testing, privacy controls, and secure handling of financial data.

---

## 🌟 Key Highlights

* **Modern Banking UI:** Clean interface built with Streamlit and custom HTML/CSS.
* **Branding:** Seamless integration with full logo in the header and emblem icon for the sidebar and favicon.
* **Credit Decision Workflow:** Automated loan application evaluation and default-risk assessment using integrated ML models.
* **Portfolio Analytics:** Real-time visual tracking of key lending metrics:
  * Total application volume & recent applications table
  * Approval vs. rejection ratios (Donut chart)
  * Default vs. non-default distributions (Donut chart)
  * Risk-level distribution across applicants
  * Portfolio averages (Credit score, loan amount, trends)
* **Model Intelligence:** Performance analytics, model metrics, and risk model comparison pages.
* **Session History:** Tracks interactive decisions within the current session.

---

## 📁 Repository Structure

```text
loan-mitra/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── logo.png             # Full logo for app header
│   └── icon.png             # Emblem icon for sidebar/favicon
├── models/
│   ├── loan_mitra_best_model.pkl
│   ├── loan_mitra_risk_model.pkl
│   ├── model_comparison.csv
│   └── risk_model_comparison.csv
└── data/
    └── loan_applicants_5000.csv
