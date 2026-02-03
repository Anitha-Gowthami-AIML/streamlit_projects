# =========================
# IMPORTS (VERY IMPORTANT)
# =========================
import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# BASE DIRECTORY (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "layoff_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

# =========================
# PAGE CONFIG (FIRST CALL)
# =========================
st.set_page_config(
    page_title="Layoff Risk Prediction",
    page_icon="📉",
    layout="wide"
)

# =========================
# LOAD MODEL FILES
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# =========================
# LABEL → ENCODING MAPS
# =========================
PRIMARY_SKILL_MAP = {
    "Data Science": 0,
    "Software Development": 1,
    "Cloud / DevOps": 2,
    "Testing / QA": 3,
    "Support / Operations": 4
}

INDUSTRY_MAP = {
    "IT Services": 0,
    "Product-Based Tech": 1,
    "Finance": 2,
    "Healthcare": 3,
    "Manufacturing": 4
}

ROLE_DEMAND_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

COMPANY_SIZE_MAP = {
    "Small": 0,
    "Medium": 1,
    "Large": 2
}

SALARY_BAND_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

# =========================
# SIDEBAR – FEATURE GUIDE
# =========================
with st.sidebar:
    st.markdown("## 📘 Feature Guide")

    st.markdown("""
    **Primary Skill**
    - Data Science → AI / ML roles  
    - Software Development → App / Backend  
    - Cloud / DevOps → Infrastructure  
    - Testing / QA → Quality  
    - Support → Operations  

    **Role Demand**
    - Low → Few openings  
    - Medium → Stable  
    - High → Actively hiring  

    **Company Size**
    - Small → < 100  
    - Medium → 100–1000  
    - Large → 1000+  

    **Industry Layoff Risk**
    - 0.0–0.3 → Stable  
    - 0.4–0.6 → Moderate risk  
    - 0.7–1.0 → High risk
    """)

# =========================
# MAIN HEADER
# =========================
st.title("📉 Layoff Risk Prediction System")
st.caption("Predict employee layoff risk using ML")

# =========================
# INPUT FORM
# =========================
with st.form("layoff_form"):
    c1, c2 = st.columns(2)

    with c1:
        experience = st.number_input("Experience (Years)", 0, 25, 5)
        primary_skill = st.selectbox("Primary Skill", list(PRIMARY_SKILL_MAP.keys()))
        certification = st.radio("Certification", ["No", "Yes"], horizontal=True)
        upskilling = st.radio("Upskilling Last Year", ["No", "Yes"], horizontal=True)
        industry = st.selectbox("Industry", list(INDUSTRY_MAP.keys()))

    with c2:
        skill_demand = st.slider("Skill Demand (1–10)", 1, 10, 5)
        industry_layoff_risk = st.slider("Industry Layoff Risk", 0.0, 1.0, 0.3)
        role_demand = st.selectbox("Role Demand", list(ROLE_DEMAND_MAP.keys()))
        company_size = st.selectbox("Company Size", list(COMPANY_SIZE_MAP.keys()))
        salary_band = st.selectbox("Salary Band", list(SALARY_BAND_MAP.keys()))

    submit = st.form_submit_button("🔮 Predict Risk")

# =========================
# PREDICTION
# =========================
if submit:
    input_df = pd.DataFrame([[
        experience,
        PRIMARY_SKILL_MAP[primary_skill],
        1 if certification == "Yes" else 0,
        1 if upskilling == "Yes" else 0,
        INDUSTRY_MAP[industry],
        skill_demand,
        industry_layoff_risk,
        ROLE_DEMAND_MAP[role_demand],
        COMPANY_SIZE_MAP[company_size],
        SALARY_BAND_MAP[salary_band]
    ]], columns=feature_names)

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input).max()

    st.divider()

    if prediction == 1:
        st.error(f"⚠️ **High Layoff Risk**  \nConfidence: **{probability:.2f}**")
    else:
        st.success(f"✅ **Low Layoff Risk**  \nConfidence: **{probability:.2f}**")

    # =========================
    # LIVE EXPLANATION
    # =========================
    if industry_layoff_risk <= 0.3:
        st.info("🟢 Stable industry with minimal layoffs")
    elif industry_layoff_risk <= 0.6:
        st.warning("🟡 Industry facing moderate uncertainty")
    else:
        st.error("🔴 High layoffs reported in this industry")
