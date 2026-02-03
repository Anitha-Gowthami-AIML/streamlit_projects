import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ======================================================
# BASE DIRECTORY (IMPORTANT FOR FOLDER DEPLOYMENT)
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "layoff_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Layoff Risk Prediction",
    page_icon="📉",
    layout="wide"
)

# ======================================================
# LOAD MODEL FILES
# ======================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ======================================================
# LABEL MAPPINGS
# ======================================================
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

ROLE_DEMAND_MAP = {"Low": 0, "Medium": 1, "High": 2}
COMPANY_SIZE_MAP = {"Small": 0, "Mid": 1, "Large": 2}
SALARY_BAND_MAP = {"Low": 0, "Medium": 1, "High": 2}

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>
.block-container { padding: 0rem; }
.header {
    background: linear-gradient(90deg, #7b0000, #b30000);
    padding: 25px;
    text-align: center;
    color: white;
    font-size: 34px;
    font-weight: 700;
    margin-top: 40px;
}
.sidebar-box {
    background: linear-gradient(180deg, #7b0000, #400000);
    height: 100vh;
    padding: 25px;
    color: white;
}
.card {
    background-color: #ffffff;
    padding: 35px;
    border-radius: 12px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    text-align: center;
    margin-left: auto;
    margin-right: auto;
    display: block;
}
.stButton > button {
    background: linear-gradient(90deg, #7b0000, #b30000);
    color: white;
    font-size: 50px;
    height: 62px;
    width: 220px;
    border-radius: 10px;
}
.result-box {
    /* Removed margin-top to allow for better horizontal alignment with button */
    padding: 18px;
    border-radius: 5px;
    font-size: 18px;
    font-weight: 600;
    height: 52px; /* Match button height */
    width: 220px; /* Match button width */
    display: flex; /* Use flexbox for centering content */
    align-items: center; /* Vertically center */
    justify-content: center; /* Horizontally center */
    text-align: center; /* Fallback and for multiline text within flex item */
}
.low-risk { background-color: #e6fff1; color: #006b3c; }
.high-risk { background-color: #ffe6e6; color: #8b0000; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="header">📉 Layoff Risk Prediction System</div>', unsafe_allow_html=True)

# ======================================================
# LAYOUT
# ======================================================
left_col, right_col = st.columns([1.2, 3.8])

# ================= SIDEBAR =================
with left_col:
    st.markdown("""
    <div class="sidebar-box">
        <h2>ℹ️ Feature Guide</h2>
        <b>Primary Skill</b>
        <ul>
            <li>Data Science</li>
            <li>Software Dev</li>
            <li>Cloud / DevOps</li>
            <li>Testing / QA</li>
            <li>Support</li>
        </ul>
        <b>Role Demand</b>
        <ul>
            <li>Low → Less hiring</li>
            <li>Medium → Stable</li>
            <li>High → Actively hiring</li>
        </ul>
        <b>Company Size</b>
        <ul>
            <li>Small → &lt;100</li>
            <li>Mid → 100–1000</li>
            <li>Large → &gt;1000</li>
        </ul>
        <hr>
        <p style="font-size:13px;">User-friendly labels mapped to ML encodings</p>
    </div>
    """, unsafe_allow_html=True)

# ================= MAIN FORM =================
with right_col:
    st.markdown('<div class="section-title">🧑‍💼 Employee Details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        experience = st.number_input("Experience (Years)", 0, 20, 5)
        primary_skill_label = st.selectbox("Primary Skill", list(PRIMARY_SKILL_MAP.keys()))
        certification = st.selectbox("Certification", ["No", "Yes"])
        upskilling_last_year = st.selectbox("Upskilling in Last Year", ["No", "Yes"])
        industry_label = st.selectbox("Industry", list(INDUSTRY_MAP.keys()))

    with c2:
        skill_demand = st.slider("Skill Demand (1–10)", 1, 10, 5)
        industry_layoff_risk = st.slider("Industry Layoff Risk", 0.0, 1.0, 0.3)

        # ======================================================
        # LIVE EXPLANATION (ADDED — NO DISTURBANCE)
        # ======================================================
        if industry_layoff_risk <= 0.3:
            st.info("🟢 Stable industry with minimal layoffs")
        elif industry_layoff_risk <= 0.6:
            st.warning("🟡 Industry facing moderate uncertainty")
        else:
            st.error("🔴 High layoffs reported in this industry")

        role_demand_label = st.selectbox("Role Demand", list(ROLE_DEMAND_MAP.keys()))
        company_size_label = st.selectbox("Company Size", list(COMPANY_SIZE_MAP.keys()))
        salary_band_label = st.selectbox("Salary Band", list(SALARY_BAND_MAP.keys()))

    # Use columns to position the button and result side-by-side and shifted right
    col_spacing_left, col_btn, col_res, col_spacing_right = st.columns([0.7, 0.5, 0.5, 0.8]) # Adjusted ratio for shifting right

    with col_btn:
        predict_button = st.button("🔮 Predict Layoff Risk")

    # Placeholder for the result, placed in the right column next to the button
    with col_res:
        result_placeholder = st.empty()

    if predict_button:
        input_df = pd.DataFrame([[
            experience,
            PRIMARY_SKILL_MAP[primary_skill_label],
            1 if certification == "Yes" else 0,
            1 if upskilling_last_year == "Yes" else 0,
            INDUSTRY_MAP[industry_label],
            skill_demand,
            industry_layoff_risk,
            ROLE_DEMAND_MAP[role_demand_label],
            COMPANY_SIZE_MAP[company_size_label],
            SALARY_BAND_MAP[salary_band_label]
        ]], columns=feature_names)

        scaled_input = scaler.transform(input_df)
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input).max()

        if pred == 1:
            with result_placeholder: # Write into the placeholder in the right column
                st.markdown(
                    f'<div class="result-box high-risk">⚠️ High Layoff Risk<br>Confidence: {prob:.2f}</div>',
                    unsafe_allow_html=True
                )
        else:
            with result_placeholder: # Write into the placeholder in the right column
                st.markdown(
                    f'<div class="result-box low-risk">✅ Low Layoff Risk<br>Confidence: {prob:.2f}</div>',
                    unsafe_allow_html=True
                )
