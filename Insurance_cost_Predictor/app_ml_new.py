import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import base64

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🩺",
    layout="centered"
)
import streamlit as st

# -----------------------------
# BACKGROUND IMAGE (from GitHub)
# -----------------------------
def set_bg_url(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
        }}

        /* Title banner styling */
        .title-box {{
            background: linear-gradient(90deg,#0284c7,#06b6d4);
            padding:18px 25px;
            border-radius:14px;
            text-align:center;
            color:white;
            margin:20px auto 25px auto;
            box-shadow:0 6px 18px rgba(0,0,0,0.2);
            max-width:720px;     /* keeps banner compact */
        }}

        .title-box h1{{
            font-size:32px;
            font-weight:600;
            margin-bottom:5px;
            white-space:nowrap;  /* prevents wrapping */
        }}

        .title-box p{{
            font-size:15px;
            opacity:0.9;
            margin:0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use your raw GitHub URL
set_bg_url("https://raw.githubusercontent.com/Anitha-Gowthami-AIML/streamlit_projects/main/Insurance_cost_Predictor/bg_image.png")

# -----------------------------
# TITLE SECTION
# -----------------------------
st.markdown(
"""
<div class="title-box">
<h1>🩺 Medical Insurance Cost Predictor</h1>
<p>AI Powered Healthcare Cost Estimation</p>
</div>
""",
unsafe_allow_html=True
)

# -------------------------------------------------------
# TITLE
# -------------------------------------------------------

st.markdown(
"""
<div class="title-box">
<h1>🩺 Medical Insurance Cost Predictor</h1>
<p>AI Powered Healthcare Cost Estimation</p>
</div>
""",
unsafe_allow_html=True
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

model = joblib.load("model.pkl")

# -------------------------------------------------------
# USER INPUTS
# -------------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 65, 30)

    bmi = st.slider("BMI", 15.0, 45.0, 25.0)

    children = st.slider("Children", 0, 5, 0)

with col2:

    gender = st.selectbox("Gender", ["male", "female"])

    smoker = st.selectbox("Smoker", ["yes", "no"])

    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

# -------------------------------------------------------
# ENCODING INPUTS
# -------------------------------------------------------

gender_val = 1 if gender == "male" else 0
smoker_val = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# -------------------------------------------------------
# MODEL INPUT
# -------------------------------------------------------

input_data = np.array([[

    age,
    gender_val,
    bmi,
    children,
    smoker_val,
    region_northwest,
    region_southeast,
    region_southwest

]])

# -------------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------------

if st.button("Predict Insurance Cost"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("Estimated Cost", f"${prediction:,.2f}")
    col2.metric("Model", "Random Forest")
    col3.metric("Model R² Score", "0.89")

    # Risk indicator

    if prediction < 5000:
        st.success("Low Insurance Cost Risk")

    elif prediction < 15000:
        st.warning("Medium Insurance Cost Risk")

    else:
        st.error("High Insurance Cost Risk")

    # ---------------------------------------------------
    # SHOW INPUT VARIABLES USED
    # ---------------------------------------------------

    st.subheader("Input Variables Used")

    input_df = pd.DataFrame({

        "Feature": [
            "Age",
            "Gender",
            "BMI",
            "Children",
            "Smoker",
            "Region"
        ],

        "Value": [
            age,
            gender,
            bmi,
            children,
            smoker,
            region
        ]

    })

    st.table(input_df)

    # ---------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------

    if hasattr(model, "feature_importances_"):

        st.subheader("Feature Importance")

        features = [
            "Age",
            "Gender",
            "BMI",
            "Children",
            "Smoker",
            "Northwest",
            "Southeast",
            "Southwest"
        ]

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # DOWNLOAD REPORT
    # ---------------------------------------------------

    report = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "BMI": [bmi],
        "Children": [children],
        "Smoker": [smoker],
        "Region": [region],
        "Predicted Cost": [prediction]
    })

    csv = report.to_csv(index=False)

    st.download_button(
        label="Download Prediction Report",
        data=csv,
        file_name="insurance_prediction.csv",
        mime="text/csv"
    )
