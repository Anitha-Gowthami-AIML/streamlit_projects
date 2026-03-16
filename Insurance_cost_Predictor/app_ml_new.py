# -----------------------------
# BACKGROUND IMAGE FROM GITHUB
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

        .title-box {{
            background: linear-gradient(90deg,#0284c7,#06b6d4);
            padding:18px 25px;
            border-radius:14px;
            text-align:center;
            color:white;
            margin:20px auto 25px auto;
            box-shadow:0 6px 18px rgba(0,0,0,0.2);
            max-width:720px;
        }}

        .title-box h1 {{
            font-size:32px;
            font-weight:600;
            margin-bottom:5px;
            white-space:nowrap;
        }}

        .title-box p {{
            font-size:15px;
            opacity:0.9;
            margin:0;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use your raw GitHub URL for the image
set_bg_url("https://raw.githubusercontent.com/Anitha-Gowthami-AIML/streamlit_projects/main/Insurance_cost_Predictor/bg_image.png")
