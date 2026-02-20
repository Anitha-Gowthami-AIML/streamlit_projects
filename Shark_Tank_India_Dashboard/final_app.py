import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import base64

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Shark Tank India Dashboard",
    layout="wide"
)

# =====================================================
# THEME
# =====================================================
st.markdown("""
<style>
.main-header {
    background-color: #0A1F44;
    padding: 22px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    #margin-bottom: 20px;
    margin: 0 auto 20px auto;  /* centers the box horizontally */
    width: 80%;               /* change this to your desired width */
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #0A1F44;
    margin-top: 15px;
}
div[data-testid="metric-container"] {
    background-color: #F2F4F8;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
import os



#================================================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(__file__)
    csv_path = os.path.join(BASE_DIR, "Shark_Tank_India_input_file.csv")
    df = pd.read_csv(csv_path)

    #df = pd.read_csv("Shark_Tank_India_input_file.csv")
    df = df.drop_duplicates()
    df['deal'] = df['total_sharks_invested'] > 0
    df['equity_per_shark'] = (
        df['deal_equity'] / df['total_sharks_invested']
    ).where(df['deal'], 0)
    return df

df = load_data()
successful = df[df['deal'] == True]

# =====================================================
# SIDEBAR IMAGE
# =====================================================
import os

BASE_DIR = os.path.dirname(__file__)
sidebar_img_path = os.path.join(BASE_DIR, "all_sharks.jpg")
if os.path.exists(sidebar_img_path):
    st.sidebar.image(sidebar_img_path)

#if os.path.exists("all_sharks.jpg"):
#   st.sidebar.image("all_sharks.jpg")

page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📊 Executive Overview",
    "🦈 Shark Analysis",
    "📈 Advanced Insights",
    "📚  Category Insights"
])

# =====================================================
# TOP RIGHT LOGO
# =====================================================
def add_logo():
    logo_path = os.path.join(BASE_DIR, "shark_tank_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
        
    #if os.path.exists("shark_tank_logo.png"):
     #   with open("shark_tank_logo.png", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <style>
        .top-right-logo {{
            position: fixed;
            top: 60px;
            right: 20px;
            z-index: 9999;
        }}
        .top-right-logo img {{
            width: 110px;
            height: auto;
        }}
        </style>
        <div class="top-right-logo">
            <img src="data:image/png;base64,{encoded}">
        </div>
        """, unsafe_allow_html=True)

add_logo()

# =====================================================
# Home
# =====================================================
if page == "🏠 Home":

    st.markdown('<div class="main-header">🦈 Shark Tank India Analytics Dashboard</div>', unsafe_allow_html=True)

    st.markdown("""
    **Shark Tank India** is India’s leading startup investment reality show 
    where entrepreneurs pitch their business ideas to experienced investors called 
    *Sharks*. The show provides funding, national visibility, mentorship, and strategic 
    business direction. Beyond investment, Sharks help founders scale operations, 
    strengthen branding, improve valuation, and build sustainable growth models.

    This dashboard presents deep analytical insights into:
    - Investment behavior of Sharks  
    - Deal success patterns  
    - Multi-shark investments  
    - Valuation categories  
    - Sector distribution  
    - Advanced investment insights  
    """)

# =====================================================
# EXECUTIVE OVERVIEW
# =====================================================
elif page == "📊 Executive Overview":

    st.markdown('<div class="main-header">📊 Executive Overview</div>', unsafe_allow_html=True)

    total_episodes = df['episode_number'].nunique()
    total_sharks = len([c for c in df.columns if c.endswith("_deal")])
    total_pitches = df.shape[0]
    success_deals = df['deal'].sum()
    success_rate = round(df['deal'].mean() * 100, 2)
    total_investment = successful['deal_amount'].sum()
    avg_deal = successful['deal_amount'].mean()
    median_deal = successful['deal_amount'].median()
    avg_equity = successful['deal_equity'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Episodes", total_episodes)
    col2.metric("Total Sharks", total_sharks)
    col3.metric("Total Pitches", total_pitches)
    col4.metric("Successful Deals", int(success_deals))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Success Rate", f"{success_rate}%")
    col6.metric("Total Investment (₹ Cr)", f"{total_investment:.2f}")
    col7.metric("Average Deal (₹ Cr)", f"{avg_deal:.2f}")
    col8.metric("Median Deal (₹ Cr)", f"{median_deal:.2f}")

    st.metric("Average Equity % Given", f"{avg_equity:.2f}%")

# =====================================================
# SHARK ANALYSIS
# =====================================================
elif page == "🦈 Shark Analysis":

    st.markdown('<div class="main-header">🦈 Shark Analysis</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["About Sharks", "Sharks Comparison"])

    # ------------------ TAB 1: About Sharks ------------------
    with tab1:

        # Get shark columns dynamically
        deal_cols = [col for col in df.columns if col.endswith("_deal")]
        shark_list = [col.replace("_deal", "") for col in deal_cols]

        selected = st.selectbox("Select Shark", shark_list)
        col_name = selected + "_deal"

        total_deals = df[col_name].sum()
        total_inv = successful.loc[
            successful[col_name] == True, 'amount_per_shark'
        ].sum()
        avg_deal = total_inv / total_deals if total_deals > 0 else 0

        col1, col2 = st.columns([1, 2])
#==========================================
        shark_images = {
            "aman": os.path.join(BASE_DIR, "Aman Gupta.png"),
            "anupam": os.path.join(BASE_DIR, "Anupam Mittal.png"),
            "ashneer": os.path.join(BASE_DIR, "Ashneer Grover.png"),
            "namita": os.path.join(BASE_DIR, "Namita Thapar.png"),
            "peyush": os.path.join(BASE_DIR, "Peyush Bansal.png"),
            "vineeta": os.path.join(BASE_DIR, "Vineeta Singh.png"),
            "ghazal": os.path.join(BASE_DIR, "Ghazal Alagh.png")
        }

 
        #shark_images = {
         #   "aman": "Aman Gupta.png",
          #  "anupam": "Anupam Mittal.png",
           # "ashneer": "Ashneer Grover.png",
           # "namita": "Namita Thapar.png",
            #"peyush": "Peyush Bansal.png",
            #"vineeta": "Vineeta Singh.png",
            #"ghazal": "Ghazal Alagh.png"
        #}

        shark_summaries = {
            "aman": "Expert in consumer electronics, branding and digital marketing.",
            "anupam": "Specialist in digital platforms and consumer behavior strategy.",
            "namita": "Healthcare and pharma leader with strong compliance expertise.",
            "peyush": "D2C and e-commerce scaling expert.",
            "vineeta": "D2C branding and strategic growth specialist.",
            "ashneer": "Aggressive investor known for sharp valuation decisions.",
            "ghazal": "Focuses on lifestyle and emerging consumer brands."
        }

        with col1:
            img_file = shark_images.get(selected.lower())
            if img_file and os.path.exists(img_file):
                st.image(img_file, width=220)

        with col2:
            st.metric("Total Deals", int(total_deals))
            st.metric("Total Investment (₹ Cr)", f"{total_inv:.2f}")
            st.metric("Average Deal Size (₹ Cr)", f"{avg_deal:.2f}")
            st.write(shark_summaries.get(selected.lower(), ""))

    # ------------------ TAB 2: Sharks Comparison ------------------
    with tab2:

        # Summary DataFrame
        shark_df = pd.DataFrame({
            "Shark": shark_list,
            "Deals": [df[col].sum() for col in deal_cols],
            "Total Investment": [successful.loc[successful[col]==True,'amount_per_shark'].sum() for col in deal_cols]
        }).sort_values("Deals", ascending=False)

        # Top 3 sharks by number of deals
        st.subheader("Top 3 Sharks by Number of Investments")
        top3 = shark_df.head(3)
        fig, ax = plt.subplots()
        ax.bar(top3['Shark'], top3['Deals'], color='skyblue')
        ax.set_ylabel("Number of Deals")
        st.pyplot(fig)

        # Top 3 sharks by total investment
        st.subheader("Top 3 Sharks by Total Investment")
        top3_inv = shark_df.sort_values("Total Investment", ascending=False).head(3)
        fig2, ax2 = plt.subplots()
        ax2.bar(top3_inv['Shark'], top3_inv['Total Investment'], color='orange')
        ax2.set_ylabel("Total Investment (₹ Cr)")
        st.pyplot(fig2)
# =====================================================
# ADVANCED INSIGHTS
# =====================================================
elif page == "📈 Advanced Insights":

    st.markdown('<div class="main-header">📈 Advanced Business Insights</div>', unsafe_allow_html=True)

    more_than_1cr = df[df['pitcher_ask_amount'] > 100]
    funded_1cr = more_than_1cr[more_than_1cr['deal']]

    col1, col2 = st.columns(2)
    col1.metric("Pitches Asked > ₹1 Cr", more_than_1cr.shape[0])
    col2.metric("Of Those Funded", funded_1cr.shape[0])

    multi_shark = successful[successful['total_sharks_invested'] > 1]
    multi_percent = round((multi_shark.shape[0] / successful.shape[0]) * 100,2)
    st.metric("Multi-Shark Deals (%)", f"{multi_percent}%")

    exceeded = df[df['deal_amount'] > df['pitcher_ask_amount']]
    st.subheader("Deals Where Investment Exceeded Ask")
    if exceeded.shape[0] > 0:
        st.dataframe(exceeded[['brand_name','pitcher_ask_amount','deal_amount']])
    else:
        st.write("No such cases found.")

# =====================================================
#  Category Insights
# =====================================================
elif page == "📚  Category Insights":

    
    st.markdown('<div class="main-header">📚  Category Insights</div>', unsafe_allow_html=True)
    
    st.subheader("Mega Big Category Distribution")
    
    category_counts = df['Mega_Big_Category_idea'].value_counts()
    
    # Smaller figure size
    fig, ax = plt.subplots(figsize=(6,4))   # 👈 decrease size here
    
    ax.bar(category_counts.index, category_counts.values)
    
    # Proper x-axis formatting
    ax.set_xticks(range(len(category_counts.index)))
    ax.set_xticklabels(category_counts.index, rotation=30, ha='right', fontsize=8)
    
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_xlabel("Category", fontsize=9)
    
    plt.tight_layout()  # 👈 prevents label cut-off
    st.pyplot(fig)
    
    
    st.subheader("Valuation Category Distribution")
    
    valuation_counts = df['Valuation_category'].value_counts()
    
    fig2, ax2 = plt.subplots(figsize=(6,4))  # 👈 decrease size here
    
    ax2.bar(valuation_counts.index, valuation_counts.values)
    
    ax2.set_xticks(range(len(valuation_counts.index)))
    ax2.set_xticklabels(valuation_counts.index, rotation=30, ha='right', fontsize=8)
    
    ax2.tick_params(axis='y', labelsize=8)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_xlabel("Valuation Category", fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    
    

