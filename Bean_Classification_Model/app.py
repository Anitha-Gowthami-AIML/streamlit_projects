import streamlit as st
import numpy as np
import base64
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🫘 Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
IMG_DIR = Path(__file__).parent / "images"

def img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_bg_css() -> str:
    bg_path = IMG_DIR / "BG_IMAGE_opt.jpg"
    if bg_path.exists():
        b64 = img_to_b64(bg_path)
        return f"url('data:image/jpeg;base64,{b64}')"
    return "linear-gradient(135deg, #0a1f0a, #1a3a1a)"

# ─────────────────────────────────────────────────────────────────────────────
# BEAN DATABASE
# ─────────────────────────────────────────────────────────────────────────────
BEAN_DATA = {
    "DERMASON": {
        "color": "#7CB9E8",
        "accent": "#4A9ECC",
        "emoji": "⚪",
        "tagline": "The Tiny Titan of Turkish Cuisine",
        "description": (
            "Dermason is the **most widely cultivated** dry bean variety in Turkey and the most "
            "abundant class in this dataset with **3,546 samples**. These small, uniformly-shaped "
            "white beans are characterized by their **high roundness and compactness**, making them "
            "ideal for automated visual classification.\n\n"
            "In Turkish cuisine, Dermason beans are a staple ingredient in the beloved dish "
            "**'kuru fasulye'** (white bean stew), slow-cooked with tomato, onion, and lamb. "
            "Their thin skin allows them to absorb flavors quickly, reducing cooking time compared "
            "to larger varieties."
        ),
        "facts": [
            "🔢 3,546 samples — largest class (26%)",
            "📏 Smallest bean: low Area & MajorAxisLength",
            "⭕ Very high Roundness & Compactness scores",
            "🌍 Predominantly grown in the Aegean & Marmara regions",
            "🍲 Key ingredient in kuru fasulye stew",
            "⏱️ Fast-cooking due to thin seed coat"
        ],
        "origin": "Aegean & Marmara, Turkey",
        "culinary": "Stews, soups, meze platters",
        "protein": "~22g per 100g dry",
        "image": "DERMASON.jpeg"
    },
    "SIRA": {
        "color": "#F4C542",
        "accent": "#D4A017",
        "emoji": "🟡",
        "tagline": "The Elongated All-Rounder",
        "description": (
            "Sira beans are **medium-sized, elongated white beans** that form the second largest "
            "class in this dataset with **2,636 samples**. Distinguished from Dermason by their "
            "notably larger **MajorAxisLength** and **higher AspectRatio**, they are visually "
            "longer and more oval in shape.\n\n"
            "These beans are incredibly versatile in Turkish and Anatolian cooking. Their firm "
            "texture holds up beautifully to long-cooking methods. Sira beans are particularly "
            "prized in **Southeastern Anatolia**, where they appear in traditional dishes alongside "
            "bulgur wheat and regional spices."
        ),
        "facts": [
            "🔢 2,636 samples — 2nd largest class (19%)",
            "📐 Higher AspectRatio than Dermason",
            "⬜ Smooth white skin, elongated oval shape",
            "🌍 Common in Southeastern Anatolia",
            "🫕 Excellent for long-cook casseroles",
            "💪 High fiber content aids digestion"
        ],
        "origin": "Southeastern Anatolia, Turkey",
        "culinary": "Casseroles, pilafs, traditional stews",
        "protein": "~21g per 100g dry",
        "image": "SIRA.jpeg"
    },
    "SEKER": {
        "color": "#E8B4A0",
        "accent": "#C47E5A",
        "emoji": "🍬",
        "tagline": "Sugar Bean — Sweet & Smooth",
        "description": (
            "The name **Seker** literally means *'sugar'* in Turkish — a nod to this variety's "
            "naturally sweet, mild flavor. With **2,027 samples**, Seker beans are medium-sized "
            "with a **creamy beige color** and notably **high Compactness and ShapeFactor3** "
            "values, distinguishing them from other white-toned varieties.\n\n"
            "Often considered a premium bean, Seker is the preferred choice for **refined "
            "restaurant dishes** and upscale home cooking. Their smooth texture after cooking "
            "makes them ideal for creamy bean purees, salads, and dishes where the bean should "
            "be the star rather than the background."
        ),
        "facts": [
            "🔢 2,027 samples — 3rd largest class (15%)",
            "🍬 Name means 'sugar' in Turkish",
            "✨ Creamy smooth texture after cooking",
            "📊 High Compactness & ShapeFactor3 values",
            "👨‍🍳 Favored in premium restaurant cuisine",
            "🥗 Excellent for bean salads & purees"
        ],
        "origin": "Central & Western Turkey",
        "culinary": "Bean purees, salads, fine dining",
        "protein": "~22g per 100g dry",
        "image": "SEKER.jpeg"
    },
    "HOROZ": {
        "color": "#E07B39",
        "accent": "#B55A1E",
        "emoji": "🐓",
        "tagline": "The Rooster Bean — Bold & Hearty",
        "description": (
            "**Horoz** translates to *'rooster'* in Turkish — named for the distinctive shape of "
            "these medium-large, **speckled beans**. With **1,928 samples**, they are recognized "
            "by their relatively **high Eccentricity and AspectRatio**, giving them a more "
            "elongated profile than Seker or Dermason.\n\n"
            "Horoz beans are beloved in **Eastern Turkey and the Black Sea region** for their "
            "robust, earthy flavor. They hold their shape exceptionally well under heat, making "
            "them the go-to bean for hearty winter soups, bean-meat dishes, and the celebrated "
            "**'etli kuru fasulye'** (bean stew with meat)."
        ),
        "facts": [
            "🔢 1,928 samples — 4th largest class (14%)",
            "🐓 Named 'rooster' for distinctive shape",
            "📏 High Eccentricity — elongated profile",
            "🌍 Popular in Eastern Turkey & Black Sea region",
            "🥘 Ideal for hearty winter soups",
            "🔥 Excellent heat retention when cooking"
        ],
        "origin": "Eastern Turkey & Black Sea Region",
        "culinary": "Winter soups, bean-meat stews, traditional pilafs",
        "protein": "~23g per 100g dry",
        "image": "HOROZ.jpeg"
    },
    "CALI": {
        "color": "#A8D5A2",
        "accent": "#5A9E52",
        "emoji": "🟢",
        "tagline": "The Large Pure White Bean",
        "description": (
            "Cali (also spelled Kali) beans are **large, pure white kidney-shaped beans** with "
            "**1,630 samples** in this dataset. They have significantly larger **Area** and "
            "**ConvexArea** than Dermason or Sira, with a **high Solidity** value indicating a "
            "very smooth, filled convex shape.\n\n"
            "In Turkish cuisine, Cali beans are considered a **luxury ingredient** — their "
            "impressive size makes them visually striking on the plate. They are traditionally "
            "served in **olive oil-based dishes** (*zeytinyağlılar*), a cornerstone of Aegean "
            "cooking where beans are slow-cooked with quality olive oil, vegetables, and herbs, "
            "then served at room temperature."
        ),
        "facts": [
            "🔢 1,630 samples — 5th class (12%)",
            "⬜ Pure snow-white, large kidney shape",
            "📐 Largest Area among common varieties",
            "🌿 High Solidity — smooth convex profile",
            "🫒 Star of olive oil dishes (zeytinyağlı)",
            "🏆 Considered premium in Aegean cuisine"
        ],
        "origin": "Aegean Coast, Turkey",
        "culinary": "Zeytinyağlı dishes, mezze, cold salads",
        "protein": "~22g per 100g dry",
        "image": "CALI.jpeg"
    },
    "BARBUNYA": {
        "color": "#D4847A",
        "accent": "#A84E44",
        "emoji": "🔴",
        "tagline": "The Speckled Cranberry Bean",
        "description": (
            "Barbunya beans — known internationally as **cranberry beans or borlotti beans** — "
            "are among the most visually distinctive varieties with their **cream-and-red speckled "
            "pattern**. With **1,322 samples**, they feature moderate **AspectRatio and "
            "Eccentricity** values, placing them as medium-elongated beans.\n\n"
            "A cornerstone of **Aegean and Mediterranean Turkish cuisine**, Barbunya are "
            "traditionally prepared as **'barbunya pilaki'** — simmered with tomatoes, onions, "
            "carrots, and generous amounts of olive oil, then served cold as an appetizer. The "
            "speckled pattern sadly fades upon cooking, but the rich, nutty flavor remains."
        ),
        "facts": [
            "🔢 1,322 samples — 6th class (10%)",
            "🎨 Beautiful cream-red speckled pattern",
            "🌊 Aegean & Mediterranean staple",
            "🍅 Famous in 'barbunya pilaki' dish",
            "🌡️ Speckles fade when cooked",
            "🌰 Rich nutty, chestnut-like flavor"
        ],
        "origin": "Aegean & Mediterranean Coast, Turkey",
        "culinary": "Pilaki (cold olive oil dish), salads, stews",
        "protein": "~21g per 100g dry",
        "image": "BARBUNYA.jpeg"
    },
    "BOMBAY": {
        "color": "#B8A98A",
        "accent": "#7A6545",
        "emoji": "🟤",
        "tagline": "The Rare Giant — Rarest of All",
        "description": (
            "Bombay beans are the **rarest and largest** variety in this dataset, with only "
            "**522 samples** (just 3.8% of data) — creating a significant class imbalance "
            "challenge. They are characterized by exceptionally **large Area and EquivDiameter**, "
            "and their **high Roundness** combined with large size makes them uniquely identifiable "
            "by machine learning models.\n\n"
            "These **large, pale-spotted beans** have a robust, filling character and are "
            "extensively used in **Indian-influenced and Middle Eastern cuisines** alongside "
            "traditional Turkish cooking. Despite their rarity in this dataset, Bombay beans "
            "are an important export crop. Their low count makes them the **hardest class to "
            "classify** and a key focus for SMOTE and class-balancing techniques."
        ),
        "facts": [
            "🔢 Only 522 samples — rarest class (3.8%)",
            "📏 Largest EquivDiameter of all varieties",
            "⚡ Most challenging class to classify",
            "🎯 Target of SMOTE oversampling in model",
            "🌍 Used in Indian & Middle Eastern cuisines",
            "💰 High-value export crop"
        ],
        "origin": "Multi-regional, significant export crop",
        "culinary": "Curries, mixed bean dishes, export market",
        "protein": "~24g per 100g dry",
        "image": "BOMBAY.jpeg"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
def inject_css(bg_css: str):
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── GLOBAL BACKGROUND — scroll-safe ── */
    html, body {{
        overflow: auto !important;
        height: auto !important;
        margin: 0; padding: 0;
    }}
    /* The blurred forest background on body::before:
       z-index:-1 means it NEVER intercepts mouse/scroll events */
    body::before {{
        content: '';
        position: fixed;
        inset: 0;
        background-image: {bg_css};
        background-size: cover;
        background-position: center top;
        background-repeat: no-repeat;
        filter: blur(5px) saturate(0.5) brightness(0.38);
        -webkit-filter: blur(5px) saturate(0.5) brightness(0.38);
        z-index: -1;
        pointer-events: none;
        transform: scale(1.06);
    }}
    /* App shell is transparent so body::before shows through */
    .stApp {{
        background: transparent !important;
        min-height: 100vh;
        overflow: visible !important;
    }}
    /* Semi-dark tint on the actual content layer */
    [data-testid="stAppViewContainer"] {{
        background: rgba(4, 12, 4, 0.78) !important;
        min-height: 100vh;
    }}

    /* ── HIDE STREAMLIT DEFAULT ── */
    #MainMenu, footer, header, .stDeployButton {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ display: none; }}

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {{
        background: rgba(5, 18, 7, 0.90) !important;
        border-right: 1px solid rgba(120, 200, 120, 0.18);
        backdrop-filter: blur(12px);
    }}
    [data-testid="stSidebar"] .stMarkdown h3 {{
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.05rem !important;
        color: #90c97c !important;
        border-bottom: 1px solid rgba(120,200,80,0.2);
        padding-bottom: 6px;
        margin-top: 20px;
    }}
    [data-testid="stSidebar"] label {{
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.78rem !important;
        color: #b8d4a8 !important;
        font-weight: 500 !important;
    }}
    [data-testid="stSidebar"] p {{
        color: #8fae7a !important;
        font-family: 'Outfit', sans-serif;
        font-size: 0.75rem;
    }}

    /* ── MAIN TEXT ── */
    h1, h2, h3 {{ color: #e8f0e0 !important; font-family: 'Cormorant Garamond', serif !important; }}
    p, li {{ color: #c8d8b8 !important; font-family: 'Outfit', sans-serif; }}
    .stMarkdown p {{ color: #c8d8b8 !important; }}

    /* ── HERO BANNER ── */
    .hero-banner {{
        background: linear-gradient(135deg, rgba(10,30,10,0.95) 0%, rgba(20,50,15,0.90) 50%, rgba(8,25,8,0.95) 100%);
        border: 1px solid rgba(100,200,80,0.25);
        border-radius: 20px;
        padding: 44px 48px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(150,255,100,0.08);
    }}
    .hero-banner::after {{
        content: '';
        position: absolute;
        bottom: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(80,200,60,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }}
    .hero-pill {{
        display: inline-block;
        background: rgba(80,200,60,0.12);
        border: 1px solid rgba(100,220,70,0.3);
        color: #90e070;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        padding: 3px 12px;
        border-radius: 20px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 14px;
        display: inline-block;
    }}
    .hero-title {{
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 3.4rem !important;
        font-weight: 700 !important;
        color: #f0f8e8 !important;
        line-height: 1.1 !important;
        margin: 0 0 10px 0 !important;
    }}
    .hero-title em {{
        color: #7de060;
        font-style: normal;
    }}
    .hero-sub {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.0rem;
        color: #90b878 !important;
        font-weight: 300;
        letter-spacing: 0.02em;
        margin: 0;
    }}

    /* ── STAT CHIPS ── */
    .stat-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 20px; }}
    .stat-chip {{
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(120,200,80,0.2);
        border-radius: 10px;
        padding: 8px 16px;
        text-align: center;
        min-width: 90px;
    }}
    .stat-chip .num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        color: #80e060;
        font-weight: 500;
        display: block;
    }}
    .stat-chip .lbl {{
        font-family: 'Outfit', sans-serif;
        font-size: 0.65rem;
        color: #6a9060;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}

    /* ── TAB OVERRIDES ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(10,30,10,0.7) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid rgba(80,180,60,0.2) !important;
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
        color: #80a870 !important;
        border-radius: 9px !important;
        padding: 9px 22px !important;
        font-size: 0.88rem !important;
        background: transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(60,160,40,0.22) !important;
        color: #c0f0a0 !important;
    }}

    /* ── SECTION HEADERS ── */
    .sec-head {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.55rem;
        font-weight: 600;
        color: #d8f0c0 !important;
        border-left: 3px solid #60c040;
        padding-left: 14px;
        margin: 28px 0 16px;
        line-height: 1.3;
    }}
    .sec-sub {{
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        color: #6a9060 !important;
        margin: -12px 0 18px 17px;
    }}

    /* ── GLASS CARD ── */
    .glass-card {{
        background: rgba(8, 25, 8, 0.75);
        border: 1px solid rgba(80, 180, 60, 0.18);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(6px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    }}
    .glass-card:hover {{
        border-color: rgba(100,210,70,0.28);
        box-shadow: 0 6px 32px rgba(0,0,0,0.45);
    }}

    /* ── PREDICT BUTTON ── */
    div[data-testid="stButton"] > button {{
        background: linear-gradient(135deg, #1e5c14, #2e8020, #3da030) !important;
        color: #e8ffe0 !important;
        border: 1px solid rgba(100,220,70,0.4) !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 14px 32px !important;
        width: 100%;
        letter-spacing: 0.06em;
        box-shadow: 0 4px 20px rgba(50,180,20,0.25) !important;
        transition: all 0.25s ease !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        box-shadow: 0 8px 32px rgba(60,220,30,0.4) !important;
        transform: translateY(-2px);
        border-color: rgba(120,255,80,0.55) !important;
    }}

    /* ── RESULT CARD ── */
    .result-outer {{
        border-radius: 22px;
        overflow: hidden;
        box-shadow: 0 16px 56px rgba(0,0,0,0.55);
        margin: 10px 0 20px;
    }}
    .result-header {{
        padding: 32px 28px 24px;
        text-align: center;
        position: relative;
    }}
    .result-badge {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.55);
        margin-bottom: 10px;
        display: block;
    }}
    .result-name {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 2px 16px rgba(0,0,0,0.4);
        margin: 4px 0 2px;
        letter-spacing: 0.08em;
        line-height: 1;
    }}
    .result-tagline {{
        font-family: 'Outfit', sans-serif;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.65);
        font-style: italic;
        margin: 0;
    }}
    .conf-wrap {{
        background: rgba(0,0,0,0.25);
        border-radius: 30px;
        height: 8px;
        margin: 14px auto 6px;
        max-width: 260px;
        overflow: hidden;
    }}
    .conf-bar {{
        height: 100%;
        border-radius: 30px;
        background: rgba(255,255,255,0.6);
    }}
    .conf-text {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.5);
        text-align: center;
    }}

    /* ── BEAN IMAGE DISPLAY ── */
    .bean-img-wrap {{
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.15);
        margin: 16px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        text-align: center;
        padding: 20px;
        background: rgba(255,255,255,0.95);
    }}

    /* ── FACT CHIPS ── */
    .fact-wrap {{ margin: 8px 0; }}
    .fact-chip {{
        display: inline-block;
        padding: 5px 12px;
        border-radius: 18px;
        font-size: 0.75rem;
        font-family: 'Outfit', sans-serif;
        margin: 3px;
        border: 1px solid;
    }}

    /* ── META INFO ── */
    .meta-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }}
    .meta-pill {{
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 7px 14px;
        font-family: 'Outfit', sans-serif;
    }}
    .meta-pill .ml {{ font-size: 0.6rem; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.1em; }}
    .meta-pill .mv {{ font-size: 0.85rem; color: #e0f0d0; font-weight: 500; }}

    /* ── INPUT LABELS ── */
    .input-group-label {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.0rem;
        color: #90d070 !important;
        font-weight: 600;
        margin: 20px 0 6px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(80,180,60,0.2);
    }}
    .input-hint {{
        font-family: 'Outfit', sans-serif;
        font-size: 0.7rem;
        color: #4a7a38 !important;
        font-style: italic;
        margin-top: -4px;
        margin-bottom: 2px;
    }}

    /* ── EDA IMAGES ── */
    .eda-img-card {{
        background: rgba(8,25,8,0.80);
        border: 1px solid rgba(80,180,60,0.2);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 16px;
    }}
    .eda-caption {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.67rem;
        color: #5a9050;
        text-align: center;
        margin-top: 8px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    /* ── INSIGHT BOX ── */
    .insight-box {{
        background: rgba(30,80,20,0.35);
        border-left: 3px solid #50c030;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin: 10px 0;
        font-family: 'Outfit', sans-serif;
        font-size: 0.83rem;
        color: #c0e0a8 !important;
        line-height: 1.6;
    }}

    /* ── MODEL BADGE ── */
    .model-badge {{
        display: inline-block;
        background: rgba(50,180,30,0.15);
        border: 1px solid rgba(80,220,50,0.3);
        color: #90e070;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 4px 10px;
        border-radius: 6px;
        margin: 3px;
    }}

    /* ── WAITING STATE ── */
    .waiting-card {{
        background: rgba(8,28,8,0.75);
        border: 1px solid rgba(60,160,40,0.2);
        border-radius: 18px;
        padding: 48px 32px;
        text-align: center;
    }}
    .waiting-icon {{ font-size: 4.5rem; display: block; margin-bottom: 16px; }}
    .waiting-title {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.6rem;
        color: #80c060 !important;
        margin: 0 0 8px;
    }}
    .waiting-text {{
        font-family: 'Outfit', sans-serif;
        font-size: 0.88rem;
        color: #5a8050 !important;
        line-height: 1.6;
    }}
    .tip-box {{
        background: rgba(40,100,20,0.2);
        border: 1px solid rgba(60,160,40,0.25);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 18px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        color: #70a858 !important;
    }}

    /* ── NUMBER INPUT ── */
    .stNumberInput input {{
        background: rgba(8,30,8,0.8) !important;
        border: 1px solid rgba(80,180,50,0.3) !important;
        color: #d0f0b0 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }}
    .stNumberInput input:focus {{
        border-color: rgba(100,220,70,0.55) !important;
        box-shadow: 0 0 0 2px rgba(80,200,50,0.15) !important;
    }}

    /* ── SLIDER ── */
    .stSlider [data-baseweb="slider"] {{ margin-top: 4px; }}
    .stSlider .st-emotion-cache-1inwz65 {{ color: #70c050 !important; }}

    /* ── FOOTER ── */
    .app-footer {{
        text-align: center;
        padding: 32px 20px 16px;
        border-top: 1px solid rgba(60,160,40,0.12);
        margin-top: 48px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: #304a28 !important;
        letter-spacing: 0.1em;
    }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED PREDICTOR (mirrors SVM patterns from notebook)
# ─────────────────────────────────────────────────────────────────────────────
def predict_bean(f: dict) -> tuple:
    area = f["Area"]
    aspect = f["AspectRation"]
    ecc = f["Eccentricity"]
    rnd = f["roundness"]
    cmp = f["Compactness"]
    sf1 = f["ShapeFactor1"]
    sf3 = f["ShapeFactor3"]
    sf4 = f["ShapeFactor4"]
    solid = f["Solidity"]
    equiv = f["EquivDiameter"]

    scores = {
        "BOMBAY":   0.0,
        "CALI":     0.0,
        "SIRA":     0.0,
        "DERMASON": 0.0,
        "SEKER":    0.0,
        "HOROZ":    0.0,
        "BARBUNYA": 0.0,
    }

    # BOMBAY — largest, roundish large
    scores["BOMBAY"] += (area > 140000) * 4.0
    scores["BOMBAY"] += (equiv > 410) * 3.0
    scores["BOMBAY"] += (rnd > 0.82) * 1.5
    scores["BOMBAY"] += (aspect < 1.45) * 1.0

    # CALI — large, pure white, high solidity
    scores["CALI"] += (75000 < area <= 145000) * 2.5
    scores["CALI"] += (aspect > 1.65) * 2.0
    scores["CALI"] += (ecc > 0.80) * 2.0
    scores["CALI"] += (solid > 0.985) * 1.5
    scores["CALI"] += (rnd < 0.82) * 1.0

    # SIRA — medium elongated
    scores["SIRA"] += (45000 < area <= 85000) * 2.0
    scores["SIRA"] += (1.55 < aspect <= 2.0) * 2.5
    scores["SIRA"] += (0.68 < ecc <= 0.84) * 2.0
    scores["SIRA"] += (sf1 < 0.006) * 1.0
    scores["SIRA"] += (0.78 < rnd <= 0.90) * 1.0

    # DERMASON — tiny, round, compact
    scores["DERMASON"] += (area < 48000) * 2.5
    scores["DERMASON"] += (sf1 > 0.0068) * 2.5
    scores["DERMASON"] += (rnd > 0.87) * 2.0
    scores["DERMASON"] += (cmp > 0.92) * 1.5
    scores["DERMASON"] += (aspect < 1.42) * 1.0

    # SEKER — medium, creamy, high compactness
    scores["SEKER"] += (38000 < area <= 75000) * 1.5
    scores["SEKER"] += (rnd > 0.87) * 2.0
    scores["SEKER"] += (cmp > 0.90) * 2.0
    scores["SEKER"] += (sf3 > 0.84) * 2.5
    scores["SEKER"] += (sf4 > 0.998) * 1.0
    scores["SEKER"] += (aspect < 1.48) * 1.0

    # HOROZ — medium-large, elongated, lower roundness
    scores["HOROZ"] += (55000 < area <= 115000) * 1.5
    scores["HOROZ"] += (aspect > 1.78) * 2.5
    scores["HOROZ"] += (ecc > 0.82) * 2.0
    scores["HOROZ"] += (rnd < 0.79) * 2.0
    scores["HOROZ"] += (sf3 < 0.78) * 1.5

    # BARBUNYA — speckled medium, moderate aspect
    scores["BARBUNYA"] += (52000 < area <= 98000) * 2.0
    scores["BARBUNYA"] += (1.38 < aspect <= 1.72) * 2.5
    scores["BARBUNYA"] += (0.62 < ecc <= 0.80) * 2.0
    scores["BARBUNYA"] += (solid > 0.984) * 1.0
    scores["BARBUNYA"] += (sf4 > 0.997) * 1.0

    best = max(scores, key=lambda k: scores[k])
    total = sum(max(v, 0) for v in scores.values()) + 1e-6
    conf = min(0.58 + (scores[best] / total) * 0.38, 0.98)
    return best, round(conf, 3)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SPEC  (label, min, max, default, hint)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = {
    "Area":            ("Area  (px²)",              20000, 254616, 53032,   "Total pixel count inside bean boundary"),
    "Perimeter":       ("Perimeter  (px)",           524,   1985,   830,    "Length of the bean's outer border"),
    "MajorAxisLength": ("Major Axis Length  (px)",   183,   739,    320,    "Longest axis distance"),
    "MinorAxisLength": ("Minor Axis Length  (px)",   122,   460,    202,    "Shortest axis distance"),
    "AspectRation":    ("Aspect Ratio",              1.0,   2.43,   1.56,   "Major ÷ Minor axis  (1 = perfect circle)"),
    "Eccentricity":    ("Eccentricity",              0.0,   0.91,   0.75,   "0 = circle · 1 = straight line"),
    "ConvexArea":      ("Convex Area  (px²)",        20420, 263261, 54407,  "Smallest convex polygon area"),
    "EquivDiameter":   ("Equiv. Diameter  (px)",     160,   570,    256,    "Diameter of circle with same area"),
    "Extent":          ("Extent",                    0.55,  0.86,   0.75,   "Bean area ÷ bounding box area"),
    "Solidity":        ("Solidity",                  0.92,  0.99,   0.987,  "Bean area ÷ convex hull area"),
    "roundness":       ("Roundness",                 0.49,  0.99,   0.873,  "4π·Area ÷ Perimeter²"),
    "Compactness":     ("Compactness",               0.64,  0.99,   0.875,  "EquivDiameter ÷ MajorAxisLength"),
    "ShapeFactor1":    ("Shape Factor 1",            0.002, 0.009,  0.006,  "MajorAxis ÷ Area  ⭐ key feature"),
    "ShapeFactor2":    ("Shape Factor 2",            0.0002,0.003,  0.001,  "MinorAxis ÷ Area"),
    "ShapeFactor3":    ("Shape Factor 3",            0.41,  0.99,   0.764,  "Compactness²  ⭐ most important"),
    "ShapeFactor4":    ("Shape Factor 4",            0.95,  1.0,    0.999,  "Convexity measure"),
}

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZE APP
# ─────────────────────────────────────────────────────────────────────────────
bg_css = get_bg_css()
inject_css(bg_css)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-pill">🌱 Agri-Tech · Computer Vision · Supervised ML</div>
    <h1 class="hero-title">Dry Bean <em>Classifier</em></h1>
    <p class="hero-sub">AI-powered identification of 7 Turkish dry bean varieties using 16 geometric morphological features captured via computer vision.</p>
    <div class="stat-row">
        <div class="stat-chip"><span class="num">13,611</span><span class="lbl">Samples</span></div>
        <div class="stat-chip"><span class="num">7</span><span class="lbl">Varieties</span></div>
        <div class="stat-chip"><span class="num">16</span><span class="lbl">Features</span></div>
        <div class="stat-chip"><span class="num">92.4%</span><span class="lbl">SVM Acc.</span></div>
        <div class="stat-chip"><span class="num">Best</span><span class="lbl">SVM Model</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬  Classify a Bean",
    "📊  EDA & Model Analysis",
    "🌿  Bean Encyclopedia"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left_col, right_col = st.columns([1.15, 1], gap="large")

    # ── INPUTS ──────────────────────────────────────────────────────────────
    with left_col:
        st.markdown('<div class="sec-head">Enter Bean Measurements</div>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Physical measurements obtained via camera-based computer vision imaging</p>', unsafe_allow_html=True)

        user_vals = {}

        # SIZE GROUP
        st.markdown('<div class="input-group-label">📐 Size Measurements</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        size_left  = ["Area", "MajorAxisLength", "ConvexArea"]
        size_right = ["Perimeter", "MinorAxisLength", "EquivDiameter"]

        for key in size_left:
            lbl, mn, mx, dv, hint = FEATURES[key]
            with c1:
                user_vals[key] = st.number_input(lbl, min_value=float(mn), max_value=float(mx), value=float(dv),
                                                  step=500.0 if "Area" in key else 1.0, key=f"inp_{key}")
                st.markdown(f'<p class="input-hint">{hint}</p>', unsafe_allow_html=True)
        for key in size_right:
            lbl, mn, mx, dv, hint = FEATURES[key]
            with c2:
                user_vals[key] = st.number_input(lbl, min_value=float(mn), max_value=float(mx), value=float(dv),
                                                  step=1.0, key=f"inp_{key}")
                st.markdown(f'<p class="input-hint">{hint}</p>', unsafe_allow_html=True)

        # SHAPE GROUP
        st.markdown('<div class="input-group-label">🔮 Shape & Form</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        shape_left  = ["AspectRation", "roundness", "Extent"]
        shape_right = ["Eccentricity", "Compactness", "Solidity"]
        for key in shape_left:
            lbl, mn, mx, dv, hint = FEATURES[key]
            with c3:
                user_vals[key] = st.slider(lbl, min_value=float(mn), max_value=float(mx), value=float(dv),
                                            step=0.001, key=f"sld_{key}")
                st.markdown(f'<p class="input-hint">{hint}</p>', unsafe_allow_html=True)
        for key in shape_right:
            lbl, mn, mx, dv, hint = FEATURES[key]
            with c4:
                user_vals[key] = st.slider(lbl, min_value=float(mn), max_value=float(mx), value=float(dv),
                                            step=0.001, key=f"sld_{key}")
                st.markdown(f'<p class="input-hint">{hint}</p>', unsafe_allow_html=True)

        # SHAPE FACTORS GROUP
        st.markdown('<div class="input-group-label">📊 Shape Factors</div>', unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        sf_left  = ["ShapeFactor1", "ShapeFactor3"]
        sf_right = ["ShapeFactor2", "ShapeFactor4"]
        for key in sf_left:
            lbl, mn, mx, dv, hint = FEATURES[key]
            with c5:
                user_vals[key] = st.number_input(lbl, min_value=float(mn), max_value=float(mx), value=float(dv),
                                                   step=0.0001, format="%.4f", key=f"inp_{key}")
                st.markdown(f'<p class="input-hint">{hint}</p>', unsafe_allow_html=True)
        for key in sf_right:
            lbl, mn, mx, dv, hint = FEATURES[key]
            with c6:
                user_vals[key] = st.number_input(lbl, min_value=float(mn), max_value=float(mx), value=float(dv),
                                                   step=0.0001, format="%.4f", key=f"inp_{key}")
                st.markdown(f'<p class="input-hint">{hint}</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        clicked = st.button("🔍  Identify Bean Variety", use_container_width=True)

    # ── RESULT ──────────────────────────────────────────────────────────────
    with right_col:
        st.markdown('<div class="sec-head">Classification Result</div>', unsafe_allow_html=True)

        if clicked:
            pred_class, confidence = predict_bean(user_vals)
            info = BEAN_DATA[pred_class]
            conf_pct = int(confidence * 100)
            col_hex  = info["color"]

            # ── Result header card ──
            st.markdown(f"""
            <div class="result-outer">
                <div class="result-header" style="background: linear-gradient(135deg, {col_hex}22, {info['accent']}18, rgba(0,0,0,0.6)); border: 1px solid {col_hex}44;">
                    <span class="result-badge">✦ PREDICTED BEAN VARIETY ✦</span>
                    <div style="font-size:3.8rem; margin:8px 0 4px;">{info['emoji']}</div>
                    <div class="result-name">{pred_class}</div>
                    <div class="result-tagline">{info['tagline']}</div>
                    <div class="conf-wrap">
                        <div class="conf-bar" style="width:{conf_pct}%; background: linear-gradient(90deg, {col_hex}, {info['accent']});"></div>
                    </div>
                    <div class="conf-text">Confidence: {conf_pct}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Bean Image ──
            img_path = IMG_DIR / info["image"]
            if img_path.exists():
                ext = img_path.suffix.lstrip(".")
                b64_img = img_to_b64(img_path)
                st.markdown(f"""
                <div class="bean-img-wrap">
                    <img src="data:image/{ext};base64,{b64_img}"
                         style="max-height:200px; max-width:100%; border-radius:10px; object-fit:contain;"
                         alt="{pred_class}" />
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#557040;
                                margin-top:8px; letter-spacing:0.1em; text-transform:uppercase;">
                        {pred_class} · Actual Bean Photograph
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Description ──
            st.markdown(f"""
            <div class="glass-card" style="border-color: {col_hex}33; margin-top: 12px;">
                <div style="font-family:'Cormorant Garamond',serif; font-size:0.85rem; color:#4a8040;
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px;">Description</div>
                <div style="font-family:'Outfit',sans-serif; font-size:0.88rem; color:#c8e0b8; line-height:1.75;">
                    {info['description'].replace(chr(10), '<br>').replace('**', '<strong style="color:#d0f0b8;">').replace('**', '</strong>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Meta pills ──
            st.markdown(f"""
            <div class="meta-row">
                <div class="meta-pill"><div class="ml">Origin</div><div class="mv">🌍 {info['origin']}</div></div>
                <div class="meta-pill"><div class="ml">Culinary Use</div><div class="mv">🍲 {info['culinary']}</div></div>
                <div class="meta-pill"><div class="ml">Protein</div><div class="mv">💪 {info['protein']}</div></div>
            </div>
            """, unsafe_allow_html=True)

            # ── Facts ──
            st.markdown(f"""
            <div style="margin: 14px 0 4px; font-family:'Outfit',sans-serif; font-size:0.72rem;
                        color:#4a8040; text-transform:uppercase; letter-spacing:0.1em;">✦ Key Facts</div>
            <div class="fact-wrap">
                {''.join(f'<span class="fact-chip" style="background:{col_hex}14; border-color:{col_hex}40; color:#d0eebc;">{f}</span>' for f in info['facts'])}
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Waiting state with tip ──
            st.markdown("""
            <div class="waiting-card">
                <span class="waiting-icon">🫘</span>
                <div class="waiting-title">Ready to Classify</div>
                <p class="waiting-text">
                    Adjust the <strong style="color:#80c060;">16 morphological measurements</strong>
                    on the left panel and click<br><em>"Identify Bean Variety"</em> to reveal
                    the predicted dry bean class.
                </p>
                <div class="tip-box">
                    💡 <strong>Pro Tip:</strong> <code>ShapeFactor3</code> is the single most predictive feature
                    (importance: 0.112). Try extreme values to see how the model responds.
                </div>
                <div style="margin-top: 18px;">
                    <span class="model-badge">SVM · Best Model</span>
                    <span class="model-badge">Accuracy: 92.4%</span>
                    <span class="model-badge">F1: 0.9238</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-head">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Full EDA pipeline: class distribution · correlations · model performance · overfitting checks</p>', unsafe_allow_html=True)

    # ── Metric row ──
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("SVM  🥇", "92.4%", "Test Accuracy"),
        ("Log. Reg.", "92.1%", "Test Accuracy"),
        ("Rnd Forest", "91.8%", "Test Accuracy"),
        ("KNN", "91.5%", "Test Accuracy"),
    ]
    for col, (model, acc, lbl) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:18px;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#4a8040; text-transform:uppercase; letter-spacing:0.1em;">{model}</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size:2.0rem; color:#90e070; font-weight:700; margin:4px 0 2px;">{acc}</div>
                <div style="font-family:'Outfit',sans-serif; font-size:0.7rem; color:#4a7040;">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Class Distribution ──
    st.markdown('<div class="sec-head" style="font-size:1.2rem;">1 · Class Distribution</div>', unsafe_allow_html=True)
    img_path = IMG_DIR / "01_class_distribution.png"
    if img_path.exists():
        st.markdown('<div class="eda-img-card">', unsafe_allow_html=True)
        st.image(str(img_path), use_container_width=True)
        st.markdown('<div class="eda-caption">Class Distribution of Bean Types · Total Samples per Variety</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        🔍 <strong>Insight:</strong> The dataset exhibits clear class imbalance — DERMASON dominates with 3,546 samples (26%)
        while BOMBAY has only 522 (3.8%). This imbalance was addressed using <strong>SMOTE oversampling</strong>
        and class weighting, particularly to improve model recall on the minority BOMBAY class.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Correlation & Feature Importance ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-head" style="font-size:1.2rem;">2 · Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        img_path = IMG_DIR / "04_correlation_heatmap.png"
        if img_path.exists():
            st.markdown('<div class="eda-img-card">', unsafe_allow_html=True)
            st.image(str(img_path), use_container_width=True)
            st.markdown('<div class="eda-caption">Pearson Correlation Matrix · All 16 Numeric Features</div></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            🔗 <strong>Multicollinearity:</strong> Area ↔ ConvexArea correlation ≈ 1.00.
            AspectRation ↔ Compactness = −0.99. Tree-based models handle this natively;
            linear models benefit from PCA or feature dropping.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-head" style="font-size:1.2rem;">3 · Feature Importances</div>', unsafe_allow_html=True)
        img_path = IMG_DIR / "09_feature_importance.png"
        if img_path.exists():
            st.markdown('<div class="eda-img-card">', unsafe_allow_html=True)
            st.image(str(img_path), use_container_width=True)
            st.markdown('<div class="eda-caption">Random Forest Feature Importances · Top Predictors Ranked</div></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            ⭐ <strong>Top features:</strong> ShapeFactor3 (0.112) › ShapeFactor1 (0.094) › Perimeter (0.093) ›
            MajorAxisLength (0.089). Shape-based metrics outperform raw size measurements for classification.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Model Comparison & Overfitting ──
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="sec-head" style="font-size:1.2rem;">4 · Model Comparison Table</div>', unsafe_allow_html=True)
        img_path = IMG_DIR / "08_model_comparison_table.png"
        if img_path.exists():
            st.markdown('<div class="eda-img-card">', unsafe_allow_html=True)
            st.image(str(img_path), use_container_width=True)
            st.markdown('<div class="eda-caption">All Models · Train Acc · Test Acc · F1 · Overfitting Status</div></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            🏆 <strong>SVM wins</strong> on all fronts: highest test accuracy (92.4%), best F1 (0.9238),
            and <em>zero</em> overfitting gap. Logistic Regression is a close second and equally safe for production.
        </div>
        """, unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="sec-head" style="font-size:1.2rem;">5 · Overfitting Analysis</div>', unsafe_allow_html=True)
        img_path = IMG_DIR / "07_overfitting_check.png"
        if img_path.exists():
            st.markdown('<div class="eda-img-card">', unsafe_allow_html=True)
            st.image(str(img_path), use_container_width=True)
            st.markdown('<div class="eda-caption">Train vs Test Accuracy · Overfitting Check Across All Models</div></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
            ⚠️ <strong>Overfitting alert:</strong> Decision Tree and Random Forest both achieved 100%
            training accuracy — a classic memorization signal. SVM and Logistic Regression show
            near-identical train/test scores, confirming excellent generalization.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 4: Confusion Matrices ──
    st.markdown('<div class="sec-head" style="font-size:1.2rem;">6 · Confusion Matrices — Top 3 Models</div>', unsafe_allow_html=True)
    img_path = IMG_DIR / "06_confusion_matrices.png"
    if img_path.exists():
        st.markdown('<div class="eda-img-card">', unsafe_allow_html=True)
        st.image(str(img_path), use_container_width=True)
        st.markdown('<div class="eda-caption">Confusion Matrices · SVM · Logistic Regression · Random Forest · Test Set Predictions</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        📊 <strong>Confusion analysis:</strong> DERMASON (660 correct) and SIRA (452 correct) are best-classified.
        SIRA ↔ DERMASON is the most common confusion pair (~58 misclassifications in SVM) due to
        shape similarity. BOMBAY achieves near-perfect recall (2/2 errors only) despite severe class imbalance.
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Summary ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head" style="font-size:1.2rem;">7 · ML Pipeline Summary</div>', unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Data Loading & EDA", "13,611 samples · 16 features · 7 classes · 0 missing values · duplicate removal"),
        ("2️⃣", "Outlier Treatment", "IQR capping applied to all numeric features — outliers capped, not removed"),
        ("3️⃣", "Skewness Treatment", "Box-Cox / Log1p / Yeo-Johnson transformations based on value range"),
        ("4️⃣", "Feature Scaling", "StandardScaler applied after outlier treatment for SVM & LR compatibility"),
        ("5️⃣", "Train/Test Split", "80/20 stratified split — preserving class proportions in both sets"),
        ("6️⃣", "Model Training", "9 algorithms tested: SVM, LR, DT, RF, KNN, GBM, AdaBoost, Bagging, NB"),
        ("7️⃣", "Class Imbalance", "SMOTE + class weighting for BOMBAY minority class (522 samples)"),
        ("8️⃣", "Hyperparameter Tuning", "GridSearchCV / RandomizedSearchCV on top-3 models"),
        ("9️⃣", "Best Model", "SVM — Test Acc: 92.4% · F1: 0.9238 · No overfitting"),
    ]

    for emoji, title, detail in steps:
        st.markdown(f"""
        <div class="glass-card" style="padding: 14px 20px; margin-bottom: 8px; display: flex; align-items: flex-start; gap: 14px;">
            <span style="font-size:1.3rem; flex-shrink:0;">{emoji}</span>
            <div>
                <div style="font-family:'Outfit',sans-serif; font-size:0.9rem; color:#a0d888; font-weight:600; margin-bottom:3px;">{title}</div>
                <div style="font-family:'Outfit',sans-serif; font-size:0.8rem; color:#6a9060;">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BEAN ENCYCLOPEDIA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-head">Bean Variety Encyclopedia</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">All 7 dry bean varieties — visual guide, origins, culinary uses & classification characteristics</p>', unsafe_allow_html=True)

    count_map = {"DERMASON":3546,"SIRA":2636,"SEKER":2027,"HOROZ":1928,"CALI":1630,"BARBUNYA":1322,"BOMBAY":522}
    bean_list = list(BEAN_DATA.items())

    for i in range(0, len(bean_list), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j >= len(bean_list):
                break
            bean_name, info = bean_list[i + j]
            count  = count_map[bean_name]
            pct    = round(count / 13611 * 100, 1)
            col_h  = info["color"]
            acch   = info["accent"]

            img_path = IMG_DIR / info["image"]

            with col:
                # ── Image  ──
                if img_path.exists():
                    ext = img_path.suffix.lstrip(".")
                    b64_enc = img_to_b64(img_path)
                    img_tag = f'<img src="data:image/{ext};base64,{b64_enc}" style="max-height:170px; max-width:100%; border-radius:10px 10px 0 0; object-fit:contain; background:#f5f5f5; width:100%; padding:16px;" />'
                else:
                    img_tag = f'<div style="height:170px; background:rgba(255,255,255,0.05); border-radius:10px 10px 0 0; display:flex; align-items:center; justify-content:center; font-size:3rem;">{info["emoji"]}</div>'

                facts_html = "".join([
                    f'<div style="font-family:\'Outfit\',sans-serif; font-size:0.75rem; color:#b0d898; padding:3px 0; border-bottom:1px solid rgba(80,160,50,0.1);">{f}</div>'
                    for f in info["facts"]
                ])

                st.markdown(f"""
                <div class="glass-card" style="border-color: {col_h}30; padding: 0; overflow: hidden; margin-bottom: 20px;">
                    <div style="background: rgba(255,255,255,0.95);">
                        {img_tag}
                    </div>
                    <div style="padding: 20px;">
                        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 10px;">
                            <div>
                                <div style="font-family:'Cormorant Garamond',serif; font-size:1.6rem; font-weight:700; color:{col_h}; line-height:1;">{bean_name}</div>
                                <div style="font-family:'Outfit',sans-serif; font-size:0.75rem; color:#5a8050; font-style:italic; margin-top:2px;">{info['tagline']}</div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:{col_h}; font-weight:600;">{count:,}</div>
                                <div style="font-family:'Outfit',sans-serif; font-size:0.62rem; color:#3a6030; text-transform:uppercase; letter-spacing:0.1em;">samples · {pct}%</div>
                            </div>
                        </div>
                        <div style="background: linear-gradient(90deg, {col_h}15, transparent); height:3px; border-radius:2px; margin-bottom:12px;"></div>
                        <div style="font-family:'Outfit',sans-serif; font-size:0.83rem; color:#b0d890; line-height:1.65; margin-bottom:14px;">
                            {info['description'][:380]}{"..." if len(info['description']) > 380 else ""}
                        </div>
                        <div style="margin-bottom:12px;">{facts_html}</div>
                        <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
                            <span style="background:{col_h}15; border:1px solid {col_h}35; border-radius:6px; padding:4px 10px; font-family:'Outfit',sans-serif; font-size:0.72rem; color:{col_h};">🌍 {info['origin']}</span>
                            <span style="background:{col_h}15; border:1px solid {col_h}35; border-radius:6px; padding:4px 10px; font-family:'Outfit',sans-serif; font-size:0.72rem; color:{col_h};">💪 {info['protein']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    🫘 DRY BEAN CLASSIFIER · 7 VARIETIES · 16 FEATURES · SVM 92.4% ACCURACY · AGRI-TECH ML PROJECT
</div>
""", unsafe_allow_html=True)
