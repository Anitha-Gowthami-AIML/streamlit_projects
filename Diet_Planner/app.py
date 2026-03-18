"""NourishAI — Smart Recipe & Meal Plan Recommender  v4"""

import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="NourishAI — Smart Meal Planner",
                   page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")

from recommender_engine import (
    generate_recipes, generate_users, generate_ratings,
    train_models, recommend_for_new_user, build_meal_plan,
    CUISINES, DIET_TYPES, HEALTH_GOALS, ALLERGENS, DAYS,
    PERSONA_NAMES, INDIAN_REGIONS, get_relevant_allergens,
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#0A0604 !important;}
.main .block-container{padding:2rem 2.5rem;max-width:1200px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0F0A05 0%,#1C0F06 60%,#130A04 100%) !important;border-right:1px solid rgba(210,100,30,0.20);}
[data-testid="stSidebar"] *{color:#F5E6D8 !important;}
[data-testid="stSidebar"] label{color:rgba(245,230,216,0.65) !important;font-size:0.78rem;text-transform:uppercase;letter-spacing:.08em;font-weight:500;}
.stButton>button{background:linear-gradient(135deg,#C45A10,#8B3A08) !important;color:#FFF0D8 !important;border:none !important;border-radius:10px !important;font-weight:500 !important;font-size:0.95rem !important;padding:10px 28px !important;}
.stButton>button:hover{opacity:0.88 !important;}
div[data-testid="stMetric"]{background:linear-gradient(145deg,#1A1008,#211408);border:1px solid rgba(210,100,30,0.22);border-radius:12px;padding:14px 16px;}
div[data-testid="stMetric"] label{color:rgba(245,230,216,0.55) !important;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#FF9A50 !important;font-family:'Playfair Display',serif;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.04);border-radius:10px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{color:rgba(245,230,216,0.55) !important;font-size:0.88rem !important;border-radius:8px !important;padding:8px 18px !important;}
.stTabs [aria-selected="true"]{background:rgba(210,100,30,0.28) !important;color:#FFB878 !important;}
.stRadio>div{flex-direction:row !important;gap:8px;flex-wrap:wrap;}
.stRadio label{color:rgba(245,230,216,0.70) !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_data_and_models():
    recipes = generate_recipes()
    users   = generate_users(n=300)
    ratings = generate_ratings(users, recipes, n=8000)
    models  = train_models(recipes, users, ratings)
    return recipes, users, ratings, models


def _sb_section(title):
    st.markdown(
        '<div style="font-family:Playfair Display,serif;font-size:1.05rem;font-weight:600;color:#FF9A50;margin:20px 0 10px;padding-bottom:6px;border-bottom:1px solid rgba(210,100,30,0.25);">'
        + title + '</div>', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:16px 0 8px;">'
            '<div style="font-size:2.2rem;">🍽️</div>'
            '<div style="font-family:Playfair Display,serif;font-size:1.3rem;font-weight:700;color:#FF9A50;margin-top:6px;">NourishAI</div>'
            '<div style="font-size:0.72rem;color:rgba(245,230,216,0.40);text-transform:uppercase;letter-spacing:.1em;">Smart Meal Planner</div>'
            '</div><hr style="border-color:rgba(210,100,30,0.20);margin:12px 0 4px;"/>',
            unsafe_allow_html=True)

        _sb_section("Your Profile")
        diet = st.selectbox("Dietary preference", DIET_TYPES, index=0)
        goal = st.selectbox("Health goal", HEALTH_GOALS, index=0)

        _sb_section("Tastes & Restrictions")
        fav_cuisines = st.multiselect("Favourite cuisines", options=CUISINES,
                                      default=["Indian","Italian"], max_selections=5)

        indian_region = "All Regions"
        if "Indian" in (fav_cuisines or []):
            indian_region = st.selectbox("Indian region", INDIAN_REGIONS, index=0,
                                         help="Filter Indian recipes by region")

        relevant_allergens = get_relevant_allergens(diet)
        allergies = st.multiselect("Allergies / intolerances", options=relevant_allergens,
                                   default=[], help="Only allergens relevant to your diet are shown")

        _sb_section("Targets")
        calorie_target = st.slider("Calories per meal (kcal)", 200, 900, 480, 10)
        max_prep       = st.slider("Max prep time (minutes)",  10,  90,  30,  5)
        age            = st.slider("Your age", 18, 70, 28, 1)
        days_plan      = st.selectbox("Meal plan — number of days", [3,5,7], index=2,
                                      help="3 meals per day: Breakfast, Lunch, Dinner")
        st.markdown("<br/>", unsafe_allow_html=True)
        go = st.button("✨  Get my recommendations", use_container_width=True)

    return {"diet_type":diet,"health_goal":goal,
            "fav_cuisines":fav_cuisines if fav_cuisines else ["Indian"],
            "indian_region":indian_region,"allergies":allergies,
            "calorie_target":calorie_target,"max_prep_min":max_prep,
            "age":age,"days_plan":days_plan}, go


# ── Card helpers — ALL style= on single lines, no newlines inside attributes ──
CUIS_EMOJI = {"Italian":"🇮🇹","Mexican":"🇲🇽","Japanese":"🇯🇵","Indian":"🇮🇳",
               "Mediterranean":"🫒","American":"🍔","Thai":"🇹🇭","Chinese":"🥢",
               "French":"🇫🇷","Middle Eastern":"🧆"}
MEAL_EMOJI = {"Breakfast":"☀️","Lunch":"🌤️","Dinner":"🌙","Snack":"🍎"}
REG_EMOJI  = {"North India":"🏔️","South India":"🌴","East India":"🌿","West India":"🌊"}

def _stars(v):
    f = int(round(float(v)))
    return '<span style="color:#FFD700;">'+"★"*f+'</span><span style="color:rgba(255,255,255,0.20);">'+"☆"*(5-f)+'</span>'

def _nutr_bar(label, val, pct, colour):
    v   = float(val)
    p   = float(pct)
    row = '<div style="display:flex;justify-content:space-between;font-size:0.68rem;color:rgba(245,230,216,0.45);margin-bottom:2px;"><span>'+label+'</span><span>'+f'{v:.0f}g'+'</span></div>'
    bar = '<div style="background:rgba(255,255,255,0.07);border-radius:3px;height:4px;margin-bottom:5px;"><div style="width:'+f'{p:.0f}'+'%;height:4px;border-radius:3px;background:'+colour+';"></div></div>'
    return row + bar

def recipe_card(r):
    emoji     = CUIS_EMOJI.get(r["cuisine"],"🍴")
    yt        = str(r.get("youtube_url","#"))
    score_pct = int(float(r.get("score",0))*100)
    p, c, f   = float(r["protein_g"]), float(r["carbs_g"]), float(r["fat_g"])
    total     = p+c+f+0.001
    nutr      = _nutr_bar("Protein",p,p/total*100,"#78FFEE") + _nutr_bar("Carbs",c,c/total*100,"#FFD700") + _nutr_bar("Fat",f,f/total*100,"#FF9A50")
    region    = str(r.get("region",""))
    reg_html  = ""
    if region:
        re      = REG_EMOJI.get(region,"📍")
        reg_html = '<span style="display:inline-block;background:rgba(255,160,50,0.14);border:1px solid rgba(255,160,50,0.30);color:#FFC080;font-size:0.68rem;padding:2px 8px;border-radius:50px;margin-left:6px;">'+re+" "+region+'</span>'
    meal_em   = MEAL_EMOJI.get(str(r["meal_type"]),"🍴")
    meal_html = '<span style="display:inline-block;background:rgba(120,180,255,0.12);border:1px solid rgba(120,180,255,0.25);color:#A0C8FF;font-size:0.68rem;padding:2px 9px;border-radius:50px;margin-left:4px;">'+meal_em+" "+str(r["meal_type"])+'</span>'
    stars_html= _stars(r["avg_rating"])
    cal       = int(r["calories"])
    prep      = int(r["prep_min"])
    rat       = float(r["avg_rating"])
    name      = str(r["name"])

    out  = '<div style="background:linear-gradient(145deg,#1A1008,#211408);border:1px solid rgba(210,100,30,0.24);border-radius:16px;padding:18px;margin-bottom:14px;position:relative;">'
    out += '<div style="position:absolute;top:14px;right:14px;background:rgba(210,100,30,0.26);border:1px solid rgba(210,100,30,0.48);color:#FFB070;font-size:0.70rem;font-weight:600;padding:3px 9px;border-radius:50px;">Match '+str(score_pct)+'%</div>'
    out += '<div style="font-family:Playfair Display,serif;font-size:1.04rem;font-weight:600;color:#FFF0E0;margin:0 0 6px;line-height:1.3;padding-right:80px;">'+name+'</div>'
    out += '<div style="margin-bottom:11px;">'
    out += '<span style="display:inline-block;background:rgba(210,100,30,0.16);border:1px solid rgba(210,100,30,0.32);color:#FFAA60;font-size:0.68rem;font-weight:500;letter-spacing:.06em;text-transform:uppercase;padding:2px 9px;border-radius:50px;">'+emoji+" "+str(r["cuisine"])+'</span>'
    out += reg_html + meal_html + '</div>'
    out += '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:11px;">'
    out += '<span style="font-size:0.80rem;color:rgba(245,230,216,0.50);">🔥 <span style="color:#FFD4A0;font-weight:500;">'+str(cal)+' kcal</span></span>'
    out += '<span style="font-size:0.80rem;color:rgba(245,230,216,0.50);">⏱️ <span style="color:#FFD4A0;font-weight:500;">'+str(prep)+' min</span></span>'
    out += '<span style="font-size:0.80rem;">'+stars_html+' <span style="color:#FFD4A0;font-weight:500;">'+f'{rat:.1f}'+'</span></span>'
    out += '</div>'
    out += '<div style="margin-bottom:13px;">'+nutr+'</div>'
    out += '<a href="'+yt+'" target="_blank" rel="noopener noreferrer" style="display:inline-flex;align-items:center;gap:6px;background:rgba(200,30,30,0.16);border:1px solid rgba(220,60,60,0.38);color:#FF9090;font-size:0.78rem;font-weight:500;padding:6px 13px;border-radius:8px;text-decoration:none;">▶ Watch on YouTube</a>'
    out += '</div>'
    return out


def render_meal_plan(plan_df):
    for day in [d for d in DAYS if d in plan_df["day"].values]:
        rows = plan_df[plan_df["day"]==day]
        if rows.empty: continue
        day_cal   = int(rows["calories"].sum())
        rows_html = ""
        for _, row in rows.iterrows():
            yt   = str(row.get("youtube_url","#"))
            reg  = str(row.get("region",""))
            reg_s= (' <span style="font-size:0.65rem;color:rgba(255,180,80,0.50);">· '+reg+'</span>') if reg else ""
            me   = MEAL_EMOJI.get(str(row["meal"]),"🍴")
            rows_html += (
                '<div style="display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                '<div style="min-width:78px;font-size:0.68rem;font-weight:500;letter-spacing:.06em;text-transform:uppercase;color:rgba(245,230,216,0.36);">'+me+" "+str(row["meal"])+'</div>'
                '<div style="flex:1;font-size:0.87rem;color:#F0DCC8;">'+str(row["name"])+reg_s
                +' <a href="'+yt+'" target="_blank" rel="noopener noreferrer" style="font-size:0.68rem;color:#FF8080;text-decoration:none;">▶ YT</a></div>'
                '<div style="font-size:0.76rem;color:rgba(255,180,80,0.62);white-space:nowrap;">'+str(int(row["calories"]))+' kcal</div>'
                '</div>'
            )
        day_html = (
            '<div style="background:linear-gradient(145deg,#18100A,#1E130A);border:1px solid rgba(210,100,30,0.16);border-radius:14px;padding:16px 18px;margin-bottom:12px;">'
            '<div style="font-family:Playfair Display,serif;font-size:1rem;font-weight:600;color:#FF9A50;margin-bottom:10px;padding-bottom:7px;border-bottom:1px solid rgba(210,100,30,0.16);">'
            +day+'<span style="font-size:0.74rem;font-weight:400;color:rgba(245,230,216,0.34);margin-left:10px;">Total ~'+str(day_cal)+' kcal</span>'
            +'</div>'+rows_html+'</div>'
        )
        st.markdown(day_html, unsafe_allow_html=True)


def render_landing():
    st.markdown(
        '<div style="text-align:center;padding:50px 40px 36px;">'
        '<div style="font-size:3.6rem;margin-bottom:12px;">🥗</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1.7rem;font-weight:600;color:#FFF0D8;margin-bottom:10px;">Tell us about your tastes</div>'
        '<div style="font-size:0.93rem;color:rgba(245,230,216,0.46);max-width:400px;margin:0 auto;line-height:1.7;">'
        'Set your preferences in the sidebar, then hit '
        '<strong style="color:#FF9A50;">Get my recommendations</strong>.'
        '</div></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,(icon,title,desc) in zip([c1,c2,c3],[
        ("🧬","Hybrid ML Engine","SVD · Content similarity · Persona clustering"),
        ("🎯","Cuisine Hard Filter","Only your selected cuisines appear in results"),
        ("🗺️","Regional Indian","North · South · East · West India recipes"),
    ]):
        col.markdown(
            '<div style="background:linear-gradient(145deg,#1A1008,#211408);border:1px solid rgba(210,100,30,0.18);border-radius:14px;padding:22px 18px;text-align:center;">'
            '<div style="font-size:1.8rem;margin-bottom:10px;">'+icon+'</div>'
            '<div style="font-family:Playfair Display,serif;font-size:0.96rem;font-weight:600;color:#FFD4A0;margin-bottom:5px;">'+title+'</div>'
            '<div style="font-size:0.78rem;color:rgba(245,230,216,0.40);line-height:1.6;">'+desc+'</div>'
            '</div>', unsafe_allow_html=True)


def main():
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(15,10,5,0.85) 0%,rgba(80,30,10,0.73) 50%,rgba(15,10,5,0.85) 100%),url(https://images.unsplash.com/photo-1543353071-873f17a7a088?w=1600&q=80) center/cover no-repeat;border-radius:20px;padding:52px 42px 48px;margin-bottom:26px;">'
        '<div style="display:inline-flex;align-items:center;gap:6px;background:rgba(210,100,30,0.26);border:1px solid rgba(210,100,30,0.48);color:#FFCFA0;font-size:0.74rem;font-weight:500;letter-spacing:.06em;padding:5px 14px;border-radius:50px;margin-bottom:16px;">✦ AI-Powered · Personalised · Nutritionally Balanced</div>'
        '<div style="font-family:Playfair Display,serif;font-size:2.9rem;font-weight:700;color:#FFF8F0;line-height:1.15;margin:0 0 10px;">Your smart<br/>meal planner</div>'
        '<div style="font-size:1.02rem;color:rgba(255,240,220,0.70);font-weight:300;max-width:480px;line-height:1.65;">Discover recipes tailored to your diet, taste, and health goals. Now with regional Indian cuisine.</div>'
        '</div>', unsafe_allow_html=True)

    with st.spinner("Warming up the kitchen… 👨‍🍳"):
        recipes, users, ratings, models = load_data_and_models()

    profile, go = render_sidebar()

    if not go:
        render_landing()
        return

    with st.spinner("Finding your perfect recipes…"):
        time.sleep(0.3)
        recs = recommend_for_new_user(profile, recipes, ratings, models, top_n=48)

    if recs.empty:
        st.warning("⚠️ No recipes matched your filters — try relaxing the calorie range or prep time.")
        return

    persona = str(recs.iloc[0].get("persona","Explorer"))
    st.markdown(
        '<div style="display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,rgba(210,100,30,0.20),rgba(180,70,10,0.16));border:1px solid rgba(210,100,30,0.38);color:#FFB878;font-size:0.86rem;font-weight:500;padding:8px 18px;border-radius:50px;margin:4px 0 20px;">'
        '🧑‍🍳 Your culinary persona: <strong>'+persona+'</strong></div>',
        unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Recipes found",  len(recs))
    c2.metric("Avg calories",   f'{recs["calories"].mean():.0f} kcal')
    c3.metric("Avg prep time",  f'{recs["prep_min"].mean():.0f} min')
    c4.metric("Cuisines",       recs["cuisine"].nunique())
    st.markdown("<br/>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🍴  Recipe Recommendations", "📅  Meal Plan"])

    with tab1:
        st.markdown(
            '<div style="font-family:Playfair Display,serif;font-size:1.45rem;font-weight:600;color:#FFF0D8;margin:8px 0 16px;">✦ Top picks for you</div>',
            unsafe_allow_html=True)

        bfast  = recs[recs["meal_type"]=="Breakfast"].reset_index(drop=True)
        lunch  = recs[recs["meal_type"]=="Lunch"].reset_index(drop=True)
        dinner = recs[recs["meal_type"]=="Dinner"].reset_index(drop=True)
        snack  = recs[recs["meal_type"]=="Snack"].reset_index(drop=True)

        mt1, mt2, mt3, mt4 = st.tabs([
            f"☀️  Breakfast  ({len(bfast)})",
            f"🌤️  Lunch  ({len(lunch)})",
            f"🌙  Dinner  ({len(dinner)})",
            f"🍎  Snack  ({len(snack)})",
        ])

        def _render_cards(df):
            if df.empty:
                st.info("No recipes in this category. Try widening your cuisine selection or calorie range.")
                return
            col_a, col_b = st.columns(2)
            for i, (_, row) in enumerate(df.iterrows()):
                (col_a if i % 2 == 0 else col_b).markdown(
                    recipe_card(row.to_dict()), unsafe_allow_html=True)

        with mt1: _render_cards(bfast)
        with mt2: _render_cards(lunch)
        with mt3: _render_cards(dinner)
        with mt4: _render_cards(snack)


    with tab2:
        plan_df = build_meal_plan(recs, days=profile["days_plan"])
        if plan_df.empty:
            st.warning("Not enough variety for a full plan. Try widening your cuisine selection.")
            return
        region_note = ""
        if profile.get("indian_region","All Regions")!="All Regions":
            region_note = " · "+profile["indian_region"]+" focus"
        st.markdown(
            '<div style="font-family:Playfair Display,serif;font-size:1.45rem;font-weight:600;color:#FFF0D8;margin:8px 0 18px;">'
            'Your '+str(profile["days_plan"])+'-day meal plan'
            '<span style="font-size:0.82rem;font-weight:400;color:rgba(245,230,216,0.40);margin-left:10px;">3 meals/day'+region_note+'</span></div>',
            unsafe_allow_html=True)
        total_cal = int(plan_df["calories"].sum())
        daily_avg = total_cal//profile["days_plan"]
        n1,n2,n3 = st.columns(3)
        n1.metric("Total plan calories", f"{total_cal:,} kcal")
        n2.metric("Daily average",       f"{daily_avg} kcal")
        n3.metric("Your daily target",   f"{profile['calorie_target']*3} kcal")
        st.markdown("<br/>", unsafe_allow_html=True)
        render_meal_plan(plan_df)


if __name__=="__main__":
    main()
