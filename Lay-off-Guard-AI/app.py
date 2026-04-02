"""
╔══════════════════════════════════════════════════════════╗
║         LayoffGuard AI — Streamlit Risk Predictor        ║
║   XGBoost · LightGBM · RandomForest · GBM · LR · ANN    ║
╚══════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json, os, warnings, time
warnings.filterwarnings("ignore")

# ── Page config MUST be first ──────────────────────────────────
st.set_page_config(
    page_title="LayoffGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS — blurred office BG + glassmorphism theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
  --bg:        #090e1a;
  --card:      rgba(15,22,40,0.82);
  --border:    rgba(0,212,170,0.18);
  --accent:    #00d4aa;
  --accent2:   #ff6b6b;
  --accent3:   #ffd93d;
  --accent4:   #7c6af7;
  --text:      #e6edf3;
  --muted:     #8b949e;
  --glass:     rgba(255,255,255,0.04);
}

/* ── Full-page blurred office background ── */
.stApp {
  background:
    linear-gradient(135deg,rgba(0,0,0,0.72) 0%,rgba(7,12,28,0.85) 100%),
    url("https://images.unsplash.com/photo-1497366216548-37526070297c?w=1600&q=80&auto=format&fit=crop")
    center/cover fixed no-repeat !important;
  font-family: 'Space Grotesk', sans-serif !important;
  color: var(--text) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding-top:1.2rem !important; padding-bottom:2rem !important;}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: rgba(9,14,26,0.92) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * {color: var(--text) !important;}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: var(--muted) !important; font-size:0.82rem !important; }

/* ── Inputs ── */
.stSelectbox>div>div,
.stTextInput>div>div>input,
.stNumberInput>div>div>input {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(0,212,170,0.25) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}
.stSelectbox>div>div:hover,
.stTextInput>div>div>input:focus,
.stNumberInput>div>div>input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
.stSlider [data-baseweb="track"] > div:first-child {
  background: var(--accent) !important;
}

/* ── Buttons ── */
.stButton>button {
  background: linear-gradient(135deg, var(--accent), #00a882) !important;
  color: #000 !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.65rem 2rem !important;
  letter-spacing: 0.03em;
  transition: all .25s ease !important;
  box-shadow: 0 4px 20px rgba(0,212,170,0.35) !important;
}
.stButton>button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 30px rgba(0,212,170,0.55) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.04) !important;
  border-radius: 12px !important;
  padding: 4px !important;
  gap: 4px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted) !important;
  border-radius: 8px !important;
  font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent), #00a882) !important;
  color: #000 !important;
  font-weight: 600 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 1rem 1.2rem !important;
  backdrop-filter: blur(16px);
}
[data-testid="stMetricLabel"] {color: var(--muted) !important; font-size:0.8rem !important;}
[data-testid="stMetricValue"] {color: var(--accent) !important; font-size:1.5rem !important; font-weight:700 !important;}
[data-testid="stMetricDelta"] {font-size:0.78rem !important;}

/* ── Plot backgrounds transparent ── */
.stPlotlyChart, .stPyplot { background: transparent !important; }

/* ── Expander ── */
.stExpander {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
.stExpander summary { color: var(--accent) !important; font-weight: 600 !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Progress bar ── */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent4)) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPER — Matplotlib dark theme
# ══════════════════════════════════════════════════════════════
DARK   = "#090e1a"
CARD   = "#0f1628"
ACC    = "#00d4aa"
ACC2   = "#ff6b6b"
ACC3   = "#ffd93d"
ACC4   = "#7c6af7"
TXT    = "#e6edf3"
MUT    = "#8b949e"

def dark_fig(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("none")
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d40")
    ax.tick_params(colors=TXT)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color(ACC)
    return fig, ax

def dark_fig_multi(rows, cols, w=14, h=6):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor("none")
    for ax in np.array(axes).flatten():
        ax.set_facecolor(CARD)
        for s in ax.spines.values():
            s.set_edgecolor("#1e2d40")
        ax.tick_params(colors=TXT)
        ax.xaxis.label.set_color(TXT)
        ax.yaxis.label.set_color(TXT)
        ax.title.set_color(ACC)
    return fig, axes


# ══════════════════════════════════════════════════════════════
#  DATA GENERATION (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generate_dataset(n=6000):
    np.random.seed(42)

    job_levels   = ["Junior","Mid-Level","Senior","Lead","Manager","Director","VP"]
    departments  = ["Engineering","Sales","Marketing","HR","Finance","Operations",
                    "Customer Success","Product","Data Science","Legal"]
    emp_types    = ["Full-Time","Contract","Part-Time"]
    ai_levels    = ["None","Basic","Intermediate","Advanced","Expert"]
    cloud_levels = ["None","Basic","Intermediate","Advanced"]
    company_sz   = ["Startup","SME","Mid-Market","Enterprise"]
    fund_stages  = ["Bootstrapped","Seed","Series A","Series B","Series C+","Public","PE-Backed"]
    perf_trends  = ["Improving","Stable","Declining"]

    d = {}
    d["age"]                        = np.random.normal(38,10,n).clip(22,65).astype(int)
    d["years_at_company"]           = np.random.exponential(4,n).clip(0.1,30).round(1)
    d["years_in_industry"]          = (d["years_at_company"]+np.random.exponential(3,n)).clip(0,40).round(1)
    d["job_level"]                  = np.random.choice(job_levels,n,p=[0.20,0.28,0.22,0.12,0.10,0.05,0.03])
    d["department"]                 = np.random.choice(departments,n,p=[0.22,0.18,0.12,0.07,0.09,0.10,0.08,0.08,0.04,0.02])
    d["employment_type"]            = np.random.choice(emp_types,n,p=[0.75,0.17,0.08])
    d["last_performance_rating"]    = np.random.choice([1,2,3,4,5],n,p=[0.05,0.12,0.35,0.33,0.15])
    d["performance_trend"]          = np.random.choice(perf_trends,n,p=[0.30,0.45,0.25])
    d["kpi_achievement_pct"]        = np.random.normal(88,18,n).clip(20,130).round(1)
    d["projects_completed"]         = np.random.poisson(4.5,n).clip(0,20)
    d["projects_failed"]            = np.random.poisson(0.8,n).clip(0,8)
    d["technical_skills_score"]     = np.random.normal(65,18,n).clip(10,100).round(1)
    d["upskilling_hours_per_year"]  = np.random.exponential(35,n).clip(0,300).round(0)
    d["certifications_count"]       = np.random.poisson(1.8,n).clip(0,15)
    d["ai_ml_proficiency"]          = np.random.choice(ai_levels,n,p=[0.20,0.30,0.28,0.15,0.07])
    d["cloud_skills"]               = np.random.choice(cloud_levels,n,p=[0.25,0.30,0.28,0.17])
    d["cross_functional_skills"]    = np.random.normal(55,20,n).clip(0,100).round(1)
    d["automation_risk_score"]      = np.random.normal(45,22,n).clip(0,100).round(1)
    d["salary_above_market_pct"]    = np.random.normal(5,20,n).round(1)
    d["bonus_received"]             = np.random.choice([0,1],n,p=[0.30,0.70])
    d["equity_vesting_months"]      = np.random.choice([0,12,24,36,48],n,p=[0.20,0.15,0.25,0.30,0.10])
    d["company_revenue_growth_pct"] = np.random.normal(8,25,n).clip(-60,80).round(1)
    d["dept_budget_change_pct"]     = np.random.normal(-2,20,n).clip(-80,50).round(1)
    d["company_layoff_history"]     = np.random.choice([0,1,2],n,p=[0.45,0.35,0.20])
    d["industry_disruption_score"]  = np.random.normal(50,20,n).clip(0,100).round(1)
    d["company_size"]               = np.random.choice(company_sz,n,p=[0.15,0.25,0.30,0.30])
    d["funding_stage"]              = np.random.choice(fund_stages,n,p=[0.10,0.08,0.12,0.12,0.15,0.28,0.15])
    d["absenteeism_days"]           = np.random.poisson(4,n).clip(0,60)
    d["overtime_hours_per_week"]    = np.random.exponential(4,n).clip(0,30).round(1)
    d["remote_work_pct"]            = np.random.choice([0,25,50,75,100],n,p=[0.10,0.10,0.30,0.20,0.30])
    d["manager_relationship_score"] = np.random.normal(65,20,n).clip(0,100).round(1)
    d["employee_engagement_score"]  = np.random.normal(62,22,n).clip(0,100).round(1)
    d["internal_mobility_attempts"] = np.random.poisson(0.5,n).clip(0,5)
    d["internal_recognition_awards"]= np.random.poisson(1.2,n).clip(0,10)
    d["mentorship_involvement"]     = np.random.choice([0,1],n,p=[0.55,0.45])
    d["cross_dept_projects"]        = np.random.poisson(1.5,n).clip(0,10)
    d["linkedin_completeness"]      = np.random.normal(70,20,n).clip(0,100).round(1)
    d["macroeconomic_stress_index"] = np.random.normal(55,15,n).clip(10,100).round(1)
    d["sector_layoff_rate_pct"]     = np.random.normal(6,4,n).clip(0,25).round(1)
    d["job_market_demand_score"]    = np.random.normal(60,20,n).clip(10,100).round(1)

    df = pd.DataFrame(d)

    # Risk scoring
    risk = np.zeros(n)
    risk += (5 - df["last_performance_rating"]) * 8
    risk += (df["performance_trend"] == "Declining") * 15
    risk += (100 - df["kpi_achievement_pct"].clip(0,100)) * 0.25
    risk -= df["upskilling_hours_per_year"] * 0.10
    risk -= df["certifications_count"] * 3
    risk -= df["technical_skills_score"] * 0.20
    risk += df["automation_risk_score"] * 0.20
    risk += df["ai_ml_proficiency"].map({"None":10,"Basic":5,"Intermediate":2,"Advanced":-3,"Expert":-6})
    risk += df["cloud_skills"].map({"None":6,"Basic":3,"Intermediate":0,"Advanced":-4})
    risk -= df["company_revenue_growth_pct"] * 0.30
    risk += (df["dept_budget_change_pct"] < -10) * 15
    risk += df["company_layoff_history"] * 8
    risk += df["industry_disruption_score"] * 0.15
    risk += df["salary_above_market_pct"] * 0.15
    risk += (df["bonus_received"] == 0) * 8
    risk += (df["employment_type"] == "Contract") * 18
    risk += (df["employment_type"] == "Part-Time") * 10
    risk += df["funding_stage"].map({"Seed":12,"Series A":8,"Bootstrapped":6,
                                      "Series B":3,"Series C+":0,"Public":-2,"PE-Backed":1})
    risk += df["company_size"].map({"Startup":10,"SME":4,"Mid-Market":0,"Enterprise":-3})
    risk += df["absenteeism_days"] * 1.2
    risk -= df["employee_engagement_score"] * 0.10
    risk -= df["manager_relationship_score"] * 0.08
    risk += df["macroeconomic_stress_index"] * 0.15
    risk += df["sector_layoff_rate_pct"] * 0.80
    risk += df["job_level"].map({"Junior":5,"Mid-Level":8,"Senior":2,
                                  "Lead":-2,"Manager":-5,"Director":-8,"VP":-10})
    risk += np.where(df["years_at_company"] < 1, 15, 0)
    risk += np.where(df["years_at_company"] > 20, 6, 0)

    r = (risk - risk.mean()) / risk.std()
    prob = (1 / (1 + np.exp(-0.7 * r)))
    prob = (prob * 0.85 + np.random.uniform(0, 0.15, n)).clip(0.02, 0.98)

    df["layoff_probability"] = prob.round(4)
    df["layoff_risk"]        = (prob > 0.50).astype(int)
    return df


# ══════════════════════════════════════════════════════════════
#  MODEL TRAINING (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_models():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
    import xgboost as xgb
    import lightgbm as lgb
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks as cb

    df = generate_dataset()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_enc = df.copy()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col])
        le_dict[col] = le

    feat_cols = [c for c in df_enc.columns if c not in ["layoff_risk","layoff_probability"]]
    X = df_enc[feat_cols].values
    y = df_enc["layoff_risk"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    results = {}

    # XGBoost
    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.06,
                           subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                           eval_metric="logloss", random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    yp = m.predict_proba(X_te)[:,1]
    results["XGBoost"] = dict(model=m, prob=yp, pred=(yp>0.5).astype(int),
                               acc=accuracy_score(y_te,yp>0.5), f1=f1_score(y_te,yp>0.5),
                               auc=roc_auc_score(y_te,yp), fi=m.feature_importances_,
                               roc=roc_curve(y_te,yp), cm=confusion_matrix(y_te,yp>0.5),
                               needs_scale=False)

    # LightGBM
    m = lgb.LGBMClassifier(n_estimators=400, max_depth=6, learning_rate=0.06,
                            num_leaves=63, subsample=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
          callbacks=[lgb.early_stopping(40,verbose=False), lgb.log_evaluation(-1)])
    yp = m.predict_proba(X_te)[:,1]
    results["LightGBM"] = dict(model=m, prob=yp, pred=(yp>0.5).astype(int),
                                acc=accuracy_score(y_te,yp>0.5), f1=f1_score(y_te,yp>0.5),
                                auc=roc_auc_score(y_te,yp), fi=m.feature_importances_,
                                roc=roc_curve(y_te,yp), cm=confusion_matrix(y_te,yp>0.5),
                                needs_scale=False)

    # Random Forest
    m = RandomForestClassifier(n_estimators=250, max_depth=12, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    yp = m.predict_proba(X_te)[:,1]
    results["Random Forest"] = dict(model=m, prob=yp, pred=(yp>0.5).astype(int),
                                     acc=accuracy_score(y_te,yp>0.5), f1=f1_score(y_te,yp>0.5),
                                     auc=roc_auc_score(y_te,yp), fi=m.feature_importances_,
                                     roc=roc_curve(y_te,yp), cm=confusion_matrix(y_te,yp>0.5),
                                     needs_scale=False)

    # Gradient Boosting
    m = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                                    subsample=0.8, random_state=42)
    m.fit(X_tr, y_tr)
    yp = m.predict_proba(X_te)[:,1]
    results["Gradient Boosting"] = dict(model=m, prob=yp, pred=(yp>0.5).astype(int),
                                         acc=accuracy_score(y_te,yp>0.5), f1=f1_score(y_te,yp>0.5),
                                         auc=roc_auc_score(y_te,yp), fi=m.feature_importances_,
                                         roc=roc_curve(y_te,yp), cm=confusion_matrix(y_te,yp>0.5),
                                         needs_scale=False)

    # Logistic Regression
    m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    m.fit(X_tr_sc, y_tr)
    yp = m.predict_proba(X_te_sc)[:,1]
    results["Logistic Regression"] = dict(model=m, prob=yp, pred=(yp>0.5).astype(int),
                                           acc=accuracy_score(y_te,yp>0.5), f1=f1_score(y_te,yp>0.5),
                                           auc=roc_auc_score(y_te,yp),
                                           fi=np.abs(m.coef_[0]),
                                           roc=roc_curve(y_te,yp), cm=confusion_matrix(y_te,yp>0.5),
                                           needs_scale=True)

    # ANN
    tf.random.set_seed(42)
    ann = keras.Sequential([
        layers.Input(shape=(X_tr_sc.shape[1],)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(32, activation="relu"), layers.Dropout(0.15),
        layers.Dense(1, activation="sigmoid"),
    ])
    ann.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(name="auc")])
    hist = ann.fit(X_tr_sc, y_tr, validation_split=0.15, epochs=120, batch_size=256, verbose=0,
                   callbacks=[cb.EarlyStopping("val_auc",patience=18,restore_best_weights=True,mode="max"),
                               cb.ReduceLROnPlateau("val_loss",factor=0.5,patience=8,min_lr=1e-6)])
    yp = ann.predict(X_te_sc, verbose=0).flatten()
    results["ANN"] = dict(model=ann, prob=yp, pred=(yp>0.5).astype(int),
                           acc=accuracy_score(y_te,yp>0.5), f1=f1_score(y_te,yp>0.5),
                           auc=roc_auc_score(y_te,yp), fi=None,
                           roc=roc_curve(y_te,yp), cm=confusion_matrix(y_te,yp>0.5),
                           needs_scale=True, history=hist.history)

    # Ensemble
    w = {"XGBoost":0.30,"LightGBM":0.28,"Random Forest":0.20,"Gradient Boosting":0.14,"ANN":0.08}
    ep = sum(results[k]["prob"]*w[k] for k in w)
    results["⭐ Ensemble"] = dict(prob=ep, pred=(ep>0.5).astype(int),
                                   acc=accuracy_score(y_te,ep>0.5), f1=f1_score(y_te,ep>0.5),
                                   auc=roc_auc_score(y_te,ep),
                                   roc=roc_curve(y_te,ep), cm=confusion_matrix(y_te,ep>0.5),
                                   fi=None, needs_scale=False)

    # Average feature importance
    fi_sum = np.zeros(len(feat_cols))
    count  = 0
    for k in ["XGBoost","LightGBM","Random Forest","Gradient Boosting"]:
        fi_sum += results[k]["fi"]
        count  += 1
    avg_fi = fi_sum / count

    return (results, feat_cols, scaler, le_dict, df, X_te, y_te,
            X_te_sc, avg_fi, w)


# ══════════════════════════════════════════════════════════════
#  CARD HTML HELPER
# ══════════════════════════════════════════════════════════════
def card(content: str, border_color: str = "#00d4aa") -> None:
    st.markdown(f"""
    <div style="
      background:rgba(15,22,40,0.82);
      border:1px solid {border_color}33;
      border-left:3px solid {border_color};
      border-radius:14px;
      padding:1.2rem 1.4rem;
      backdrop-filter:blur(16px);
      margin-bottom:0.8rem;
    ">{content}</div>""", unsafe_allow_html=True)


def risk_badge(label: str, color: str, emoji: str) -> str:
    return f"""
    <div style="
      display:inline-flex;align-items:center;gap:10px;
      background:{color}22;border:2px solid {color};
      border-radius:999px;padding:0.6rem 1.6rem;
      font-size:1.4rem;font-weight:700;color:{color};
      box-shadow:0 0 24px {color}44;
    ">{emoji} {label}</div>"""


# ══════════════════════════════════════════════════════════════
#  SIDEBAR — Input form
# ══════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 1.5rem">
          <div style="font-size:2.2rem">⚡</div>
          <div style="font-size:1.3rem;font-weight:700;color:#00d4aa;letter-spacing:0.05em">LayoffGuard AI</div>
          <div style="font-size:0.75rem;color:#8b949e;margin-top:4px">Career Risk Intelligence Platform</div>
        </div>
        <hr style="border-color:rgba(0,212,170,0.18);margin:0 0 1.2rem">
        """, unsafe_allow_html=True)

        inp = {}

        st.markdown("### 👤 Profile")
        inp["age"]              = st.slider("Age", 22, 65, 32)
        inp["years_at_company"] = st.slider("Years at Current Company", 0.0, 30.0, 2.5, 0.5)
        inp["years_in_industry"]= st.slider("Total Industry Experience (yrs)", 0.0, 40.0, 5.0, 0.5)
        inp["job_level"]        = st.selectbox("Job Level", ["Junior","Mid-Level","Senior","Lead","Manager","Director","VP"])
        inp["department"]       = st.selectbox("Department", ["Engineering","Sales","Marketing","HR","Finance","Operations","Customer Success","Product","Data Science","Legal"])
        inp["employment_type"]  = st.selectbox("Employment Type", ["Full-Time","Contract","Part-Time"])

        st.markdown("---")
        st.markdown("### 📊 Performance")
        inp["last_performance_rating"] = st.slider("Last Performance Rating (1–5)", 1, 5, 3)
        inp["performance_trend"]       = st.selectbox("Performance Trend", ["Improving","Stable","Declining"])
        inp["kpi_achievement_pct"]     = st.slider("KPI Achievement %", 20, 130, 88)
        inp["projects_completed"]      = st.slider("Projects Completed (Last Year)", 0, 20, 4)
        inp["projects_failed"]         = st.slider("Projects Failed (Last Year)", 0, 8, 1)

        st.markdown("---")
        st.markdown("### 🧠 Skills & Learning")
        inp["technical_skills_score"]    = st.slider("Technical Skills Score (0–100)", 0, 100, 65)
        inp["upskilling_hours_per_year"] = st.slider("Upskilling Hours / Year", 0, 300, 35)
        inp["certifications_count"]      = st.slider("Certifications Earned", 0, 15, 2)
        inp["ai_ml_proficiency"]         = st.selectbox("AI/ML Proficiency", ["None","Basic","Intermediate","Advanced","Expert"])
        inp["cloud_skills"]              = st.selectbox("Cloud Skills", ["None","Basic","Intermediate","Advanced"])
        inp["cross_functional_skills"]   = st.slider("Cross-Functional Skills (0–100)", 0, 100, 55)
        inp["automation_risk_score"]     = st.slider("Automation Risk Score (0–100)", 0, 100, 45)

        st.markdown("---")
        st.markdown("### 💰 Compensation")
        inp["salary_above_market_pct"] = st.slider("Salary vs Market (%)", -40, 60, 5)
        inp["bonus_received"]          = st.selectbox("Bonus Received Last Year", [1, 0], format_func=lambda x: "Yes" if x else "No")
        inp["equity_vesting_months"]   = st.selectbox("Equity Vesting Cliff (months)", [0,12,24,36,48])

        st.markdown("---")
        st.markdown("### 🏢 Company & Market")
        inp["company_revenue_growth_pct"] = st.slider("Company Revenue Growth %", -60, 80, 8)
        inp["dept_budget_change_pct"]     = st.slider("Dept Budget Change %", -80, 50, -2)
        inp["company_layoff_history"]     = st.selectbox("Prior Layoff Rounds", [0,1,2])
        inp["industry_disruption_score"]  = st.slider("Industry Disruption Score", 0, 100, 50)
        inp["company_size"]               = st.selectbox("Company Size", ["Startup","SME","Mid-Market","Enterprise"])
        inp["funding_stage"]              = st.selectbox("Funding Stage", ["Bootstrapped","Seed","Series A","Series B","Series C+","Public","PE-Backed"])

        st.markdown("---")
        st.markdown("### 🤝 Work Behavior")
        inp["absenteeism_days"]            = st.slider("Absenteeism Days / Year", 0, 60, 4)
        inp["overtime_hours_per_week"]     = st.slider("Overtime Hours / Week", 0.0, 30.0, 4.0, 0.5)
        inp["remote_work_pct"]             = st.select_slider("Remote Work %", [0,25,50,75,100], value=50)
        inp["manager_relationship_score"]  = st.slider("Manager Relationship Score", 0, 100, 65)
        inp["employee_engagement_score"]   = st.slider("Employee Engagement Score", 0, 100, 62)
        inp["internal_mobility_attempts"]  = st.slider("Internal Mobility Attempts", 0, 5, 0)
        inp["internal_recognition_awards"] = st.slider("Internal Recognition Awards", 0, 10, 1)
        inp["mentorship_involvement"]      = st.selectbox("Mentorship Involvement", [1,0], format_func=lambda x: "Yes" if x else "No")
        inp["cross_dept_projects"]         = st.slider("Cross-Dept Projects", 0, 10, 1)
        inp["linkedin_completeness"]       = st.slider("LinkedIn Profile Completeness %", 0, 100, 70)

        st.markdown("---")
        st.markdown("### 🌍 Macro Factors")
        inp["macroeconomic_stress_index"] = st.slider("Macroeconomic Stress Index", 10, 100, 55)
        inp["sector_layoff_rate_pct"]     = st.slider("Sector Layoff Rate %", 0.0, 25.0, 6.0, 0.5)
        inp["job_market_demand_score"]    = st.slider("Job Market Demand Score", 10, 100, 60)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ PREDICT LAYOFF RISK", use_container_width=True)

    return inp, predict_btn


# ══════════════════════════════════════════════════════════════
#  RISK COMPUTATION
# ══════════════════════════════════════════════════════════════
def compute_risk(inp, results, feat_cols, scaler, le_dict, weights):
    from sklearn.preprocessing import LabelEncoder
    cat_map = {
        "job_level":        ["Junior","Mid-Level","Senior","Lead","Manager","Director","VP"],
        "department":       ["Engineering","Sales","Marketing","HR","Finance","Operations","Customer Success","Product","Data Science","Legal"],
        "employment_type":  ["Full-Time","Contract","Part-Time"],
        "performance_trend":["Improving","Stable","Declining"],
        "ai_ml_proficiency":["None","Basic","Intermediate","Advanced","Expert"],
        "cloud_skills":     ["None","Basic","Intermediate","Advanced"],
        "company_size":     ["Startup","SME","Mid-Market","Enterprise"],
        "funding_stage":    ["Bootstrapped","Seed","Series A","Series B","Series C+","Public","PE-Backed"],
    }

    row = {}
    for f in feat_cols:
        val = inp.get(f, 0)
        if f in cat_map:
            le = le_dict.get(f)
            if le:
                try:    val = int(le.transform([str(val)])[0])
                except: val = 0
            else:
                options = cat_map[f]
                val = options.index(val) if val in options else 0
        row[f] = float(val)

    X_raw = np.array([[row[f] for f in feat_cols]])
    X_sc  = scaler.transform(X_raw)

    probs = {}
    for name, res in results.items():
        if name == "⭐ Ensemble":
            continue
        m   = res["model"]
        X_in = X_sc if res["needs_scale"] else X_raw
        if name == "ANN":
            p = float(m.predict(X_in, verbose=0).flatten()[0])
        else:
            p = float(m.predict_proba(X_in)[0][1])
        probs[name] = p

    ens = sum(probs[k] * weights[k] for k in weights if k in probs)
    total_w = sum(weights[k] for k in weights if k in probs)
    ens /= total_w

    return ens, probs, X_raw, X_sc


# ══════════════════════════════════════════════════════════════
#  RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
def get_recs(inp, prob):
    recs = {"🚨 Critical Actions": [], "⚠️ Important Actions": [],
            "💡 Suggested Improvements": [], "✅ Your Strengths": []}

    if inp.get("last_performance_rating", 3) <= 2:
        recs["🚨 Critical Actions"].append(("📉","Improve Performance Rating",
            "Request a PIP review, set 30-60-90 day goals, seek mentorship immediately."))
    if inp.get("automation_risk_score", 50) > 70:
        recs["🚨 Critical Actions"].append(("🤖","Automation-Proof Your Role",
            "Develop creative, strategic & interpersonal skills AI cannot replicate."))
    if inp.get("employment_type") == "Contract":
        recs["🚨 Critical Actions"].append(("📄","Convert to Permanent Status",
            "Negotiate full-time conversion or build a strong freelance safety net."))
    if inp.get("company_layoff_history", 0) >= 2:
        recs["🚨 Critical Actions"].append(("🏚️","Company Layoff History is High",
            "Start networking externally and update your resume now as a precaution."))

    if inp.get("upskilling_hours_per_year", 0) < 40:
        recs["⚠️ Important Actions"].append(("📚","Increase Learning Investment",
            f"You log ~{int(inp.get('upskilling_hours_per_year',0))}h/yr. Target 80+ hours — Coursera, Udemy, O'Reilly."))
    if inp.get("ai_ml_proficiency") in ["None","Basic"]:
        recs["⚠️ Important Actions"].append(("🧠","Build AI/ML Skills",
            "Complete fast.ai, Google ML Crash Course, or Andrew Ng's Deep Learning Spec."))
    if inp.get("certifications_count", 0) < 2:
        recs["⚠️ Important Actions"].append(("🏅","Earn Industry Certifications",
            "AWS / GCP / Azure / PMP / Scrum — certs boost market value 15-30%."))
    if inp.get("cloud_skills") in ["None","Basic"]:
        recs["⚠️ Important Actions"].append(("☁️","Develop Cloud Expertise",
            "AWS Solutions Architect Associate is the #1 demanded cloud cert right now."))
    if inp.get("dept_budget_change_pct", 0) < -20:
        recs["⚠️ Important Actions"].append(("💸","Dept Budget Cut Detected",
            "Department cuts >20% precede layoffs. Quantify your ROI to leadership now."))

    if inp.get("cross_dept_projects", 0) < 2:
        recs["💡 Suggested Improvements"].append(("🤝","Cross-Functional Visibility",
            "Volunteer for inter-department projects — reduces single-point-of-failure risk."))
    if inp.get("internal_recognition_awards", 0) == 0:
        recs["💡 Suggested Improvements"].append(("🌟","Seek Internal Recognition",
            "Present wins at all-hands meetings, document achievements, nominate peers."))
    if inp.get("mentorship_involvement") == 0:
        recs["💡 Suggested Improvements"].append(("👨‍🏫","Join Mentorship Programs",
            "Mentors/mentees are consistently rated as high-value employees in surveys."))
    if inp.get("linkedin_completeness", 70) < 80:
        recs["💡 Suggested Improvements"].append(("💼","Optimize LinkedIn Profile",
            "Complete profiles get 40× more opportunities. Add skills, projects, endorsements."))

    if inp.get("kpi_achievement_pct", 80) >= 100:
        recs["✅ Your Strengths"].append(("🎯","KPI Overachiever","Exceeding targets makes you hard to replace."))
    if inp.get("certifications_count", 0) >= 3:
        recs["✅ Your Strengths"].append(("🏅","Well-Certified","Multiple certs signal continuous growth mindset."))
    if inp.get("employee_engagement_score", 60) >= 80:
        recs["✅ Your Strengths"].append(("💪","High Engagement","Engagement above average — you're a culture carrier."))
    if inp.get("cross_dept_projects", 0) >= 3:
        recs["✅ Your Strengths"].append(("🔗","Cross-Functional Leader","Visible and valued across multiple teams."))
    if inp.get("upskilling_hours_per_year", 0) >= 80:
        recs["✅ Your Strengths"].append(("📚","Continuous Learner","Strong L&D investment sets you apart."))
    if inp.get("ai_ml_proficiency") in ["Advanced","Expert"]:
        recs["✅ Your Strengths"].append(("🤖","AI/ML Expert","Among the most in-demand skills globally right now."))

    return recs


# ══════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════
def plot_feature_importance(feat_cols, avg_fi, top_n=20):
    fi_series = pd.Series(avg_fi, index=feat_cols).sort_values(ascending=False).head(top_n)
    fig, ax = dark_fig(12, 7)
    colors = [ACC if i < 5 else ACC4 if i < 10 else "#4fc3f7" for i in range(len(fi_series))]
    ax.barh(fi_series.index[::-1], fi_series.values[::-1],
            color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(f"Top {top_n} Features Driving Layoff Risk", fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.25, color="#1e2d40")
    ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=ACC,label="Top 5"),
               mpatches.Patch(color=ACC4,label="Top 6–10"),
               mpatches.Patch(color="#4fc3f7",label="Top 11–20")]
    ax.legend(handles=patches, facecolor=CARD, edgecolor="#1e2d40", labelcolor=TXT)
    plt.tight_layout()
    return fig


def plot_heatmap(df):
    from sklearn.preprocessing import LabelEncoder
    df_enc = df.copy()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
    num_cols = df_enc.select_dtypes(include=np.number).columns.tolist()
    corr_cols = pd.Series(
        df_enc[num_cols].corrwith(df_enc["layoff_risk"]).abs()
    ).sort_values(ascending=False).head(14).index.tolist()
    if "layoff_risk" not in corr_cols:
        corr_cols.append("layoff_risk")
    corr = df_enc[corr_cols].corr()
    cmap = LinearSegmentedColormap.from_list("lg", [ACC4, CARD, ACC], N=256)
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("none")
    ax.set_facecolor(CARD)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#0d1a2e",
                annot_kws={"size": 8, "color": TXT}, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap (Top 14 + Target)", fontsize=13,
                 fontweight="bold", color=ACC, pad=14)
    ax.tick_params(axis="x", rotation=40, labelsize=8, colors=TXT)
    ax.tick_params(axis="y", rotation=0,  labelsize=8, colors=TXT)
    ax.collections[0].colorbar.ax.tick_params(colors=TXT)
    plt.tight_layout()
    return fig


def plot_roc(results, y_te):
    fig, ax = dark_fig(9, 7)
    ax.plot([0,1],[0,1],"--",color="#4a5568",linewidth=1.5,label="Random (0.50)")
    palette = [ACC, ACC2, ACC3, ACC4, "#4fc3f7", "#fd79a8", "#a8edea"]
    for (name, res), col in zip(results.items(), palette):
        fpr, tpr, _ = res["roc"]
        ax.plot(fpr, tpr, linewidth=2.2, color=col,
                label=f"{name} ({res['auc']:.4f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(facecolor=CARD, edgecolor="#1e2d40", labelcolor=TXT, fontsize=9)
    ax.xaxis.grid(True,alpha=0.2); ax.yaxis.grid(True,alpha=0.2)
    plt.tight_layout(); return fig


def plot_model_comparison(results):
    metrics = ["acc","f1","auc"]
    labels  = ["Accuracy","F1 Score","ROC-AUC"]
    fig, axes = dark_fig_multi(1, 3, 16, 5)
    palette = [ACC, ACC3, ACC4, ACC2, "#4fc3f7", "#fd79a8", "#a8edea"]
    for ax, metric, label in zip(axes, metrics, labels):
        names = list(results.keys())
        vals  = [results[n][metric] for n in names]
        bars  = ax.bar(range(len(names)), vals, color=palette[:len(names)], edgecolor="none", width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(min(vals)*0.94, 1.01)
        ax.yaxis.grid(True,alpha=0.2); ax.set_axisbelow(True)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=TXT)
    fig.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold",
                 color=ACC, y=1.02)
    plt.tight_layout(); return fig


def plot_confusion_matrix(results):
    names = ["XGBoost","LightGBM","ANN","⭐ Ensemble"]
    names = [n for n in names if n in results][:4]
    fig, axes = dark_fig_multi(1, len(names), len(names)*3+2, 4)
    if len(names)==1: axes = [axes]
    cmap = LinearSegmentedColormap.from_list("cm",[CARD, ACC4],N=100)
    for ax, name in zip(axes, names):
        cm = results[name]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    linewidths=1, linecolor=DARK,
                    annot_kws={"size":13,"fontweight":"bold","color":TXT})
        ax.set_title(name, color=ACC, fontsize=10)
        ax.set_xlabel("Predicted",fontsize=9); ax.set_ylabel("Actual",fontsize=9)
        ax.set_xticklabels(["No Risk","At Risk"],color=TXT)
        ax.set_yticklabels(["No Risk","At Risk"],rotation=0,color=TXT)
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold", color=ACC, y=1.02)
    plt.tight_layout(); return fig


def plot_ann_history(history):
    fig, axes = dark_fig_multi(1, 2, 12, 4)
    axes[0].plot(history["loss"],     color=ACC2, lw=2, label="Train Loss")
    axes[0].plot(history["val_loss"], color=ACC,  lw=2, label="Val Loss")
    axes[0].set_title("ANN — Loss"); axes[0].legend(facecolor=CARD, edgecolor="#1e2d40", labelcolor=TXT)
    axes[0].yaxis.grid(True,alpha=0.2)
    axes[1].plot(history["auc"],     color=ACC3, lw=2, label="Train AUC")
    axes[1].plot(history["val_auc"], color=ACC4, lw=2, label="Val AUC")
    axes[1].set_title("ANN — AUC"); axes[1].legend(facecolor=CARD, edgecolor="#1e2d40", labelcolor=TXT)
    axes[1].yaxis.grid(True,alpha=0.2)
    plt.tight_layout(); return fig


def plot_risk_gauge(prob: float):
    """Circular gauge chart — returns tight base64 PNG."""
    fig, ax = plt.subplots(figsize=(3.0, 1.6), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_position([0.0, -0.08, 1.0, 1.08])

    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [0.9]*200, linewidth=10, color="#1e2d40", solid_capstyle="round")

    colors_gauge = [(0.0, ACC), (0.4, ACC3), (0.65, "#ff8c00"), (1.0, ACC2)]
    def interp_color(t):
        for i in range(len(colors_gauge)-1):
            t0, c0 = colors_gauge[i]
            t1, c1 = colors_gauge[i+1]
            if t0 <= t <= t1:
                r = (t-t0)/(t1-t0)
                def hex2rgb(h): return [int(h[j:j+2],16)/255 for j in (1,3,5)]
                rgb0, rgb1 = hex2rgb(c0), hex2rgb(c1)
                return tuple(rgb0[j]*(1-r)+rgb1[j]*r for j in range(3))
        return (1,0,0)

    fill_theta = np.linspace(np.pi, np.pi - np.pi*prob, 200)
    colors_arr = [interp_color(i/199*prob) for i in range(200)]
    for i in range(len(fill_theta)-1):
        ax.plot(fill_theta[i:i+2], [0.9,0.9], linewidth=10,
                color=colors_arr[i], solid_capstyle="round")

    angle = np.pi - np.pi * prob
    ax.annotate("", xy=(angle, 0.70), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=TXT, lw=2.0, mutation_scale=12))

    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, np.pi)
    ax.axis("off")

    for v, lbl in [(0,"0%"),(0.25,"25%"),(0.5,"50%"),(0.75,"75%"),(1,"100%")]:
        a = np.pi - np.pi*v
        ax.text(a, 1.04, lbl, ha="center", va="center", fontsize=6,
                color=MUT, fontfamily="monospace")

    col = (ACC if prob < 0.25 else
           ACC3 if prob < 0.5 else
           "#ff8c00" if prob < 0.75 else ACC2)
    ax.text(np.pi/2, 0.32, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=19, fontweight="bold", color=col, fontfamily="monospace")
    ax.text(np.pi/2, 0.08, "RISK SCORE", ha="center", va="center",
            fontsize=6, color=MUT, fontfamily="monospace")

    # ── Convert to base64 PNG — bypasses Streamlit stretching ──
    import io, base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                transparent=True, pad_inches=0.0)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return b64

def plot_radar(inp_vals, feat_cols, avg_fi):
    """Radar chart of top 8 risk drivers for this user."""
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    top8_feats = pd.Series(avg_fi, index=feat_cols).sort_values(ascending=False).head(8).index.tolist()
    labels = [f.replace("_","\n") for f in top8_feats]

    # Normalize input values to 0–1
    vals = []
    for f in top8_feats:
        raw = inp_vals.get(f, 0)
        try: raw = float(raw)
        except: raw = 0
        vals.append(raw)

    # Simple min-max scale using dataset stats (approximate)
    vals_norm = [min(max(v/100, 0), 1) for v in vals]

    num_vars = len(labels)
    angles   = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    vals_r   = vals_norm + vals_norm[:1]
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    fig.patch.set_facecolor("none")
    ax.set_facecolor(CARD)
    ax.plot(angles, vals_r, "o-", linewidth=2.5, color=ACC)
    ax.fill(angles, vals_r, alpha=0.22, color=ACC)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color=TXT)
    ax.set_yticklabels([]); ax.set_ylim(0, 1.1)
    for a in angles[:-1]:
        ax.plot([a,a],[0,1],"--",color="#1e2d40",linewidth=0.8)
    ax.spines["polar"].set_color("#1e2d40")
    ax.grid(color="#1e2d40", linewidth=0.8)
    ax.set_title("Risk Driver Radar", fontsize=11, fontweight="bold",
                 color=ACC, pad=22)
    plt.tight_layout(); return fig


def plot_risk_dist(df):
    fig, axes = dark_fig_multi(1,2,13,5)
    axes[0].hist(df["layoff_probability"][df["layoff_risk"]==0], bins=50,
                 color=ACC, alpha=0.7, label="No Layoff", edgecolor="none")
    axes[0].hist(df["layoff_probability"][df["layoff_risk"]==1], bins=50,
                 color=ACC2, alpha=0.7, label="Layoff Risk", edgecolor="none")
    axes[0].set_xlabel("Risk Probability"); axes[0].set_ylabel("Count")
    axes[0].set_title("Risk Probability Distribution", fontsize=11)
    axes[0].legend(facecolor=CARD, edgecolor="#1e2d40", labelcolor=TXT)
    axes[0].yaxis.grid(True,alpha=0.2)

    dept_risk = df.groupby("department")["layoff_risk"].mean().sort_values(ascending=False)
    bar_colors = [ACC2 if v>0.5 else ACC for v in dept_risk.values]
    axes[1].bar(range(len(dept_risk)), dept_risk.values, color=bar_colors, edgecolor="none", width=0.65)
    axes[1].set_xticks(range(len(dept_risk)))
    axes[1].set_xticklabels(dept_risk.index, rotation=35, ha="right", fontsize=8)
    axes[1].set_title("Avg Layoff Risk by Department", fontsize=11)
    axes[1].set_ylabel("Avg Risk Rate")
    axes[1].yaxis.grid(True,alpha=0.2); axes[1].set_axisbelow(True)
    plt.tight_layout(); return fig


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div style="
      background:linear-gradient(90deg,rgba(0,212,170,0.08),rgba(124,106,247,0.08));
      border:1px solid rgba(0,212,170,0.2);
      border-radius:18px;padding:1.4rem 2rem;margin-bottom:1.5rem;
      backdrop-filter:blur(20px);
    ">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem">
        <div>
          <div style="font-size:1.9rem;font-weight:800;letter-spacing:-0.02em;color:#e6edf3">
            ⚡ LayoffGuard <span style="color:#00d4aa">AI</span>
          </div>
          <div style="color:#8b949e;font-size:0.9rem;margin-top:4px">
            Career Risk Intelligence · 6 ML Models + ANN Ensemble · 40+ Real-World Variables
          </div>
        </div>
        <div style="display:flex;gap:1rem;flex-wrap:wrap">
          <div style="background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);
            border-radius:8px;padding:0.4rem 0.9rem;font-size:0.8rem;color:#00d4aa">
            🤖 XGBoost · LightGBM · RF · GBM · LR · ANN
          </div>
          <div style="background:rgba(124,106,247,0.1);border:1px solid rgba(124,106,247,0.3);
            border-radius:8px;padding:0.4rem 0.9rem;font-size:0.8rem;color:#7c6af7">
            📊 6,000 Training Records
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data & models ─────────────────────────────────────
    with st.spinner("🔄 Loading AI models (first run trains them — takes ~2 min)…"):
        (results, feat_cols, scaler, le_dict, df,
         X_te, y_te, X_te_sc, avg_fi, weights) = train_models()

    best_name = max(results, key=lambda k: results[k]["auc"])

    # ── Sidebar input ──────────────────────────────────────────
    inp, predict_btn = render_sidebar()

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Risk Prediction",
        "📊 Model Analytics",
        "🔥 Feature Insights",
        "📈 Dataset Overview",
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — Prediction
    # ══════════════════════════════════════════════════════════
with tab1:
        if predict_btn:
            with st.spinner("🧠 Analyzing risk profile…"):
                prob, model_probs, X_raw, X_sc = compute_risk(
                    inp, results, feat_cols, scaler, le_dict, weights)
                time.sleep(0.4)

            # ── Risk level ────────────────────────────────────
            if prob < 0.25:
                lvl, col, emoji, pop_emoji = "LOW RISK", ACC, "🟢", "🎉🥳✨🙌💪🎊"
            elif prob < 0.50:
                lvl, col, emoji, pop_emoji = "MODERATE RISK", ACC3, "🟡", "⚠️🤔💡📚🔍🛠️"
            elif prob < 0.75:
                lvl, col, emoji, pop_emoji = "HIGH RISK", "#ff8c00", "🟠", "😰⚡🚨📉💸🏃"
            else:
                lvl, col, emoji, pop_emoji = "CRITICAL RISK", ACC2, "🔴", "🚨😱💀⛔🔥🆘"

            # ── Emoji pop-up ──────────────────────────────────
            st.markdown(f"""
            <div style="text-align:center;font-size:3rem;
              animation:popIn 0.5s cubic-bezier(0.34,1.56,0.64,1);margin:0.5rem 0">
              {pop_emoji}
            </div>
            <style>
            @keyframes popIn {{
              0% {{ transform:scale(0.2); opacity:0 }}
              100% {{ transform:scale(1); opacity:1 }}
            }}
            </style>
            """, unsafe_allow_html=True)

            # ── Risk badge ────────────────────────────────────
            st.markdown(f"""
            <div style="text-align:center;margin:0.8rem 0 1.4rem">
              {risk_badge(lvl, col, emoji)}
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + metrics row ───────────────────────────
            gc, mc1, mc2, mc3 = st.columns([1.2, 1, 1, 1])
            with gc:
                gauge_b64 = plot_risk_gauge(prob)
                st.markdown(f"""
                <div style="
                  display:flex;align-items:center;
                  justify-content:center;height:100%;
                  padding-top:0.3rem;
                ">
                  <img src="data:image/png;base64,{gauge_b64}"
                       style="width:100%;max-width:260px;height:auto;
                              display:block;margin:0 auto;"
                       alt="Risk Gauge"/>
                </div>
                """, unsafe_allow_html=True)
            with mc1:
                st.metric("Ensemble Risk", f"{prob*100:.1f}%",
                          delta="▲ Above Safe Zone" if prob > 0.35 else "▼ In Safe Zone")
            with mc2:
                best_prob = model_probs.get(best_name.replace("⭐ ",""), prob)
                st.metric(f"Best Model ({best_name})", f"{best_prob*100:.1f}%")
            with mc3:
                delta_from_avg = prob - 0.42
                st.metric("vs Population Avg",
                          f"{'+' if delta_from_avg>0 else ''}{delta_from_avg*100:.1f}%",
                          delta="Higher than avg" if delta_from_avg>0 else "Lower than avg")

            st.markdown("<hr style='border-color:rgba(0,212,170,0.15)'>", unsafe_allow_html=True)

            # ── Per-model breakdown ───────────────────────────
            st.markdown("#### 🤖 Per-Model Predictions")
            model_cols = st.columns(len(model_probs))
            p_colors = [ACC, ACC3, "#4fc3f7", ACC4, ACC2]
            for col_obj, (mname, mp), pc in zip(model_cols, model_probs.items(), p_colors):
                with col_obj:
                    st.markdown(f"""
                    <div style="
                      background:rgba(15,22,40,0.85);
                      border:1px solid {pc}33;border-top:3px solid {pc};
                      border-radius:12px;padding:0.9rem;text-align:center;
                    ">
                      <div style="font-size:0.72rem;color:#8b949e;margin-bottom:4px">{mname}</div>
                      <div style="font-size:1.5rem;font-weight:700;color:{pc}">{mp*100:.1f}%</div>
                      <div style="font-size:0.7rem;color:#8b949e;margin-top:4px">
                        {"🔴 High" if mp>0.5 else "🟢 Low"}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Radar + Risk bar ──────────────────────────────
            rc1, rc2 = st.columns([1, 1.5])
            with rc1:
                st.markdown("#### 🕸️ Risk Driver Radar")
                st.pyplot(plot_radar(inp, feat_cols, avg_fi),
                          use_container_width=True, clear_figure=True)
            with rc2:
                st.markdown("#### 📊 Model Probability Comparison")
                fig, ax = dark_fig(8, 5)
                names_bar = list(model_probs.keys())
                vals_bar  = [model_probs[n]*100 for n in names_bar]
                b_colors  = [ACC2 if v>50 else ACC3 if v>35 else ACC for v in vals_bar]
                bars = ax.bar(range(len(names_bar)), vals_bar, color=b_colors,
                              edgecolor="none", width=0.55)
                ax.axhline(50, color=ACC2, linewidth=1.2, linestyle="--",
                           alpha=0.6, label="50% threshold")
                ax.set_xticks(range(len(names_bar)))
                ax.set_xticklabels(names_bar, rotation=25, ha="right", fontsize=9)
                ax.set_ylabel("Risk Probability (%)")
                ax.set_title("Your Risk Score Across All Models", fontsize=11)
                ax.set_ylim(0, 105)
                ax.yaxis.grid(True, alpha=0.2)
                ax.set_axisbelow(True)
                ax.legend(facecolor=CARD, edgecolor="#1e2d40", labelcolor=TXT)
                for bar, val in zip(bars, vals_bar):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                            f"{val:.1f}%", ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color=TXT)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True, clear_figure=True)

            # ── Recommendations ───────────────────────────────
            st.markdown("<hr style='border-color:rgba(0,212,170,0.15)'>", unsafe_allow_html=True)
            st.markdown("## 🗺️ Personalized Action Plan")
            recs = get_recs(inp, prob)
            sec_cols = {"🚨 Critical Actions": ACC2,
                        "⚠️ Important Actions": ACC3,
                        "💡 Suggested Improvements": ACC4,
                        "✅ Your Strengths": ACC}

            for section, items in recs.items():
                if not items:
                    continue
                col_s = sec_cols[section]
                st.markdown(f"""
                <div style="color:{col_s};font-size:1rem;font-weight:600;
                  margin:1.2rem 0 0.5rem;letter-spacing:0.02em">{section}
                </div>""", unsafe_allow_html=True)
                for icon, title, detail in items:
                    st.markdown(f"""
                    <div style="
                      background:rgba(15,22,40,0.82);
                      border:1px solid {col_s}33;border-left:3px solid {col_s};
                      border-radius:12px;padding:0.9rem 1.2rem;margin-bottom:0.6rem;
                      backdrop-filter:blur(12px);
                    ">
                      <div style="font-size:0.95rem;font-weight:600;
                        color:{col_s};margin-bottom:3px">
                        {icon} {title}
                      </div>
                      <div style="font-size:0.83rem;color:#8b949e;
                        line-height:1.5">{detail}</div>
                    </div>""", unsafe_allow_html=True)

        else:
            # ── Placeholder ───────────────────────────────────
            st.markdown("""
            <div style="
              text-align:center;padding:4rem 2rem;
              background:rgba(15,22,40,0.6);border:1px dashed rgba(0,212,170,0.3);
              border-radius:20px;backdrop-filter:blur(16px);
            ">
              <div style="font-size:4rem;margin-bottom:1rem">🎯</div>
              <div style="font-size:1.5rem;font-weight:700;color:#00d4aa;margin-bottom:0.6rem">
                Ready to Assess Your Risk
              </div>
              <div style="color:#8b949e;font-size:0.95rem;max-width:480px;
                margin:0 auto;line-height:1.7">
                Fill in your professional details in the sidebar, then hit
                <strong style="color:#00d4aa">⚡ PREDICT LAYOFF RISK</strong> to get
                your AI-powered risk score, feature breakdown, and personalized upskilling roadmap.
              </div>
              <div style="margin-top:2rem;display:flex;justify-content:center;
                gap:2rem;flex-wrap:wrap">
                <div style="color:#8b949e;font-size:0.85rem">🤖 6 ML Models + ANN</div>
                <div style="color:#8b949e;font-size:0.85rem">📊 40+ Variables</div>
                <div style="color:#8b949e;font-size:0.85rem">🗺️ Action Roadmap</div>
                <div style="color:#8b949e;font-size:0.85rem">🎯 Instant Results</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            palette_m = [ACC, ACC3, ACC4, ACC2, "#4fc3f7"]
            for col_obj, (name, res), pc in zip(
                    [c1,c2,c3,c4,c5], list(results.items())[:5], palette_m):
                with col_obj:
                    st.markdown(f"""
                    <div style="
                      background:rgba(15,22,40,0.82);border:1px solid {pc}33;
                      border-top:3px solid {pc};border-radius:12px;
                      padding:0.8rem;text-align:center;
                    ">
                      <div style="font-size:0.7rem;color:#8b949e">{name}</div>
                      <div style="font-size:1.2rem;font-weight:700;color:{pc}">
                        {res['auc']:.4f}
                      </div>
                      <div style="font-size:0.68rem;color:#8b949e">ROC-AUC</div>
                    </div>""", unsafe_allow_html=True)
    # ══════════════════════════════════════════════════════════
    # TAB 2 — Model Analytics
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 📈 Model Performance Dashboard")

        # Summary table
        summary = {n: {"Accuracy":f"{r['acc']:.4f}","F1":f"{r['f1']:.4f}","ROC-AUC":f"{r['auc']:.4f}"}
                   for n,r in results.items()}
        df_summary = pd.DataFrame(summary).T.reset_index().rename(columns={"index":"Model"})
        st.dataframe(df_summary.style.set_properties(**{
            "background-color":"rgba(15,22,40,0.6)",
            "color":"#e6edf3","border":"1px solid rgba(0,212,170,0.15)"
        }), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📉 ROC Curves")
        st.pyplot(plot_roc(results, y_te), use_container_width=True, clear_figure=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏆 Model Comparison")
        st.pyplot(plot_model_comparison(results), use_container_width=True, clear_figure=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔲 Confusion Matrices")
        st.pyplot(plot_confusion_matrix(results), use_container_width=True, clear_figure=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 ANN Training History")
        st.pyplot(plot_ann_history(results["ANN"]["history"]),
                  use_container_width=True, clear_figure=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — Feature Insights
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🔥 Feature Importance (Avg: XGBoost + LightGBM + RF + GBM)")
        st.pyplot(plot_feature_importance(feat_cols, avg_fi),
                  use_container_width=True, clear_figure=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🌡️ Correlation Heatmap")
        st.pyplot(plot_heatmap(df), use_container_width=True, clear_figure=True)

        # Top 10 table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Top 20 Feature Importance Scores")
        fi_df = (pd.Series(avg_fi, index=feat_cols)
                   .sort_values(ascending=False)
                   .head(20)
                   .reset_index())
        fi_df.columns = ["Feature","Importance"]
        fi_df["Importance"] = fi_df["Importance"].round(6)
        fi_df["Rank"] = range(1, len(fi_df)+1)
        fi_df = fi_df[["Rank","Feature","Importance"]]
        st.dataframe(fi_df.style.set_properties(**{
            "background-color":"rgba(15,22,40,0.6)",
            "color":"#e6edf3","border":"1px solid rgba(0,212,170,0.15)"
        }), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — Dataset Overview
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 📊 Dataset Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Records", f"{len(df):,}")
        with c2: st.metric("Features", len(feat_cols))
        with c3: st.metric("Layoff Rate", f"{df['layoff_risk'].mean()*100:.1f}%")
        with c4: st.metric("Avg Risk Score", f"{df['layoff_probability'].mean()*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📉 Risk Distribution Analysis")
        st.pyplot(plot_risk_dist(df), use_container_width=True, clear_figure=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Sample Data Preview")
        st.dataframe(df.head(50).style.set_properties(**{
            "background-color":"rgba(15,22,40,0.6)",
            "color":"#e6edf3","border":"1px solid rgba(0,212,170,0.1)"
        }), use_container_width=True)

        # Describe
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📐 Descriptive Statistics")
        st.dataframe(df.describe().round(2).style.set_properties(**{
            "background-color":"rgba(15,22,40,0.6)",
            "color":"#e6edf3","border":"1px solid rgba(0,212,170,0.1)"
        }), use_container_width=True)

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <div style="
      text-align:center;padding:1.5rem;margin-top:2rem;
      border-top:1px solid rgba(0,212,170,0.12);
      color:#4a5568;font-size:0.78rem;
    ">
      ⚡ LayoffGuard AI · Built with Streamlit · XGBoost · LightGBM · TensorFlow · scikit-learn<br>
      <span style="color:rgba(0,212,170,0.4)">Synthetic data for demonstration. Not financial or career advice.</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
