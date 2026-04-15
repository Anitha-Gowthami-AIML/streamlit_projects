"""
Unsupervised Machine Learning Model Dashboard — Anime Recommendations & Student Clustering
Compatible with: Streamlit 1.56.0, pandas 3.x, scikit-learn 1.8, matplotlib 3.10
"""

import os
import io
import re
import random
import warnings
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE                        # top-level, never inside a function
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, mean_squared_error)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

try:
    from kneed import KneeLocator
    KNEED_OK = True
except ImportError:
    KNEED_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Unsupervised models Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#fff0f3 0%,#ffe4e1 30%,#ffd6cc 60%,#fff5f0 100%);
    background-attachment: fixed;
}
.stApp::before {
    content:""; position:fixed; top:0; left:0; right:0; bottom:0;
    background-image:
        radial-gradient(circle,rgba(255,182,193,.18) 1px,transparent 1px),
        radial-gradient(circle,rgba(221,160,221,.12) 1px,transparent 1px),
        radial-gradient(circle,rgba(255,218,185,.15) 1px,transparent 1px);
    background-size:28px 28px,19px 19px,37px 37px;
    background-position:0 0,9px 9px,5px 20px;
    pointer-events:none; z-index:0;
}
.block-container{position:relative;z-index:1;}
.stTabs [data-baseweb="tab-list"]{
    background:rgba(255,255,255,.55);border-radius:18px;padding:6px 10px;
    backdrop-filter:blur(8px);box-shadow:0 4px 18px rgba(231,84,128,.10);gap:8px;
}
.stTabs [data-baseweb="tab"]{
    border-radius:12px;font-weight:600;font-size:.97rem;
    color:#8B3A62 !important;padding:8px 22px;transition:all .2s;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(90deg,#ffb3c6,#c7ceea) !important;
    color:#4a0030 !important;box-shadow:0 2px 10px rgba(231,84,128,.18);
}
.metric-card{
    background:rgba(255,255,255,.72);border-radius:16px;padding:16px 20px;
    box-shadow:0 4px 16px rgba(231,84,128,.10);border:1px solid rgba(255,182,193,.35);
    backdrop-filter:blur(6px);text-align:center;margin-bottom:10px;
}
.metric-card h3{color:#8B3A62;font-size:1.5rem;margin:0;}
.metric-card p{color:#5a2040;font-size:.85rem;margin:4px 0 0;}
.section-title{
    background:linear-gradient(90deg,#e75480,#c9a0dc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:1.5rem;font-weight:800;margin:1.2rem 0 .5rem;
}
.explain-box{
    background:rgba(255,255,255,.65);border-left:4px solid #e75480;
    border-radius:10px;padding:14px 18px;margin:10px 0 18px;
    font-size:.93rem;color:#4a2040;backdrop-filter:blur(4px);
    box-shadow:0 2px 10px rgba(231,84,128,.07);
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PASTEL = ['#FFB3C6','#B5EAD7','#C7CEEA','#FFDAC1','#FF9AA2',
          '#A8D8EA','#E2F0CB','#D4A5A5','#957DAD','#F3B0C3',
          '#FFD1DC','#AEC6CF','#FFEAA7','#DDA0DD','#98D8C8']
FIG_BG = '#FFF5F2'

# ── Helper functions ──────────────────────────────────────────────────────────
def ax_style(ax):
    """Apply consistent pastel style to a matplotlib Axes."""
    ax.set_facecolor('#FFF8F6')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.28)


def show_fig(fig):
    """Save a matplotlib figure to a PNG buffer and display it full-width.
    width='stretch' is the correct value in Streamlit 1.56+."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=FIG_BG)
    buf.seek(0)
    st.image(buf, width='stretch')   # 'stretch' fills the container in Streamlit 1.56
    plt.close(fig)


def wc_color(word, font_size, position, orientation, random_state=None, **kwargs):
    """Random pastel colour picker for word clouds."""
    colors = ['#E75480','#C9A0DC','#FF9AA2','#F4A460','#6A9FB5',
              '#85C1E9','#F9A8D4','#F1948A','#BB8FCE','#76D7C4',
              '#FFB347','#77DD77','#AEC6CF','#DEB887','#B0C4DE']
    return random.choice(colors)


def clean_string_cols(df):
    """Convert any pandas StringDtype columns to plain object dtype.
    Pandas 3.x with PyArrow loads CSV string columns as StringDtype.
    Explicitly convert them so .map(), .fillna() etc. behave consistently."""
    # Use 'str' not 'string' — 'string' triggers a pandas 3.x deprecation
    str_cols = df.select_dtypes(include=['str', 'string']).columns.tolist()
    if str_cols:
        df[str_cols] = df[str_cols].astype(object)
    return df


# ── File finding ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(name):
    for candidate in [os.path.join(DATA_DIR, name),
                      os.path.join('/mnt/user-data/uploads', name),
                      name]:
        if os.path.exists(candidate):
            return candidate
    return None


# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_anime():
    path = find_file('anime.csv')
    if path is None:
        st.error("anime.csv not found — place it alongside app.py.")
        st.stop()
    df = pd.read_csv(path)
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')
    return df


@st.cache_data(show_spinner=False)
def load_students():
    path = find_file('03_Clustering_Marketing.csv')
    if path is None:
        st.error("03_Clustering_Marketing.csv not found.")
        st.stop()
    return pd.read_csv(path)


# ═════════════════════════════════════════════════════════════════════════════
# CLUSTERING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_clustering(_df_raw):
    df = _df_raw.drop_duplicates().copy()

    # ── pandas 3.x: convert StringDtype cols to plain object ──────────────────
    df = clean_string_cols(df)

    # ── Age: strip stray letters using vectorised .str (ArrowDtype-safe) ──────
    df['age'] = pd.to_numeric(
        df['age'].astype(str).str.replace(r'[A-Za-z]', '', regex=True).str.strip(),
        errors='coerce'
    ).clip(13, 22)

    # ── Gender: encode as integer ─────────────────────────────────────────────
    df['gender'] = df['gender'].astype(object).map({'M': 1, 'F': 0}).fillna(-1)

    # ── Impute numeric nulls with median ──────────────────────────────────────
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # ── log1p on highly skewed columns ────────────────────────────────────────
    skewness = df[num_cols].skew(numeric_only=True)
    df_t = df.copy()
    for col in skewness[skewness.abs() > 1.0].index:
        if (df_t[col] >= 0).all():
            df_t[col] = np.log1p(df_t[col])

    # ── IQR outlier capping ───────────────────────────────────────────────────
    for col in num_cols:
        q1, q3 = df_t[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df_t[col] = df_t[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)

    # ── Scale ─────────────────────────────────────────────────────────────────
    X_scaled = RobustScaler().fit_transform(df_t[num_cols])

    # ── PCA: keep enough components for 90% cumulative variance ───────────────
    pca_full = PCA(random_state=42).fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cumvar >= 0.90)) + 1
    X_pca  = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)

    # ── K-Means sweep ─────────────────────────────────────────────────────────
    K_range = list(range(2, 11))
    inertia_list, sil_km = [], []
    for k in K_range:
        km  = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=42)
        lbl = km.fit_predict(X_pca)
        inertia_list.append(km.inertia_)
        sil_km.append(silhouette_score(X_pca, lbl))

    elbow_k = 5
    if KNEED_OK:
        kl = KneeLocator(K_range, inertia_list, curve='convex', direction='decreasing')
        elbow_k = kl.knee or 5

    best_k    = K_range[int(np.argmax(sil_km))]
    km_labels = KMeans(n_clusters=best_k, init='k-means++',
                       n_init=20, random_state=42).fit_predict(X_pca)
    km_sil = silhouette_score(X_pca, km_labels)
    km_db  = davies_bouldin_score(X_pca, km_labels)
    km_ch  = calinski_harabasz_score(X_pca, km_labels)

    # ── Hierarchical sweep ────────────────────────────────────────────────────
    sil_hier = []
    for k in K_range:
        lbl = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_pca)
        sil_hier.append(silhouette_score(X_pca, lbl))

    best_k_ag   = K_range[int(np.argmax(sil_hier))]
    hier_labels = AgglomerativeClustering(
        n_clusters=best_k_ag, linkage='ward').fit_predict(X_pca)
    hier_sil = silhouette_score(X_pca, hier_labels)
    hier_db  = davies_bouldin_score(X_pca, hier_labels)
    hier_ch  = calinski_harabasz_score(X_pca, hier_labels)

    # ── DBSCAN: k-distance knee + grid search ─────────────────────────────────
    nn = NearestNeighbors(n_neighbors=5).fit(X_pca)
    dist, _ = nn.kneighbors(X_pca)
    k_dist = np.sort(dist[:, 4])[::-1]
    knee_idx = len(k_dist) // 10
    if KNEED_OK:
        kl2 = KneeLocator(range(len(k_dist)), k_dist,
                          curve='convex', direction='decreasing')
        knee_idx = kl2.knee if kl2.knee else knee_idx
    eps_knee = round(float(k_dist[knee_idx]), 3)
    eps_cands = np.linspace(max(0.1, eps_knee * 0.5), eps_knee * 2.0, 8).round(3)

    best_sil_db, best_db_labels = -1, None
    for eps in eps_cands:
        for ms in [3, 5, 7, 10]:
            lbl  = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_pca)
            n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
            mask = lbl != -1
            if n_cl >= 2 and mask.sum() > 100:
                s = silhouette_score(X_pca[mask], lbl[mask])
                if s > best_sil_db:
                    best_sil_db, best_db_labels = s, lbl.copy()

    if best_db_labels is None:
        best_db_labels = km_labels.copy()

    mask_db    = best_db_labels != -1
    n_cl_db    = len(set(best_db_labels)) - (1 if -1 in best_db_labels else 0)
    dbscan_sil = silhouette_score(X_pca[mask_db], best_db_labels[mask_db])
    dbscan_db  = davies_bouldin_score(X_pca[mask_db], best_db_labels[mask_db])
    dbscan_ch  = calinski_harabasz_score(X_pca[mask_db], best_db_labels[mask_db])

    # ── t-SNE: reduce sample to 1000 pts — fine visually, saves ~3× memory/time ─
    # sklearn 1.5+: parameter renamed from n_iter → max_iter
    n_sample   = min(1000, len(X_pca))          # was 3000; 1000 is plenty for viz
    idx_sample = np.random.choice(len(X_pca), n_sample, replace=False)
    safe_perp  = min(30, n_sample - 1)          # perplexity must be < n_sample
    X_tsne     = TSNE(
        n_components=2, perplexity=safe_perp,
        max_iter=300,           # was 500; 300 is enough for cluster separation
        random_state=42
    ).fit_transform(X_pca[idx_sample])

    return dict(
        X_pca=X_pca, X_tsne=X_tsne, idx_sample=idx_sample,
        km_labels=km_labels, hier_labels=hier_labels, db_labels=best_db_labels,
        best_k=best_k, best_k_ag=best_k_ag, n_cl_db=n_cl_db,
        km_sil=km_sil,     hier_sil=hier_sil,     dbscan_sil=dbscan_sil,
        km_db=km_db,       hier_db=hier_db,        dbscan_db=dbscan_db,
        km_ch=km_ch,       hier_ch=hier_ch,         dbscan_ch=dbscan_ch,
        K_range=K_range,   inertia=inertia_list,   sil_km=sil_km,
        sil_hier=sil_hier, elbow_k=elbow_k,
        pca_full=pca_full, n_comp=n_comp,
        df_clean=df,       num_cols=num_cols,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ANIME PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_anime_pipeline(_anime):
    # ── Content-based: TF-IDF on genres ───────────────────────────────────────
    # MEMORY FIX: Do NOT build the full 12294×12294 cosine similarity matrix
    # (that is 1.2 GB and kills the free-tier container with an OOM / EOF).
    # Instead we store only the sparse TF-IDF matrix (~KB) and compute
    # cosine similarity ON DEMAND for a single query row (one vector vs all).
    anime_cb = _anime.dropna(subset=['genre']).copy().reset_index(drop=True)
    anime_cb['genre_clean'] = (anime_cb['genre']
                               .str.replace(', ', ' ')
                               .str.replace('-', ''))
    tfidf_vec = TfidfVectorizer(token_pattern=r"[a-zA-Z0-9]+")
    tfidf_mat = tfidf_vec.fit_transform(anime_cb['genre_clean'])
    # tfidf_mat is a sparse CSR matrix — very small (~400 KB for 12k anime × 40 genres)
    anime_idx = pd.Series(anime_cb.index, index=anime_cb['name'])

    # ── Collaborative SVD ─────────────────────────────────────────────────────
    # MEMORY FIX: cap at 200 users × 300 items (was 300 × 500 = 150k floats → fine,
    # but reducing further speeds up the sweep and stays well under 1 GB total).
    np.random.seed(42)
    top_names    = _anime.nlargest(300, 'members')['name'].values   # was 500
    n_users      = 200                                               # was 300
    n_items      = len(top_names)
    UI           = np.zeros((n_users, n_items))
    base_ratings = (_anime.set_index('name')
                    .loc[top_names, 'rating']
                    .fillna(7.0).values)
    for u in range(n_users):
        n_rated = np.random.randint(10, 60)
        items   = np.random.choice(n_items, n_rated, replace=False)
        UI[u, items] = np.clip(
            base_ratings[items] + np.random.randn(n_rated) * 1.5, 1, 10)

    rmse_dict = {}
    for k in [5, 10, 20]:          # dropped k=50 — saves time & memory
        sv   = TruncatedSVD(n_components=k, random_state=42)
        U_   = sv.fit_transform(UI)
        R_   = np.dot(U_, sv.components_)
        mask = UI > 0
        rmse_dict[k] = round(
            float(np.sqrt(mean_squared_error(UI[mask], R_[mask]))), 4)

    best_k_svd = min(rmse_dict, key=rmse_dict.get)
    svd_final  = TruncatedSVD(n_components=best_k_svd, random_state=42)
    R_final    = np.dot(svd_final.fit_transform(UI), svd_final.components_)
    rec_df     = pd.DataFrame(R_final, columns=top_names)

    svd_ev = TruncatedSVD(n_components=min(30, n_items - 1), random_state=42)
    svd_ev.fit(UI)

    return dict(
        anime_cb=anime_cb,
        tfidf_mat=tfidf_mat,    # sparse matrix — tiny; used for on-demand sim
        anime_idx=anime_idx,
        top_names=top_names, UI=UI, rec_df=rec_df,
        svd_final=svd_final, rmse_dict=rmse_dict, best_k_svd=best_k_svd,
        svd_ev=svd_ev,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:28px 0 10px;">
  <span style="font-size:2.8rem;">🌸</span>
  <h1 style="font-size:2.4rem;font-weight:900;
             background:linear-gradient(90deg,#e75480,#c9a0dc,#6a9fb5);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;">
    Unsupervised Machine Learning Model - Dashboard
  </h1>
  <p style="color:#8B3A62;font-size:1.05rem;margin-top:6px;">
    Student Clustering &nbsp;|&nbsp; Anime Recommendation System
    &nbsp;|&nbsp; SVD · KMeans · Hierarchical · DBSCAN
  </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎓 Student Clustering",
                       "🎌 Anime Recommendations"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — STUDENT CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Student Social Network Profile Clustering</p>',
                unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    Clusters <b>15,000 high school students</b> using word-count features from their social
    network profiles (2006–2009). We compare K-Means, Hierarchical (Ward), and DBSCAN after
    PCA reduction, measuring quality with Silhouette, Davies-Bouldin, and Calinski-Harabasz.
    </div>""", unsafe_allow_html=True)

    with st.spinner("🌸 Running clustering pipeline — ~60 s the first time…"):
        df_raw = load_students()
        clust  = run_clustering(df_raw)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("#### 📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    for col, label, sil, db, ch, k in [
        (c1, "K-Means",      clust['km_sil'],     clust['km_db'],     clust['km_ch'],     clust['best_k']),
        (c2, "Hierarchical", clust['hier_sil'],   clust['hier_db'],   clust['hier_ch'],   clust['best_k_ag']),
        (c3, "DBSCAN",       clust['dbscan_sil'], clust['dbscan_db'], clust['dbscan_ch'], clust['n_cl_db']),
    ]:
        col.markdown(f"""<div class="metric-card">
          <p><b>{label}</b>&nbsp;(K={k})</p>
          <h3>{sil:.4f}</h3><p>Silhouette ↑</p>
          <p>Davies-Bouldin: <b>{db:.4f}</b> ↓&nbsp;|&nbsp;CH: <b>{ch:.1f}</b> ↑</p>
        </div>""", unsafe_allow_html=True)

    # ── EDA ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Exploratory Data Analysis")
    st.markdown("""<div class="explain-box">
    Gender and graduation-year distributions reveal the demographic mix. "Music," "sports," and
    "dance" dominate word-feature means — these naturally become the axes the clusters form around.
    The correlation heatmap shows co-occurring interests (e.g. church ↔ bible ↔ jesus).
    </div>""", unsafe_allow_html=True)

    df_viz    = df_raw.drop_duplicates().copy()
    df_viz    = clean_string_cols(df_viz)
    word_cols = [c for c in df_viz.columns
                 if c not in ['gradyear','gender','age','NumberOffriends']]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(FIG_BG)

    gd = df_viz['gender'].value_counts()
    axes[0,0].bar(gd.index, gd.values,
                  color=['#FFB3C6','#B5EAD7','#C7CEEA'], edgecolor='white', lw=1.5)
    axes[0,0].set_title("Gender Distribution", fontweight='bold')

    df_viz['gradyear'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0,1], color=PASTEL[:4], edgecolor='white', lw=1.5)
    axes[0,1].set_title("Students per Graduation Year", fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=0)

    age_c = pd.to_numeric(
        df_viz['age'].astype(str).str.replace(r'[A-Za-z]','',regex=True).str.strip(),
        errors='coerce').dropna()
    age_c = age_c[(age_c >= 13) & (age_c <= 22)]
    axes[0,2].hist(age_c, bins=18, color='#FFDAC1', edgecolor='white', lw=1.2)
    axes[0,2].set_title("Age Distribution (13–22)", fontweight='bold')

    axes[1,0].hist(pd.to_numeric(df_viz['NumberOffriends'], errors='coerce').clip(upper=200),
                   bins=30, color='#C7CEEA', edgecolor='white', lw=1.2)
    axes[1,0].set_title("Friends Count (clipped @200)", fontweight='bold')

    wmeans = df_viz[word_cols].apply(pd.to_numeric, errors='coerce').mean()
    wmeans = wmeans.sort_values(ascending=False).head(15)
    axes[1,1].barh(wmeans.index[::-1], wmeans.values[::-1],
                   color=PASTEL[:15], edgecolor='white')
    axes[1,1].set_title("Top 15 Word Features (Mean)", fontweight='bold')

    top12 = wmeans.head(12).index.tolist()
    num_top12 = df_viz[top12].apply(pd.to_numeric, errors='coerce')
    sns.heatmap(num_top12.corr(), ax=axes[1,2], cmap='RdPu',
                linewidths=0.5, cbar_kws={'shrink':0.7}, annot=False)
    axes[1,2].set_title("Correlation — Top 12 Words", fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor('#FFF8F6')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    show_fig(fig)

    # ── Word Cloud ────────────────────────────────────────────────────────────
    st.markdown("### 🌸 Student Profile Keywords — Word Cloud")
    st.markdown("""<div class="explain-box">
    Word size = total mention count across all 15,000 profiles. Music, sports, and dance dominate,
    hinting at the major lifestyle segments the algorithms surface.
    </div>""", unsafe_allow_html=True)

    word_freq = (df_raw[word_cols]
                 .apply(pd.to_numeric, errors='coerce')
                 .sum().to_dict())
    wc_obj = WordCloud(
        width=900, height=340, background_color='#FFF0F3',
        max_words=60, color_func=wc_color,
        prefer_horizontal=0.85, collocations=False, min_font_size=10
    ).generate_from_frequencies(word_freq)
    fig_wc, ax_wc = plt.subplots(figsize=(13, 4.5))
    fig_wc.patch.set_facecolor('#FFF0F3')
    ax_wc.imshow(wc_obj, interpolation='bilinear')
    ax_wc.axis('off')
    plt.tight_layout()
    show_fig(fig_wc)

    # ── PCA ───────────────────────────────────────────────────────────────────
    st.markdown("### 📐 PCA — Dimensionality Reduction")
    st.markdown("""<div class="explain-box">
    PCA rotates the feature space to maximise variance along each new axis. We keep only
    the components needed for <b>90% cumulative variance</b>, discarding the rest as noise.
    </div>""", unsafe_allow_html=True)

    pca_full = clust['pca_full']
    n_comp   = clust['n_comp']
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    K_all    = range(1, len(cumvar) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(FIG_BG)
    axes[0].bar(K_all, pca_full.explained_variance_ratio_ * 100,
                color=PASTEL[0], edgecolor='white', alpha=0.85)
    axes[0].axvline(n_comp, color='#8B3A62', ls='--', lw=2,
                    label=f'90% at PC={n_comp}')
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Variance (%)")
    axes[0].set_title("Scree Plot", fontweight='bold')
    axes[0].legend()

    axes[1].plot(K_all, cumvar * 100, 'o-', color='#E75480', lw=2, ms=4)
    axes[1].fill_between(K_all, cumvar * 100, alpha=0.15, color='#E75480')
    axes[1].axhline(90, color='#6A5ACD', ls='--', lw=1.8, label='90% variance')
    axes[1].axvline(n_comp, color='#8B3A62', ls='--', lw=1.5)
    axes[1].set_xlabel("Number of PCs")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Explained Variance", fontweight='bold')
    axes[1].legend()

    for ax in axes:
        ax_style(ax)
    plt.tight_layout()
    show_fig(fig)

    # ── K-Means Elbow + Silhouette ────────────────────────────────────────────
    st.markdown("### 🔵 K-Means — Elbow & Silhouette Sweep")
    st.markdown("""<div class="explain-box">
    The <b>Elbow Method</b> finds where adding more clusters gives diminishing WCSS reduction.
    The <b>Silhouette Score</b> directly measures cluster separation (higher = better, max 1.0).
    We choose K based on the highest silhouette.
    </div>""", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(FIG_BG)
    axes[0].plot(clust['K_range'], clust['inertia'],
                 'o-', color='#E75480', lw=2, ms=7)
    axes[0].axvline(clust['elbow_k'], color='#8B3A62', ls='--', lw=2,
                    label=f"Elbow K={clust['elbow_k']}")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow Method", fontweight='bold'); axes[0].legend()

    axes[1].plot(clust['K_range'], clust['sil_km'],
                 's-', color='#6A5ACD', lw=2, ms=7)
    axes[1].axvline(clust['best_k'], color='#8B3A62', ls='--', lw=2,
                    label=f"Best K={clust['best_k']}")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette vs K", fontweight='bold'); axes[1].legend()

    for ax in axes:
        ax_style(ax)
    plt.tight_layout()
    show_fig(fig)

    # ── Hierarchical + Dendrogram ─────────────────────────────────────────────
    st.markdown("### 🌿 Hierarchical Clustering — Silhouette + Dendrogram")
    st.markdown("""<div class="explain-box">
    Ward linkage minimises the increase in total within-cluster variance at each merge.
    The dendrogram (300-point sample) shows the tree of merges — long vertical lines are
    natural breakpoints where the algorithm should stop.
    </div>""", unsafe_allow_html=True)

    sample_d = np.random.choice(len(clust['X_pca']), 300, replace=False)
    Z        = linkage(clust['X_pca'][sample_d], method='ward')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(FIG_BG)

    axes[0].plot(clust['K_range'], clust['sil_hier'],
                 'd-', color='#E75480', lw=2, ms=7)
    axes[0].axvline(clust['best_k_ag'], color='#8B3A62', ls='--', lw=2,
                    label=f"Best K={clust['best_k_ag']}")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Hierarchical — Silhouette vs K", fontweight='bold')
    axes[0].legend()
    ax_style(axes[0])

    dendrogram(Z, ax=axes[1], truncate_mode='lastp', p=15,
               color_threshold=0, above_threshold_color='#C7CEEA',
               leaf_font_size=8)
    axes[1].set_title("Dendrogram (300-sample, Ward)", fontweight='bold')
    axes[1].set_facecolor('#FFF8F6')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    show_fig(fig)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("### 🏆 All Three Methods — Side-by-Side Comparison")
    st.markdown("""<div class="explain-box">
    <b>Silhouette ↑</b>: similarity of a point to its own cluster vs neighbours (0–1, higher better).
    <b>Davies-Bouldin ↓</b>: avg ratio of within-cluster scatter to between-cluster separation.
    <b>Calinski-Harabasz ↑</b>: ratio of between- to within-cluster dispersion.
    </div>""", unsafe_allow_html=True)

    results_df = pd.DataFrame({
        'Method'             : ['K-Means', 'Hierarchical', 'DBSCAN'],
        'Optimal K'          : [clust['best_k'],  clust['best_k_ag'],  clust['n_cl_db']],
        'Silhouette ↑'       : [round(clust['km_sil'],4),    round(clust['hier_sil'],4),    round(clust['dbscan_sil'],4)],
        'Davies-Bouldin ↓'   : [round(clust['km_db'],4),     round(clust['hier_db'],4),     round(clust['dbscan_db'],4)],
        'Calinski-Harabasz ↑': [round(clust['km_ch'],1),     round(clust['hier_ch'],1),     round(clust['dbscan_ch'],1)],
    })
    # st.dataframe default width='stretch' in Streamlit 1.56, no extra arg needed
    st.dataframe(results_df.set_index('Method'))

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(FIG_BG)
    bars = ax.bar(['K-Means','Hierarchical','DBSCAN'],
                  results_df['Silhouette ↑'],
                  color=PASTEL[:3], edgecolor='white', lw=1.5, width=0.5)
    for bar, val in zip(bars, results_df['Silhouette ↑']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.4f}", ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel("Silhouette Score (higher = better)")
    ax.set_title("Clustering Silhouette Comparison", fontweight='bold', fontsize=13)
    ax.set_ylim(0, 0.75)
    ax_style(ax)
    plt.tight_layout()
    show_fig(fig)

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    st.markdown("### 🔮 t-SNE 2D Cluster Visualisation")
    st.markdown("""<div class="explain-box">
    t-SNE projects the PCA space into 2D while preserving local neighbourhood structure,
    revealing non-linear cluster shapes. Tight, distinct blobs = good separation.
    (Plotted on up to 3,000 sampled points for speed.)
    </div>""", unsafe_allow_html=True)

    X_tsne     = clust['X_tsne']
    idx_sample = clust['idx_sample']
    # matplotlib 3.7+: use plt.colormaps instead of the old plt.cm accessor
    cmap_p     = plt.colormaps.get_cmap('Pastel1').resampled(10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(FIG_BG)
    label_sets = [
        (clust['km_labels'][idx_sample],   f"K-Means (K={clust['best_k']})"),
        (clust['hier_labels'][idx_sample],  f"Hierarchical (K={clust['best_k_ag']})"),
        (clust['db_labels'][idx_sample],    f"DBSCAN (K={clust['n_cl_db']})"),
    ]
    for ax, (lbl, title) in zip(axes, label_sets):
        for i, cl in enumerate(sorted(set(lbl))):
            mask  = lbl == cl
            label = "Noise" if cl == -1 else f"Cluster {cl}"
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=[cmap_p(i)], s=6, alpha=0.75, label=label)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_facecolor('#FFF8F6')
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(set(lbl)) <= 8:
            ax.legend(markerscale=4, fontsize=8, loc='best')
    plt.tight_layout()
    show_fig(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANIME RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Anime Recommendation System</p>',
                unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    Two complementary recommenders on <b>12,294 anime</b>. (1) <b>Content-Based</b>: TF-IDF on
    genres + cosine similarity. (2) <b>Collaborative SVD</b>: decomposes a user-item rating
    matrix into latent preference factors — the same idea powering Netflix recommendations.
    RMSE measures how accurately SVD reconstructs known ratings.
    </div>""", unsafe_allow_html=True)

    with st.spinner("🌸 Running anime pipeline…"):
        anime = load_anime()
        ap    = run_anime_pipeline(anime)

    # ── EDA ───────────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Anime Dataset EDA")
    st.markdown("""<div class="explain-box">
    Ratings cluster between 6–8, TV series dominate, and members follow a power-law distribution.
    A few blockbusters have millions of members while most have only hundreds — hence we
    log-transform members before any distance computation.
    </div>""", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.patch.set_facecolor(FIG_BG)

    axes[0,0].hist(anime['rating'].dropna(), bins=28,
                   color='#FFB3C6', edgecolor='white', lw=1.2)
    axes[0,0].set_title("Rating Distribution", fontweight='bold')
    axes[0,0].set_xlabel("Rating")

    tc = anime['type'].value_counts()
    axes[0,1].bar(tc.index, tc.values,
                  color=PASTEL[:len(tc)], edgecolor='white', lw=1.5)
    axes[0,1].set_title("Anime Type Distribution", fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=20)

    axes[0,2].hist(np.log1p(anime['members']), bins=28,
                   color='#C7CEEA', edgecolor='white', lw=1.2)
    axes[0,2].set_title("Member Count (log scale)", fontweight='bold')
    axes[0,2].set_xlabel("log(1 + members)")

    top15r = anime.nlargest(15, 'rating')[['name','rating']].reset_index(drop=True)
    axes[1,0].barh(top15r['name'][::-1], top15r['rating'][::-1],
                   color=PASTEL[:15], edgecolor='white')
    axes[1,0].set_title("Top 15 Highest Rated Anime", fontweight='bold')
    axes[1,0].tick_params(axis='y', labelsize=7.5)

    top15m = anime.nlargest(15, 'members')[['name','members']].reset_index(drop=True)
    axes[1,1].barh(top15m['name'][::-1], top15m['members'][::-1],
                   color=PASTEL[:15], edgecolor='white')
    axes[1,1].set_title("Top 15 Most Popular (Members)", fontweight='bold')
    axes[1,1].tick_params(axis='y', labelsize=7.5)

    sc = axes[1,2].scatter(np.log1p(anime['members']), anime['rating'],
                            c=anime['rating'], cmap='RdPu', s=7, alpha=0.55)
    plt.colorbar(sc, ax=axes[1,2], label='Rating')
    axes[1,2].set_title("Rating vs Popularity (log)", fontweight='bold')
    axes[1,2].set_xlabel("log(1+members)")
    axes[1,2].set_ylabel("Rating")

    for ax in axes.flat:
        ax.set_facecolor('#FFF8F6')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    show_fig(fig)

    # ── Genre Word Cloud ──────────────────────────────────────────────────────
    st.markdown("### 🎌 Genre Word Cloud")
    st.markdown("""<div class="explain-box">
    Word size = number of anime in that genre. Comedy, Action, and Adventure dominate.
    Niche genres like Josei or Dementia create strong signals for fans who enjoy them.
    </div>""", unsafe_allow_html=True)

    all_genres  = []
    for g in anime['genre'].dropna():
        all_genres.extend([x.strip() for x in g.split(',')])
    genre_freq2 = Counter(all_genres)

    wc2 = WordCloud(
        width=980, height=380, background_color='#FFF5F0',
        max_words=80, color_func=wc_color,
        prefer_horizontal=0.85, collocations=False, min_font_size=9
    ).generate_from_frequencies(genre_freq2)
    fig_wc2, ax_wc2 = plt.subplots(figsize=(13, 4.5))
    fig_wc2.patch.set_facecolor('#FFF5F0')
    ax_wc2.imshow(wc2, interpolation='bilinear')
    ax_wc2.axis('off')
    plt.tight_layout()
    show_fig(fig_wc2)

    # ── Skewness ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 Skewness & log1p Transformation")
    st.markdown("""<div class="explain-box">
    Members and episodes are extremely right-skewed. log(1+x) compresses the long tail so
    distance metrics work proportionally. Rating is already near-Gaussian.
    </div>""", unsafe_allow_html=True)

    anime_num = anime[['rating','members','episodes']].copy()
    anime_t   = anime_num.copy()
    for col in ['members','episodes']:
        anime_t[col] = np.log1p(anime_t[col].fillna(0))
    skew_b = anime_num.skew(numeric_only=True)
    skew_a = anime_t.skew(numeric_only=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor(FIG_BG)
    for i, col in enumerate(['rating','members','episodes']):
        axes[0,i].hist(anime_num[col].dropna(), bins=28,
                       color=PASTEL[i], edgecolor='white')
        axes[0,i].set_title(f"{col} Before (skew={skew_b[col]:.2f})", fontweight='bold')
        axes[1,i].hist(anime_t[col].dropna(), bins=28,
                       color=PASTEL[i+3], edgecolor='white')
        axes[1,i].set_title(f"{col} After log1p (skew={skew_a[col]:.2f})", fontweight='bold')
    for ax in axes.flat:
        ax.set_facecolor('#FFF8F6')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    show_fig(fig)

    # ── SVD Accuracy ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 SVD — Latent Factor Model Accuracy")
    st.markdown("""<div class="explain-box">
    SVD factorises the user-item matrix into U·Σ·Vᵀ, capturing hidden taste patterns.
    We sweep k from 5–50 and measure RMSE on observed ratings. Lower RMSE = better
    reconstruction. The right chart shows cumulative variance captured per additional factor.
    </div>""", unsafe_allow_html=True)

    rmse_dict  = ap['rmse_dict']
    best_k_svd = ap['best_k_svd']
    cum_ev     = np.cumsum(ap['svd_ev'].explained_variance_ratio_) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(FIG_BG)

    axes[0].plot(list(rmse_dict.keys()), list(rmse_dict.values()),
                 'o-', color='#E75480', lw=2.5, ms=8)
    axes[0].axvline(best_k_svd, color='#8B3A62', ls='--', lw=2,
                    label=f'Best k={best_k_svd}  RMSE={rmse_dict[best_k_svd]:.4f}')
    axes[0].set_xlabel("Latent Factors (k)")
    axes[0].set_ylabel("RMSE on Observed Ratings")
    axes[0].set_title("SVD RMSE vs Latent Factors", fontweight='bold')
    axes[0].legend()

    ev_range = range(1, len(cum_ev) + 1)
    axes[1].plot(ev_range, cum_ev, 'o-', color='#6A5ACD', lw=2, ms=4)
    axes[1].fill_between(ev_range, cum_ev, alpha=0.15, color='#6A5ACD')
    axes[1].axhline(80, color='#E75480', ls='--', lw=1.5, label='80%')
    axes[1].axhline(90, color='#8B3A62', ls='--', lw=1.5, label='90%')
    axes[1].set_xlabel("SVD Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("SVD Cumulative Explained Variance", fontweight='bold')
    axes[1].legend()

    for ax in axes:
        ax_style(ax)
    plt.tight_layout()
    show_fig(fig)

    # ── Top genres bar ────────────────────────────────────────────────────────
    st.markdown("### 🎭 Top 20 Anime Genres")
    genre_series = pd.Series(genre_freq2).sort_values(ascending=False)[:20]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.barh(genre_series.index[::-1], genre_series.values[::-1],
            color=PASTEL[:20], edgecolor='white', lw=1.2)
    ax.set_title("Top 20 Anime Genres by Frequency", fontweight='bold', fontsize=13)
    ax.set_xlabel("Number of Anime")
    ax_style(ax)
    plt.tight_layout()
    show_fig(fig)

    # ── Interactive recommenders ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔎 Try the Recommenders")
    col_cb, col_svd = st.columns(2)

    with col_cb:
        st.markdown("#### 📚 Content-Based (Genre TF-IDF)")
        all_titles = sorted(ap['anime_cb']['name'].tolist())
        default    = ('Sword Art Online' if 'Sword Art Online' in all_titles
                      else all_titles[0])
        chosen = st.selectbox("Pick an anime:", all_titles,
                              index=all_titles.index(default), key="cb_sel")
        n_cb   = st.slider("# Recommendations:", 5, 20, 10, key="cb_n")

        if st.button("🌸 Get Content Recs", key="btn_cb"):
            idx = ap['anime_idx'][chosen]
            if isinstance(idx, pd.Series):
                idx = int(idx.iloc[0])
            # On-demand cosine similarity: compute ONE row vs all (no 1.2 GB matrix)
            query_vec  = ap['tfidf_mat'][idx]           # sparse row vector
            sim_scores = cosine_similarity(query_vec, ap['tfidf_mat']).flatten()
            sim_scores[idx] = -1                        # exclude the query itself
            top_idx    = np.argsort(sim_scores)[::-1][:n_cb]
            sims       = [(i, sim_scores[i]) for i in top_idx]
            rec  = ap['anime_cb'].iloc[[i[0] for i in sims]][
                ['name','genre','rating']].copy()
            rec['similarity'] = [round(s[1], 4) for s in sims]
            rec = rec.reset_index(drop=True)
            st.dataframe(rec)   # default width='stretch' in Streamlit 1.56

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(FIG_BG)
            ax.barh(rec['name'][::-1], rec['similarity'][::-1],
                    color=PASTEL[:n_cb], edgecolor='white', lw=1.2)
            ax.set_title(f"Genre Similarity to '{chosen}'", fontweight='bold')
            ax.set_xlabel("Cosine Similarity")
            ax_style(ax)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            show_fig(fig)

    with col_svd:
        st.markdown("#### 🤖 Collaborative SVD (User-Based)")
        user_id = st.slider("User ID (0–199):", 0, 199, 42, key="svd_uid")
        n_svd   = st.slider("# Recommendations:", 5, 20, 10, key="svd_n")

        if st.button("🌸 Get SVD Recs", key="btn_svd"):
            pred  = ap['rec_df'].iloc[user_id]
            seen  = ap['top_names'][ap['UI'][user_id] > 0]
            cands = pred.drop(labels=seen, errors='ignore')
            top_n = cands.nlargest(n_svd).reset_index()
            top_n.columns = ['Anime', 'Predicted Rating']
            top_n['Predicted Rating'] = top_n['Predicted Rating'].round(2)
            st.dataframe(top_n)   # default width='stretch' in Streamlit 1.56

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(FIG_BG)
            ax.barh(top_n['Anime'][::-1], top_n['Predicted Rating'][::-1],
                    color=PASTEL[:n_svd], edgecolor='white', lw=1.2)
            ax.set_title(f"SVD Predicted Ratings — User {user_id}", fontweight='bold')
            ax.set_xlabel("Predicted Rating")
            ax_style(ax)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            show_fig(fig)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;color:#8B3A62;font-size:.88rem;padding:10px 0;">
      SVD Best k = <b>{best_k_svd}</b> &nbsp;|&nbsp;
      RMSE = <b>{rmse_dict[best_k_svd]:.4f}</b> &nbsp;|&nbsp;
      Anime dataset: <b>{len(anime):,}</b> titles &nbsp;|&nbsp;
      Built with 🌸 Streamlit 1.56
    </div>
    """, unsafe_allow_html=True)
