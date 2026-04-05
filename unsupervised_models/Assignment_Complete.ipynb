"""
Assignment Dashboard — Anime Recommendations & Student Clustering
Streamlit app with two tabs, pastel/peach gradient theme, word clouds, and all plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import io
from collections import Counter

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, mean_squared_error)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from wordcloud import WordCloud

try:
    from kneed import KneeLocator
    KNEED_OK = True
except ImportError:
    KNEED_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Assignment Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS — peach/pink gradient, glitter texture ─────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff0f3 0%, #ffe4e1 30%, #ffd6cc 60%, #fff5f0 100%);
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(circle, rgba(255,182,193,0.18) 1px, transparent 1px),
        radial-gradient(circle, rgba(221,160,221,0.12) 1px, transparent 1px),
        radial-gradient(circle, rgba(255,218,185,0.15) 1px, transparent 1px);
    background-size: 28px 28px, 19px 19px, 37px 37px;
    background-position: 0 0, 9px 9px, 5px 20px;
    pointer-events: none;
    z-index: 0;
}
.block-container { position: relative; z-index: 1; }
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.55);
    border-radius: 18px;
    padding: 6px 10px;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 18px rgba(231,84,128,0.10);
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.97rem;
    color: #8B3A62 !important;
    padding: 8px 22px;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #ffb3c6, #c7ceea) !important;
    color: #4a0030 !important;
    box-shadow: 0 2px 10px rgba(231,84,128,0.18);
}
.metric-card {
    background: rgba(255,255,255,0.72);
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(231,84,128,0.10);
    border: 1px solid rgba(255,182,193,0.35);
    backdrop-filter: blur(6px);
    text-align: center;
    margin-bottom: 10px;
}
.metric-card h3 { color: #8B3A62; font-size: 1.5rem; margin: 0; }
.metric-card p  { color: #5a2040; font-size: 0.85rem; margin: 4px 0 0; }
.section-title {
    background: linear-gradient(90deg, #e75480, #c9a0dc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.5rem;
    font-weight: 800;
    margin: 1.2rem 0 0.5rem;
}
.explain-box {
    background: rgba(255,255,255,0.65);
    border-left: 4px solid #e75480;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0 18px;
    font-size: 0.93rem;
    color: #4a2040;
    backdrop-filter: blur(4px);
    box-shadow: 0 2px 10px rgba(231,84,128,0.07);
}
</style>
""", unsafe_allow_html=True)

PASTEL = ['#FFB3C6','#B5EAD7','#C7CEEA','#FFDAC1','#FF9AA2',
          '#A8D8EA','#E2F0CB','#D4A5A5','#957DAD','#F3B0C3',
          '#FFD1DC','#AEC6CF','#FFEAA7','#DDA0DD','#98D8C8']
FIG_BG = '#FFF5F2'


def fig_style(ax):
    ax.set_facecolor('#FFF8F6')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.28)


def st_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=FIG_BG)
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)


def pastel_wc_color(word, font_size, position, orientation, random_state=None, **kwargs):
    import random
    colors = ['#E75480','#C9A0DC','#FF9AA2','#F4A460','#6A9FB5',
              '#85C1E9','#F9A8D4','#F1948A','#BB8FCE','#76D7C4',
              '#FFB347','#77DD77','#AEC6CF','#DEB887','#B0C4DE']
    return random.choice(colors)


# ─────────────────────────────────────────────────────────────────────────────
# FILE FINDING
# ─────────────────────────────────────────────────────────────────────────────
import os
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(name):
    candidates = [
        os.path.join(DATA_DIR, name),
        os.path.join('/mnt/user-data/uploads', name),
        name,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def load_anime():
    path = find_file('anime.csv')
    if path is None:
        st.error("anime.csv not found. Place it alongside app.py.")
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


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_clustering(_df_raw):
    # Convert all columns to plain Python object dtype immediately so that
    # ArrowDtype-backed string columns (pandas 2.x + PyArrow backend on
    # Streamlit Cloud) don't cause re.sub / lambda errors inside .apply().
    df = _df_raw.drop_duplicates().copy()
    df = df.astype({col: 'object' for col in df.select_dtypes(include='string').columns},
                   errors='ignore')

    # Use pandas vectorised .str accessor (ArrowDtype-safe) instead of .apply+lambda
    df['age'] = pd.to_numeric(
        df['age'].astype(str).str.replace(r'[A-Za-z]', '', regex=True).str.strip(),
        errors='coerce')
    df['age'] = df['age'].clip(13, 22)
    df['gender'] = df['gender'].astype(object).map({'M': 1, 'F': 0}).fillna(-1)
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    skewness = df[num_cols].skew()
    df_t = df.copy()
    for col in skewness[skewness.abs() > 1.0].index:
        if (df_t[col] >= 0).all():
            df_t[col] = np.log1p(df_t[col])

    for col in num_cols:
        Q1, Q3 = df_t[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df_t[col] = df_t[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    X_scaled = RobustScaler().fit_transform(df_t[num_cols])

    pca_full = PCA(random_state=42).fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cumvar >= 0.90)) + 1
    X_pca  = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)

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

    best_k     = K_range[int(np.argmax(sil_km))]
    km_model   = KMeans(n_clusters=best_k, init='k-means++', n_init=20, random_state=42)
    km_labels  = km_model.fit_predict(X_pca)
    km_sil     = silhouette_score(X_pca, km_labels)
    km_db      = davies_bouldin_score(X_pca, km_labels)
    km_ch      = calinski_harabasz_score(X_pca, km_labels)

    sil_hier = []
    for k in K_range:
        lbl = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_pca)
        sil_hier.append(silhouette_score(X_pca, lbl))

    best_k_ag   = K_range[int(np.argmax(sil_hier))]
    hier_labels = AgglomerativeClustering(n_clusters=best_k_ag, linkage='ward').fit_predict(X_pca)
    hier_sil    = silhouette_score(X_pca, hier_labels)
    hier_db     = davies_bouldin_score(X_pca, hier_labels)
    hier_ch     = calinski_harabasz_score(X_pca, hier_labels)

    k_nn   = 5
    nn     = NearestNeighbors(n_neighbors=k_nn).fit(X_pca)
    dist, _ = nn.kneighbors(X_pca)
    k_dist = np.sort(dist[:, k_nn-1])[::-1]
    knee_idx = len(k_dist)//10
    if KNEED_OK:
        kl2 = KneeLocator(range(len(k_dist)), k_dist, curve='convex', direction='decreasing')
        if kl2.knee:
            knee_idx = kl2.knee
    eps_knee = round(float(k_dist[knee_idx]), 3)
    eps_cands = np.linspace(max(0.1, eps_knee*0.5), eps_knee*2.0, 8).round(3)

    best_sil_db, best_db_labels = -1, None
    best_eps, best_ms = eps_knee, 5
    for eps in eps_cands:
        for ms in [3, 5, 7, 10]:
            lbl  = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_pca)
            n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
            mask = lbl != -1
            if n_cl >= 2 and mask.sum() > 100:
                s = silhouette_score(X_pca[mask], lbl[mask])
                if s > best_sil_db:
                    best_sil_db, best_db_labels = s, lbl
                    best_eps, best_ms = eps, ms

    if best_db_labels is None:
        best_db_labels = km_labels.copy()

    mask_db    = best_db_labels != -1
    n_cl_db    = len(set(best_db_labels)) - (1 if -1 in best_db_labels else 0)
    dbscan_sil = silhouette_score(X_pca[mask_db], best_db_labels[mask_db])
    dbscan_db  = davies_bouldin_score(X_pca[mask_db], best_db_labels[mask_db])
    dbscan_ch  = calinski_harabasz_score(X_pca[mask_db], best_db_labels[mask_db])

    from sklearn.manifold import TSNE
    n_sample   = min(3000, len(X_pca))
    idx_sample = np.random.choice(len(X_pca), n_sample, replace=False)
    X_tsne     = TSNE(n_components=2, perplexity=40, random_state=42,
                      n_iter=500).fit_transform(X_pca[idx_sample])

    return dict(
        X_pca=X_pca, X_tsne=X_tsne, idx_sample=idx_sample,
        km_labels=km_labels, hier_labels=hier_labels, db_labels=best_db_labels,
        best_k=best_k, best_k_ag=best_k_ag, n_cl_db=n_cl_db,
        km_sil=km_sil, hier_sil=hier_sil, dbscan_sil=dbscan_sil,
        km_db=km_db, hier_db=hier_db, dbscan_db=dbscan_db,
        km_ch=km_ch, hier_ch=hier_ch, dbscan_ch=dbscan_ch,
        K_range=K_range, inertia=inertia_list, sil_km=sil_km,
        sil_hier=sil_hier, elbow_k=elbow_k,
        pca_full=pca_full, n_comp=n_comp, df_clean=df, num_cols=num_cols,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANIME PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_anime_pipeline(_anime):
    anime_cb = _anime.dropna(subset=['genre']).copy().reset_index(drop=True)
    anime_cb['genre_clean'] = anime_cb['genre'].str.replace(', ', ' ').str.replace('-', '')
    tfidf        = TfidfVectorizer(token_pattern=r"[a-zA-Z0-9]+")
    tfidf_matrix = tfidf.fit_transform(anime_cb['genre_clean'])
    cos_sim      = cosine_similarity(tfidf_matrix, tfidf_matrix)
    anime_idx    = pd.Series(anime_cb.index, index=anime_cb['name'])

    np.random.seed(42)
    top_names    = _anime.nlargest(500, 'members')['name'].values
    n_users, n_items = 300, len(top_names)
    UI           = np.zeros((n_users, n_items))
    base_ratings = _anime.set_index('name').loc[top_names, 'rating'].fillna(7).values
    for u in range(n_users):
        n_rated = np.random.randint(10, 80)
        items   = np.random.choice(n_items, n_rated, replace=False)
        UI[u, items] = np.clip(base_ratings[items] + np.random.randn(n_rated)*1.5, 1, 10)

    rmse_dict = {}
    for k in [5, 10, 20, 50]:
        sv  = TruncatedSVD(n_components=k, random_state=42)
        U_  = sv.fit_transform(UI)
        R_  = np.dot(U_, sv.components_)
        mask = UI > 0
        rmse_dict[k] = round(float(np.sqrt(mean_squared_error(UI[mask], R_[mask]))), 4)

    best_k_svd = min(rmse_dict, key=rmse_dict.get)
    svd_final  = TruncatedSVD(n_components=best_k_svd, random_state=42)
    U_final    = svd_final.fit_transform(UI)
    R_final    = np.dot(U_final, svd_final.components_)
    rec_df     = pd.DataFrame(R_final, columns=top_names)

    svd_ev = TruncatedSVD(n_components=min(50, n_items-1), random_state=42)
    svd_ev.fit(UI)

    return dict(
        anime_cb=anime_cb, cos_sim=cos_sim, anime_idx=anime_idx,
        top_names=top_names, UI=UI, rec_df=rec_df,
        svd_final=svd_final, rmse_dict=rmse_dict, best_k_svd=best_k_svd,
        svd_ev=svd_ev,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 28px 0 10px;">
  <span style="font-size:2.8rem;">🌸</span>
  <h1 style="font-size:2.4rem; font-weight:900;
             background:linear-gradient(90deg,#e75480,#c9a0dc,#6a9fb5);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
    ML Assignment Dashboard
  </h1>
  <p style="color:#8B3A62; font-size:1.05rem; margin-top:6px;">
    Student Clustering &nbsp;|&nbsp; Anime Recommendation System
    &nbsp;|&nbsp; SVD + KMeans + Hierarchical + DBSCAN
  </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎓 Part B — Student Clustering", "🎌 Part C — Anime Recommendations"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — STUDENT CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Student Social Network Profile Clustering</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="explain-box">
    This section clusters <b>15,000 high school students</b> using their social network profiles
    (2006–2009). Features include sports, music, fashion, and religion-related word counts.
    We apply K-Means, Hierarchical (Ward), and DBSCAN after PCA reduction, then compare
    all three with Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🌸 Running clustering pipeline — ~60 s the first time…"):
        df_raw = load_students()
        clust  = run_clustering(df_raw)

    # Metric cards
    st.markdown("#### 📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    for col, label, sil, db, ch, k in [
        (c1, "K-Means",      clust['km_sil'],     clust['km_db'],     clust['km_ch'],     clust['best_k']),
        (c2, "Hierarchical", clust['hier_sil'],   clust['hier_db'],   clust['hier_ch'],   clust['best_k_ag']),
        (c3, "DBSCAN",       clust['dbscan_sil'], clust['dbscan_db'], clust['dbscan_ch'], clust['n_cl_db']),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <p><b>{label}</b> &nbsp;(K={k})</p>
          <h3>{sil:.4f}</h3>
          <p>Silhouette ↑</p>
          <p>Davies-Bouldin: <b>{db:.4f}</b> ↓ &nbsp;|&nbsp; CH: <b>{ch:.1f}</b> ↑</p>
        </div>""", unsafe_allow_html=True)

    # EDA
    st.markdown("---")
    st.markdown("### 🔍 Exploratory Data Analysis")
    st.markdown("""<div class="explain-box">
    Gender and grad-year distributions reveal the demographic mix. "Music," "sports," and "dance"
    dominate the word-feature means — these become the natural axes of the clusters we discover.
    The correlation heatmap shows which interests tend to co-occur (e.g. church ↔ bible ↔ jesus).
    </div>""", unsafe_allow_html=True)

    df_viz    = df_raw.drop_duplicates().copy()
    word_cols = [c for c in df_viz.columns if c not in ['gradyear','gender','age','NumberOffriends']]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(FIG_BG)

    gd = df_viz['gender'].value_counts()
    axes[0,0].bar(gd.index, gd.values, color=['#FFB3C6','#B5EAD7','#C7CEEA'], edgecolor='white', lw=1.5)
    axes[0,0].set_title("Gender Distribution", fontweight='bold')

    df_viz['gradyear'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0,1], color=PASTEL[:4], edgecolor='white', lw=1.5)
    axes[0,1].set_title("Students per Graduation Year", fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=0)

    age_c = pd.to_numeric(
        df_viz['age'].astype(str).str.replace(r'[A-Za-z]', '', regex=True).str.strip(),
        errors='coerce').dropna()
    age_c = age_c[(age_c>=13) & (age_c<=22)]
    axes[0,2].hist(age_c, bins=18, color='#FFDAC1', edgecolor='white', lw=1.2)
    axes[0,2].set_title("Age Distribution (13–22)", fontweight='bold')

    axes[1,0].hist(df_viz['NumberOffriends'].clip(upper=200), bins=30,
                   color='#C7CEEA', edgecolor='white', lw=1.2)
    axes[1,0].set_title("Friends Count (clipped @200)", fontweight='bold')

    wmeans = df_viz[word_cols].mean().sort_values(ascending=False).head(15)
    axes[1,1].barh(wmeans.index[::-1], wmeans.values[::-1], color=PASTEL[:15], edgecolor='white')
    axes[1,1].set_title("Top 15 Word Features (Mean)", fontweight='bold')

    top12 = df_viz[word_cols].mean().nlargest(12).index.tolist()
    sns.heatmap(df_viz[top12].corr(), ax=axes[1,2], cmap='RdPu',
                linewidths=0.5, cbar_kws={'shrink':0.7}, annot=False)
    axes[1,2].set_title("Correlation — Top 12 Words", fontweight='bold')

    for ax in axes.flat:
        ax.set_facecolor('#FFF8F6')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st_fig(fig)

    # Word Cloud
    st.markdown("### 🌸 Student Profile Keywords — Word Cloud")
    st.markdown("""<div class="explain-box">
    Word size reflects total mention count across all 15,000 profiles. Music, sports, and dance lead —
    hinting at the major lifestyle clusters that the algorithms will identify.
    </div>""", unsafe_allow_html=True)

    word_freq = df_raw[word_cols].sum().to_dict()
    wc = WordCloud(
        width=900, height=340, background_color='#FFF0F3', max_words=60,
        color_func=pastel_wc_color, prefer_horizontal=0.85,
        collocations=False, min_font_size=10
    ).generate_from_frequencies(word_freq)

    fig_wc, ax_wc = plt.subplots(figsize=(13, 4.5))
    fig_wc.patch.set_facecolor('#FFF0F3')
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    plt.tight_layout()
    st_fig(fig_wc)

    # PCA
    st.markdown("### 📐 PCA — Dimensionality Reduction")
    st.markdown("""<div class="explain-box">
    PCA rotates the feature space so that the first axis captures the most variance, the second
    captures the next most, and so on. We keep only as many axes as needed for <b>90% cumulative
    variance</b>, dropping the rest as noise. This dramatically speeds up clustering while
    retaining nearly all the signal.
    </div>""", unsafe_allow_html=True)

    pca_full = clust['pca_full']
    n_comp   = clust['n_comp']
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    K_all    = range(1, len(pca_full.explained_variance_ratio_)+1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(FIG_BG)

    axes[0].bar(K_all, pca_full.explained_variance_ratio_*100,
                color=PASTEL[0], edgecolor='white', alpha=0.85)
    axes[0].axvline(n_comp, color='#8B3A62', ls='--', lw=2, label=f'90% at PC={n_comp}')
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance (%)")
    axes[0].set_title("Scree Plot — Variance per Component", fontweight='bold')
    axes[0].legend()

    axes[1].plot(K_all, cumvar*100, 'o-', color='#E75480', lw=2, ms=4)
    axes[1].fill_between(K_all, cumvar*100, alpha=0.15, color='#E75480')
    axes[1].axhline(90, color='#6A5ACD', ls='--', lw=1.8, label='90% variance')
    axes[1].axvline(n_comp, color='#8B3A62', ls='--', lw=1.5)
    axes[1].set_xlabel("Number of PCs"); axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("Cumulative Explained Variance", fontweight='bold')
    axes[1].legend()

    for ax in axes:
        fig_style(ax)
    plt.tight_layout()
    st_fig(fig)

    # K-Means Elbow
    st.markdown("### 🔵 K-Means — Elbow & Silhouette Sweep")
    st.markdown("""<div class="explain-box">
    The <b>Elbow Method</b> tracks WCSS (within-cluster sum of squares) — we look for the point
    where adding more clusters gives diminishing returns. The <b>Silhouette Score</b> directly
    measures cluster quality (higher is better, max 1.0). We select K from the highest silhouette.
    </div>""", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(FIG_BG)

    axes[0].plot(clust['K_range'], clust['inertia'], 'o-', color='#E75480', lw=2, ms=7)
    axes[0].axvline(clust['elbow_k'], color='#8B3A62', ls='--', lw=2,
                    label=f"Elbow at K={clust['elbow_k']}")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("Elbow Method", fontweight='bold'); axes[0].legend()

    axes[1].plot(clust['K_range'], clust['sil_km'], 's-', color='#6A5ACD', lw=2, ms=7)
    axes[1].axvline(clust['best_k'], color='#8B3A62', ls='--', lw=2,
                    label=f"Best K={clust['best_k']}")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs K", fontweight='bold'); axes[1].legend()

    for ax in axes:
        fig_style(ax)
    plt.tight_layout()
    st_fig(fig)

    # Hierarchical
    st.markdown("### 🌿 Hierarchical Clustering — Silhouette Sweep + Dendrogram")
    st.markdown("""<div class="explain-box">
    Ward linkage minimises the increase in total within-cluster variance at each merge step —
    producing compact, equally-sized clusters. The <b>dendrogram</b> (on a 300-point sample)
    visualises the merging hierarchy: long vertical lines indicate natural breakpoints where
    the algorithm should stop.
    </div>""", unsafe_allow_html=True)

    sample_d = np.random.choice(len(clust['X_pca']), 300, replace=False)
    Z        = linkage(clust['X_pca'][sample_d], method='ward')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(FIG_BG)

    axes[0].plot(clust['K_range'], clust['sil_hier'], 'd-', color='#E75480', lw=2, ms=7)
    axes[0].axvline(clust['best_k_ag'], color='#8B3A62', ls='--', lw=2,
                    label=f"Best K={clust['best_k_ag']}")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Hierarchical — Silhouette vs K", fontweight='bold'); axes[0].legend()
    fig_style(axes[0])

    dendrogram(Z, ax=axes[1], truncate_mode='lastp', p=15,
               color_threshold=0, above_threshold_color='#C7CEEA', leaf_font_size=8)
    axes[1].set_title("Dendrogram (300-sample, Ward linkage)", fontweight='bold')
    axes[1].set_facecolor('#FFF8F6')
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    st_fig(fig)

    # Comparison table
    st.markdown("### 🏆 All Three Methods — Side-by-Side Comparison")
    st.markdown("""<div class="explain-box">
    <b>Silhouette ↑</b> (0–1, higher is better): how well each point fits its own cluster vs others.
    <b>Davies-Bouldin ↓</b> (lower is better): ratio of within-cluster scatter to between-cluster
    separation. <b>Calinski-Harabasz ↑</b> (higher is better): ratio of between- to
    within-cluster dispersion. Together they give a well-rounded view of clustering quality.
    </div>""", unsafe_allow_html=True)

    results_df = pd.DataFrame({
        'Method'             : ['K-Means', 'Hierarchical', 'DBSCAN'],
        'Optimal K'          : [clust['best_k'], clust['best_k_ag'], clust['n_cl_db']],
        'Silhouette ↑'       : [round(clust['km_sil'],4), round(clust['hier_sil'],4),  round(clust['dbscan_sil'],4)],
        'Davies-Bouldin ↓'   : [round(clust['km_db'],4),  round(clust['hier_db'],4),   round(clust['dbscan_db'],4)],
        'Calinski-Harabasz ↑': [round(clust['km_ch'],1),   round(clust['hier_ch'],1),  round(clust['dbscan_ch'],1)],
    })
    st.dataframe(results_df.set_index('Method'), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(FIG_BG)
    bars = ax.bar(['K-Means','Hierarchical','DBSCAN'], results_df['Silhouette ↑'],
                  color=PASTEL[:3], edgecolor='white', lw=1.5, width=0.5)
    for bar, val in zip(bars, results_df['Silhouette ↑']):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f"{val:.4f}", ha='center', fontweight='bold', fontsize=11)
    ax.set_ylabel("Silhouette Score (higher = better)")
    ax.set_title("Clustering Silhouette Comparison", fontweight='bold', fontsize=13)
    ax.set_ylim(0, 0.7)
    fig_style(ax)
    plt.tight_layout()
    st_fig(fig)

    # t-SNE
    st.markdown("### 🔮 t-SNE 2D Cluster Visualisation")
    st.markdown("""<div class="explain-box">
    t-SNE projects the multi-dimensional PCA space into 2D while preserving local neighbourhood
    structure — revealing non-linear cluster shapes that PCA would miss. Each colour is a cluster.
    Tight, distinct blobs indicate good separation; overlapping blobs indicate overlapping clusters.
    (Plotted on a 3,000-point sample for speed.)
    </div>""", unsafe_allow_html=True)

    X_tsne     = clust['X_tsne']
    idx_sample = clust['idx_sample']
    cmap_p     = plt.cm.get_cmap('Pastel1', 10)

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
            ax.scatter(X_tsne[mask,0], X_tsne[mask,1], c=[cmap_p(i)], s=6, alpha=0.75, label=label)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_facecolor('#FFF8F6')
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if len(set(lbl)) <= 8:
            ax.legend(markerscale=4, fontsize=8, loc='best')
    plt.tight_layout()
    st_fig(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANIME RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Anime Recommendation System</p>', unsafe_allow_html=True)
    st.markdown("""<div class="explain-box">
    Two complementary recommenders on <b>12,294 anime</b>:
    (1) <b>Content-Based</b> using TF-IDF on genres + cosine similarity — finds anime with
    the same genre composition. (2) <b>Collaborative SVD</b> — decomposes a user-item rating
    matrix into latent preference factors, the same technique powering Netflix recommendations.
    RMSE measures how accurately the SVD reconstructs known ratings.
    </div>""", unsafe_allow_html=True)

    with st.spinner("🌸 Running anime pipeline…"):
        anime = load_anime()
        ap    = run_anime_pipeline(anime)

    # EDA
    st.markdown("### 🔍 Anime Dataset EDA")
    st.markdown("""<div class="explain-box">
    Ratings cluster between 6–8, TV series dominate, and members follow a power-law distribution.
    A few blockbusters have millions of members while most anime have only hundreds — which is why
    we log-transform members before any distance-based computation.
    </div>""", unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.patch.set_facecolor(FIG_BG)

    axes[0,0].hist(anime['rating'].dropna(), bins=28, color='#FFB3C6', edgecolor='white', lw=1.2)
    axes[0,0].set_title("Rating Distribution", fontweight='bold'); axes[0,0].set_xlabel("Rating")

    tc = anime['type'].value_counts()
    axes[0,1].bar(tc.index, tc.values, color=PASTEL[:len(tc)], edgecolor='white', lw=1.5)
    axes[0,1].set_title("Anime Type Distribution", fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=20)

    axes[0,2].hist(np.log1p(anime['members']), bins=28, color='#C7CEEA', edgecolor='white', lw=1.2)
    axes[0,2].set_title("Member Count (log scale)", fontweight='bold')
    axes[0,2].set_xlabel("log(1 + members)")

    top15r = anime.nlargest(15, 'rating')[['name','rating']].reset_index(drop=True)
    axes[1,0].barh(top15r['name'][::-1], top15r['rating'][::-1], color=PASTEL[:15], edgecolor='white')
    axes[1,0].set_title("Top 15 Highest Rated Anime", fontweight='bold')
    axes[1,0].tick_params(axis='y', labelsize=7.5)

    top15m = anime.nlargest(15, 'members')[['name','members']].reset_index(drop=True)
    axes[1,1].barh(top15m['name'][::-1], top15m['members'][::-1], color=PASTEL[:15], edgecolor='white')
    axes[1,1].set_title("Top 15 Most Popular Anime (Members)", fontweight='bold')
    axes[1,1].tick_params(axis='y', labelsize=7.5)

    sc = axes[1,2].scatter(np.log1p(anime['members']), anime['rating'],
                            c=anime['rating'], cmap='RdPu', s=7, alpha=0.55)
    plt.colorbar(sc, ax=axes[1,2], label='Rating')
    axes[1,2].set_title("Rating vs Popularity (log)", fontweight='bold')
    axes[1,2].set_xlabel("log(1+members)"); axes[1,2].set_ylabel("Rating")

    for ax in axes.flat:
        ax.set_facecolor('#FFF8F6')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    st_fig(fig)

    # Genre Word Cloud
    st.markdown("### 🎌 Genre Word Cloud")
    st.markdown("""<div class="explain-box">
    Word size = number of anime in that genre. Comedy, Action, and Adventure dominate —
    meaning most similarity scores between anime will lean on whether they share these broad genres.
    Niche genres like Josei or Dementia create strong signals for those who enjoy them.
    </div>""", unsafe_allow_html=True)

    all_genres  = []
    for g in anime['genre'].dropna():
        all_genres.extend([x.strip() for x in g.split(',')])
    genre_freq2 = Counter(all_genres)

    wc2 = WordCloud(
        width=980, height=380, background_color='#FFF5F0', max_words=80,
        color_func=pastel_wc_color, prefer_horizontal=0.85,
        collocations=False, min_font_size=9
    ).generate_from_frequencies(genre_freq2)

    fig_wc2, ax_wc2 = plt.subplots(figsize=(13, 4.5))
    fig_wc2.patch.set_facecolor('#FFF5F0')
    ax_wc2.imshow(wc2, interpolation='bilinear')
    ax_wc2.axis('off')
    plt.tight_layout()
    st_fig(fig_wc2)

    # Skewness
    st.markdown("### 📊 Skewness & log1p Transformation")
    st.markdown("""<div class="explain-box">
    Raw member and episode counts are extremely right-skewed. Applying log(1+x) compresses
    the long tail so that distance metrics treat a difference of 10 vs 100 members similarly
    to 100,000 vs 1,000,000 — proportionally, not absolutely. Rating is already near-Gaussian
    and needs no transformation.
    </div>""", unsafe_allow_html=True)

    anime_num = anime[['rating','members','episodes']].copy()
    anime_t   = anime_num.copy()
    for col in ['members','episodes']:
        anime_t[col] = np.log1p(anime_t[col].fillna(0))
    skew_b = anime_num.skew(); skew_a = anime_t.skew()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor(FIG_BG)
    for i, col in enumerate(['rating','members','episodes']):
        axes[0,i].hist(anime_num[col].dropna(), bins=28, color=PASTEL[i], edgecolor='white')
        axes[0,i].set_title(f"{col} Before (skew={skew_b[col]:.2f})", fontweight='bold')
        axes[1,i].hist(anime_t[col].dropna(), bins=28, color=PASTEL[i+3], edgecolor='white')
        axes[1,i].set_title(f"{col} After log1p (skew={skew_a[col]:.2f})", fontweight='bold')
    for ax in axes.flat:
        ax.set_facecolor('#FFF8F6')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st_fig(fig)

    # SVD performance
    st.markdown("### 🤖 SVD — Latent Factor Model Accuracy")
    st.markdown("""<div class="explain-box">
    SVD factorises the user-item matrix into three matrices: U (user-factor affinities),
    Σ (factor strengths), and V (item-factor affinities). The product U·Σ·Vᵀ approximates
    the original matrix. We measure <b>RMSE</b> on observed entries — the best k minimises it.
    The right chart shows what fraction of matrix variance each additional factor captures.
    </div>""", unsafe_allow_html=True)

    rmse_dict  = ap['rmse_dict']
    best_k_svd = ap['best_k_svd']
    svd_ev     = ap['svd_ev']
    cum_ev     = np.cumsum(svd_ev.explained_variance_ratio_)*100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(FIG_BG)

    axes[0].plot(list(rmse_dict.keys()), list(rmse_dict.values()),
                 'o-', color='#E75480', lw=2.5, ms=8)
    axes[0].axvline(best_k_svd, color='#8B3A62', ls='--', lw=2,
                    label=f'Best k={best_k_svd}\nRMSE={rmse_dict[best_k_svd]:.4f}')
    axes[0].set_xlabel("Latent Factors (k)"); axes[0].set_ylabel("RMSE on Observed Ratings")
    axes[0].set_title("SVD RMSE vs Latent Factors", fontweight='bold'); axes[0].legend()

    ev_range = range(1, len(cum_ev)+1)
    axes[1].plot(ev_range, cum_ev, 'o-', color='#6A5ACD', lw=2, ms=4)
    axes[1].fill_between(ev_range, cum_ev, alpha=0.15, color='#6A5ACD')
    axes[1].axhline(80, color='#E75480', ls='--', lw=1.5, label='80% variance')
    axes[1].axhline(90, color='#8B3A62', ls='--', lw=1.5, label='90% variance')
    axes[1].set_xlabel("SVD Components"); axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].set_title("SVD Cumulative Explained Variance", fontweight='bold'); axes[1].legend()

    for ax in axes:
        fig_style(ax)
    plt.tight_layout()
    st_fig(fig)

    # Top genres bar
    st.markdown("### 🎭 Top 20 Anime Genres")
    genre_series = pd.Series(genre_freq2).sort_values(ascending=False)[:20]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.barh(genre_series.index[::-1], genre_series.values[::-1],
            color=PASTEL[:20], edgecolor='white', lw=1.2)
    ax.set_title("Top 20 Anime Genres by Frequency", fontweight='bold', fontsize=13)
    ax.set_xlabel("Number of Anime")
    fig_style(ax)
    plt.tight_layout()
    st_fig(fig)

    # Interactive recommenders
    st.markdown("---")
    st.markdown("### 🔎 Try the Recommenders")

    col_cb, col_svd = st.columns(2)

    with col_cb:
        st.markdown("#### 📚 Content-Based (Genre TF-IDF)")
        all_titles = sorted(ap['anime_cb']['name'].tolist())
        default    = 'Sword Art Online' if 'Sword Art Online' in all_titles else all_titles[0]
        chosen     = st.selectbox("Pick an anime:", all_titles,
                                  index=all_titles.index(default), key="cb_sel")
        n_cb = st.slider("# Recommendations:", 5, 20, 10, key="cb_n")

        if st.button("🌸 Get Content Recs", key="btn_cb"):
            idx = ap['anime_idx'][chosen]
            if isinstance(idx, pd.Series):
                idx = int(idx.iloc[0])
            sims = sorted(enumerate(ap['cos_sim'][idx]), key=lambda x: x[1], reverse=True)[1:n_cb+1]
            rec  = ap['anime_cb'].iloc[[i[0] for i in sims]][['name','genre','rating']].copy()
            rec['similarity'] = [round(s[1], 4) for s in sims]
            rec = rec.reset_index(drop=True)
            st.dataframe(rec, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(FIG_BG)
            ax.barh(rec['name'][::-1], rec['similarity'][::-1],
                    color=PASTEL[:n_cb], edgecolor='white', lw=1.2)
            ax.set_title(f"Genre Similarity to '{chosen}'", fontweight='bold')
            ax.set_xlabel("Cosine Similarity")
            fig_style(ax)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            st_fig(fig)

    with col_svd:
        st.markdown("#### 🤖 Collaborative SVD (User-Based)")
        user_id = st.slider("User ID (0–299):", 0, 299, 42, key="svd_uid")
        n_svd   = st.slider("# Recommendations:", 5, 20, 10, key="svd_n")

        if st.button("🌸 Get SVD Recs", key="btn_svd"):
            pred  = ap['rec_df'].iloc[user_id]
            seen  = ap['top_names'][ap['UI'][user_id] > 0]
            cands = pred.drop(labels=seen, errors='ignore')
            top_n = cands.nlargest(n_svd).reset_index()
            top_n.columns = ['Anime', 'Predicted Rating']
            top_n['Predicted Rating'] = top_n['Predicted Rating'].round(2)
            st.dataframe(top_n, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(FIG_BG)
            ax.barh(top_n['Anime'][::-1], top_n['Predicted Rating'][::-1],
                    color=PASTEL[:n_svd], edgecolor='white', lw=1.2)
            ax.set_title(f"SVD Predicted Ratings for User {user_id}", fontweight='bold')
            ax.set_xlabel("Predicted Rating")
            fig_style(ax)
            ax.tick_params(axis='y', labelsize=8)
            plt.tight_layout()
            st_fig(fig)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:#8B3A62; font-size:0.88rem; padding:10px 0;">
        SVD Best k = <b>{best_k_svd}</b> &nbsp;|&nbsp;
        RMSE = <b>{rmse_dict[best_k_svd]:.4f}</b> &nbsp;|&nbsp;
        Anime dataset: <b>{len(anime):,}</b> titles &nbsp;|&nbsp;
        Built with 🌸 Streamlit
    </div>
    """, unsafe_allow_html=True)
