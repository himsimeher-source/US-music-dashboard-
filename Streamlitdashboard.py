import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atlantic RC – US Top 50 Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {font-size:2.2rem; font-weight:800; color:#1DB954; margin-bottom:0.2rem;}
    .sub-title  {font-size:1rem;  color:#888; margin-bottom:1.5rem;}
    .kpi-card   {background:#1e1e2e; border-radius:12px; padding:18px; text-align:center;
                 border-left:4px solid #1DB954;}
    .kpi-value  {font-size:2rem; font-weight:700; color:#1DB954;}
    .kpi-label  {font-size:0.85rem; color:#aaa; margin-top:4px;}
    [data-testid="stSidebar"] {background:#0d1117;}
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING & CACHING ───────────────────────────────────────────────────
@st.cache_data
def load_data(path='Atlantic_United_States.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df['duration_min'] = df['duration_ms'] / 60_000
    df['month']        = df['date'].dt.month
    df['week']         = df['date'].dt.isocalendar().week.astype(int)
    df['top10']        = (df['position'] <= 10).astype(int)
    df['top_tier']     = pd.cut(df['position'], bins=[0,10,25,50],
                                labels=['Top 10','Top 25','Top 50'])
    return df

@st.cache_data
def compute_features(df):
    song_stats = df.groupby('song').agg(
        days_on_chart   = ('date',         'nunique'),
        avg_rank        = ('position',     'mean'),
        best_rank       = ('position',     'min'),
        rank_volatility = ('position',     'std'),
        avg_popularity  = ('popularity',   'mean'),
        max_popularity  = ('popularity',   'max'),
        is_explicit     = ('is_explicit',  'first'),
        album_type      = ('album_type',   'first'),
        duration_min    = ('duration_min', 'first'),
        total_tracks    = ('total_tracks', 'first'),
        artist          = ('artist',       'first')
    ).reset_index()
    song_stats['rank_volatility']  = song_stats['rank_volatility'].fillna(0)
    song_stats['popularity_trend'] = song_stats['max_popularity'] - song_stats['avg_popularity']

    artist_stats = df.groupby('artist').agg(
        unique_songs   = ('song',       'nunique'),
        total_days     = ('date',       'count'),
        avg_rank       = ('position',   'mean'),
        avg_popularity = ('popularity', 'mean'),
        top10_entries  = ('top10',      'sum')
    ).reset_index()
    artist_stats['dominance_index'] = (
        artist_stats['unique_songs'] * artist_stats['total_days']
        / artist_stats['total_days'].max()
    )
    return song_stats, artist_stats

# ─── LOAD DATA ──────────────────────────────────────────────────────────────────
try:
    df = load_data()
    song_stats, artist_stats = compute_features(df)
except FileNotFoundError:
    st.error("❌ Data file not found! Please check the file path.")
    st.stop()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9c/Atlantic_Records_logo.svg", width=160)
    st.markdown("## 🎛️ Filters")

    date_min, date_max = df['date'].min(), df['date'].max()
    date_range = st.date_input("Date Range",
                               value=(date_min.date(), date_max.date()),
                               min_value=date_min.date(),
                               max_value=date_max.date())

    rank_range = st.slider("Position Range", 1, 50, (1, 50))
    album_toggle = st.multiselect("Album Type",
                                  options=df['album_type'].unique().tolist(),
                                  default=df['album_type'].unique().tolist())
    artist_filter = st.multiselect("Filter by Artist (optional)",
                                   options=sorted(df['artist'].unique()))

    st.markdown("---")
    page = st.radio("📑 Navigate",
                    ["🏠 Overview", "🎵 Song Analysis",
                     "🎤 Artist Analysis", "🤖 ML Models",
                     "🔍 Clustering"])

# ─── APPLY FILTERS ────────────────────────────────────────────────────────────
if len(date_range) == 2:
    fdf = df[(df['date'].dt.date >= date_range[0]) &
             (df['date'].dt.date <= date_range[1])]
else:
    fdf = df.copy()

fdf = fdf[(fdf['position'] >= rank_range[0]) & (fdf['position'] <= rank_range[1])]
fdf = fdf[fdf['album_type'].isin(album_toggle)]
if artist_filter:
    fdf = fdf[fdf['artist'].isin(artist_filter)]

# Check if filtered data is empty
if fdf.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust your selection.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-title">🎵 Atlantic RC – US Top 50 Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">United States Top 50 Playlist Performance & Song Popularity Trend Analysis</div>', unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    def kpi(col, value, label):
        col.markdown(f'<div class="kpi-card"><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

    kpi(c1, f"{len(fdf):,}",            "Total Records")
    kpi(c2, f"{fdf['song'].nunique()}",  "Unique Songs")
    kpi(c3, f"{fdf['artist'].nunique()}","Unique Artists")
    kpi(c4, f"{fdf['popularity'].mean():.1f}", "Avg Popularity")
    kpi(c5, f"{fdf['is_explicit'].mean()*100:.0f}%", "Explicit %")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Daily Unique Songs on Playlist")
        daily = fdf.groupby('date')['song'].nunique()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(daily.index, daily.values, color='#1DB954', linewidth=1.5)
        ax.fill_between(daily.index, daily.values, alpha=0.2, color='#1DB954')
        ax.set_xlabel("Date"); ax.set_ylabel("Unique Songs")
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white'); 
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📊 Popularity by Chart Tier")
        fig, ax = plt.subplots(figsize=(7, 4))
        fdf2 = fdf.dropna(subset=['top_tier'])
        sns.boxplot(data=fdf2, x='top_tier', y='popularity', palette='Set2', ax=ax)
        ax.set_xlabel("Tier"); ax.set_ylabel("Popularity")
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white'); 
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🥧 Explicit Content Share")
        ec = fdf['is_explicit'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ['Non-Explicit','Explicit'] if ec.index[0] == 0 else ['Explicit','Non-Explicit']
        ax.pie(ec.values,
               labels=labels,
               autopct='%1.1f%%', colors=['#1DB954','#E74C3C'],
               wedgeprops={'edgecolor':'#0d1117','linewidth':2})
        fig.patch.set_facecolor('#0d1117')
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("📦 Avg Popularity by Album Type")
        alb = fdf.groupby('album_type')['popularity'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_bar = ['#1DB954','#3498DB','#9B59B6'][:len(alb)]
        ax.bar(alb.index, alb.values, color=colors_bar)
        ax.set_ylabel("Avg Popularity")
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white'); 
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – SONG ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎵 Song Analysis":
    st.title("🎵 Song-Level Performance Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top Songs by Days on Chart")
        n = st.slider("Show Top N Songs", 5, 30, 15)
        fsong = song_stats[song_stats['song'].isin(fdf['song'].unique())]
        top_s = fsong.nlargest(n, 'days_on_chart')[['song','artist','days_on_chart','avg_rank','avg_popularity','best_rank']]
        top_s.columns = ['Song','Artist','Days','Avg Rank','Avg Pop','Best Rank']
        st.dataframe(top_s.reset_index(drop=True), use_container_width=True, height=450)

    with col2:
        st.subheader("📊 Days on Chart Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(fsong['days_on_chart'], bins=35, color='#1DB954', edgecolor='#0d1117', linewidth=0.5)
        ax.axvline(fsong['days_on_chart'].median(), color='red', ls='--', linewidth=1.5,
                   label=f"Median = {fsong['days_on_chart'].median():.0f}")
        ax.set_xlabel("Days on Chart"); ax.set_ylabel("Count")
        ax.legend(); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white'); 
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()

    st.subheader("🔍 Song Deep Dive – Rank Timeline")
    search_song = st.selectbox("Select a Song", sorted(fdf['song'].unique()))
    song_df = fdf[fdf['song'] == search_song].sort_values('date')
    if not song_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(song_df['date'], song_df['position'], 'o-', color='#1DB954', markersize=4)
        ax.invert_yaxis(); ax.set_xlabel("Date"); ax.set_ylabel("Rank (Lower = Better)")
        ax.set_title(f'Rank Timeline: {search_song}')
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white'); 
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days on Chart",  song_df['date'].nunique())
        c2.metric("Best Rank",      song_df['position'].min())
        c3.metric("Avg Rank",       f"{song_df['position'].mean():.1f}")
        c4.metric("Avg Popularity", f"{song_df['popularity'].mean():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – ARTIST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎤 Artist Analysis":
    st.title("🎤 Artist Dominance & Performance")
    fartist = artist_stats[artist_stats['artist'].isin(fdf['artist'].unique())]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏅 Artist Dominance Leaderboard")
        top_a = fartist.nlargest(20, 'dominance_index')[
            ['artist','unique_songs','total_days','avg_popularity','top10_entries','dominance_index']]
        top_a.columns = ['Artist','Songs','Days','Avg Pop','Top10 Entries','Dominance']
        top_a['Dominance'] = top_a['Dominance'].round(2)
        st.dataframe(top_a.reset_index(drop=True), use_container_width=True, height=500)

    with col2:
        st.subheader("📊 Top 10 Artists – Total Chart Days")
        top10a = fartist.nlargest(10, 'total_days')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top10a['artist'], top10a['total_days'], color='#1DB954')
        ax.set_xlabel("Total Chart Days"); ax.invert_yaxis()
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white'); 
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()

    st.subheader("📈 Artist Rank Timeline")
    sel_artist = st.selectbox("Select Artist", sorted(fdf['artist'].unique()))
    adf = fdf[fdf['artist'] == sel_artist].sort_values('date')
    if not adf.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(adf['date'], adf['position'], 'o-', color='#1DB954', markersize=3)
        axes[0].invert_yaxis(); axes[0].set_title("Rank over Time")
        axes[0].set_xlabel("Date"); axes[0].set_ylabel("Rank")
        axes[1].plot(adf['date'], adf['popularity'], color='#E74C3C', linewidth=1.5)
        axes[1].set_title("Popularity over Time")
        axes[1].set_xlabel("Date"); axes[1].set_ylabel("Popularity")
        for ax in axes:
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
            ax.tick_params(colors='white'); 
            for spine in ax.spines.values():
                spine.set_color('#333')
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.title("🤖 Machine Learning Models")

    tab1, tab2 = st.tabs(["🎯 Classification (Top-10)", "📈 Regression (Popularity)"])

    with tab1:
        st.subheader("🎯 ML Task 1 – Predict Top-10 Placement")
        st.info("**Goal:** Predict whether a song will appear in the Top 10 on a given day.\n\n"
                "**Features:** Popularity, Duration, Total Tracks, Explicit Flag, Month, Album Type")

        with st.spinner("Training Random Forest Classifier..."):
            le = LabelEncoder()
            df['album_type_enc'] = le.fit_transform(df['album_type'])
            X = df[['popularity','duration_min','total_tracks','is_explicit','month','album_type_enc']].copy()
            X['is_explicit'] = X['is_explicit'].astype(int)
            y = df['top10']
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            acc = (pred == y_te).mean()

        st.success(f"✅ Random Forest Accuracy: **{acc*100:.2f}%**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_te, pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                        xticklabels=['Not Top 10','Top 10'],
                        yticklabels=['Not Top 10','Top 10'])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
            ax.tick_params(colors='white'); 
            for spine in ax.spines.values():
                spine.set_color('#333')
            st.pyplot(fig); plt.close()
        with col2:
            st.subheader("Feature Importance")
            feat_names = ['Popularity','Duration','Total Tracks','Explicit','Month','Album Type']
            importance = clf.feature_importances_
            idx = np.argsort(importance)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh([feat_names[i] for i in idx], importance[idx], color='#1DB954')
            ax.set_xlabel("Importance Score")
            fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
            ax.tick_params(colors='white'); 
            for spine in ax.spines.values():
                spine.set_color('#333')
            st.pyplot(fig); plt.close()

        st.subheader("Classification Report")
        report = classification_report(y_te, pred, target_names=['Not Top 10','Top 10'], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))

        st.subheader("🔮 Predict a Song's Chart Tier")
        c1, c2, c3 = st.columns(3)
        pop_val  = c1.slider("Popularity Score", 0, 100, 88)
        dur_val  = c2.slider("Duration (min)", 1.0, 8.0, 3.5, 0.1)
        exp_val  = c3.selectbox("Explicit?", [False, True])
        c4, c5   = st.columns(2)
        trk_val  = c4.slider("Total Tracks in Album", 1, 50, 14)
        mon_val  = c5.slider("Month", 1, 12, 6)
        alb_val  = st.selectbox("Album Type", df['album_type'].unique())
        alb_enc  = le.transform([alb_val])[0]
        inp      = pd.DataFrame([[pop_val, dur_val, trk_val, int(exp_val), mon_val, alb_enc]],
                                columns=['popularity','duration_min','total_tracks','is_explicit','month','album_type_enc'])
        pred_val = clf.predict(inp)[0]
        prob_val = clf.predict_proba(inp)[0][1]
        if pred_val == 1:
            st.success(f"🏆 **Top 10 likely!** Probability: {prob_val*100:.1f}%")
        else:
            st.warning(f"📉 **Not Top 10.** Top-10 probability: {prob_val*100:.1f}%")

    with tab2:
        st.subheader("📈 ML Task 2 – Predict Average Popularity Score")
        st.info("**Goal:** Predict a song's average popularity from its chart behaviour.\n\n"
                "**Features:** Days on Chart, Avg Rank, Best Rank, Rank Volatility, Duration, Total Tracks, Explicit")

        with st.spinner("Training RF Regressor..."):
            fsong2 = song_stats.copy()
            RFEATS = ['days_on_chart','avg_rank','best_rank','rank_volatility',
                      'duration_min','total_tracks','is_explicit']
            Xr = fsong2[RFEATS].copy()
            Xr['is_explicit'] = Xr['is_explicit'].astype(int)
            yr = fsong2['avg_popularity']
            Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
            rfr = RandomForestRegressor(n_estimators=100, random_state=42)
            rfr.fit(Xr_tr, yr_tr)
            rpr = rfr.predict(Xr_te)
            r2  = r2_score(yr_te, rpr)
            mae = mean_absolute_error(yr_te, rpr)

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("MAE",       f"{mae:.2f}")
        col3.metric("Songs used", len(fsong2))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(yr_te, rpr, alpha=0.5, color='#1DB954', s=30, edgecolors='none')
        lims = [min(yr_te.min(), rpr.min()), max(yr_te.max(), rpr.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
        ax.set_xlabel("Actual Popularity"); ax.set_ylabel("Predicted Popularity")
        ax.set_title("Actual vs Predicted Popularity (RF Regressor)")
        ax.legend(); fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white'); 
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Clustering":
    st.title("🔍 Song Clustering – KMeans Segmentation")
    st.info("Songs are grouped into behavioural clusters using KMeans on 5 chart features.")

    CFEATS = ['days_on_chart','avg_rank','avg_popularity','rank_volatility','duration_min']
    Xc = song_stats[CFEATS].fillna(0)
    sc = StandardScaler()
    Xc_s = sc.fit_transform(Xc)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    song_stats['cluster'] = km.fit_predict(Xc_s)
    cluster_names = {0:'Flash Hit', 1:'One-Hit Wonder', 2:'Chart Dominator', 3:'Mid-Tier Performer'}
    song_stats['cluster_name'] = song_stats['cluster'].map(cluster_names)

    pca = PCA(n_components=2, random_state=42)
    Xpca = pca.fit_transform(Xc_s)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🗺️ PCA 2D Cluster View")
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ['#1DB954','#E74C3C','#3498DB','#F39C12']
        for c in range(4):
            mask = song_stats['cluster'] == c
            ax.scatter(Xpca[mask,0], Xpca[mask,1], label=cluster_names[c],
                       color=colors[c], alpha=0.7, s=30, edgecolors='none')
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=9)
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white'); 
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📊 Cluster Profiles")
        cp = song_stats.groupby('cluster_name')[CFEATS].mean().round(2)
        st.dataframe(cp, use_container_width=True)
        dist = song_stats['cluster_name'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(dist.index, dist.values, color=colors[:len(dist)])
        ax.set_xlabel("Cluster"); ax.set_ylabel("# Songs")
        fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
        ax.tick_params(colors='white', axis='x', rotation=20)
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig); plt.close()

    st.subheader("🔎 Songs in Each Cluster")
    selected_cluster = st.selectbox("Select Cluster", list(cluster_names.values()))
    cluster_songs = song_stats[song_stats['cluster_name'] == selected_cluster][
        ['song','artist','days_on_chart','avg_rank','avg_popularity','rank_volatility']
    ].sort_values('avg_popularity', ascending=False)
    cluster_songs.columns = ['Song','Artist','Days','Avg Rank','Avg Pop','Volatility']
    st.dataframe(cluster_songs.reset_index(drop=True), use_container_width=True, height=400)
