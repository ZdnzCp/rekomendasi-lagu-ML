import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="Rekomendasi Lagu Berdasarkan Lagu Favoritmu",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Background gradient */
    .stApp {
        background: linear-gradient(160deg, #1a0b2e 0%, #16213e 45%, #0f3460 100%);
    }

    /* Hero header */
    .hero-box {
        text-align: center;
        padding: 2.2rem 1rem 1.8rem 1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(255,94,148,0.15), rgba(120,80,255,0.15));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.8rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff5e94, #ff9a5e, #ffd35e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #d0d0e8;
        font-size: 0.95rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Section labels */
    .section-label {
        color: #ff9a5e;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
        letter-spacing: 0.3px;
    }

    /* Text input & selectbox tweaks */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #f0f0f5 !important;
    }
    .stTextInput input::placeholder {
        color: #9090b0 !important;
    }

    /* Slider label */
    .stSlider label, .stTextInput label, .stSelectbox label {
        color: #d0d0e8 !important;
        font-weight: 500 !important;
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff5e94, #ff9a5e);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        border-radius: 14px;
        border: none;
        box-shadow: 0 6px 18px rgba(255,94,148,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(255,94,148,0.5);
        color: white;
    }

    /* Result card container */
    .song-card {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: background 0.15s ease, transform 0.15s ease;
    }
    .song-card:hover {
        background: rgba(255,255,255,0.09);
        transform: translateX(3px);
    }
    .song-rank {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ff9a5e;
        min-width: 2.2rem;
    }
    .song-info { flex: 1; padding-left: 0.6rem; }
    .song-title { font-weight: 700; font-size: 1.02rem; color: #f5f5fa; }
    .song-artist { font-size: 0.87rem; color: #b8b8d0; margin-top: 1px; }
    .song-meta { text-align: right; }
    .genre-badge {
        display: inline-block;
        background: rgba(120,80,255,0.25);
        color: #c9b8ff;
        border: 1px solid rgba(120,80,255,0.4);
        border-radius: 20px;
        padding: 2px 11px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .similarity-score {
        font-size: 0.85rem;
        color: #7ee8a0;
        font-weight: 700;
    }
    .popularity-score {
        font-size: 0.75rem;
        color: #9090b0;
    }

    /* Divider styling */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* Caption footer */
    .footer-box {
        text-align: center;
        color: #8888a8;
        font-size: 0.8rem;
        margin-top: 1.5rem;
        line-height: 1.6;
    }

    /* Success/warning/error message tweaks */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model & Data (cache supaya tidak reload tiap interaksi)
# =========================
@st.cache_resource
def load_model():
    with open("model_knn.pkl", "rb") as f:
        model_knn = pickle.load(f)
    return model_knn

@st.cache_data
def load_data():
    X = np.load("X_features.npy", allow_pickle=True).astype(np.float64)
    df = pd.read_csv("df_lookup.csv")
    return X, df

model_knn = load_model()
X, df = load_data()

# =========================
# Fungsi Rekomendasi
# =========================
def rekomendasi_lagu(track_name, artist_name=None, n=10):
    if artist_name:
        mask = (df['track_name'].str.lower() == track_name.lower()) & \
               (df['artist_name'].str.lower() == artist_name.lower())
    else:
        mask = df['track_name'].str.lower() == track_name.lower()

    if mask.sum() == 0:
        return None

    idx = df[mask].index[0]
    distances, indices = model_knn.kneighbors([X[idx]], n_neighbors=n + 1)

    rekomendasi_idx = indices.flatten()[1:]
    similarity_score = 1 - distances.flatten()[1:]

    hasil = df.iloc[rekomendasi_idx][['track_name', 'artist_name', 'genre', 'popularity']].copy()
    hasil['similarity'] = similarity_score.round(3)
    return hasil.reset_index(drop=True)

# =========================
# UI — Header
# =========================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🎵 Sistem Rekomendasi Lagu</div>
    <div class="hero-subtitle">
        Content-Based Filtering menggunakan fitur audio Spotify<br>
        (danceability, energy, tempo, valence, dan lainnya)
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# UI — Input
# =========================
st.markdown('<div class="section-label">🔍 Cari Lagu Favoritmu</div>', unsafe_allow_html=True)
keyword = st.text_input(
    "keyword",
    placeholder="contoh: Shape of You, Ed Sheeran",
    label_visibility="collapsed",
)

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown('<div class="section-label">🎚️ Jumlah Rekomendasi</div>', unsafe_allow_html=True)
    jumlah_rekomendasi = st.slider("jumlah", min_value=5, max_value=20, value=10, label_visibility="collapsed")

pilihan = None
if keyword:
    mask_cari = df['track_name'].str.contains(keyword, case=False, na=False) | \
                df['artist_name'].str.contains(keyword, case=False, na=False)
    hasil_cari = df[mask_cari]

    if hasil_cari.empty:
        st.warning("😕 Tidak ada lagu yang cocok. Coba kata kunci lain.")
    else:
        opsi = (hasil_cari['track_name'] + " — " + hasil_cari['artist_name']).drop_duplicates()
        st.markdown(f'<div class="section-label">🎯 Ditemukan {len(opsi)} lagu — pilih salah satu</div>', unsafe_allow_html=True)
        pilihan = st.selectbox(
            "pilihan",
            options=opsi.iloc[:50],
            label_visibility="collapsed",
        )

st.write("")
cari_clicked = False
if pilihan:
    cari_clicked = st.button("Cari Rekomendasi 🎧")

# =========================
# UI — Hasil
# =========================
if cari_clicked:
    track_name, artist_name = pilihan.rsplit(" — ", 1)
    hasil = rekomendasi_lagu(track_name, artist_name, n=jumlah_rekomendasi)

    if hasil is None:
        st.error("Lagu tidak ditemukan di dataset.")
    else:
        st.markdown(
            f'<div style="text-align:center; color:#7ee8a0; font-weight:600; margin: 1rem 0;">'
            f'✨ Rekomendasi lagu mirip dengan <b>{track_name} — {artist_name}</b></div>',
            unsafe_allow_html=True
        )

        for i, row in hasil.iterrows():
            st.markdown(f"""
            <div class="song-card">
                <div class="song-rank">#{i+1}</div>
                <div class="song-info">
                    <div class="song-title">{row['track_name']}</div>
                    <div class="song-artist">{row['artist_name']}</div>
                </div>
                <div class="song-meta">
                    <div class="genre-badge">{row['genre']}</div>
                    <div class="similarity-score">{row['similarity']*100:.1f}% match</div>
                    <div class="popularity-score">Popularitas: {row['popularity']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📊 Lihat sebagai tabel"):
            st.dataframe(hasil, use_container_width=True)

st.divider()
st.markdown("""
<div class="footer-box">
    🎓 Proyek Akhir Machine Learning — Content-Based Filtering (Spotify Features Dataset)<br>
    Muhammad Zaidan Yazid Ilmani &nbsp;•&nbsp; Maulana Auliya Firdaus
</div>
""", unsafe_allow_html=True)
