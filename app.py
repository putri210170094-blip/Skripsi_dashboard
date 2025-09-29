# Streamlit: library untuk membuat web app interaktif dengan Python
import streamlit as st

# Pandas dan Numpy: untuk manipulasi data dan operasi numerik
import pandas as pd
import numpy as np

# Seaborn dan Matplotlib: untuk visualisasi data
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn: untuk machine learning
# train_test_split: membagi data menjadi data latih dan uji
# DecisionTreeClassifier & plot_tree: untuk membangun dan memvisualisasikan pohon keputusan
# accuracy_score & classification_report: untuk evaluasi model
# LabelEncoder: untuk mengubah variabel kategorikal menjadi numerik
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# datetime.timedelta: untuk operasi tanggal dan waktu
from datetime import timedelta

# PySwarms: library optimisasi berbasis Particle Swarm Optimization
import pyswarms as ps

# BytesIO: untuk menangani data biner di memori (misal untuk file download)
from io import BytesIO

# export_text: untuk mengekstrak representasi teks dari pohon keputusan
from sklearn.tree import export_text

# time: untuk operasi terkait waktu (misal delay atau pengukuran waktu)
import time

# Plotly: untuk visualisasi interaktif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# base64: untuk encoding/decoding data (misal file download atau gambar)
import base64

# hashlib: untuk enkripsi atau hashing (misal password)
import hashlib

# psycopg2: library untuk koneksi dan eksekusi query ke database PostgreSQL
from psycopg2 import sql
import psycopg2

# os: untuk operasi terkait sistem seperti environment variable dan file path
import os

# dotenv: untuk memuat file .env ke environment variables
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Konfigurasi halaman
st.set_page_config(
    page_title="KLASIFIKASI DURASI RAWAT INAP PASIEN SKIZOFRENIA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling dengan background blue sky dan animasi
st.markdown("""
<style>
    /* Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
    
    /* Reset default Streamlit styles */
    .stApp {
        background: transparent !important;
    }
    
    .main .block-container {
        padding-top: 0;
        padding-bottom: 0;
        max-width: 1200px;
    }
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        overflow: auto;
        background: transparent !important;
    }
    
    /* Blue Sky Animated Background */
    .sky-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 50%, #fbc2eb 100%);
        z-index: -2;
        overflow: hidden;
    }

    .cloud {
        position: absolute;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 50%;
        box-shadow: 0 0 30px rgba(255, 255, 255, 0.7);
        animation: float linear infinite;
        opacity: 0.8;
    }
    
    @keyframes float {
        0% {
            transform: translateX(-100px) translateY(0);
        }
        100% {
            transform: translateX(calc(100vw + 100px)) translateY(-20px);
        }
    }
    
    .main {
        position: relative;
        z-index: 1;
        min-height: 100vh;
    }
    
    /* Glass effect untuk login container */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.15);
        padding:40px;
        border-radius: 8px;
        margin: 20px 20px;
        margin-bottom:20px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        text-align: center;
    }

    /* Glassmorphism Effect */
    .glass {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.2);
        padding: 40px;
        margin-bottom: 30px;
        transition: all 0.4s ease;
        overflow: hidden;
        position: relative;
    }
    
    .glass::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(to bottom right, 
            rgba(255, 255, 255, 0.1), 
            rgba(255, 255, 255, 0.05));
        transform: rotate(30deg);
        z-index: -1;
    }
    
    .glass:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 50px rgba(31, 38, 135, 0.3);
    }
    
    /* Navigation Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        justify-content: end;
        background: transparent;
        padding: 0 20px;
        margin-bottom: 40px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre;
        background: rgba(255, 255, 255, 0.39);
        border-radius: 16px;
        font-weight: 600;
        padding: 0 32px;
        color: #005;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        font-size: 1.1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.55);
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.9);
        color: #008080;
        box-shadow: 0 8px 20px rgba(13, 71, 161, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Button Styles */
    .stButton button {
        background: linear-gradient(120deg, #a1c4fd 51%, #fbc2eb 100%);
        color: white;
        border: none;
        padding: 16px 40px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
        transition: all 0.3s ease;
        font-weight: 600;
        box-shadow: 0 6px 15px rgba(13, 71, 161, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(100deg, #a1c4fd 51%, #fbc2eb 100%);
        transition: 0.5s;
    }
    
    .stButton button:hover::before {
        left: 100%;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #a1c4fd 51%, #fbc2eb 100%);
        box-shadow: 0 10px 25px rgba(33, 150, 243, 0.4);
        transform: translateY(-3px);
    }
    
    .stButton button:active {
        transform: translateY(1px);
        box-shadow: 0 4px 10px rgba(33, 150, 243, 0.4);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 16px;
        background: rgba(255, 255, 255, 0.85);
        transition: all 0.3s ease;
        font-size: 16px;
        color: #0d47a1;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stTextInput>div>div>input:focus {
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.3);
        border: 1px solid rgba(33, 150, 243, 0.5);
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #90a4ae;
    }
    
    /* Card Styles */
    .feature-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        height: 100%;
        transition: all 0.4s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #0d47a1, #1976d2);
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.2);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 20px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    .card-content {
        font-size: 1.05rem;
        color: #e3f2fd;
        line-height: 1.6;
    }
    
    /* Success and Error Messages */
    .stAlert {
        border-radius: 16px;
        backdrop-filter: blur(10px);
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 18px 10px;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 40px;
        background: linear-gradient(135deg, rgba(0,77,64,0.6), rgba(0,128,128,0.4));
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border-radius: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        line-height: 0.5;
        background-image: linear-gradient(
            to right,
            rgba(0, 77, 77, 0.25),
            rgba(0, 128, 128, 0.15),
            rgba(0, 77, 77, 0.25)
        );
        box-shadow: 0 -4px 20px rgba(0, 77, 77, 0.3);
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease-in-out;
    }

    .footer:hover {
        background: rgba(0, 128, 128, 0.2);
        background-image: linear-gradient(
            to right,
            rgba(0, 128, 128, 0.3),
            rgba(0, 128, 128, 0.2),
            rgba(0, 128, 128, 0.3)
        );
        box-shadow: 0 -6px 25px rgba(0, 77, 77, 0.5);
        transform: translateY(-2px);
    }
    
    /* Loading Animation Pastel Dream Style */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 120px;
    }

    .loading-spinner {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        border: 6px solid transparent;
        border-top: 6px solid #a1c4fd;  /* biru pastel */
        border-right: 6px solid #fbc2eb; /* pink pastel */
        animation: spin 1.2s linear infinite, glow 2s ease-in-out infinite alternate;
        box-shadow: 0 0 15px rgba(161, 196, 253, 0.5);
    }

    /* Spin effect */
    @keyframes spin {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Glow soft pastel */
    @keyframes glow {
        0%   { box-shadow: 0 0 10px rgba(161, 196, 253, 0.3), 0 0 20px rgba(251, 194, 235, 0.2); }
        100% { box-shadow: 0 0 20px rgba(161, 196, 253, 0.7), 0 0 30px rgba(251, 194, 235, 0.5); }
    }

    
    /* Login Form Specific Styles */
    .login-container {
        max-width: 500px;
        margin: 10px;
        padding: 50px 0;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 40px;
    }
    
    .login-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #004;
        margin-bottom: 10px;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.2);
    }
    
    .login-subtitle {
        font-size: 1.1rem;
        color: #008080;
        opacity: 0.9;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .glass {
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 20px;
            font-size: 1rem;
        }
        
        .feature-card {
            margin-bottom: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# JavaScript untuk animasi awan dan efek interaktif
st.markdown("""
<script>
// Function to create clouds
function createClouds() {
    const sky = document.querySelector('.sky-background');
    if (!sky) return;
    
    // Clear existing clouds
    sky.innerHTML = '';
    
    // Create multiple clouds
    for (let i = 0; i < 7; i++) {
        const cloud = document.createElement('div');
        cloud.classList.add('cloud');
        
        // Random size and position
        const width = Math.random() * 120 + 80;
        const height = width * 0.5;
        const top = Math.random() * 60;
        const left = -width;
        const duration = Math.random() * 40 + 40;
        
        cloud.style.width = `${width}px`;
        cloud.style.height = `${height}px`;
        cloud.style.top = `${top}%`;
        cloud.style.left = `${left}px`;
        cloud.style.animationDuration = `${duration}s`;
        cloud.style.animationDelay = `${Math.random() * 20}s`;
        
        // Add some variation to cloud shape
        cloud.style.borderRadius = '50%';
        
        sky.appendChild(cloud);
    }
}

// Add ripple effect to buttons
function addRippleEffect() {
    const buttons = document.querySelectorAll('.stButton button');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const x = e.pageX - this.offsetLeft;
            const y = e.pageY - this.offsetTop;
            
            const ripple = document.createElement('span');
            ripple.classList.add('ripple-effect');
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
}

// Run when page loads
window.addEventListener('load', function() {
    createClouds();
    addRippleEffect();
});

// Recreate clouds when page changes (for Streamlit)
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(createClouds, 100);
    setTimeout(addRippleEffect, 500);
});
</script>
""", unsafe_allow_html=True)

# Tambah background biru langit
st.markdown('<div class="sky-background"></div>', unsafe_allow_html=True)

# Fungsi hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Inisialisasi koneksi database
def init_db_connection():
    try:
        # Konfigurasi koneksi database
        conn = psycopg2.connect(
            user=os.getenv("ST_USER"),
            password=os.getenv("ST_PASSWORD"),
            host=os.getenv("ST_HOST"),
            port=os.getenv("ST_PORT"),
            dbname=os.getenv("ST_DATABASE")
        )
        
        return conn
    except Exception as e:
        st.error(f"❌ Gagal terhubung ke database: {str(e)}")
        return None

# Fungsi untuk membuat tabel jika belum ada
def create_tables(conn):
    try:
        cursor = conn.cursor()
        
        # Buat tabel users jika belum ada
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Buat tabel patients jika belum ada
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                age INTEGER NOT NULL,
                gender VARCHAR(10) NOT NULL,
                duration INTEGER NOT NULL,
                classification VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        st.info("✅ Tabel database sudah siap")
    except Exception as e:
        st.error(f"❌ Gagal membuat tabel: {str(e)}")

# Fungsi untuk login dengan animasi loading
def login_user(email, password):
    try:
        hashed_password = hash_password(password)
        conn = init_db_connection()
        
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, email, password, name FROM users WHERE email = %s AND password = %s",
                (email, hashed_password)
            )
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if user:
                # Convert tuple to dictionary
                user_dict = {
                    "id": user[0],
                    "email": user[1],
                    "password": user[2],
                    "name": user[3]
                }
                return user_dict
            else:
                return None
        else:
            # Fallback ke data dummy jika koneksi database gagal
            dummy_users = [
                {"email": "admin@rsudmuyangkute.com", "password": hash_password("admin123"), "name": "Administrator"},
                {"email": "dokter@rsudmuyangkute.com", "password": hash_password("dokter123"), "name": "Dokter Spesialis"}
            ]
            
            for user in dummy_users:
                if user["email"] == email and user["password"] == hashed_password:
                    return user
            return None
    except Exception as e:
        st.error(f"Error logging in: {e}")
        return None

# Fungsi untuk menampilkan animasi loading
def show_loading_animation():
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
    </div>
    """, unsafe_allow_html=True)

# Fungsi untuk halaman login
def show_login_page():
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    
    # Login header
    st.markdown("<h2 class='login-title' style='color:#006;'>🔐 Masuk ke Sistem</h2>", unsafe_allow_html=True)
    st.markdown("<p class='login-subtitle'>Silakan masukkan kredensial Anda untuk mengakses sistem</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        (col1,) = st.columns([1])

        with col1:
            email = st.text_input("📧 Email", placeholder="Masukkan email Anda", key="login_email")
            password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password Anda", key="login_password")
            
            submit_button = st.form_submit_button("🚀 Masuk", use_container_width=True)

            # State untuk loading
            if "loading" not in st.session_state:
                st.session_state.loading = False

            if submit_button:
                if email and password:
                    st.session_state.loading = True
                    placeholder = st.empty()  # container sementara
                    with placeholder:
                        show_loading_animation()

                    time.sleep(1.5)
                    user = login_user(email, password)

                    st.session_state.loading = False  # matikan loading
                    placeholder.empty()  # hapus loading

                    if user:
                        st.session_state.user = user
                        st.session_state.page = "dashboard"
                        st.success("✅ Login berhasil! Mengalihkan...")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("❌ Email atau password salah")
                else:
                    st.warning("⚠️ Harap isi semua field")

    # Tutup div glass
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== FUNGSI MAPE =====================
def calculate_mape(y_true, y_pred):
    """
    Menghitung Mean Absolute Percentage Error (MAPE)
    Rumus: (1/n) * Σ(|y_true - y_pred| / y_true) * 100%
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Hindari pembagian 0 dengan mengganti nilai 0 kecil (epsilon)
    epsilon = 1e-10  
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

# ===================== FUNGSI PRA_PROSES DATA =====================
@st.cache_data(show_spinner="Memproses data...")
def preprocess_data(df_raw):
    df = df_raw.copy()

    # 1. Pembersihan Kolom Tidak Relevan
    kolom_hapus = ['No', 'Nomor', 'ID'] 
    for kolom in kolom_hapus:
        if kolom in df.columns:
            df.drop(columns=[kolom], inplace=True)

    # 2. Konversi Tanggal
    if 'Tanggal Masuk' in df.columns and 'Tanggal Keluar' in df.columns:
        df['Tanggal Masuk'] = pd.to_datetime(df['Tanggal Masuk'])
        df['Tanggal Keluar'] = pd.to_datetime(df['Tanggal Keluar'])
        df['Durasi Rawat Inap (Hari)'] = (df['Tanggal Keluar'] - df['Tanggal Masuk']).dt.days
        df = df[df['Durasi Rawat Inap (Hari)'] >= 0]
    else:
        st.error("❌ Kolom tanggal tidak ditemukan. Pastikan ada kolom 'Tanggal Masuk' dan 'Tanggal Keluar'.")
        st.stop()

    # 3. Menangani Data Kosong
    df.dropna(inplace=True)
    
    # 4. Klasifikasi Durasi
    def classify_duration(days):
        if days <= 5: return 'Singkat'
        elif 6 <= days <= 10: return 'Sedang'
        else: return 'Lama'
    
    df['Kategori Durasi'] = df['Durasi Rawat Inap (Hari)'].apply(classify_duration)

    # 5. Encoding
    if 'Diagnosa' in df.columns:
        if 'Skizofrenia' in df['Diagnosa'].unique():
            df = df[df['Diagnosa'].str.contains('Skizofrenia', case=False, na=False)]
        df.drop(columns=['Diagnosa'], inplace=True)

    le = LabelEncoder()
    df['Kategori Durasi Encoded'] = le.fit_transform(df['Kategori Durasi'])
    
    # Inisialisasi label_encoder di session state
    if "label_encoder" not in st.session_state:
        st.session_state.label_encoder = LabelEncoder()
    df['Kategori Durasi Encoded'] = st.session_state.label_encoder.fit_transform(df['Kategori Durasi'])    

    return df

@st.cache_resource(show_spinner="Melatih model C4.5 dan mengoptimalkan dengan PSO...")
def train_models(X_train, X_test, y_train, y_test):
    # 1. Model dasar (baseline)
    dt_base = DecisionTreeClassifier(random_state=42)
    dt_base.fit(X_train, y_train)
    y_pred_base = dt_base.predict(X_test)

    # 2. Hitung akurasi dasar
    base_accuracy = accuracy_score(y_test, y_pred_base)

    # 3. Hitung MAPE untuk model dasar
    base_mape = calculate_mape(y_test, y_pred_base)

    # 4. Fungsi untuk optimasi PSO
    def f_objective(params):
        n_particles = params.shape[0]
        fitness = []
        for i in range(n_particles):
            max_depth = int(params[i, 0])          # Parameter 1: max_depth
            min_samples = int(params[i, 1])        # Parameter 2: min_samples_split
            
            # Batasan nilai parameter
            max_depth = max(1, min(max_depth, 20))  # Pastikan 1 <= max_depth <= 20
            min_samples = max(2, min(min_samples, 10))  # Pastikan 2 <= min_samples <= 10

            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples,
                criterion='entropy',
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            fitness.append(-accuracy)  # Minimalkan negatif akurasi
        
        return np.array(fitness)
    
    # 5. Konfigurasi PSO
    bounds = (np.array([1, 2]), np.array([20, 10]))  # Batasan parameter
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

    # 6. Jalankan optimasi
    cost, pos = optimizer.optimize(f_objective, iters=50, verbose=False)

    # 7. Tetapkan nilai optimal
    best_max_depth = int(pos[0])        # Parameter optimal 1
    best_min_samples = int(pos[1])      # Parameter optimal 2

    # 8. Bangun Model dengan Parameter Optimal
    dt_optimized = DecisionTreeClassifier(
        max_depth=best_max_depth,
        min_samples_split=best_min_samples,
        criterion='entropy',
        random_state=42
    )
    dt_optimized.fit(X_train, y_train)
    y_pred_optim = dt_optimized.predict(X_test)

    # 9. Hitung akurasi model optimasi
    optim_accuracy = accuracy_score(y_test, y_pred_optim)

    # 10. Hitung MAPE untuk model optimasi
    optim_mape = calculate_mape(y_test, y_pred_optim)

    # 11. Kembalikan hasil
    return {
        "base_model": dt_base,
        "base_accuracy": base_accuracy,
        "base_mape": base_mape,
        "optimized_model": dt_optimized,
        "optim_accuracy": optim_accuracy,
        "optim_mape": optim_mape,
        "y_test": y_test,
        "y_pred_base": y_pred_base,
        "y_pred_optim": y_pred_optim,
        "best_params": {
            "max_depth": best_max_depth,
            "min_samples": best_min_samples
        }
    }

# Fungsi untuk halaman dashboard
def show_dashboard():
    # Dashboard styling
    st.markdown("""
    <style>
        /* Global Styles */
        html, body, [class*="css"] {
            background: transparent !important;
        }
        
        .main {
            background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 50%, #fbc2eb 100%) !important;
            padding: 0;
        }
        
        .stApp {
            background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 50%, #fbc2eb 100%) !important;
        }
        
        /* Header Styles */
        .main-header {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 0 0 20px 20px; 
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            color: #2b5876;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #2b5876;
        }
        
        .main-subtitle {
            font-size: 1.2rem;
            color: #4b86b4;
            margin-bottom: 1rem;
        }
        
        /* Card Styles */
        .custom-card {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.5);
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .custom-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(31, 38, 135, 0.2);
        }
        
        /* Button Styles */
        .stButton>button {
            background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
            color: #2b5876;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.7rem 1.5rem;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(161, 196, 253, 0.3);
        }
        
        .stButton>button:hover {
            background: linear-gradient(90deg, #c2e9fb 0%, #fbc2eb 100%);
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(194, 233, 251, 0.4);
            color: #2b5876;
        }
        
        /* Sidebar Styles */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.2) !important;
            backdrop-filter: blur(10px);
            width:290px;
            border-right: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        [data-testid="stSidebar"] .stButton>button {
            background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
            width: 100%;
            color: #2b5876;
        }
        
        [data-testid="stSidebar"] .stButton>button:hover {
            background: linear-gradient(90deg, #c2e9fb 0%, #fbc2eb 100%);
            color: #2b5876;
        }
        
        [data-testid="stSidebar"] .stExpander {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        [data-testid="stSidebar"] .streamlit-expanderHeader {
            color: #2b5876;
            font-weight: 600;
        }
        
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] .stMarkdown li,
        [data-testid="stSidebar"] .stText {
            color: #2b5876 !important;
        }
        
        .sidebar-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 2rem;
            text-align: center;
            color: #2b5876;
        }
        
        /* Tab Styles */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.5);
            border-radius: 10px 10px 0 0;
            padding: 1rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            color: #2b5876;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
            color: #2b5876;
        }
        
        /* Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(5px);
            border-radius: 15px;
            padding: 1.5rem;
            color: #2b5876;
            text-align: center;
            box-shadow: 0 8px 20px rgba(161, 196, 253, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        .metric-title {
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: #4b86b4;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #2b5876;
        }
        
        /* Animation Keyframes */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 1s ease forwards;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
        }
        
        /* File Uploader */
        .stFileUploader > div > div {
            border: 2px dashed #a1c4fd;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.5);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            font-weight: 600;
            color: #2b5876;
        }
        
        /* Upload Success Message */
        .upload-success {
            background: rgba(46, 204, 113, 0.2);
            border: 1px solid #2ecc71;
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            color: #27ae60;
            text-align: center;
        }
        
        /* Download Link */
        .download-link {
            display: inline-block;
            background: linear-gradient(90deg, #a1c4fd 0%, #c2e9fb 100%);
            color: #2b5876;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-decoration: none;
            margin-top: 10px;
            margin-bottom:10px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .download-link:hover {
            background: linear-gradient(90deg, #c2e9fb 0%, #fbc2eb 100%);
            color: #2b5876;
            text-decoration: none;
        }
    </style>
    """, unsafe_allow_html=True)

    # ===================== JUDUL & HEADER =====================
  
    
    st.markdown("""
    <div class="main-header fade-in">
        <h1 class="main-title">🏥 Klasifikasi Durasi Rawat Inap Pasien Skizofrenia</h1>
        <p class="main-subtitle">Analisis Canggih dengan Algoritma C4.5 dan Particle Swarm Optimization (PSO)</p>
    </div>
    """, unsafe_allow_html=True)

    # ===================== SIDEBAR =====================
    with st.sidebar:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.3); 
                    backdrop-filter: blur(10px);
                    padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;
                    border: 1px solid rgba(255, 255, 255, 0.3);">
            <h1 style="color: #2b5876; margin: 0; font-size: 1.5rem;">📊 Menu Analisis</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        st.markdown(f"""
        {st.session_state.user['name']} <br>
        {st.session_state.user['email']}
        """, unsafe_allow_html=True)
        

        # Inisialisasi state
        if "show_logout_confirm" not in st.session_state:
            st.session_state.show_logout_confirm = False
        if "logged_out" not in st.session_state:
            st.session_state.logged_out = False

        # Tombol utama Logout
        with st.container():
            if st.button("🚪 Keluar", use_container_width=True):
                st.session_state.show_logout_confirm = True

        # Jika user klik keluar → tampilkan konfirmasi
        if st.session_state.show_logout_confirm:
            st.warning("Apakah Anda yakin ingin keluar dari sistem?")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Ya",  use_container_width=True):
                    st.session_state.clear()   # clear semua state
                    st.session_state.logged_out = True  # tandai sudah logout
                    st.session_state.show_logout_confirm = False  # sembunyikan konfirmasi

            with col2:
                if st.button("❌ Batal",  use_container_width=True):
                    st.session_state.show_logout_confirm = False  # sembunyikan konfirmasi

        # Tampilkan pesan sukses kalau logout
        if st.session_state.logged_out:
            st.success("Anda telah logout")
            time.sleep(1)
            st.session_state.logged_out = False  # reset agar tidak muncul lagi
            st.rerun()

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "📁 Unggah Data Pasien (Excel)", 
            type=["xlsx"],
            help="File harus mengandung kolom: Tanggal Masuk, Tanggal Keluar"
        )
        
        # Tampilkan status file yang diunggah
        if uploaded_file is not None:
            st.markdown(f"""
            <div class="upload-success">
                <p style="margin: 0; font-weight: bold;">✅ File berhasil diunggah!</p>
                <p style="margin: 0; font-size: 0.9rem;">Nama file: {uploaded_file.name}</p>
                <p style="margin: 0; font-size: 0.9rem;">Ukuran: {uploaded_file.size / 1024:.2f} KB</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("ℹ️ Panduan Penggunaan", expanded=True):
            st.info("""
            1. **Unggah file Excel** dengan data pasien
            2. Data akan diproses secara otomatis
            3. Jelajahi berbagai tab untuk melihat analisis
            4. Gunakan fitur prediksi untuk kasus baru
            """)
            
            # Link download template
            st.markdown("""
            <div style="margin-top: 0.2rem;">
                <p style="margin-bottom: 0.5rem; font-weight: bold;">📥 Template Dataset:</p>
                <a href="https://xhydpuugdxcrhmvtwzbd.supabase.co/storage/v1/object/public/dataset/DataMentah.xlsx" class="download-link" target="_blank">
                    Template Data.xlsx
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="padding: 1rem; background: rgba(255, 255, 255, 0.3); border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.3);">
            <h4 style="margin-bottom: 0.5rem; color: #2b5876;">📋 Informasi Dataset</h4>
            <p style="font-size: 0.9rem; margin-bottom: 0; color: #2b5876;">Pastikan dataset Anda mengandung:</p>
            <ul style="font-size: 0.8rem; color: #2b5876;">
                <li>Kolom Tanggal Masuk</li>
                <li>Kolom Tanggal Keluar</li>
                <li>Kolom Diagnosa</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
                    **Dikembangkan oleh:** <br>
                    Putri Agustina Dewi <br>
                    Teknik Informatika, Universitas Malikussaleh <br>
                    © 2025
                    """, unsafe_allow_html=True)

    # ===================== MAIN APP =====================
    df_raw = None

    if uploaded_file:
        try:
            with st.spinner('🔄 Memuat dan memproses data...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                df_raw = pd.read_excel(uploaded_file)
                
                df_processed = preprocess_data(df_raw)
            
            # Verifikasi kolom
            required_cols = ['Durasi Rawat Inap (Hari)', 'Kategori Durasi Encoded']
            if not all(col in df_processed.columns for col in required_cols):
                st.error("❌ Format data tidak valid. Pastikan file memiliki kolom yang dibutuhkan.")
                st.stop()
                
            # Siapkan data training
            X = df_processed[['Durasi Rawat Inap (Hari)']]
            y = df_processed['Kategori Durasi Encoded']
            
            if len(df_processed) < 2:
                st.error("⚠ Data terlalu sedikit. Minimal diperlukan 2 sampel.")
                st.stop()
                
            if y.nunique() < 2:
                st.warning("⚠ Hanya ada 1 kategori durasi. Tidak bisa dilakukan klasifikasi.")
                st.stop()
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42, 
                stratify=y
            )
            
            with st.spinner("🧠 Melatih model dengan algoritma C4.5 dan PSO..."):
                model_results = train_models(X_train, X_test, y_train, y_test)
            
            # ===================== TABBED INTERFACE =====================
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Ikhtisar Data", 
                "📈 Perbandingan Model", 
                "🌳 Pohon Keputusan", 
                "🔍 Prediksi Baru", 
                "🏥 Estimasi Ruangan"
            ])
            
            with tab1:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="custom-card"><h3>📋 Preview Data</h3>', unsafe_allow_html=True)
                    
                    # Buat DataFrame untuk preview dengan nomor urut mulai dari 1
                    df_preview = df_processed.copy()
                    df_preview.insert(0, 'No', range(1, len(df_preview) + 1))
                    
                    # Fitur pencarian
                    search_term = st.text_input("🔍 Cari data...", placeholder="Ketik untuk mencari di semua kolom")
                    
                    if search_term:
                        # Filter data berdasarkan pencarian
                        mask = df_preview.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                        df_display = df_preview[mask]
                    else:
                        df_display = df_preview
                    
                    # Tampilkan informasi jumlah data yang ditampilkan
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.5); padding: 8px; border-radius: 8px; margin-bottom: 10px;">
                        <p style="margin: 0; font-size: 0.9rem; color: #2b5876;">
                        <strong>📊 Menampilkan:</strong> {len(df_display)} dari {len(df_preview)} records
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tampilkan data
                    st.dataframe(df_display, use_container_width=True, hide_index=True)
                    
                    # Download button untuk data lengkap
                    if len(df_display) > 0:
                        csv = df_display.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Data yang Ditampilkan (CSV)",
                            data=csv,
                            file_name="data_pasien.csv",
                            mime="text/csv"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
                    
                with col2:
                    st.markdown('<div class="custom-card"><h3>📈 Statistik Data</h3>', unsafe_allow_html=True)
                    st.metric("Total Sampel", len(df_processed))
                    st.metric("Jumlah Fitur", len(df_processed.columns))
                    st.metric("Rata-rata Durasi", f"{df_processed['Durasi Rawat Inap (Hari)'].mean():.1f} Hari")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown('<div class="custom-card"><h3>📊 Distribusi Durasi</h3>', unsafe_allow_html=True)
                    fig1 = px.histogram(
                        df_processed, 
                        x='Durasi Rawat Inap (Hari)',
                        nbins=15,
                        color_discrete_sequence=['#a1c4fd']
                    )
                    fig1.update_layout(
                        yaxis_title='Jumlah Pasien',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col4:
                    st.markdown('<div class="custom-card"><h3>🍩 Distribusi Kategori</h3>', unsafe_allow_html=True)
                    kategori_count = df_processed['Kategori Durasi'].value_counts()
                    fig2 = px.pie(
                        values=kategori_count.values,
                        names=kategori_count.index,
                        color_discrete_sequence=['#a1c4fd', '#c2e9fb', '#fbc2eb']
                    )
                    fig2.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="custom-card"><h3>📌 Informasi Data</h3>', unsafe_allow_html=True)
                col5, col6, col7 = st.columns(3)
                
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Singkat (1-5 hari)</div>
                        <div class="metric-value">{len(df_processed[df_processed['Kategori Durasi'] == 'Singkat'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col6:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Sedang (6-10 hari)</div>
                        <div class="metric-value">{len(df_processed[df_processed['Kategori Durasi'] == 'Sedang'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col7:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Lama (>10 hari)</div>
                        <div class="metric-value">{len(df_processed[df_processed['Kategori Durasi'] == 'Lama'])}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="custom-card"><h3>⚙️ Parameter Model Optimal</h3>', unsafe_allow_html=True)
                col8, col9 = st.columns(2)
                
                with col8:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Max Depth</div>
                        <div class="metric-value">{model_results['best_params']['max_depth']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col9:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Min Samples Split</div>
                        <div class="metric-value">{model_results['best_params']['min_samples']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                col10, col11 = st.columns(2)
                
                with col10:
                    st.markdown('<div class="custom-card"><h3>📊 Model C4.5 Dasar</h3>', unsafe_allow_html=True)
                    st.metric("Akurasi", f"{model_results['base_accuracy']:.2%}", delta=None)
                    st.metric("MAPE", f"{model_results['base_mape']:.2f}%", delta=None)
                    
                    # Visualisasi akurasi
                    fig_base = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = model_results['base_accuracy'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Akurasi Model Dasar (%)"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#a1c4fd"},
                            'steps': [
                                {'range': [0, 70], 'color': "rgba(255, 255, 255, 0.3)"},
                                {'range': [70, 90], 'color': "rgba(194, 233, 251, 0.5)"},
                                {'range': [90, 100], 'color': "rgba(161, 196, 253, 0.7)"}
                            ],
                        }
                    ))
                    fig_base.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_base, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col11:
                    st.markdown('<div class="custom-card"><h3>🚀 Model C4.5+PSO</h3>', unsafe_allow_html=True)
                    improvement = model_results['optim_accuracy'] - model_results['base_accuracy']
                    st.metric("Akurasi", f"{model_results['optim_accuracy']:.2%}", 
                             delta=f"{improvement:.2%}" if improvement > 0 else None)
                    st.metric("MAPE", f"{model_results['optim_mape']:.2f}%", 
                             delta=f"{- (model_results['optim_mape'] - model_results['base_mape']):.2f}%" 
                             if model_results['optim_mape'] < model_results['base_mape'] else None)
                    
                    # Visualisasi akurasi
                    fig_optim = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = model_results['optim_accuracy'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Akurasi Model Optimasi (%)"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#fbc2eb"},
                            'steps': [
                                {'range': [0, 70], 'color': "rgba(255, 255, 255, 0.3)"},
                                {'range': [70, 90], 'color': "rgba(194, 233, 251, 0.5)"},
                                {'range': [90, 100], 'color': "rgba(251, 194, 235, 0.7)"}
                            ],
                        }
                    ))
                    fig_optim.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_optim, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Laporan klasifikasi
                st.markdown('<div class="custom-card"><h3>📋 Laporan Klasifikasi</h3>', unsafe_allow_html=True)
                col12, col13 = st.columns(2)
                
                with col12:
                    st.markdown("**Model Dasar**")
                    st.code(
                        classification_report(
                            model_results['y_test'], 
                            model_results['y_pred_base'],
                            target_names=['Singkat', 'Sedang', 'Lama']
                        )
                    )
                    
                with col13:
                    st.markdown("**Model Optimasi**")
                    st.code(
                        classification_report(
                            model_results['y_test'], 
                            model_results['y_pred_optim'],
                            target_names=['Singkat', 'Sedang', 'Lama']
                        )
                    )
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Download hasil prediksi
                output_df = X_test.copy()
                if "label_encoder" in st.session_state:
                    output_df['Aktual'] = st.session_state['label_encoder'].inverse_transform(model_results['y_test'])
                    output_df['Prediksi (Dasar)'] = st.session_state['label_encoder'].inverse_transform(model_results['y_pred_base'])
                    output_df['Prediksi (Optimasi)'] = st.session_state['label_encoder'].inverse_transform(model_results['y_pred_optim'])
                else:
                    st.error("Label encoder belum diinisialisasi.")
                    st.stop()
                
                excel_buffer = BytesIO()
                output_df.to_excel(excel_buffer, index=False)
                
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (Excel)",
                    data=excel_buffer,
                    file_name="prediksi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="custom-card"><h3>🌳 Visualisasi Pohon Keputusan</h3>', unsafe_allow_html=True)
                
                with st.expander("ℹ️ Tentang visualisasi ini"):
                    st.write("""
                    Pohon ini menunjukkan proses pengambilan keputusan model C4.5 yang dioptimasi.
                    Setiap node menunjukkan kondisi pembagian berdasarkan durasi rawat inap.
                    """)
                
                # Visualisasi pohon keputusan
                plt.figure(figsize=(20, 12))
                plot_tree(
                    model_results['optimized_model'],
                    feature_names=['Durasi (hari)'],
                    class_names=['Singkat', 'Sedang', 'Lama'],
                    filled=True,
                    rounded=True,
                    fontsize=10,
                    max_depth=3,  # Menunjukkan 3 level pertama untuk kejelasan
                    impurity=False,
                    proportion=True
                )
                st.pyplot(plt)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Representasi teks
                st.markdown('<div class="custom-card"><h3>📝 Representasi Teks</h3>', unsafe_allow_html=True)
                tree_text = export_text(
                    model_results['optimized_model'],
                    feature_names=['Durasi'],
                    max_depth=3
                )
                st.code(tree_text)
                
                st.download_button(
                    "📥 Unduh Struktur Pohon (TXT)",
                    tree_text,
                    file_name="struktur_pohon.txt"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown('<div class="custom-card"><h3>🔮 Prediksi Pasien Baru</h3>', unsafe_allow_html=True)
                st.markdown("""
                <div style='color:#666; margin-bottom:20px;'>
                     Masukkan tanggal masuk dan keluar pasien untuk memprediksi kategori durasi rawat inap.
                </div>
                """, unsafe_allow_html=True)

                with st.form("form_prediksi"):
                    col14, col15 = st.columns(2)
                    with col14:
                        tgl_masuk = st.date_input("Tanggal Masuk", key="tgl_masuk")
                    with col15:
                        tgl_keluar = st.date_input("Tanggal Keluar", 
                                                    value=tgl_masuk + timedelta(days=7),
                                                    key="tgl_keluar")

                    submit_button = st.form_submit_button("📊 Prediksi Kategori Durasi")

                if submit_button:
                    durasi = (tgl_keluar - tgl_masuk).days
            
                    if durasi < 0:
                        st.error("⛔ Tanggal keluar tidak boleh sebelum tanggal masuk!")
                    else:
                        if 'label_encoder' in st.session_state and 'optimized_model' in model_results:
                            try:
                                # Prediksi kategori
                                prediksi = model_results['optimized_model'].predict([[durasi]])[0]
                                kategori = st.session_state['label_encoder'].inverse_transform([prediksi])[0]
                        
                                # Animasi hasil prediksi
                                with st.spinner('Memprediksi...'):
                                    time.sleep(1)
                                
                                st.success(f"✅ Kategori Durasi Prediksi: **{kategori}**")
                        
                                # Visualisasi hasil prediksi
                                col16, col17 = st.columns([1, 2])
                                
                                with col16:
                                    # Tampilkan indikator kategori
                                    if kategori == 'Singkat':
                                        color = "#a1c4fd"
                                        icon = "⏱️"
                                    elif kategori == 'Sedang':
                                        color = "#c2e9fb"
                                        icon = "⏳"
                                    else:
                                        color = "#fbc2eb"
                                        icon = "⌛"
                                    
                                    fig_pred = go.Figure(go.Indicator(
                                        mode = "number+delta",
                                        value = durasi,
                                        number = {'font': {'size': 40}, 'prefix': icon},
                                        title = {'text': "Durasi (Hari)", 'font': {'size': 20}},
                                        delta = {'reference': 5, 'position': "bottom"},
                                        domain = {'x': [0, 1], 'y': [0, 1]}
                                    ))
                                    
                                    fig_pred.update_layout(
                                        height=300,
                                        paper_bgcolor=color,
                                        font={'color': "#2b5876"}
                                    )
                                    st.plotly_chart(fig_pred, use_container_width=True)
                                
                                with col17:
                                    # Rekomendasi
                                    st.markdown(f"""
                                    <div style="background: rgba(255, 255, 255, 0.7); 
                                                padding: 1.5rem; border-radius: 15px; color: #2b5876;
                                                border: 1px solid rgba(255, 255, 255, 0.5);">
                                        <h3>📋 Rekomendasi Manajemen Ruangan</h3>
                                        <p><strong>Kategori: {kategori}</strong></p>
                                    """, unsafe_allow_html=True)
                                    
                                    if kategori == 'Singkat':
                                        st.markdown("""
                                        - *Ruangan*: Gunakan ruangan dengan turnover cepat
                                        - *Perawatan*: Persiapan discharge mulai hari ke-3
                                        - *Sumber Daya*: 1 perawat per 5 pasien
                                        """)
                                    elif kategori == 'Sedang':
                                        st.markdown("""
                                        - *Ruangan*: Ruang perawatan standar
                                        - *Perawatan*: Evaluasi mingguan
                                        - *Sumber Daya*: 1 perawat per 3 pasien
                                        """)
                                    else:
                                        st.markdown("""
                                        - *Ruangan*: Ruang perawatan jangka panjang
                                        - *Perawatan*: Evaluasi harian
                                        - *Sumber Daya*: 1 perawat per 2 pasien
                                        """)
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                        
                            except Exception as e:
                                st.error(f"⚠ Error dalam prediksi: {str(e)}")
                        else:
                            st.error("🔴 Sistem belum siap! Pastikan:")
                            st.write("1. Data pasien sudah diunggah")
                            st.write("2. Model sudah selesai dilatih")
                            st.stop()
                st.markdown('</div>', unsafe_allow_html=True)

            with tab5:
                st.markdown('<div class="custom-card"><h3>🏥 Estimasi Kebutuhan Ruangan</h3>', unsafe_allow_html=True)
                
                # Hitung distribusi pasien
                distribusi = df_processed['Kategori Durasi'].value_counts().reset_index()
                distribusi.columns = ['Kategori', 'Jumlah Pasien']
                
                # Visualisasi distribusi
                fig_dist = px.bar(
                    distribusi,
                    x='Kategori',
                    y='Jumlah Pasien',
                    color='Kategori',
                    color_discrete_sequence=['#a1c4fd', '#c2e9fb', '#fbc2eb'],
                    text='Jumlah Pasien'
                )
                fig_dist.update_layout(
                    title='Distribusi Pasien Berdasarkan Kategori Durasi',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Rekomendasi alokasi ruangan
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.7); 
                            padding: 1.5rem; border-radius: 15px; color: #2b5876; margin-top: 1rem;
                            border: 1px solid rgba(255, 255, 255, 0.5);">
                    <h3>📊 Rekomendasi Alokasi Ruangan</h3>
                    <p>Berdasarkan data historis:</p>
                """, unsafe_allow_html=True)
                
                col18, col19, col20 = st.columns(3)
                
                with col18:
                    singkat = distribusi[distribusi['Kategori'] == 'Singkat']['Jumlah Pasien'].values
                    singkat_val = singkat[0] if len(singkat) > 0 else 0
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                        <h4>Singkat</h4>
                        <h2>{singkat_val}</h2>
                        <p>{singkat_val/len(df_processed)*100:.1f}% dari total</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col19:
                    sedang = distribusi[distribusi['Kategori'] == 'Sedang']['Jumlah Pasien'].values
                    sedang_val = sedang[0] if len(sedang) > 0 else 0
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                        <h4>Sedang</h4>
                        <h2>{sedang_val}</h2>
                        <p>{sedang_val/len(df_processed)*100:.1f}% dari total</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col20:
                    lama = distribusi[distribusi['Kategori'] == 'Lama']['Jumlah Pasien'].values
                    lama_val = lama[0] if len(lama) > 0 else 0
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.5); border-radius: 10px;">
                        <h4>Lama</h4>
                        <h2>{lama_val}</h2>
                        <p>{lama_val/len(df_processed)*100:.1f}% dari total</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                    <p style="margin-top: 1.5rem;"><strong>📋 Saran Alokasi:</strong></p>
                    <ul>
                        <li>Ruangan singkat: 5-10% dari total kapasitas</li>
                        <li>Ruangan sedang: 15-20% dari total kapasitas</li>
                        <li>Ruangan panjang: 30-40% dari total kapasitas</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"""
            ❌ Terjadi kesalahan:
            {str(e)}
            """)
            st.write("Pastikan format file sesuai dan data lengkap.")

    else:
        # Tampilan awal sebelum upload file
        col21, col22, col23 = st.columns([1, 2, 1])
        
        with st.container():
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem; background: rgba(255, 255, 255, 0.8); 
                        border-radius: 20px; margin-top: 2rem; box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.5);">
                <h2 style="color: #2b5876; font-weight:700;">Selamat Datang, {st.session_state.user['name']}</h2>
                <p style="color: #5d707f; font-size: 1.1rem;">
                    Silakan unggah file data pasien untuk memulai analisis klasifikasi durasi rawat inap
                </p>
                <div style="font-size: 4rem; margin: 1.5rem 0;">📊</div>
                <p style="color: #5d707f;">
                    Gunakan menu di sidebar untuk mengunggah file Excel Anda
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Fitur animasi
            st.markdown("""
            <div style="text-align: center; margin-top: 2rem;">
                <lottie-player src="https://assets1.lottiefiles.com/packages/lf20_ukaaZq.json"  
                    background="transparent" speed="1" style="width: 300px; height: 300px; margin: 0 auto;" 
                    loop autoplay>
                </lottie-player>
            </div>
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            """, unsafe_allow_html=True)

# Fungsi untuk menampilkan halaman utama/beranda
def show_home_page():
    # ======================
    # Header section dengan efek paralaks
    # ======================
    st.markdown("""
    <div style="padding: 50px 0 40px 0; text-align: center; line-height:1.3;">
        <div style="font-size: 42px; font-weight: 800; color: #005;">
            KLASIFIKASI DURASI RAWAT INAP
        </div>
        <div style="font-size: 35px; margin-top: 4px; font-weight: 600; color: #006;">
            PASIEN SKIZOFRENIA DI RSUD MUYANG KUTE
        </div>
        <div style="font-size: 25px; margin-top: 2px; font-weight: 500; color: #006;">
            Menggunakan Kombinasi C4.5 dan Particle Swarm Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # Introduction section (Tentang Sistem)
    # ======================
    st.markdown("<div>", unsafe_allow_html=True)
    st.markdown("""
    <h3 class='glass' style='color: #0d47a1; text-align: center; margin-bottom: 30px;'>Tentang Sistem</h3>
    <p style='text-align: justify; line-height: 1.8; font-size: 1.1rem;'>
    Sistem ini dirancang untuk mengklasifikasikan durasi rawat inap pasien skizofrenia di RSUD Muyang Kute 
    dengan memanfaatkan kombinasi algoritma C4.5 dan Particle Swarm Optimization (PSO). Pendekatan ini 
    menghasilkan model prediksi yang akurat untuk membantu manajemen rumah sakit dalam perencanaan sumber daya 
    dan perawatan pasien yang lebih efektif.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # Features section (Fitur Utama)
    # ======================
    st.markdown("<h2 style='text-align: center; color: #fff; margin: 60px 0 40px 0; text-shadow: 1px 1px 3px rgba(0,0,0,0.2);'>Fitur Utama</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)  # Membagi layout menjadi 3 kolom

    with col1:
        # Feature 1: Klasifikasi Akurat
        st.markdown("""
        <div class='feature-card'>
        <div class='card-title'>🎯 Klasifikasi Akurat</div>
        <div class='card-content' style='text-align: justify; font-size:16px; color: #004d4d;'>
        Menggunakan kombinasi algoritma C4.5 dan PSO untuk menghasilkan 
        klasifikasi durasi rawat inap dengan akurasi tinggi.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Feature 2: Optimasi Parameter
        st.markdown("""
        <div class='feature-card'>
        <div class='card-title'>⚙️ Optimasi Parameter </div>
        <div class='card-content' style='text-align: justify; font-size:16px; color: #004d4d;'>
        PSO digunakan untuk mengoptimasi parameter algoritma C4.5, 
        meningkatkan performa klasifikasi secara signifikan.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Feature 3: Dashboard Interaktif
        st.markdown("""
        <div class='feature-card'>
        <div class='card-title'>📊 Dashboard</div>
        <div class='card-content' style='text-align: justify; font-size:16px; color: #004d4d;'>
        Melakukan perhitungan & visualisasi data yang interaktif dan informatif untuk 
        memudahkan analisis hasil klasifikasi.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ======================
    # Methodology section (Metodologi)
    # ======================
    st.markdown("<div style='margin-top: 60px;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='glass' style='color: #0d47a1; text-align: center; margin-bottom: 40px;'>Metodologi</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # Algoritma C4.5
        st.markdown("""
        <div style="padding: 20px; background: rgba(13, 71, 161, 0.1); border-radius: 16px; margin-bottom: 10px;">
        <h4 style='color: #0d47a1; margin-bottom: 8px;'>Algoritma C4.5</h4>
        <p style='text-align: justify; color: #37474f;'>Algoritma C4.5 digunakan untuk membangun pohon keputusan yang dapat mengklasifikasikan 
        durasi rawat inap berdasarkan berbagai faktor klinis dan demografis pasien.</p>

        <h4 style='color: #0d47a1; margin-top: 10px; margin-bottom: 8px;'>Keunggulan:</h4>
        - Dapat menangani data numerik dan kategorikal  <br>
        - Melakukan pemilihan fitur otomatis  <br>
        - Mampu mengatasi nilai yang hilang  
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Particle Swarm Optimization
        st.markdown("""
        <div style="padding: 20px; background: rgba(13, 71, 161, 0.1); border-radius: 16px; margin-bottom: 10px;">
        <h4 style='color: #0d47a1; margin-bottom: 8px;'>Particle Swarm Optimization</h4>
        <p style='text-align: justify; color: #37474f;'>
        PSO digunakan untuk mengoptimasi parameter dari algoritma C4.5, 
        sehingga menghasilkan model klasifikasi dengan performa terbaik.</p>

        <h4 style='color: #0d47a1; margin-top: 10px; margin-bottom: 8px;'>Keunggulan:</h4> 
        - Konvergensi yang cepat  <br>
        - Menghindari terjebak di optimum lokal  <br>
        - Efisien dalam pencarian solusi optimal 
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # System Status section (Status Sistem & Database)
    # ======================
    st.markdown("<div style='margin-top: 60px;'>", unsafe_allow_html=True)
    st.markdown("<h3 class='glass' style='color: #0d47a1; text-align: center; margin-bottom: 30px;'>Status Sistem</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # Koneksi Database
        st.markdown("<div class='feature-card card-title'>🔌 Koneksi Database</div></div>", unsafe_allow_html=True)
        if st.button("Test Koneksi Database", use_container_width=True):
            conn = init_db_connection()  # Fungsi koneksi ke database
            if conn:
                st.success("✅ Terhubung ke database Supabase")
                create_tables(conn)       # Fungsi untuk membuat tabel jika belum ada
                conn.close()
            else:
                st.error("❌ Gagal terhubung ke database")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Status layanan lain
        st.markdown("<div class='feature-card card-title'>📈 Status Layanan", unsafe_allow_html=True)
        st.markdown("""
        <div class='card-content' style='line-height: 1; color:#005;' >
        <p>✅ Streamlit: Berjalan</p>
        <p>✅ Visualisasi: Aktif</p>
        <p>🔌 Database: Perlu diuji</p>
        <p>✅ Authentication: Siap</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    
    # Footer
    st.markdown("""
    <div class='footer'>
    <p style=font-size: 17px; opacity: 1;">© 2025 RSUD Muyang Kute - Sistem Klasifikasi Durasi Rawat Inap Pasien Skizofrenia</p>
    <p style="font-size: 14px; opacity: 1;">Dikembangkan dengan ❤️ Putri Agustina Dewi untuk pelayanan kesehatan yang lebih baik</p>
    </div>
    """, unsafe_allow_html=True)

# Fungsi utama aplikasi Streamlit
def main():
    # Inisialisasi session state
    # Digunakan untuk menyimpan data antar rerender, seperti halaman aktif dan user login
    if "page" not in st.session_state:
        st.session_state.page = "home"  # halaman default adalah 'home'
    if "user" not in st.session_state:
        st.session_state.user = None     # user belum login, set ke None
    
    # Logika navigasi halaman
    if st.session_state.page == "home":
        # Membuat tab navigasi di halaman beranda
        tabs = st.tabs(["🏠 Beranda", "🔐 Masuk"])
        
        # Konten tab "Beranda"
        with tabs[0]:
            show_home_page()  # memanggil fungsi untuk menampilkan halaman home
        
        # Konten tab "Masuk"
        with tabs[1]:
            show_login_page()  # memanggil fungsi untuk menampilkan halaman login
    
    # Jika user sudah login dan page diarahkan ke dashboard
    elif st.session_state.page == "dashboard":
        show_dashboard()  # menampilkan halaman dashboard

# Menjalankan aplikasi jika file ini dieksekusi langsung
if __name__ == "__main__":
    main()
