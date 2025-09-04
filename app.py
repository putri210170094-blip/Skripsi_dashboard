import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import pyswarms as ps
from io import BytesIO
from sklearn.tree import export_text

# ===================== KONFIGURASI HALAMAN =====================
st.set_page_config(
    page_title="Klasifikasi Durasi Rawat Inap Pasien Skizofrenia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== STYLING =====================
st.markdown("""
<style>
    .main {
        background-color: #f5f9fc;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 1rem;
    }
    h1 {
        color: #2b5876;
        border-bottom: 2px solid #4b86b4;
        padding-bottom: 10px;
    }
    h2 {
        color: #3a7ca5;
        margin-top: 1.5rem;
    }
    .stButton>button {
        background-color: #4b86b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #2b5876;
    }
    .highlight-box {
        background-color: #e7f0f7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===================== JUDUL =====================
st.title("🧠 Klasifikasi Durasi Rawat Inap Pasien Skizofrenia")
st.markdown("""
<div style='color:#5d707f; font-size:1.1rem;'>
    Aplikasi ini digunakan untuk menganalisis dan mengklasifikasikan lama rawat inap pasien skizofrenia 
    di RSUD Muyang Kute menggunakan kombinasi algoritma C4.5 dan Particle Swarm Optimization (PSO).
</div>
""", unsafe_allow_html=True)

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

# ===================== MAIN APP =====================
df_raw = None
uploaded_file = st.sidebar.file_uploader(
    "📁 Unggah Data Pasien (Excel)", 
    type=["xlsx"],
    help="File harus mengandung kolom: Tanggal Masuk, Tanggal Keluar"
)

if uploaded_file:
    try:
        df_raw = pd.read_excel(uploaded_file)
        st.sidebar.success("✅ File berhasil diunggah!")
        
        with st.spinner("Memproses data..."):
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
        
        with st.spinner("Melatih model..."):
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
            st.subheader("Ikhtisar Data")
            st.dataframe(df_processed.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Distribusi Durasi")
                fig1, ax1 = plt.subplots()
                sns.histplot(df_processed['Durasi Rawat Inap (Hari)'], bins=15, kde=True, ax=ax1)
                ax1.set_xlabel("Hari")
                st.pyplot(fig1)
                
            with col2:
                st.markdown("### Distribusi Kategori")
                fig2, ax2 = plt.subplots()
                df_processed['Kategori Durasi'].value_counts().plot.pie(
                    autopct='%1.1f%%', 
                    colors=['#66b3ff','#99ccff','#cce6ff'],
                    ax=ax2
                )
                st.pyplot(fig2)
            
            st.markdown(f"""
            <div class="highlight-box">
                <h4>📌 Informasi Data</h4>
                <ul>
                    <li>Total sampel: <strong>{len(df_processed)}</strong></li>
                    <li>Durasi singkat (1-5 hari): <strong>{len(df_processed[df_processed['Kategori Durasi'] == 'Singkat'])}</strong></li>
                    <li>Durasi sedang (6-10 hari): <strong>{len(df_processed[df_processed['Kategori Durasi'] == 'Sedang'])}</strong></li>
                    <li>Durasi lama (>10 hari): <strong>{len(df_processed[df_processed['Kategori Durasi'] == 'Lama'])}</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Hasil Optimasi")
            st.write(f"Parameter Terbaik: Max Depth = {model_results['best_params']['max_depth']}, Min Samples = {model_results['best_params']['min_samples']}")
            st.write(f"MAPE Dasar: {model_results['base_mape']:.2f}%") 
            st.write(f"MAPE Model Optimasi: {model_results['optim_mape']:.2f}%") 
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Model C4.5 Dasar")
                st.metric("Akurasi", f"{model_results['base_accuracy']:.2%}")
                st.metric("MAPE", f"{model_results['base_mape']:.2f}%")
                st.code(
                    classification_report(
                        model_results['y_test'], 
                        model_results['y_pred_base'],
                        target_names=['Singkat', 'Sedang', 'Lama']
                    )
                )
                
            with col2:
                st.markdown("#### Model C4.5+PSO")
                st.metric("Akurasi", f"{model_results['optim_accuracy']:.2%}")
                st.metric("MAPE", f"{model_results['optim_mape']:.2f}%")
                st.code(
                    classification_report(
                        model_results['y_test'], 
                        model_results['y_pred_optim'],
                        target_names=['Singkat', 'Sedang', 'Lama']
                    )
                )
                
            improvement = model_results['optim_accuracy'] - model_results['base_accuracy']
            st.markdown(f"""
            <div class="highlight-box">
                <h4>🔍 Temuan Utama</h4>
                <p>Model optimasi menunjukkan peningkatan akurasi sebesar <strong>{improvement:.2%}</strong></p>
                <p>Parameter terbaik dari PSO:</p>
                <ul>
                    <li>Kedalaman maksimal: <strong>{model_results['best_params']['max_depth']}</strong></li>
                    <li>Minimum sampel split: <strong>{model_results['best_params']['min_samples']}</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Simpan prediksi ke Excel
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
            st.download_button(
                label="📥 Unduh Hasil Prediksi",
                data=excel_buffer,
                file_name="prediksi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with tab3:
            st.subheader("Visualisasi Pohon Keputusan")
            
            with st.expander("ℹ Tentang visualisasi ini"):
                st.write("""
                Pohon ini menunjukkan proses pengambilan keputusan model C4.5 yang dioptimasi.
                Setiap node menunjukkan kondisi pembagian berdasarkan durasi rawat inap.
                """)
            
            plt.figure(figsize=(25, 15))
            plot_tree(
                model_results['optimized_model'],
                feature_names=['Durasi (hari)'],
                class_names=['Singkat', 'Sedang', 'Lama'],
                filled=True,
                rounded=True,
                fontsize=10,
                max_depth=2  # Menunjukkan 2 level pertama untuk kejelasan
            )
            st.pyplot(plt)
            
            st.subheader("Representasi Teks")
            tree_text = export_text(
                model_results['optimized_model'],
                feature_names=['Durasi'],
                max_depth=2
            )
            st.code(tree_text)
            
            st.download_button(
                "📥 Unduh Struktur Pohon",
                tree_text,
                file_name="struktur_pohon.txt"
            )
        
        with tab4:
            st.subheader("Prediksi Pasien Baru")
            st.markdown("""
            <div style='color:#666; margin-bottom:20px;'>
                 Masukkan tanggal masuk dan keluar pasien untuk memprediksi kategori durasi rawat inap.
            </div>
            """, unsafe_allow_html=True)

            with st.form("form_prediksi"):
                col1, col2 = st.columns(2)
                with col1:
                    tgl_masuk = st.date_input("Tanggal Masuk", key="tgl_masuk")
                with col2:
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
                    
                            st.success(f"✅ Kategori Durasi Prediksi: *{kategori}*")
                    
                    # ========== BAGIAN REKOMENDASI LENGKAP ==========
                            st.markdown("""
                            <div class="highlight-box">
                                <h3>📌 Rekomendasi Manajemen Ruangan</h3>
                                <p><strong>Kategori: {kategori}</strong></p>
                            """.format(kategori=kategori), unsafe_allow_html=True)
                    
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
                    
                            st.markdown("""
                            </div>
                            """, unsafe_allow_html=True)
                    
                        except Exception as e:
                            st.error(f"⚠ Error dalam prediksi: {str(e)}")
                    else:
                        st.error("🔴 Sistem belum siap! Pastikan:")
                        st.write("1. Data pasien sudah diunggah")
                        st.write("2. Model sudah selesai dilatih")
                        st.stop()

        with tab5:
            st.subheader("Estimasi Kebutuhan Ruangan")
            
            # Hitung distribusi pasien
            distribusi = df_processed['Kategori Durasi'].value_counts().reset_index()
            distribusi.columns = ['Kategori', 'Jumlah Pasien']
            
            # Tampilkan tabel distribusi
            st.write("### Distribusi Pasien Berdasarkan Durasi")
            st.dataframe(distribusi)
            
            # Visualisasi
            fig, ax = plt.subplots()
            sns.barplot(
                x='Kategori',
                y='Jumlah Pasien',
                data=distribusi,
                palette=['#66b3ff','#99ccff','#cce6ff'],
                ax=ax
            )
            ax.set_title('Jumlah Pasien per Kategori')
            st.pyplot(fig)
            
            # Rekomendasi
            st.markdown("""
            <div class="highlight-box">
                <h3>📊 Rekomendasi Alokasi Ruangan</h3>
                <p>Berdasarkan data historis:</p>
                <ul>
            """, unsafe_allow_html=True)
            
            for _, row in distribusi.iterrows():
                st.markdown(f"""
                <li><strong>{row['Kategori']}</strong>: {row['Jumlah Pasien']} pasien ({row['Jumlah Pasien']/len(df_processed):.1%})</li>
                """, unsafe_allow_html=True)
                
            st.markdown("""
                </ul>
                <p><strong>Saran:</strong></p>
                <ul>
                    <li>Ruangan singkat: 5-10% dari total kapasitas</li>
                    <li>Ruangan sedang: 15-20% dari total kapasitas</li>
                    <li>Ruangan panjang: 30-40% dari total kapasitas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"""
        ❌ Terjadi kesalahan:
        {str(e)}
        """)
        st.write("Pastikan format file sesuai dan data lengkap.")

else:
    st.info("📌 Silakan unggah file data pasien untuk memulai")

# ===================== FOOTER =====================
st.sidebar.markdown("---")
st.sidebar.markdown("*Dikembangkan oleh:*")
st.sidebar.markdown("Putri Agustina Dewi")
st.sidebar.markdown("Teknik Informatika, Universitas Malikussaleh")
st.sidebar.markdown("2025")

with st.sidebar.expander("ℹ Tentang Aplikasi"):
    st.write("""
    Aplikasi ini digunakan untuk analisis durasi rawat inap pasien skizofrenia
    menggunakan algoritma C4.5 dan PSO.
    """)