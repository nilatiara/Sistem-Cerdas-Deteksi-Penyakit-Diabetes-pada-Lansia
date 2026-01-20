import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Diabetes Lansia",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS (UNTUK TAMPILAN KEREN)
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4A90E2;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        border: 2px solid #4A90E2;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# FUNGSI LOAD & TRAIN MODEL (DENGAN CACHING)
# =========================
@st.cache_resource
def load_and_train_model():
    # Load Data (Pastikan file diabetes.csv ada di folder yang sama)
    try:
        data = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        return None, None, "File CSV tidak ditemukan!"

    # Filter lansia
    data_lansia = data[data['Age'] >= 60]

    if data_lansia.empty:
         return None, None, "Data lansia tidak cukup!"

    X = data_lansia.drop("Outcome", axis=1)
    y = data_lansia["Outcome"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, None

# Load Model di awal
model, scaler, error_msg = load_and_train_model()

# =========================
# SIDEBAR INFORMASI
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Panel Informasi")
    st.info(
        """
        **Tentang Aplikasi:**
        Sistem ini menggunakan Machine Learning (Logistic Regression) untuk mendeteksi risiko diabetes dini pada lansia.
        
        **Target Pengguna:**
        Lansia (Usia ≥ 60 tahun).
        """
    )
    st.warning("⚠️ **Disclaimer:** Hasil prediksi ini hanyalah alat bantu skrining awal dan bukan pengganti diagnosa dokter.")

# =========================
# MAIN INTERFACE
# =========================
if error_msg:
    st.error(f"Error: {error_msg}")
else:
    st.markdown('<div class="main-header">🩺 MediCheck: Deteksi Diabetes Lansia</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Masukkan parameter kesehatan untuk analisis risiko berbasis AI</div>', unsafe_allow_html=True)

    # Container form agar terlihat rapi
    with st.container():
        st.write("---")
        # Layout 2 Kolom untuk Input
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("👤 Data Fisik")
            age = st.number_input("Usia (Tahun)", min_value=60, max_value=120, value=65, help="Wajib di atas 60 tahun")
            bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, format="%.1f", help="Berat (kg) / Tinggi² (m)")
            pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, step=1)
            skin_thickness = st.number_input("Ketebalan Kulit (mm)", min_value=0)

        with col2:
            st.subheader("🩸 Data Klinis")
            glucose = st.number_input("Kadar Glukosa (mg/dL)", min_value=0)
            blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0)
            insulin = st.number_input("Kadar Insulin (mu U/ml)", min_value=0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", help="Riwayat keturunan diabetes")

        st.write("---")
        
        # Tombol Submit di tengah
        detect_btn = st.button("🔍 ANALISIS RISIKO SEKARANG")

    # =========================
    # LOGIKA PREDIKSI
    # =========================
    if detect_btn:
        if age < 60:
            st.toast("Usia harus di atas 60 tahun!", icon="🚫")
            st.error("Sistem dikhususkan untuk data lansia (Usia ≥ 60 tahun).")
        else:
            # Tampilkan spinner loading biar terlihat proses
            with st.spinner('Sedang menganalisis data kesehatan...'):
                time.sleep(1) # Efek dramatis (opsional)
                
                # Preprocessing Input
                input_data = np.array([[pregnancies, glucose, blood_pressure, 
                                        skin_thickness, insulin, bmi, dpf, age]])
                input_scaled = scaler.transform(input_data)
                
                # Prediksi
                prediksi = model.predict(input_scaled)[0]
                probabilitas = model.predict_proba(input_scaled)[0][1] # Ambil probabilitas kelas 1 (Diabetes)

            # Tampilan Hasil
            st.markdown("### Hasil Analisis")
            
            # Layout hasil
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prediksi == 1:
                    st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=120)
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/1161/1161388.png", width=120)

            with res_col2:
                if prediksi == 1:
                    st.markdown(
                        f"""
                        <div class="result-card" style="background-color: #ffcccc; border: 2px solid #ff4b4b;">
                            <h2 style="color: #ff4b4b;">⚠️ RISIKO TINGGI</h2>
                            <p>Probabilitas: <b>{probabilitas:.1%}</b></p>
                            <p>Sistem mendeteksi indikator yang mengarah pada diabetes.</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.markdown("**Rekomendasi:** Segera konsultasikan hasil ini dengan dokter spesialis penyakit dalam.")
                else:
                    st.markdown(
                        f"""
                        <div class="result-card" style="background-color: #d4edda; border: 2px solid #28a745;">
                            <h2 style="color: #28a745;">✅ RISIKO RENDAH</h2>
                            <p>Probabilitas: <b>{probabilitas:.1%}</b></p>
                            <p>Parameter kesehatan Anda terlihat normal untuk kategori lansia.</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.markdown("**Rekomendasi:** Pertahankan pola makan sehat dan olahraga ringan teratur.")
