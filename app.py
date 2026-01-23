import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Diabetes Lansia",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("diabetes.csv")

# Filter lansia
data_lansia = data[data["Age"] >= 60]

X = data_lansia.drop("Outcome", axis=1)
y = data_lansia["Outcome"]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# NORMALISASI DATA
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# EVALUASI MODEL
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# =========================
# CSS UI (FONT BESAR)
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 18px !important;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #2C7BE5;
    text-align: center;
    margin-bottom: 0.3rem;
}

.subtitle {
    font-size: 1.5rem;
    text-align: center;
    color: #555;
    margin-bottom: 2rem;
}

h2, h3 {
    font-size: 1.7rem !important;
}

label {
    font-size: 1.2rem !important;
    font-weight: 600;
}

input {
    font-size: 1.2rem !important;
    padding: 10px !important;
}

button {
    font-size: 1.3rem !important;
    padding: 12px 24px !important;
    font-weight: bold !important;
}

.result-box {
    padding: 30px;
    border-radius: 14px;
    text-align: center;
    font-size: 1.4rem;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">🩺 Sistem Deteksi Diabetes Lansia</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Aplikasi prediksi risiko diabetes berbasis Machine Learning</div>',
    unsafe_allow_html=True
)

st.write("---")

# =========================
# INFORMASI MODEL
# =========================
st.info(f"📊 Akurasi Model Logistic Regression: **{accuracy:.2f}**")

# =========================
# FORM INPUT
# =========================
with st.form("form_lansia"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Data Fisik")
        age = st.number_input("Usia (Tahun)", min_value=60, max_value=120, step=1)
        bmi = st.number_input("BMI", step=0.1)
        pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, step=1)

    with col2:
        st.subheader("🩸 Data Klinis")
        glucose = st.number_input("Kadar Glukosa", step=1)
        blood_pressure = st.number_input("Tekanan Darah", step=1)
        skin_thickness = st.number_input("Ketebalan Kulit", step=1)
        insulin = st.number_input("Kadar Insulin", step=1)
        dpf = st.number_input("Diabetes Pedigree Function", step=0.01)

    submit = st.form_submit_button("🔍 Deteksi Risiko Diabetes")

# =========================
# HASIL PREDIKSI
# =========================
if submit:
    st.write("---")
    st.subheader("📊 Hasil Deteksi")

    data_input = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    data_input = scaler.transform(data_input)
    hasil = model.predict(data_input)

    if hasil[0] == 1:
        st.markdown("""
        <div class="result-box" style="background-color:#ffe5e5; border:3px solid #ff4b4b;">
            <h2>⚠️ RISIKO DIABETES TINGGI</h2>
            <p>Disarankan segera melakukan pemeriksaan medis lanjutan.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-box" style="background-color:#e6f4ea; border:3px solid #2ecc71;">
            <h2>✅ RISIKO DIABETES RENDAH</h2>
            <p>Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.</p>
        </div>
        """, unsafe_allow_html=True)
