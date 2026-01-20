import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("diabetes.csv")

# Filter lansia
data_lansia = data[data['Age'] >= 60]

X = data_lansia.drop("Outcome", axis=1)
y = data_lansia["Outcome"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# STREAMLIT UI
# =========================
st.title("🩺 Sistem Deteksi Diabetes Lansia")
st.write("Masukkan data kesehatan lansia untuk mendeteksi risiko diabetes")

with st.form("form_lansia"):
    pregnancies = st.number_input("Jumlah kehamilan", 0, 20)
    glucose = st.number_input("Kadar glukosa")
    blood_pressure = st.number_input("Tekanan darah")
    skin_thickness = st.number_input("Ketebalan kulit")
    insulin = st.number_input("Kadar insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Usia", 60, 120)

    submit = st.form_submit_button("Deteksi")

if submit:
    if age < 60:
        st.error("Sistem hanya untuk lansia (usia ≥ 60 tahun)")
    else:
        data_input = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        data_input = scaler.transform(data_input)
        hasil = model.predict(data_input)

        if hasil[0] == 1:
            st.error("⚠️ RISIKO DIABETES TINGGI")
            st.write("Rekomendasi: Segera lakukan pemeriksaan medis.")
        else:
            st.success("✅ RISIKO DIABETES RENDAH")
            st.write("Rekomendasi: Pertahankan pola hidup sehat.")
