import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# UI
st.title("Prediksi Kualitas Air Untuk Budidaya Ikan Nila")
st.markdown("Masukkan parameter air untuk memprediksi kualitasnya.")

# Input
bod = st.number_input("BOD (mg/L)", min_value=0.0)
do = st.number_input("DO (mg/L)", min_value=0.0)
turb = st.number_input("Turbidity (cm)", min_value=0.0)
h2s = st.number_input("H2S (mg/L)", min_value=0.0)
nitrite = st.number_input("Nitrite (mg/L)", min_value=0.0)

# Fitur rekayasa
do_to_bod = do / (bod + 0.01)  # untuk mencegah pembagian nol
total_toxin = h2s + nitrite

# Pastikan urutan dan jumlah sesuai model
fitur_input = [[bod, do, turb, h2s, nitrite, do_to_bod, total_toxin]]

if st.button("Prediksi"):
    try:
        data_scaled = scaler.transform(fitur_input)
        pred = model.predict(data_scaled)[0]
        label = {0: "Buruk", 1: "Sedang", 2: "Baik"}
        st.success(f"Kualitas Air: {label[pred]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
