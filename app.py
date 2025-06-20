import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# UI
st.title("Prediksi Kualitas Air untuk Budidaya Ikan Nila")
st.markdown("Masukkan nilai parameter kualitas air berikut:")

# Input 5 fitur
bod = st.number_input("BOD (mg/L)", min_value=0.0, value=1.0)
do = st.number_input("DO (mg/L)", min_value=0.0, value=6.0)
turbidity = st.number_input("Turbidity (cm)", min_value=0.0, value=30.0)
h2s = st.number_input("H2S (mg/L)", min_value=0.0, value=0.01)
nitrite = st.number_input("Nitrite (mg/L)", min_value=0.0, value=0.05)

# Prediksi
if st.button("Prediksi"):
    data = np.array([[bod, do, turbidity, h2s, nitrite]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    
    label_map = {0: "Buruk", 1: "Sedang", 2: "Baik"}
    st.success(f"Kualitas Air: **{label_map[prediction]}**")
