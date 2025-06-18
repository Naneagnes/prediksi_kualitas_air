import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# UI
st.title("Prediksi Kualitas Air")
st.markdown("Masukkan parameter air untuk memprediksi kualitasnya.")

# Input
bod = st.number_input("BOD (mg/L)", min_value=0.0)
do = st.number_input("DO (mg/L)", min_value=0.0)
turb = st.number_input("Turbidity (cm)", min_value=0.0)
h2s = st.number_input("H2S (mg/L)", min_value=0.0)
nitrite = st.number_input("Nitrite (mg/L)", min_value=0.0)

if st.button("Prediksi"):
    data = [[bod, do, turb, h2s, nitrite]]
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    label = {0: "Buruk", 1: "Sedang", 2: "Baik"}
    st.success(f"Kualitas Air: {label[pred]}")
