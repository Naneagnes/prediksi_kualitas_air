import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_rf.pkl')  # Ganti dengan nama file model kamu

# UI
st.title("Prediksi Kualitas Air Budidaya Ikan")
st.markdown("Masukkan parameter air untuk mengetahui kualitasnya.")

# Input
bod = st.number_input('BOD (mg/L)', min_value=0.0)
do = st.number_input('DO (mg/L)', min_value=0.0)
turbidity = st.number_input('Turbidity (cm)', min_value=0.0)
h2s = st.number_input('H2S (mg L-1)', min_value=0.0)
nitrite = st.number_input('Nitrite (mg L-1)', min_value=0.0)

if st.button("Prediksi"):
    data = pd.DataFrame([[bod, do, turbidity, h2s, nitrite]],
                        columns=['BOD (mg/L)', 'DO(mg/L)', 'Turbidity (cm)', 'H2S (mg L-1 )', 'Nitrite (mg L-1 )'])
    pred = model.predict(data)[0]
    label = {0: "Buruk", 1: "Sedang", 2: "Baik"}
    st.success(f"Kualitas Air: {label[pred]}")
