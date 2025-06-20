import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load('model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# Judul
st.title("Prediksi Kualitas Air untuk Budidaya Ikan Nila")
st.markdown("Masukkan parameter air berikut:")

# Form input
bod = st.number_input("BOD (mg/L)", min_value=0.0, value=1.0)
do = st.number_input("DO (mg/L)", min_value=0.0, value=6.0)
turb = st.number_input("Turbidity (cm)", min_value=0.0, value=25.0)
h2s = st.number_input("H2S (mg/L)", min_value=0.0, value=0.01)
nitrite = st.number_input("Nitrite (mg/L)", min_value=0.0, value=0.1)

# Jika tombol diklik
if st.button("Prediksi"):
    try:
        # Hitung fitur tambahan
        do_to_bod = do / (bod + 0.01)  # Hindari pembagi 0
        total_toxin = h2s + nitrite

        # Gabungkan semua fitur dalam urutan yang sesuai
        data = np.array([[bod, do, turb, h2s, nitrite, do_to_bod, total_toxin]])

        # Normalisasi
        data_scaled = scaler.transform(data)

        # Prediksi
        pred = model.predict(data_scaled)[0]
        label_map = {0: "Buruk", 1: "Sedang", 2: "Baik"}
        st.success(f"Prediksi Kualitas Air: **{label_map[pred]}**")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
