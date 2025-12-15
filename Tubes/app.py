import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_models():
    scaler = joblib.load("Tubes/scaler_svm_rice.pkl")
    svm_model = joblib.load("Tubes/svm_rice_model.pkl")
    xgb_model = joblib.load("Tubes/xgboost_rice_model.pkl")
    return scaler, svm_model, xgb_model


scaler, svm_model, xgb_model = load_models()

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Rice Classification", layout="centered")

st.title("🍚 Rice Classification App")
st.write("Klasifikasi varietas beras **Cammeo** dan **Osmancik** menggunakan Machine Learning")

# ===============================
# MODEL SELECTION
# ===============================
model_choice = st.selectbox(
    "Pilih Model Klasifikasi",
    ["SVM", "XGBoost"]
)

# ===============================
# INPUT FEATURES
# ===============================
st.subheader("Masukkan Fitur Beras")

area = st.number_input("Area", min_value=0.0, value=12000.0)
perimeter = st.number_input("Perimeter", min_value=0.0, value=450.0)
major_axis = st.number_input("Major Axis Length", min_value=0.0, value=150.0)
minor_axis = st.number_input("Minor Axis Length", min_value=0.0, value=120.0)
eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.85)
convex_area = st.number_input("Convex Area", min_value=0.0, value=13000.0)
extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.75)

# Gabungkan input ke array
input_data = np.array([[area, perimeter, major_axis, minor_axis,
                         eccentricity, convex_area, extent]])

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Prediksi"):
    if model_choice == "SVM":
        input_scaled = scaler.transform(input_data)
        prediction = svm_model.predict(input_scaled)[0]
        prob = svm_model.decision_function(input_scaled)[0]

        st.success(f"Hasil Prediksi (SVM): **{prediction}**")
        st.write(f"Confidence Score: `{prob:.3f}`")

    else:
        prediction = xgb_model.predict(input_data)[0]
        probability = xgb_model.predict_proba(input_data)[0]

        label = "Cammeo" if prediction == 0 else "Osmancik"

        st.success(f"Hasil Prediksi (XGBoost): **{label}**")
        st.write(f"Probabilitas Cammeo : `{probability[0]:.2f}`")
        st.write(f"Probabilitas Osmancik : `{probability[1]:.2f}`")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("📊 Dataset: Rice (Cammeo and Osmancik) - UCI ML Repository")

