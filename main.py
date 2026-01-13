import streamlit as st
import numpy as np
import pickle

# Load scaler and model
scaler = pickle.load(open("Model/Scaled.pkl", "rb"))
model = pickle.load(open("Model/svc.pkl", "rb"))

# App title
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction System")
st.markdown("Predict whether a person is **Diabetic or Not** using ML")

st.divider()

# User input form
with st.form("diabetes_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1, max_value=120)

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.divider()

    if prediction == 1:
        st.error("🔴 **The person is likely DIABETIC**")
    else:
        st.success("🟢 **The person is NOT diabetic**")
