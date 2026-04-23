import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

# Title
st.title("💼 Salary Prediction App")

st.write("Predict salary based on years of experience")

# Input
experience = st.number_input("Enter Years of Experience", 0.0, 50.0, step=0.5)

# Predict button
if st.button("Predict Salary"):
    prediction = model.predict(np.array([[experience]]))
    st.success(f"Estimated Salary: {prediction[0]:.2f}")