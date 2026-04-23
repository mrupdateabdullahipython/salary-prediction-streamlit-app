import streamlit as st
import numpy as np
import joblib

st.title("💼 Salary Prediction App")

try:
    model = joblib.load("model.pkl")

    experience = st.number_input("Enter Experience", 0.0, 50.0)

    if st.button("Predict"):
        result = model.predict(np.array([[experience]]))
        st.success(f"Salary: {result[0]:.2f}")

except Exception as e:
    st.error(f"Error: {e}")