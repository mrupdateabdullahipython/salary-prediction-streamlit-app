import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Salary Predictor", page_icon="💼")

# Title
st.title("💼 Salary Prediction App")
st.markdown("Predict salary based on years of experience")

# Load model
model = joblib.load("model.pkl")

# Sidebar
st.sidebar.header("Input")
experience = st.sidebar.slider("Years of Experience", 0, 50, 1)

# Prediction
if st.sidebar.button("Predict"):
    result = model.predict(np.array([[experience]]))
    
    st.subheader("Result")
    st.success(f"Estimated Salary: {result[0]:,.2f}")
st.metric("Experience", f"{experience} years")
st.markdown("---")
st.markdown("### About")
st.write("This app uses Linear Regression to predict salary based on experience.")
import matplotlib.pyplot as plt

if st.checkbox("Show Graph"):
    X = np.arange(0, 50).reshape(-1, 1)
    y = model.predict(X)

    fig, ax = plt.subplots()
    ax.plot(X, y)
    ax.set_xlabel("Experience")
    ax.set_ylabel("Salary")

    st.pyplot(fig)