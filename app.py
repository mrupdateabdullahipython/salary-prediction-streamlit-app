import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Salary Predictor", page_icon="💼")

# Title
st.title("💼 Salary Prediction App")
st.write("Predict salary based on years of experience")

# Load model safely
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar input
st.sidebar.header("Input Data")
experience = st.sidebar.slider("Years of Experience", 0, 50, 1)

# Show metric
st.metric("Selected Experience", f"{experience} years")

# Prediction
if st.sidebar.button("Predict Salary"):
    try:
        prediction = model.predict(np.array([[experience]]))
        st.success(f"Estimated Salary: {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Show graph
if st.checkbox("Show Prediction Graph"):
    try:
        import matplotlib.pyplot as plt

X = np.arange(0, 50).reshape(-1, 1)
y = model.predict(X)

fig, ax = plt.subplots()

# Line (prediction)
ax.plot(X, y, color='blue', linewidth=2, label='Prediction Line')

# Scatter (optional example points)
ax.scatter(X, y, color='red', alpha=0.5, label='Data Points')

ax.set_xlabel("Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary vs Experience")

ax.legend()
ax.grid(True)

st.pyplot(fig)
    except Exception as e:
        st.error(f"Graph error: {e}")

# About section
st.markdown("---")
st.markdown("### About")
st.write(
    "This app uses a Linear Regression model to predict salary based on years of experience. An interactive Machine Learning web application that predicts salary using Linear Regression. Built with Python and deployed using Streamlit, this project highlights practical ML skills including data cleaning, model building, and deployment."
)