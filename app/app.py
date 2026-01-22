import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("app/model.pkl", "rb"))
scaler = pickle.load(open("app/scaler.pkl", "rb"))

st.title("AI Placement Predictor")
st.write("Enter student details to predict placement outcome.")

# User inputs
maths = st.number_input("Maths Marks")
python = st.number_input("Python Marks")
sql = st.number_input("SQL Marks")
attendance = st.number_input("Attendance %")
mini_projects = st.number_input("Mini Project Marks")
communication_score = st.number_input("Communication Skills (0-10)")
placement_readiness = st.number_input("Placement Readiness Score")

if st.button("Predict"):
    features = np.array([[
        maths,
        python,
        sql,
        attendance,
        mini_projects,
        communication_score,
        placement_readiness
    ]])

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)

    if prediction[0] == 1:
        st.success("🎉 Prediction: Placed")
    else:
        st.error("❌ Prediction: Not Placed")
