import streamlit as st
import joblib
import pandas as pd
import os

# Load the trained model (Ensure the correct path)
cwd=os.getcwd()
print(cwd)
path=os.path.join(cwd,"covid/api/model.pkl")
print(path)


model_path = "/Users/gauravjangid/Work/covid/api/model.pkl"  # Correct path inside the container
model = joblib.load(path)

# Streamlit UI
st.title("Machine Learning Model Predictor")

# Input fields
feature1 = st.number_input("age", value=0)
feature2 = st.selectbox("select Gender",[0,1])
feature3 = st.number_input("fever", value=0.0)
feature4 = st.selectbox("cough",[0,1,2])
feature5 = st.selectbox("city",[0,1,2,3])

# Predict button
if st.button("Predict"):
    features = pd.DataFrame([[feature1, feature2, feature3, feature4, feature5]])
    prediction = model.predict(features)
    st.success(f"Predicted Value: {int(prediction[0])}")