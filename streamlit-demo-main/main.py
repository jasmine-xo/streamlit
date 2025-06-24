import pickle
import streamlit as st
import numpy as np
import os

# App title
st.title("Total Litres of Pure Alcohol Predictor")

# Load model (only once)
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "model", "model.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Input fields
country = st.number_input("Insert Country (encoded)", min_value=0)
beer = st.number_input("Beer Servings", min_value=0)
spirit = st.number_input("Spirit Servings", min_value=0)
wine = st.number_input("Wine Servings", min_value=0)
continent = st.number_input("Insert Continent (encoded)", min_value=0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[country, beer, spirit, wine, continent]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Total Litres of Pure Alcohol: {round(prediction[0], 2)}")
