import pickle
import streamlit as st
import numpy as np
from os import path


st.title("Flower Classification App")


file_name = "lr_model.pkl"
with open(path.join("model", file_name), "rb") as f:
    lr_model = pickle.load(f)

sl = st.number_input("Insert a sepel length")
sw = st.number_input("Insert a sepel width")
pl = st.number_input("Insert a petal length")
pw = st.number_input("Insert a petal width")

if st.button("Predict"):
    pred = lr_model.predict(np.array([[sl, sw, pl, pw]]))
    st.write("The flower is :", pred[0])
