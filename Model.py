import streamlit as st
import joblib

# Load model
model = joblib.load(r"model/iris_random_forest.pkl")

# Map numeric predictions to species names
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.title("  Iris Flower Classifier")
sepal_length = st.slider("Sepal Length", 4.0, 8.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5)
petal_length = st.slider("Petal Length", 1.0, 7.0)
petal_width = st.slider("Petal Width", 0.1, 2.5)

if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    flower = species_map[prediction[0]]
    st.success(f"Predicted Class: {flower}")
