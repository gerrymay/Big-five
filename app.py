import streamlit as st
import joblib
import numpy as np

modelo = joblib.load("modelo_bigfive.pkl")

st.title("Predicción de Profesión con Big Five")

st.write("Ingresa tus resultados del test")

O = st.slider("Openness", 0.0, 100.0, 50.0)
C = st.slider("Conscientiousness", 0.0, 100.0, 50.0)
E = st.slider("Extraversion", 0.0, 100.0, 50.0)
A = st.slider("Agreeableness", 0.0, 100.0, 50.0)
N = st.slider("Neuroticism", 0.0, 100.0, 50.0)

if st.button("Predecir profesión"):
    datos = np.array([[O, C, E, A, N]])
    prediccion = modelo.predict(datos)
    st.success(f"Tu profesión predicha es: {prediccion[0]}")
