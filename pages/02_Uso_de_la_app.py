import streamlit as st
import streamlit_app

# Esta página delega la UI principal a streamlit_app.render_uso_app()
st.set_page_config(page_title="Uso de la app - GDP Dashboard")

st.header("Uso de la app")
st.write("Interfaz principal de predicción e ingreso de parámetros.")

# Llamamos al renderer principal
streamlit_app.render_uso_app()
