import streamlit as st

st.set_page_config(page_title="Métricas - GDP Dashboard")

st.title("Métricas")
st.write("Página destinada a métricas e indicadores: precisión, recall, tasas por región, etc.")
st.info("Placeholder: aquí añadiremos gráficos y tablas con métricas del modelo.")

# Ejemplo de tarjeta
col1, col2, col3 = st.columns(3)
col1.metric("Precision (ej)", "0.85")
col2.metric("Recall (ej)", "0.78")
col3.metric("F1 (ej)", "0.81")
