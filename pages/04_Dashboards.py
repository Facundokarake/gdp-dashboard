import streamlit as st

st.set_page_config(page_title="Dashboards - GDP Dashboard")

st.title("Dashboards")
st.write("Espacio para dashboards interactivos: mapas, series temporales y filtros.")
st.info("Placeholder: conectá fuentes de datos y añadí gráficos aquí.")

# Ejemplo de gráfico simple (placeholder)
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'mes': ['Ene','Feb','Mar','Abr','May','Jun'],
    'valor': np.random.randint(0, 100, size=6)
})

st.line_chart(df.set_index('mes'))
