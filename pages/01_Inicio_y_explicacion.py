import streamlit as st

st.set_page_config(page_title="Inicio - GDP Dashboard")

st.title("Inicio y explicación")
st.markdown(
    "Esta aplicación contiene varias secciones:\n\n"
    "- Inicio y explicación (esta página).\n"
    "- Uso de la app (interfaz principal de predicción).\n"
    "- Métricas (indicadores del rendimiento).\n"
    "- Dashboards (visualizaciones).\n\n"
    "Seleccioná la pestaña 'Uso de la app' para acceder a la interfaz de predicción basada en `streamlit_app.py`."
)

st.markdown("---")
st.subheader("Detalles rápidos")
st.write("- Los umbrales por mes y región se cargan desde `data/UMBRALES POR MES REGION.xlsx` cuando está disponible.")
st.write("- Para ejecutar localmente: `streamlit run app_index.py`.")
