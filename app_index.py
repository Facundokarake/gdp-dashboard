import streamlit as st
import streamlit_app

st.set_page_config(page_title="GDP Dashboard", layout="wide")

st.title("GDP Dashboard")
st.markdown("Seleccioná una pestaña para navegar entre las secciones principales.")

tabs = st.tabs(["Inicio y explicación", "Uso de la app", "Métricas", "Dashboards"])

with tabs[0]:
    st.header("Inicio y explicación")
    st.write(
        "Esta aplicación permite predecir casos de dengue por región climática y ver umbrales de riesgo mensuales.\n\n" \
        "Usá la pestaña 'Uso de la app' para ingresar parámetros y ejecutar predicciones. 'Métricas' y 'Dashboards' serán usadas para mostrar indicadores y visualizaciones." 
    )
    st.markdown("---")
    st.subheader("Cómo navegar")
    st.write("- Seleccioná 'Uso de la app' para acceder a la interfaz principal de predicción.\n- Seleccioná la región, definí los parámetros y presioná 'Predecir'.\n- En el expander 'Ver umbrales' podés ver los umbrales por mes de la región seleccionada.")

with tabs[1]:
    st.header("Uso de la app")
    st.write("Aquí se muestra la interfaz principal de la aplicación (inputs, predicción y umbrales).")
    # Llamamos al renderer principal
    streamlit_app.render_uso_app()

with tabs[2]:
    st.header("Métricas")
    st.write("Zona destinada a métricas clave: precisión del modelo, tasas de acierto por región, etc. (placeholder)")
    st.info("Por ahora esta página es un placeholder. Podemos integrar gráficos y tablas aquí.")

with tabs[3]:
    st.header("Dashboards")
    st.write("Dashboards con mapas y series temporales (placeholder).")
    st.info("Aquí se conectarán los dashboards interactivos.")
