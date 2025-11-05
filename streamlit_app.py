import streamlit as st
import pandas as pd
import numpy as np
import joblib
import difflib

# ---------- Config ----------
st.set_page_config(page_title="Predicción con tu modelo", page_icon="🤖", layout="centered")

def load_model():
    """Intentar cargar `model.pkl`. Devuelve (model, error_str).

    Retornamos la tupla para poder mostrar un mensaje amigable en la app
    en caso de que el pickle no sea compatible con las versiones instaladas.
    """
    try:
        m = joblib.load("model.pkl")
        return m, None
    except Exception as e:
        return None, str(e)

model, load_error = load_model()

def _get_expected_features_from_model(m):
    """Intentar inferir los nombres de columnas que el modelo espera.

    Devuelve lista de nombres o None si no se pudo determinar.
    """
    if m is None:
        return None
    # Prefer attribute directo
    feat = getattr(m, "feature_names_in_", None)
    if feat is not None:
        try:
            return list(feat)
        except Exception:
            return None
    # Si es Pipeline, buscar en steps
    try:
        # algunos estimadores guardan feature_names_in_ en steps
        if hasattr(m, "named_steps"):
            for step in m.named_steps.values():
                feat = getattr(step, "feature_names_in_", None)
                if feat is not None:
                    return list(feat)
    except Exception:
        pass
    return None

# nombres que el modelo espera (si podemos obtenerlos)
expected_features = _get_expected_features_from_model(model)

st.title("Predicción de casos de dengue")
st.caption("Carga un pipeline entrenado en scikit-learn y predice a partir de parámetros.")

st.sidebar.header("Parámetros de entrada")

# ======== INPUTS corregidos solicitados por el usuario ========
# Nombres que usaremos en la UI / DataFrame (si tu pipeline espera otros nombres, ver más abajo):
# - densidad
# - semana_epidemiologica_1 (seleccionada via calendario -> convertimos a número de semana)
# - prec_sem_prom
# - hum_sem_prom
# - temp_sem_prom

densidad = st.sidebar.number_input(
    "Densidad del departamento (personas/km²)",
    value=0.0,
    step=0.1,
    format="%.3f",
    help="Densidad del departamento en personas por km²",
)

fecha_sem = st.sidebar.date_input(
    "Fecha (seleccionar día para calcular la semana epidemiológica)",
    value=pd.to_datetime("2023-01-01").date(),
    help="Seleccioná una fecha dentro de la semana epidemiológica que querés usar; se convertirá al número de semana (1-53)",
)
try:
    # Nombre que el pipeline espera: 'semana_epidemiologica' (categoria)
    semana_epidemiologica = int(pd.to_datetime(fecha_sem).isocalendar().week)
except Exception:
    semana_epidemiologica = 1

prec_sem_prom = st.sidebar.number_input(
    "Precipitación semanal promedio (mm) — prec_sem_prom",
    value=0.0,
    step=0.1,
    format="%.3f",
    help="Precipitación semanal promedio en mm",
)

hum_sem_prom = st.sidebar.number_input(
    "Humedad semanal promedio (%) — hum_sem_prom",
    value=0.0,
    step=0.1,
    format="%.3f",
    help="Humedad relativa semanal promedio (porcentaje)",
)

temp_sem_prom = st.sidebar.number_input(
    "Temperatura semanal promedio (°C) — temp_sem_prom",
    value=0.0,
    step=0.1,
    format="%.3f",
    help="Temperatura semanal promedio en °C",
)

# Explicación rápida en la barra lateral
with st.sidebar.expander("Significado de las columnas (UI → nombres usados)"):
    st.write("- densidad → densidad del departamento (personas/km²)")
    st.write("- semana_epidemiologica → semana epidemiológica (convertida desde la fecha seleccionada)")
    st.write("- prec_sem_prom → precipitación semanal promedio (mm)")
    st.write("- hum_sem_prom → humedad semanal promedio (%)")
    st.write("- temp_sem_prom → temperatura semanal promedio (°C)")

# Armamos el DataFrame con los nombres corregidos
input_dict = {
    "densidad": [densidad],
    "semana_epidemiologica": [semana_epidemiologica],
    "prec_sem_prom": [prec_sem_prom],
    "hum_sem_prom": [hum_sem_prom],
    "temp_sem_prom": [temp_sem_prom],
}
X = pd.DataFrame(input_dict)

st.subheader("Datos a predecir")
st.dataframe(X)

# Mostrar fecha seleccionada y semana calculada para claridad
st.write(f"Fecha seleccionada: {fecha_sem} — semana epidemiológica: {semana_epidemiologica}")

if load_error is not None:
    st.error(
        "No se pudo cargar `model.pkl`. Esto suele ocurrir por incompatibilidades de versiones (numpy / scikit-learn).\n"
        f"Detalle: {load_error}\n\nRecomendación: revisá y ajustá `requirements.txt` para que coincida con las versiones usadas al guardar el pickle."
    )

# Botón de predicción
if st.button("🔮 Predecir"):
    if model is None:
        st.error("No hay un modelo cargado. Arreglá el problema de carga (ver mensaje de error arriba) antes de predecir.")
    else:
        try:
            # Antes de predecir, intentamos mapear (automáticamente) los nombres de columnas
            # de la UI a los que el modelo espera, si los conocemos.
            X_to_predict = X.copy()
            mapping_performed = None
            if expected_features is not None:
                # Si ya coinciden exactamente, no hacemos nada
                if set(expected_features) != set(X_to_predict.columns):
                    # Intentamos mapear cada expected -> mejor coincidencia en X.columns
                    available = list(X_to_predict.columns)
                    rename_map = {}
                    used = set()
                    for exp in expected_features:
                        # buscar la mejor coincidencia no usada
                        matches = difflib.get_close_matches(exp, available, n=3, cutoff=0.3)
                        chosen = None
                        for mname in matches:
                            if mname not in used:
                                chosen = mname
                                break
                        if chosen is None and len(available) > 0:
                            # fallback por similitud con ratio
                            best = None
                            best_ratio = 0.0
                            for c in available:
                                if c in used:
                                    continue
                                r = difflib.SequenceMatcher(None, exp, c).ratio()
                                if r > best_ratio:
                                    best_ratio = r
                                    best = c
                            if best is not None and best_ratio >= 0.4:
                                chosen = best
                        if chosen is not None:
                            rename_map[chosen] = exp
                            used.add(chosen)
                    if rename_map:
                        X_to_predict = X_to_predict.rename(columns=rename_map)
                        mapping_performed = rename_map
                    # Añadir columnas faltantes con NaN (el pipeline puede manejarlo o no)
                    for col in expected_features:
                        if col not in X_to_predict.columns:
                            X_to_predict[col] = np.nan

            # Clasificación: pred y proba (si el modelo lo soporta)
            y_pred = model.predict(X_to_predict)
            st.success(f"Predicción: {y_pred[0]}")

            # Si se realizó mapeo, mostramos al usuario qué cambió
            if mapping_performed is not None:
                st.info("Se realizó un mapeo automático entre columnas de la UI y las esperadas por el modelo:")
                st.write(mapping_performed)

            # Probabilidades (si existe predict_proba)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                proba_df = pd.DataFrame({
                    "clase": np.arange(len(proba)),
                    "prob": proba
                })
                st.subheader("Probabilidades")
                st.bar_chart(proba_df.set_index("clase"))
            else:
                st.info("El modelo no expone `predict_proba`.")

            # Si es regresión, cambiar por:
            # y_pred = model.predict(X)[0]
            # st.success(f"Predicción: {y_pred:.3f}")

        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")
            # Si sabemos qué columnas espera el modelo, muéstralas para ayudar al debug
            if expected_features is not None:
                st.info("El modelo indica que espera las siguientes columnas:")
                st.write(expected_features)
            else:
                st.info("No se pudieron determinar las columnas esperadas por el modelo. Comprueba el stacktrace en la terminal.")

st.divider()
st.caption("Tip: asegurate de pinnear las versiones en requirements.txt para evitar incompatibilidades del pickle.")
