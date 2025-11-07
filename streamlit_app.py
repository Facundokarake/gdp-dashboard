import streamlit as st
import pandas as pd
import numpy as np
import joblib
import difflib
from math import ceil

# ---------- Config ----------
# Nota: `st.set_page_config` se gestiona desde la página principal para evitar llamadas duplicadas

# Configuración de regiones: modelo y umbrales
REGIONES_CONFIG = {
    "ARIDO/SEMIARIDO": {
        "modelo": "model_arido_semiarido.pkl",
        "umbrales": {"bajo": 5, "medio": 20, "alto": 50}
    },
    "FRIO/MONTANA": {
        "modelo": "model_frio_montana.pkl",
        "umbrales": {"bajo": 3, "medio": 10, "alto": 30}
    },
    "SUBTROPICAL": {
        "modelo": "model_subtropical.pkl",
        "umbrales": {"bajo": 15, "medio": 50, "alto": 100}
    },
    "TEMPLADO": {
        "modelo": "model_templado.pkl",
        "umbrales": {"bajo": 10, "medio": 30, "alto": 70}
    }
}


def load_model(region):
    """Intentar cargar el modelo según la región. Devuelve (model, error_str)."""
    try:
        modelo_path = REGIONES_CONFIG[region]["modelo"]
        m = joblib.load(modelo_path)
        return m, None
    except FileNotFoundError:
        try:
            m = joblib.load("model.pkl")
            return m, None
        except Exception as e:
            return None, f"No se encontró el modelo para la región {region} ({modelo_path}). Error: {str(e)}"
    except Exception as e:
        return None, str(e)


def _get_expected_features_from_model(m):
    if m is None:
        return None
    feat = getattr(m, "feature_names_in_", None)
    if feat is not None:
        try:
            return list(feat)
        except Exception:
            return None
    try:
        if hasattr(m, "named_steps"):
            for step in m.named_steps.values():
                feat = getattr(step, "feature_names_in_", None)
                if feat is not None:
                    return list(feat)
    except Exception:
        pass
    return None


def _parse_mes_generic(v):
    try:
        return int(float(v))
    except Exception:
        s = str(v).strip().lower()
        meses = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6
        }
        for name, num in meses.items():
            if s.startswith(name[:3]):
                return num
    return None


def render_uso_app():
    """Renderiza la interfaz principal (uso de la app)."""
    st.title("Predicción de casos de dengue")
    st.caption("Sistema de predicción por región climática con umbrales de riesgo personalizados.")

    # ========== SELECTOR DE REGIÓN ==========
    st.subheader("1️⃣ Seleccionar región climática")
    region_seleccionada = st.selectbox(
        "Elegí la región climática del departamento:",
        options=list(REGIONES_CONFIG.keys()),
        help="Cada región tiene un modelo específico y umbrales de riesgo personalizados"
    )

    # Cargar modelo y umbrales base
    model, load_error = load_model(region_seleccionada)
    umbrales = REGIONES_CONFIG[region_seleccionada]["umbrales"]

    # Expander: umbrales por mes si existe el Excel
    with st.expander("📊 Ver umbrales de riesgo para esta región"):
        excel_path = "data/UMBRALES POR MES REGION.xlsx"
        try:
            df_umbr = pd.read_excel(excel_path, engine="openpyxl")
        except Exception as e:
            st.error(f"No se pudo leer {excel_path}: {e}")
            df_umbr = None

        if df_umbr is None or df_umbr.empty:
            st.info("No hay datos de umbrales disponibles en el Excel. Usando umbrales por configuración de la región.")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"🟢 **Bajo**: < {umbrales['bajo']} casos")
            with col2:
                st.markdown(f"🟠 **Medio**: {umbrales['bajo']} - {umbrales['medio']-1} casos")
            with col3:
                st.markdown(f"🔴 **Alto**: ≥ {umbrales['medio']} casos")
        else:
            df_umbr.columns = [str(c).strip().lower() for c in df_umbr.columns]
            possible_region_cols = ['clima_region', 'region', 'clima', 'region_nombre']
            possible_mes_cols = ['mes', 'mes_nombre', 'mes_name']
            region_col = next((c for c in df_umbr.columns if c in possible_region_cols), None)
            mes_col = next((c for c in df_umbr.columns if c in possible_mes_cols), None)
            q33_col = next((c for c in df_umbr.columns if 'q33' in c), None)
            q66_col = next((c for c in df_umbr.columns if 'q66' in c), None)

            if region_col is None or mes_col is None or q33_col is None or q66_col is None:
                st.warning("El archivo de umbrales no contiene las columnas esperadas (region/mes/q33/q66). Mostrando umbrales por configuración.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"🟢 **Bajo**: < {umbrales['bajo']} casos")
                with col2:
                    st.markdown(f"🟠 **Medio**: {umbrales['bajo']} - {umbrales['medio']-1} casos")
                with col3:
                    st.markdown(f"🔴 **Alto**: ≥ {umbrales['medio']} casos")
            else:
                df = df_umbr[[region_col, mes_col, q33_col, q66_col]].copy()
                df[region_col] = df[region_col].astype(str).str.strip().str.upper()
                df['mes_num'] = df[mes_col].apply(_parse_mes_generic)
                df['mes_name'] = df['mes_num'].map({1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio'})

                reg = region_seleccionada.strip().upper()
                df_reg = df[df[region_col] == reg].copy()
                df_reg = df_reg[df_reg['mes_num'].isin([1,2,3,4,5,6])].sort_values('mes_num')

                if df_reg.empty:
                    st.info(f"No se encontraron filas en el Excel para la región {region_seleccionada}.")
                    st.write(df.head())
                else:
                    df_reg[q33_col] = pd.to_numeric(df_reg[q33_col], errors='coerce')
                    df_reg[q66_col] = pd.to_numeric(df_reg[q66_col], errors='coerce')

                    def fmt_ceil_int(x):
                        if pd.isna(x):
                            return ""
                        try:
                            return str(int(ceil(float(x))))
                        except Exception:
                            return ""

                    df_reg['Bajo (< q33)'] = df_reg[q33_col].apply(lambda v: f"< {fmt_ceil_int(v)}")
                    df_reg['Medio (q33 - q66)'] = df_reg.apply(lambda r: f"{fmt_ceil_int(r[q33_col])} - {fmt_ceil_int(r[q66_col])}", axis=1)
                    df_reg['Alto (> q66)'] = df_reg[q66_col].apply(lambda v: f"> {fmt_ceil_int(v)}")

                    display_df = df_reg[['mes_num','mes_name','Bajo (< q33)','Medio (q33 - q66)','Alto (> q66)']].copy()
                    display_df = display_df.rename(columns={'mes_name':'Mes'}).drop(columns=['mes_num'])
                    try:
                        st.dataframe(display_df.style.hide_index())
                    except Exception:
                        st.dataframe(display_df)

    # ======== INPUTS ========
    st.divider()
    st.subheader("2️⃣ Ingresar parámetros")
    st.info(f"**Región:** {region_seleccionada}")

    densidad = st.number_input(
        "Densidad del departamento (personas/km²)",
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Densidad del departamento en personas por km²",
    )

    # Selector separado: mes y día (mostramos la semana epidemiológica al lado)
    month_names_ui = ['Enero','Febrero','Marzo','Abril','Mayo','Junio']
    col_m, col_d, col_w = st.columns([1,1,1])
    with col_m:
        mes_sel = st.selectbox("Mes", options=list(range(1,7)), format_func=lambda x: month_names_ui[x-1], index=0, key='mes_sel')
    with col_d:
        year_for_calc = pd.Timestamp.now().year
        try:
            days_in_month = pd.Period(f"{year_for_calc}-{mes_sel}").days_in_month
        except Exception:
            days_in_month = 31
        dia_sel = st.selectbox("Día", options=list(range(1, days_in_month+1)), index=0, key='dia_sel')

    try:
        fecha_sem = pd.Timestamp(year_for_calc, int(mes_sel), int(dia_sel)).date()
        semana_epid = int(pd.to_datetime(fecha_sem).isocalendar().week)
    except Exception:
        fecha_sem = pd.to_datetime("2023-01-01").date()
        semana_epid = 1

    with col_w:
        st.markdown(f"**Semana epidemiológica:** {semana_epid}")

    semana_epidemiologica = semana_epid

    prec_sem_prom = st.number_input(
        "Precipitación semanal promedio (mm) — prec_sem_prom",
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Precipitación semanal promedio en mm",
    )

    hum_sem_prom = st.number_input(
        "Humedad semanal promedio (%) — hum_sem_prom",
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Humedad relativa semanal promedio (porcentaje)",
    )

    temp_sem_prom = st.number_input(
        "Temperatura semanal promedio (°C) — temp_sem_prom",
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Temperatura semanal promedio en °C",
    )

    with st.expander("Significado de las columnas (UI → nombres usados)"):
        st.write("- densidad → densidad del departamento (personas/km²)")
        st.write("- semana_epidemiologica → semana epidemiológica (convertida desde la fecha seleccionada)")
        st.write("- prec_sem_prom → precipitación semanal promedio (mm)")
        st.write("- hum_sem_prom → humedad semanal promedio (%)")
        st.write("- temp_sem_prom → temperatura semanal promedio (°C)")

    input_dict = {
        "densidad": [densidad],
        "semana_epidemiologica": [semana_epidemiologica],
        "prec_sem_prom": [prec_sem_prom],
        "hum_sem_prom": [hum_sem_prom],
        "temp_sem_prom": [temp_sem_prom],
    }
    X = pd.DataFrame(input_dict)

    st.subheader("3️⃣ Datos a predecir")
    st.dataframe(X)
    st.write(f"Fecha seleccionada: {fecha_sem} — semana epidemiológica: {semana_epidemiologica}")

    expected_features = _get_expected_features_from_model(model)

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
                X_to_predict = X.copy()
                mapping_performed = None
                if expected_features is not None:
                    if set(expected_features) != set(X_to_predict.columns):
                        available = list(X_to_predict.columns)
                        rename_map = {}
                        used = set()
                        for exp in expected_features:
                            matches = difflib.get_close_matches(exp, available, n=3, cutoff=0.3)
                            chosen = None
                            for mname in matches:
                                if mname not in used:
                                    chosen = mname
                                    break
                            if chosen is None and len(available) > 0:
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
                        for col in expected_features:
                            if col not in X_to_predict.columns:
                                X_to_predict[col] = np.nan

                y_pred = model.predict(X_to_predict)
                casos = float(y_pred[0])
                casos_int = max(0, int(round(casos)))

                st.subheader("Resultado")

                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        st.metric(label="Casos estimados", value=f"{casos_int:,}".replace(",", "."))
                    with c2:
                        st.metric("Semana epid.", f"{semana_epidemiologica}")
                    with c3:
                        st.metric("Fecha elegida", pd.to_datetime(fecha_sem).strftime("%d-%m-%Y"))

                    # buscar q33/q66 para el mes seleccionado (si existen)
                    q33_val = None
                    q66_val = None
                    try:
                        excel_path = "data/UMBRALES POR MES REGION.xlsx"
                        df_umbr2 = pd.read_excel(excel_path, engine="openpyxl")
                        df_umbr2.columns = [str(c).strip().lower() for c in df_umbr2.columns]
                        possible_region_cols = ['clima_region', 'region', 'clima', 'region_nombre']
                        possible_mes_cols = ['mes', 'mes_nombre', 'mes_name']
                        region_col = next((c for c in df_umbr2.columns if c in possible_region_cols), None)
                        mes_col = next((c for c in df_umbr2.columns if c in possible_mes_cols), None)
                        q33_col = next((c for c in df_umbr2.columns if 'q33' in c), None)
                        q66_col = next((c for c in df_umbr2.columns if 'q66' in c), None)

                        if region_col and mes_col and q33_col and q66_col:
                            df_umbr2[region_col] = df_umbr2[region_col].astype(str).str.strip().str.upper()
                            df_umbr2['mes_num'] = df_umbr2[mes_col].apply(_parse_mes_generic)
                            reg_norm = region_seleccionada.strip().upper()
                            row = df_umbr2[(df_umbr2[region_col] == reg_norm) & (df_umbr2['mes_num'] == mes_sel)]
                            if not row.empty:
                                q33_val = pd.to_numeric(row[q33_col].iloc[0], errors='coerce')
                                q66_val = pd.to_numeric(row[q66_col].iloc[0], errors='coerce')
                    except Exception:
                        q33_val = None
                        q66_val = None

                    def nivel_riesgo(n, umbrales_region, q33=None, q66=None):
                        try:
                            if q33 is not None and not pd.isna(q33) and q66 is not None and not pd.isna(q66):
                                q33_i = int(ceil(float(q33)))
                                q66_i = int(ceil(float(q66)))
                                if n > q66_i:
                                    return "🔴 Alto", "#fee2e2"
                                elif n >= q33_i:
                                    return "🟠 Medio", "#ffedd5"
                                else:
                                    return "🟢 Bajo", "#dcfce7"
                        except Exception:
                            pass

                        if n >= umbrales_region.get('medio', 0):
                            return "🔴 Alto", "#fee2e2"
                        elif n >= umbrales_region.get('bajo', 0):
                            return "🟠 Medio", "#ffedd5"
                        else:
                            return "🟢 Bajo", "#dcfce7"

                    etiqueta, color = nivel_riesgo(casos_int, umbrales, q33_val, q66_val)
                    st.markdown(
                        f"""
                        <div style=""
                            display:inline-block;
                            padding:6px 10px;
                            border-radius:999px;
                            background:{color};
                            margin-bottom:10px;
                            font-weight:600;">
                            {etiqueta}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # leyenda con umbrales por mes/región si está disponible
                    caption_text = None
                    try:
                        excel_path = "data/UMBRALES POR MES REGION.xlsx"
                        df_umbr3 = pd.read_excel(excel_path, engine="openpyxl")
                        df_umbr3.columns = [str(c).strip().lower() for c in df_umbr3.columns]
                        possible_region_cols = ['clima_region', 'region', 'clima', 'region_nombre']
                        possible_mes_cols = ['mes', 'mes_nombre', 'mes_name']
                        region_col = next((c for c in df_umbr3.columns if c in possible_region_cols), None)
                        mes_col = next((c for c in df_umbr3.columns if c in possible_mes_cols), None)
                        q33_col = next((c for c in df_umbr3.columns if 'q33' in c), None)
                        q66_col = next((c for c in df_umbr3.columns if 'q66' in c), None)

                        if region_col and mes_col and q33_col and q66_col:
                            df_umbr3[region_col] = df_umbr3[region_col].astype(str).str.strip().str.upper()
                            df_umbr3['mes_num'] = df_umbr3[mes_col].apply(_parse_mes_generic)
                            reg_norm = region_seleccionada.strip().upper()
                            row = df_umbr3[(df_umbr3[region_col] == reg_norm) & (df_umbr3['mes_num'] == mes_sel)]
                            if not row.empty:
                                q33_val2 = pd.to_numeric(row[q33_col].iloc[0], errors='coerce')
                                q66_val2 = pd.to_numeric(row[q66_col].iloc[0], errors='coerce')
                                if not pd.isna(q33_val2) and not pd.isna(q66_val2):
                                    caption_text = (f"📊 Umbrales para **{region_seleccionada} — {month_names_ui[mes_sel-1]}**: "
                                                    f"Bajo < {int(ceil(q33_val2))} | Medio: {int(ceil(q33_val2))} - {int(ceil(q66_val2))} | Alto ≥ {int(ceil(q66_val2))}")
                    except Exception:
                        caption_text = None

                    if caption_text is None:
                        caption_text = (f"📊 Umbrales para **{region_seleccionada}**: Bajo < {umbrales['bajo']} | "
                                        f"Medio: {umbrales['bajo']}-{umbrales['medio']-1} | Alto ≥ {umbrales['medio']}")

                    st.caption(caption_text)

                if mapping_performed is not None:
                    st.info("Se realizó un mapeo automático entre columnas de la UI y las esperadas por el modelo:")
                    st.write(mapping_performed)

            except Exception as e:
                st.error(f"Ocurrió un error al predecir: {e}")
                if expected_features is not None:
                    st.info("El modelo indica que espera las siguientes columnas:")
                    st.write(expected_features)
                else:
                    st.info("No se pudieron determinar las columnas esperadas por el modelo. Comprueba el stacktrace en la terminal.")

    st.divider()
    st.caption("💡 Tip: Cada región tiene un modelo específico y umbrales de riesgo personalizados basados en características climáticas locales.")


if __name__ == "__main__":
    render_uso_app()
semana_epidemiologica = semana_epid

prec_sem_prom = st.number_input(
    "Precipitación semanal promedio (mm) — prec_sem_prom",
    value=0.0,
    step=0.1,
    format="%.3f",
    help="Precipitación semanal promedio en mm",
)

hum_sem_prom = st.number_input(
    "Humedad semanal promedio (%) — hum_sem_prom",
    value=0.0,
    step=0.1,
    format="%.3f",
    help="Humedad relativa semanal promedio (porcentaje)",
)

temp_sem_prom = st.number_input(
    "Temperatura semanal promedio (°C) — temp_sem_prom",
    value=0.0,
    step=0.1,
    format="%.3f",
    help="Temperatura semanal promedio en °C",
)

# Explicación rápida sobre las columnas (ahora en el contenido principal)
with st.expander("Significado de las columnas (UI → nombres usados)"):
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

st.subheader("3️⃣ Datos a predecir")
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
            
            # Convertir a casos enteros
            casos = float(y_pred[0])
            casos_int = max(0, int(round(casos)))

            # Tarjeta principal + contexto
            st.subheader("Resultado")

            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.metric(
                        label="Casos estimados",
                        value=f"{casos_int:,}".replace(",", ".")  # cambia a coma/punto según prefieras
                    )
                with c2:
                    st.metric("Semana epid.", f"{semana_epidemiologica}")
               # with c3:
                #    st.metric("Fecha elegida", pd.to_datetime(fecha_sem).strftime("%d-%m-%Y"))

                # Calcular mes para posibles umbrales por mes (usado luego para caption también)
                try:
                    mes_num_for_caption = int(pd.to_datetime(fecha_sem).month)
                except Exception:
                    mes_num_for_caption = None

                # Intentar leer umbrales por mes/región (q33/q66) desde el Excel; si no están, usaremos umbrales generales
                q33_val = None
                q66_val = None
                try:
                    excel_path = "data/UMBRALES POR MES REGION.xlsx"
                    df_umbr2 = pd.read_excel(excel_path, engine="openpyxl")
                    df_umbr2.columns = [str(c).strip().lower() for c in df_umbr2.columns]
                    possible_region_cols = ['clima_region', 'region', 'clima', 'region_nombre']
                    possible_mes_cols = ['mes', 'mes_nombre', 'mes_name']
                    region_col = next((c for c in df_umbr2.columns if c in possible_region_cols), None)
                    mes_col = next((c for c in df_umbr2.columns if c in possible_mes_cols), None)
                    q33_col = next((c for c in df_umbr2.columns if 'q33' in c), None)
                    q66_col = next((c for c in df_umbr2.columns if 'q66' in c), None)

                    if region_col and mes_col and q33_col and q66_col and mes_num_for_caption is not None:
                        df_umbr2[region_col] = df_umbr2[region_col].astype(str).str.strip().str.upper()

                        def _parse_mes(v):
                            try:
                                return int(float(v))
                            except Exception:
                                s = str(v).strip().lower()
                                meses = {
                                    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                                    'january': 1, 'february': 2, 'march': 3, 'april': 4
                                }
                                for name, num in meses.items():
                                    if s.startswith(name[:3]):
                                        return num
                            return None

                        df_umbr2['mes_num'] = df_umbr2[mes_col].apply(_parse_mes)
                        reg_norm = region_seleccionada.strip().upper()
                        row = df_umbr2[(df_umbr2[region_col] == reg_norm) & (df_umbr2['mes_num'] == mes_num_for_caption)]
                        if not row.empty:
                            q33_val = pd.to_numeric(row[q33_col].iloc[0], errors='coerce')
                            q66_val = pd.to_numeric(row[q66_col].iloc[0], errors='coerce')
                except Exception:
                    q33_val = None
                    q66_val = None

                # Badge de nivel de riesgo: priorizar umbrales por mes (q33/q66) si están presentes
                def nivel_riesgo(n, umbrales_region, q33=None, q66=None):
                    # usar umbrales por mes si ambos q33 y q66 son numéricos
                    try:
                        if q33 is not None and not pd.isna(q33) and q66 is not None and not pd.isna(q66):
                            q33_i = int(ceil(float(q33)))
                            q66_i = int(ceil(float(q66)))
                            if n > q66_i:
                                return "🔴 Alto", "#fee2e2"
                            elif n >= q33_i:
                                return "🟠 Medio", "#ffedd5"
                            else:
                                return "🟢 Bajo", "#dcfce7"
                    except Exception:
                        pass

                    # fallback a umbrales por configuración de la región
                    if n >= umbrales_region.get('medio', 0):
                        return "🔴 Alto", "#fee2e2"
                    elif n >= umbrales_region.get('bajo', 0):
                        return "🟠 Medio", "#ffedd5"
                    else:
                        return "🟢 Bajo", "#dcfce7"

                etiqueta, color = nivel_riesgo(casos_int, umbrales, q33_val, q66_val)
                st.markdown(
                    f"""
                    <div style=""
                        display:inline-block;
                        padding:6px 10px;
                        border-radius:999px;
                        background:{color};
                        margin-bottom:10px;
                        font-weight:600;">
                        {etiqueta}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Mostrar umbrales aplicados: preferir umbrales por mes del Excel si están disponibles
                try:
                    mes_num_for_caption = int(pd.to_datetime(fecha_sem).month)
                except Exception:
                    mes_num_for_caption = None

                caption_text = None
                excel_path = "data/UMBRALES POR MES REGION.xlsx"
                try:
                    df_umbr = pd.read_excel(excel_path, engine="openpyxl")
                    df_umbr.columns = [str(c).strip().lower() for c in df_umbr.columns]
                    possible_region_cols = ['clima_region', 'region', 'clima', 'region_nombre']
                    possible_mes_cols = ['mes', 'mes_nombre', 'mes_name']
                    region_col = next((c for c in df_umbr.columns if c in possible_region_cols), None)
                    mes_col = next((c for c in df_umbr.columns if c in possible_mes_cols), None)
                    q33_col = next((c for c in df_umbr.columns if 'q33' in c), None)
                    q66_col = next((c for c in df_umbr.columns if 'q66' in c), None)

                    if region_col and mes_col and q33_col and q66_col and mes_num_for_caption is not None:
                        df_umbr[region_col] = df_umbr[region_col].astype(str).str.strip().str.upper()
                        # intentar parsear mes a número
                        def _parse_mes(v):
                            try:
                                return int(float(v))
                            except Exception:
                                s = str(v).strip().lower()
                                meses = {
                                    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                                    'january': 1, 'february': 2, 'march': 3, 'april': 4
                                }
                                for name, num in meses.items():
                                    if s.startswith(name[:3]):
                                        return num
                            return None

                        df_umbr['mes_num'] = df_umbr[mes_col].apply(_parse_mes)
                        reg_norm = region_seleccionada.strip().upper()
                        row = df_umbr[(df_umbr[region_col] == reg_norm) & (df_umbr['mes_num'] == mes_num_for_caption)]
                        if not row.empty:
                            q33_val = pd.to_numeric(row[q33_col].iloc[0], errors='coerce')
                            q66_val = pd.to_numeric(row[q66_col].iloc[0], errors='coerce')
                            if not pd.isna(q33_val) and not pd.isna(q66_val):
                                month_names = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio'}
                                mname = month_names.get(mes_num_for_caption, f'Mes {mes_num_for_caption}')
                                caption_text = (f"📊 Umbrales para **{region_seleccionada} — {mname}**: "
                                                f"Bajo < {int(ceil(q33_val))} | Medio: {int(ceil(q33_val))} - {int(ceil(q66_val))} | Alto ≥ {int(ceil(q66_val))}")
                except Exception:
                    caption_text = None

                if caption_text is None:
                    # fallback a umbrales por configuración
                    caption_text = (f"📊 Umbrales para **{region_seleccionada}**: Bajo < {umbrales['bajo']} | "
                                    f"Medio: {umbrales['bajo']}-{umbrales['medio']-1} | Alto ≥ {umbrales['medio']}")

                st.caption(caption_text)

            # Si hiciste mapeo de columnas, mantené este aviso
            if mapping_performed is not None:
                st.info("Se realizó un mapeo automático entre columnas de la UI y las esperadas por el modelo:")
                st.write(mapping_performed)

        except Exception as e:
            st.error(f"Ocurrió un error al predecir: {e}")
            # Si sabemos qué columnas espera el modelo, muéstralas para ayudar al debug
            if expected_features is not None:
                st.info("El modelo indica que espera las siguientes columnas:")
                st.write(expected_features)
            else:
                st.info("No se pudieron determinar las columnas esperadas por el modelo. Comprueba el stacktrace en la terminal.")

st.divider()
st.caption("💡 Tip: Cada región tiene un modelo específico y umbrales de riesgo personalizados basados en características climáticas locales.")

