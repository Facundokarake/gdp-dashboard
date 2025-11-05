# Configuración de Modelos por Región

## Estructura de archivos

El sistema espera encontrar un modelo pickle para cada región climática:

```
/workspaces/gdp-dashboard/
├── model_arido_semiarido.pkl
├── model_frio_montana.pkl
├── model_subtropical.pkl
├── model_templado.pkl
└── model.pkl (fallback genérico)
```

## Regiones y Umbrales

### 1. ARIDO/SEMIARIDO
- **Modelo**: `model_arido_semiarido.pkl`
- **Umbrales de riesgo**:
  - 🟢 Bajo: < 5 casos
  - 🟠 Medio: 5-19 casos
  - 🔴 Alto: ≥ 20 casos

### 2. FRIO/MONTANA
- **Modelo**: `model_frio_montana.pkl`
- **Umbrales de riesgo**:
  - 🟢 Bajo: < 3 casos
  - 🟠 Medio: 3-9 casos
  - 🔴 Alto: ≥ 10 casos

### 3. SUBTROPICAL
- **Modelo**: `model_subtropical.pkl`
- **Umbrales de riesgo**:
  - 🟢 Bajo: < 15 casos
  - 🟠 Medio: 15-49 casos
  - 🔴 Alto: ≥ 50 casos

### 4. TEMPLADO
- **Modelo**: `model_templado.pkl`
- **Umbrales de riesgo**:
  - 🟢 Bajo: < 10 casos
  - 🟠 Medio: 10-29 casos
  - 🔴 Alto: ≥ 30 casos

## Cómo generar los modelos

Si actualmente tenés un solo modelo (`model.pkl`), podés:

1. **Opción temporal**: Copiar el mismo modelo para todas las regiones:
```bash
cp model.pkl model_arido_semiarido.pkl
cp model.pkl model_frio_montana.pkl
cp model.pkl model_subtropical.pkl
cp model.pkl model_templado.pkl
```

2. **Opción recomendada**: Entrenar modelos específicos por región:
   - Filtrar los datos de entrenamiento por región
   - Entrenar un modelo independiente para cada región
   - Guardar cada modelo con el nombre correspondiente

## Personalizar umbrales

Para modificar los umbrales de riesgo por región, editá el diccionario `REGIONES_CONFIG` en `streamlit_app.py`:

```python
REGIONES_CONFIG = {
    "ARIDO/SEMIARIDO": {
        "modelo": "model_arido_semiarido.pkl",
        "umbrales": {"bajo": 5, "medio": 20, "alto": 50}  # ← Ajustar aquí
    },
    # ... otras regiones
}
```

**Nota**: Los umbrales actuales son de ejemplo y deben ajustarse según:
- Datos históricos de cada región
- Capacidad del sistema de salud local
- Densidad poblacional
- Características epidemiológicas específicas
