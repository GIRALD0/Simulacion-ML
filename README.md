# Sistema de Votación Educativo · Taller Final ML

Proyecto del curso de **Machine Learning y Simulación**. Combina dos
piezas complementarias:

1. Un **dashboard Streamlit** que simula una jornada de votación
   participativa en Medellín (ciudadanos, votos, logs, mapa territorial).
2. El **taller final de Machine Learning** sobre los mismos datos: un
   notebook ejecutable que implementa K-Means, DBSCAN, Random Forest
   y PCA + regresión logística, con su documento técnico y su
   presentación ejecutiva.

> Opción A de la rúbrica — Trabajo sobre el proyecto Votación.

---

## 1. Estructura del repositorio

```text
Simulacion-ML/
├── dashboard/
│   ├── streamlit_app.py        # Aplicación Streamlit
│   └── requirements.txt        # Dependencias mínimas del dashboard
├── config/
│   └── settings.py             # Parámetros globales
├── data/                       # CSV generados (ciudadanos, votos, logs, comunas)
├── notebooks/
│   └── taller_final_ml.ipynb   # Notebook del taller (ejecutado y con outputs)
├── docs/
│   ├── documento_tecnico.md    # Informe 4-6 páginas
│   └── presentacion_ejecutiva.md  # 5 diapositivas
├── outputs/
│   ├── figuras/                # 12 figuras del EDA y los modelos
│   ├── modelos/                # K-Means, RF, PCA, LogReg serializados
│   └── comunas_con_cluster.csv # Resultado del clustering territorial
├── generate_data.py            # Script de generación de datos sintéticos
├── build_notebook.py           # Utilidad para reconstruir el notebook
├── cli.py                      # Comandos auxiliares
├── requirements.txt            # Dependencias completas del proyecto
├── Dockerfile                  # Imagen del dashboard
└── README.md                   # Este archivo
```

## 2. Requisitos

- Python 3.11 o superior (probado con 3.13).
- pip.
- Navegador para el dashboard.
- ~600 MB libres entre dependencias y artefactos.

## 3. Instalación

```powershell
cd Simulacion-ML
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En bash:

```bash
cd Simulacion-ML
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Flujo del taller final de Machine Learning

```text
1. generate_data.py   ->  data/*.csv
2. taller_final_ml.ipynb  ->  outputs/figuras + outputs/modelos
3. docs/documento_tecnico.md  ->  informe escrito
4. docs/presentacion_ejecutiva.md  ->  5 diapositivas
```

### 4.1 Generar los datasets

```bash
python generate_data.py
```

Crea cuatro archivos en `data/`:

| Archivo | Filas | Contenido |
|---|---:|---|
| `ciudadanos.csv` | 8.000 | ciudadanos habilitados (edad, género, comuna, validación, voto) |
| `votos.csv` | ~7.500 | votos con canal, dispositivo, latencia, estado, hora |
| `logs.csv` | 6.000 | eventos de auditoría con nivel y servicio |
| `comunas_agregado.csv` | 21 | KPIs agregados por comuna/corregimiento |

Los datos no son completamente aleatorios: el generador inyecta señales
realistas (latencia por canal, dependencia entre latencia y estado del
voto, brecha digital en corregimientos) para que los modelos encuentren
patrones interpretables.

### 4.2 Ejecutar el notebook

Opción A — abrir Jupyter y correr todas las celdas:

```bash
jupyter notebook notebooks/taller_final_ml.ipynb
```

Opción B — ejecutar en línea de comandos y dejar las salidas embebidas:

```bash
jupyter nbconvert --to notebook --execute --inplace \
        notebooks/taller_final_ml.ipynb \
        --ExecutePreprocessor.timeout=180
```

Al terminar quedarán en `outputs/`:

- 12 figuras PNG (EDA + modelos),
- 6 modelos `.joblib` (K-Means, Random Forest, PCA, etc.),
- `comunas_con_cluster.csv` con la asignación final de K-Means.

### 4.3 Documento técnico y presentación

- `docs/documento_tecnico.md` — informe de 4-6 páginas con introducción,
  dataset, metodología, modelos, resultados, conclusiones y recomendaciones.
- `docs/presentacion_ejecutiva.md` — cinco diapositivas en Markdown,
  listas para exportar a PDF o slides.

## 5. Algoritmos implementados

| # | Algoritmo | Grupo de la rúbrica | Métrica principal |
|---|---|---|---|
| 1 | K-Means (k=3) | No supervisado | silhouette ≈ 0.29 |
| 2 | DBSCAN (eps=0.6) | No supervisado | 5-9% de votos marcados como atípicos |
| 3 | Random Forest (350 árboles) | Supervisado | accuracy 0.877, F1-macro 0.528 |
| 4 | PCA + Reg. logística | Complementario | F1-macro 0.482 (baseline lineal) |

## 6. Ejecutar el dashboard original

El dashboard sigue siendo parte del proyecto y se usa para demostrar el
flujo Big Data:

```bash
streamlit run dashboard/streamlit_app.py
```

o vía CLI:

```bash
python cli.py dashboard
```

URL local: `http://localhost:8501`.

## 7. Reproducibilidad

- Semilla global fijada en `SEED = 42` en `generate_data.py`, el notebook
  y todos los modelos `sklearn`.
- Las versiones de las librerías están acotadas en `requirements.txt`.
- El notebook se puede regenerar desde `build_notebook.py` y volver a
  ejecutar para verificar reproducibilidad punto a punto.

## 8. Comandos útiles (resumen)

```bash
# Generar datos sintéticos
python generate_data.py

# Reconstruir el notebook desde el script (sin ejecutar)
python build_notebook.py

# Ejecutar el notebook completo y embeber salidas
jupyter nbconvert --to notebook --execute --inplace \
        notebooks/taller_final_ml.ipynb \
        --ExecutePreprocessor.timeout=180

# Dashboard
streamlit run dashboard/streamlit_app.py

# Resumen rápido del proyecto
python cli.py resumen
```

## 9. Notas finales

El proyecto está pensado para reproducirse en una sola sesión: clonar
el repo, crear el entorno virtual, ejecutar `generate_data.py` y luego
el notebook. Los modelos y figuras quedan persistidos en `outputs/` para
poder integrarlos al documento o a la presentación sin tener que volver
a entrenar.
