# Taller Final — Aplicación de Machine Learning sobre un Proyecto Real

## 1. Contexto

Durante el semestre se trabajaron fundamentos de analítica, estadística aplicada, visualización, modelos supervisados, clustering, reducción de dimensionalidad, privacidad diferencial, evaluación de modelos y proyectos aplicados.

Para la nota final, cada equipo deberá implementar **mínimo 3 algoritmos de Machine Learning** sobre el proyecto **Votacion Pro** o sobre un proyecto/dataset de interés propio.

Votacion simula una plataforma de votación participativa con ciudadanos, votos, logs, comunas/corregimientos, proyectos, canales de votación, dependencias y salidas analíticas.

---

## 2. Objetivo general

Aplicar técnicas de Machine Learning sobre un conjunto de datos real o simulado, justificando la selección de algoritmos, preparando los datos, entrenando modelos, evaluando resultados e interpretando los hallazgos obtenidos.

---

## 3. Condiciones del taller

Cada equipo debe seleccionar una de estas dos opciones:

### Opción A — Trabajar sobre Votacion

Pueden usar datos simulados del proyecto, por ejemplo:

- ciudadanos,
- votos,
- logs,
- comunas,
- canales,
- proyectos,
- dependencias,
- participación territorial,
- latencia,
- auditoría,
- errores.

### Opción B — Trabajar sobre un proyecto propio

Pueden usar un dataset relacionado con:

- salud,
- educación,
- movilidad,
- ventas,
- deportes,
- redes sociales,
- IoT,
- gobierno,
- medio ambiente,
- imágenes,
- texto,
- comportamiento de usuarios.

El dataset debe tener mínimo **500 registros**, salvo que el docente apruebe una excepción.

---

## 4. Algoritmos mínimos a implementar

Cada equipo debe implementar **mínimo 3 algoritmos** de los vistos durante el semestre.

Deben escoger al menos uno de cada grupo.

### Grupo 1 — Modelos supervisados

Pueden elegir uno o más:

- Regresión lineal.
- Regresión logística.
- KNN.
- Naive Bayes.
- SVM.
- Perceptrón.
- Random Forest.

### Grupo 2 — Modelos no supervisados

Pueden elegir uno o más:

- K-Means.
- DBSCAN.
- Mean Shift.

### Grupo 3 — Análisis complementario

Pueden elegir uno o más:

- PCA.
- Selección de características.
- Métricas de clustering.
- Privacidad diferencial con SmartNoise.
- Comparación de modelos.

---

## 5. Propuestas de aplicación sobre Votacion 

| Problema | Algoritmo recomendado | Qué se busca |
|---|---|---|
| Segmentar comunas según comportamiento de votación | K-Means | Agrupar comunas con patrones similares |
| Detectar anomalías en votos o logs | DBSCAN | Identificar comportamientos atípicos |
| Predecir si un ciudadano votará | Regresión logística / KNN | Clasificación binaria |
| Predecir nivel de participación por comuna | Regresión lineal / Random Forest | Estimación numérica |
| Reducir variables para visualizar patrones | PCA | Ver estructura de datos en 2D |
| Clasificar canal de votación más probable | Naive Bayes / KNN / SVM | Clasificación multiclase |
| Evaluar importancia de variables | Selección de características / Random Forest | Identificar variables más influyentes |

---

## 6. Estructura obligatoria del notebook

El notebook debe seguir la misma estructura trabajada durante el semestre.

### Antes de cada bloque de código

Debe explicar qué se va a hacer.

Ejemplo:

```markdown
En este bloque se cargará el dataset de votos y se revisará su estructura general para identificar columnas, tipos de datos y valores faltantes.
```

### Dentro del código

Cada línea importante debe tener comentario.

Ejemplo:

```python
# Importamos pandas para trabajar con tablas.
import pandas as pd

# Cargamos el dataset desde un archivo CSV.
df = pd.read_csv("votos.csv")

# Mostramos las primeras filas para revisar la estructura.
df.head()
```

### Después de cada bloque

Debe incluir interpretación.

Ejemplo:

```markdown
La tabla muestra que el dataset contiene información de votos por comuna, canal y proyecto. Estas variables pueden ser útiles para segmentar territorios y analizar patrones de participación.
```

---

## 7. Desarrollo mínimo esperado

### Parte 1 — Contexto del problema

Cada equipo debe explicar:

- Qué proyecto o dataset eligió.
- Qué problema quiere analizar.
- Por qué ese problema se puede abordar con Machine Learning.
- Qué valor aporta el análisis.

Ejemplo:

> El equipo trabajará con datos de Votacion para analizar patrones de participación ciudadana por comuna y detectar posibles comportamientos atípicos en votos y logs.

### Parte 2 — Carga y exploración de datos

Debe incluir:

- carga del dataset,
- dimensiones,
- tipos de datos,
- valores nulos,
- duplicados,
- estadísticas descriptivas,
- visualización inicial.

Mínimo deben incluir:

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### Parte 3 — Limpieza y preparación

Debe incluir:

- tratamiento de nulos,
- eliminación o justificación de duplicados,
- selección de variables,
- codificación de variables categóricas,
- escalado si aplica,
- separación entre entrenamiento y prueba si usan modelos supervisados.

### Parte 4 — Análisis exploratorio

Debe incluir mínimo **5 visualizaciones**, por ejemplo:

- barras por comuna,
- distribución de votos,
- boxplot de latencia,
- dispersión entre variables,
- mapa de calor de correlaciones,
- distribución por canal,
- conteo de errores por servicio.

### Parte 5 — Implementación de mínimo 3 algoritmos

Cada algoritmo debe tener:

1. Justificación.
2. Preparación de datos.
3. Entrenamiento o ejecución.
4. Resultados.
5. Métricas.
6. Interpretación.

---

## 8. Ejemplo de combinación válida

Un equipo puede entregar:

### Algoritmo 1 — K-Means

Objetivo:

> Agrupar comunas según cantidad de ciudadanos, votos, participación y proyectos seleccionados.

Métricas sugeridas:

- inertia,
- silhouette score,
- visualización PCA 2D.

### Algoritmo 2 — DBSCAN

Objetivo:

> Detectar comunas, votos o logs con comportamientos atípicos.

Métricas sugeridas:

- número de clusters,
- cantidad de ruido,
- silhouette si aplica,
- análisis de puntos etiquetados como `-1`.

### Algoritmo 3 — Regresión logística

Objetivo:

> Predecir si un ciudadano vota o no vota.

Métricas sugeridas:

- accuracy,
- precision,
- recall,
- F1-score,
- matriz de confusión.

---

## 9. Preguntas que debe responder el equipo

Cada equipo debe responder mínimo estas preguntas:

1. ¿Qué problema seleccionaron y por qué?
2. ¿Qué variables usaron y por qué?
3. ¿Qué limpieza realizaron?
4. ¿Qué algoritmo tuvo mejor desempeño?
5. ¿Qué algoritmo fue más fácil de interpretar?
6. ¿Qué patrones encontraron?
7. ¿Qué limitaciones tiene el análisis?
8. ¿Qué decisión se podría tomar con estos resultados?
9. ¿Qué mejorarían si tuvieran más tiempo?
10. ¿Cómo se podría llevar esto a producción?

---

## 10. Entregables

Cada equipo debe entregar:

### 1. Notebook ejecutable `.ipynb`

Debe contener:

- contexto,
- carga de datos,
- limpieza,
- EDA,
- mínimo 3 algoritmos,
- métricas,
- visualizaciones,
- interpretación,
- conclusiones.

### 2. Documento técnico corto

Extensión sugerida: **4 a 6 páginas**.

Debe incluir:

1. Introducción.
2. Descripción del dataset.
3. Problema abordado.
4. Metodología.
5. Algoritmos implementados.
6. Resultados.
7. Conclusiones.
8. Recomendaciones.

### 3. Presentación ejecutiva

Máximo **5 diapositivas**.

Estructura sugerida:

1. Problema y dataset.
2. Preparación de datos.
3. Modelos implementados.
4. Resultados comparativos.
5. Conclusiones y recomendaciones.

### 4. Repositorio GitHub con los integrantes del equipo e invitacion Daniel-JFA

Debe incluir:

```text
/proyecto_final_ml
│
├── data/
├── notebooks/
├── docs/
├── outputs/
├── README.md
└── requirements.txt
```

---

## 11. Rúbrica de evaluación — 20%

| Criterio | Porcentaje |
|---|---:|
| Contexto, problema y justificación | 10% |
| Limpieza y preparación de datos | 15% |
| EDA y visualizaciones | 15% |
| Implementación correcta de mínimo 3 algoritmos | 25% |
| Métricas e interpretación de resultados | 20% |
| Documento, presentación y orden del repositorio | 15% |

---

## 12. Criterios de calidad

Se valorará positivamente que el equipo:

- explique por qué eligió cada algoritmo,
- no se limite a ejecutar código,
- interprete gráficas y métricas,
- compare modelos,
- identifique limitaciones,
- proponga mejoras,
- relacione los resultados con decisiones reales.

---

## 13. Reglas importantes

- No se aceptará un trabajo que solo tenga código sin explicación.
- No se aceptará un notebook sin interpretación debajo de los resultados.
- No se aceptará entregar únicamente capturas de pantalla.
- No se aceptará usar modelos sin explicar para qué sirven.
- Cada equipo debe demostrar comprensión, no solo ejecución.

---

## 14. Final

> El objetivo de este taller no es demostrar que un modelo “corre”, sino demostrar que el equipo entiende cómo convertir datos en decisiones usando Machine Learning.
