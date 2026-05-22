# Sistema de Votacion Educativo

Sistema de Votacion es una aplicacion educativa en **Streamlit** para simular una jornada de votacion participativa en Medellin. La entrega quedo reducida a lo esencial: una interfaz web, datos sinteticos en memoria, graficas, tablas, mapa territorial y una explicacion clara del flujo tipo Big Data y Machine Learning.

No usa PostgreSQL, FastAPI, Kafka ni servicios externos de backend. La idea es que el proyecto se pueda ejecutar rapido en clase y que el foco sea entender los datos.

## 1. Estructura del proyecto

```text
entrega_final_votaciones/
├── dashboard/
│   ├── streamlit_app.py      # Aplicacion principal
│   └── requirements.txt      # Dependencias Python
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuracion simple
├── .env                      # Valores locales sugeridos
├── .env.example              # Plantilla de configuracion
├── .dockerignore
├── .gitignore
├── cli.py                    # Comandos de ayuda
├── Dockerfile                # Imagen Docker principal
├── Dockerfile.streamlit      # Dockerfile alterno para Streamlit
└── README.md                 # Esta guia
```

## 2. Requisitos

- Python 3.11 o superior.
- pip.
- Navegador web.
- Docker opcional.

## 3. Instalacion local

Desde la carpeta del proyecto:

```bash
cd entrega_final_votaciones
python -m venv .venv
source .venv/bin/activate
pip install -r dashboard/requirements.txt
```

## 4. Ejecutar la aplicacion

Opcion directa:

```bash
streamlit run dashboard/streamlit_app.py
```

Opcion con el CLI del proyecto:

```bash
python cli.py dashboard
```

Luego abre:

```text
http://localhost:8501
```

## 5. Ejecutar con Docker

```bash
docker build -t sistema-votacion-educativo .
docker run --rm -p 8501:8501 sistema-votacion-educativo
```

Luego abre:

```text
http://localhost:8501
```

## 6. Como funciona por dentro

```text
Parametros en Streamlit
        |
        v
Generacion de datos sinteticos
        |
        v
Agregaciones con pandas
        |
        v
Indicadores, tablas y mapa
        |
        v
Data Lake conceptual y variables para Machine Learning
```

La barra lateral permite cambiar:

- cantidad de ciudadanos,
- cantidad de votos,
- cantidad de logs,
- tasa de validacion,
- varianza entre comunas,
- semilla de reproducibilidad.

Cuando cambias esos valores, Streamlit recalcula la simulacion. La semilla permite repetir el mismo escenario para comparar resultados.

Los topes minimos de la simulacion estan organizados para trabajar como proyecto de Big Data:

| Fuente | Minimo permitido | Valor inicial sugerido |
| --- | ---: | ---: |
| Ciudadanos | 500.000 | 1.000.000 |
| Votos | 500.000 | 650.000 |
| Logs | 500.000 | 1.500.000 |

Para que la interfaz siga siendo rapida, el sistema calcula agregados sobre el volumen completo configurado y solo muestra una muestra de hasta 5.000 filas por fuente en las tablas de registros.

## 7. Que genera la simulacion

La app crea tres grupos de datos:

| Grupo | Que representa |
| --- | --- |
| Ciudadanos | Personas habilitadas para participar |
| Votos | Eventos de votacion por proyecto |
| Logs | Eventos de auditoria y operacion |

Con esos datos calcula:

- ciudadanos por comuna,
- votos por proyecto,
- votos por canal,
- votos por dependencia,
- votos por comuna y proyecto,
- logs por nivel,
- tasa de participacion,
- proyecto lider por territorio.

## 8. Pestañas del dashboard

| Pestana | Para que sirve |
| --- | --- |
| Aprender | Explica la arquitectura simplificada |
| Simulador | Muestra el proceso de generacion sintetica |
| Data Lake | Explica las capas raw, bronze, silver y gold |
| Resultados | Grafica votos por proyecto, canal y dependencia |
| Territorio | Compara ciudadanos y votos por comuna |
| Mapa | Visualiza participacion territorial en Medellin |
| Registros | Muestra ejemplos de ciudadanos y votos |
| Auditoria | Muestra logs sinteticos y errores |

## 9. Data Lake conceptual

La aplicacion no escribe un Data Lake real en disco. Lo explica de forma pedagogica:

| Capa | Significado |
| --- | --- |
| raw | Datos originales como llegaron |
| bronze | Datos con estructura y tipos claros |
| silver | Datos limpios, normalizados y listos para unir |
| gold | Indicadores listos para analisis y dashboard |

Este enfoque permite explicar Big Data sin obligar a instalar una base de datos o un cluster.

## 10. Machine Learning conceptual

La aplicacion prepara datos que pueden usarse para ejercicios de Machine Learning, por ejemplo:

- clasificar territorios con baja participacion,
- detectar anomalias en logs y votos,
- segmentar comunas por comportamiento de participacion,
- priorizar proyectos segun patrones de canal, territorio y dependencia.

En esta version no se entrena un modelo real; el foco es dejar los datos, variables y agregaciones listos para explicar el flujo antes de agregar librerias como scikit-learn o Spark MLlib.

## 11. Archivos importantes

- `dashboard/streamlit_app.py`: contiene la interfaz, generacion de datos, agregaciones y visualizaciones.
- `dashboard/requirements.txt`: lista las librerias necesarias.
- `config/settings.py`: centraliza valores simples de configuracion.
- `cli.py`: permite lanzar el dashboard o ver un resumen del proyecto.
- `Dockerfile`: empaqueta la aplicacion para ejecutarla en contenedor.

## 12. Comandos utiles

```bash
python cli.py resumen
python -m py_compile dashboard/streamlit_app.py cli.py config/settings.py
streamlit run dashboard/streamlit_app.py
```

## 13. Que se elimino

Se retiraron documentos, notebooks, laboratorios, pruebas y codigo viejo que describian otra arquitectura. Tambien se quitaron referencias operativas a PostgreSQL, FastAPI y Kafka porque ya no son necesarios para ejecutar esta version.
