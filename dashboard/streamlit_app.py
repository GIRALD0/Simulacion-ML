"""Dashboard educativo autonomo para Sistema de Votacion.

La aplicacion no necesita PostgreSQL ni una API externa: genera datos
sinteticos en memoria para explicar el flujo Big Data de punta a punta.
"""

import json
import random
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any, Iterable, Optional

import pandas as pd
import pydeck as pdk
import streamlit as st

try:
    from config.settings import get_settings
    settings = get_settings()
except ImportError:
    # Fallback seguro: Si el estudiante ejecuta este archivo fuera del proyecto
    # o desde una ruta incorrecta, se autoconfigura para no fallar.
    import dataclasses
    @dataclasses.dataclass
    class DummySettings:
        total_ciudadanos: int = 500_000
        total_votos: int = 500_000
        total_logs: int = 500_000
    settings = DummySettings()

MIN_SIMULATION_ROWS = 500_000
MAX_CIUDADANOS_SIMULACION = 2_200_166
MAX_EVENTOS_SIMULACION = 5_000_000
MAX_SAMPLE_ROWS = 5_000

ARCGIS_LAYER_QUERY = (
    "https://services8.arcgis.com/XnN79i2zv7tQsjdM/arcgis/rest/services/"
    "Tendencias_de_compra_uso_y_gesti%C3%B3n_posconsumo_de_las_bater%C3%ADas_Plomo_%C3%A1cido/"
    "FeatureServer/2/query"
)
COMMUNES_GEOJSON_URL = (
    f"{ARCGIS_LAYER_QUERY}?where=1%3D1"
    "&outFields=COMUNA,NOMBRE,SECTOR,ID,COMUNA_O_C"
    "&returnGeometry=true&outSR=4326&geometryPrecision=5"
    "&maxAllowableOffset=0.00035&f=geojson"
)

COMUNAS = [
    {"id": 1, "nombre": "Comuna 1 - Popular", "alias": "Popular", "coord": [-75.5442, 6.2935]},
    {"id": 2, "nombre": "Comuna 2 - Santa Cruz", "alias": "Santa Cruz", "coord": [-75.5546, 6.2983]},
    {"id": 3, "nombre": "Comuna 3 - Manrique", "alias": "Manrique", "coord": [-75.5455, 6.2733]},
    {"id": 4, "nombre": "Comuna 4 - Aranjuez", "alias": "Aranjuez", "coord": [-75.5615, 6.2762]},
    {"id": 5, "nombre": "Comuna 5 - Castilla", "alias": "Castilla", "coord": [-75.5687, 6.2922]},
    {"id": 6, "nombre": "Comuna 6 - Doce de Octubre", "alias": "Doce de Octubre", "coord": [-75.5793, 6.2990]},
    {"id": 7, "nombre": "Comuna 7 - Robledo", "alias": "Robledo", "coord": [-75.5925, 6.2775]},
    {"id": 8, "nombre": "Comuna 8 - Villa Hermosa", "alias": "Villa Hermosa", "coord": [-75.5444, 6.2473]},
    {"id": 9, "nombre": "Comuna 9 - Buenos Aires", "alias": "Buenos Aires", "coord": [-75.5529, 6.2315]},
    {"id": 10, "nombre": "Comuna 10 - La Candelaria", "alias": "La Candelaria", "coord": [-75.5681, 6.2477]},
    {"id": 11, "nombre": "Comuna 11 - Laureles-Estadio", "alias": "Laureles-Estadio", "coord": [-75.5913, 6.2506]},
    {"id": 12, "nombre": "Comuna 12 - La America", "alias": "La America", "coord": [-75.6056, 6.2545]},
    {"id": 13, "nombre": "Comuna 13 - San Javier", "alias": "San Javier", "coord": [-75.6171, 6.2570]},
    {"id": 14, "nombre": "Comuna 14 - El Poblado", "alias": "El Poblado", "coord": [-75.5663, 6.2010]},
    {"id": 15, "nombre": "Comuna 15 - Guayabal", "alias": "Guayabal", "coord": [-75.5865, 6.2133]},
    {"id": 16, "nombre": "Comuna 16 - Belen", "alias": "Belen", "coord": [-75.5992, 6.2256]},
    {"id": 50, "nombre": "Corregimiento 50 - San Sebastian de Palmitas", "alias": "San Sebastian de Palmitas", "coord": [-75.6893, 6.3278]},
    {"id": 60, "nombre": "Corregimiento 60 - San Cristobal", "alias": "San Cristobal", "coord": [-75.6396, 6.2902]},
    {"id": 70, "nombre": "Corregimiento 70 - Altavista", "alias": "Altavista", "coord": [-75.6348, 6.2270]},
    {"id": 80, "nombre": "Corregimiento 80 - San Antonio de Prado", "alias": "San Antonio de Prado", "coord": [-75.6734, 6.2161]},
    {"id": 90, "nombre": "Corregimiento 90 - Santa Elena", "alias": "Santa Elena", "coord": [-75.5168, 6.2350]},
]

PROYECTOS = [
    {"id": "PP-101", "proyecto": "Bibliotecas barriales", "dependencia": "Cultura", "eje": "Educacion"},
    {"id": "PP-205", "proyecto": "Huertas urbanas", "dependencia": "Medio Ambiente", "eje": "Sostenibilidad"},
    {"id": "PP-318", "proyecto": "Ciclorutas seguras", "dependencia": "Movilidad", "eje": "Infraestructura"},
    {"id": "PP-422", "proyecto": "Escuelas deportivas", "dependencia": "INDER", "eje": "Deporte"},
    {"id": "PP-517", "proyecto": "Cuidado adulto mayor", "dependencia": "Inclusion Social", "eje": "Bienestar"},
    {"id": "PP-633", "proyecto": "Aulas digitales", "dependencia": "Educacion", "eje": "Tecnologia"},
    {"id": "PP-704", "proyecto": "Mejoramiento de parques", "dependencia": "Infraestructura", "eje": "Espacio publico"},
]

CANALES = ["web", "movil", "punto_fisico", "kiosko"]
DISPOSITIVOS = ["android", "ios", "desktop", "tablet", "terminal"]
COMUNA_BY_ALIAS = {item["alias"]: item for item in COMUNAS}
DEPENDENCIA_BY_PROYECTO = {item["proyecto"]: item["dependencia"] for item in PROYECTOS}


st.set_page_config(page_title="Sistema de Votacion Educativo", page_icon="V", layout="wide")


def weighted_counts(total: int, labels: list[str], alpha: float, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    weights = [rng.gammavariate(alpha, 1.0) for _ in labels]
    weight_sum = sum(weights) or 1.0
    raw = [int(total * weight / weight_sum) for weight in weights]
    remainder = total - sum(raw)
    for index in range(remainder):
        raw[index % len(raw)] += 1
    return dict(zip(labels, raw))


def split_count(total: int, labels: list[str], seed: int) -> dict[str, int]:
    return weighted_counts(total, labels, 1.6, seed)


def comuna_from_alias(alias: str) -> dict[str, Any]:
    return COMUNA_BY_ALIAS[alias]


@st.cache_data(show_spinner=False)
def build_dataset(
    total_ciudadanos: int,
    total_votos: int,
    total_logs: int,
    tasa_validacion: float,
    alpha_comunas: float,
    seed: int,
) -> dict[str, Any]:
    comunas = [item["alias"] for item in COMUNAS]
    proyectos = [item["proyecto"] for item in PROYECTOS]
    dependencias = [item["dependencia"] for item in PROYECTOS]

    ciudadanos_por_comuna = weighted_counts(total_ciudadanos, comunas, alpha_comunas, seed)
    votos_por_comuna = weighted_counts(total_votos, comunas, max(alpha_comunas * 0.9, 0.05), seed + 7)
    votos_por_proyecto = split_count(total_votos, proyectos, seed + 13)
    votos_por_canal = split_count(total_votos, CANALES, seed + 17)
    logs_por_nivel = split_count(total_logs, ["INFO", "WARNING", "ERROR"], seed + 23)

    comuna_project_rows = []
    for comuna, votos_comuna in votos_por_comuna.items():
        local_projects = split_count(votos_comuna, proyectos, seed + len(comuna))
        for proyecto, total in local_projects.items():
            if total:
                comuna_project_rows.append({"comuna": comuna, "proyecto": proyecto, "total": total})

    ciudadanos_rows = [
        {"comuna": comuna, "total": total}
        for comuna, total in sorted(ciudadanos_por_comuna.items(), key=lambda item: item[1], reverse=True)
    ]
    proyecto_rows = [
        {"proyecto": proyecto, "total": total}
        for proyecto, total in sorted(votos_por_proyecto.items(), key=lambda item: item[1], reverse=True)
    ]
    canal_rows = [
        {"canal": canal, "total": total}
        for canal, total in sorted(votos_por_canal.items(), key=lambda item: item[1], reverse=True)
    ]
    dependencia_rows = []
    for dependencia in sorted(set(dependencias)):
        total = sum(row["total"] for row in proyecto_rows if project_dependency(row["proyecto"]) == dependencia)
        dependencia_rows.append({"dependencia": dependencia, "total": total})

    sample_size = min(total_ciudadanos, MAX_SAMPLE_ROWS)
    rng = random.Random(seed + 31)
    ciudadanos_sample = []
    for index in range(sample_size):
        comuna = rng.choices(comunas, weights=[ciudadanos_por_comuna[c] for c in comunas], k=1)[0]
        ciudadanos_sample.append(
            {
                "id_ciudadano": f"C-{index + 1:07d}",
                "edad": rng.randint(18, 82),
                "genero": rng.choice(["F", "M", "No reporta"]),
                "comuna": comuna,
                "validado": rng.random() <= tasa_validacion,
                "fecha_registro": (datetime(2026, 4, 15, 8, 0) + timedelta(seconds=index * 3)).isoformat(),
            }
        )

    vote_sample_size = min(total_votos, MAX_SAMPLE_ROWS)
    votos_sample = []
    for index in range(vote_sample_size):
        comuna = rng.choices(comunas, weights=[votos_por_comuna[c] for c in comunas], k=1)[0]
        project = rng.choices(PROYECTOS, weights=[votos_por_proyecto[p["proyecto"]] for p in PROYECTOS], k=1)[0]
        votos_sample.append(
            {
                "id_voto": f"V-{index + 1:07d}",
                "id_proyecto": project["id"],
                "proyecto": project["proyecto"],
                "comuna": comuna,
                "dependencia": project["dependencia"],
                "canal": rng.choice(CANALES),
                "dispositivo": rng.choice(DISPOSITIVOS),
                "latencia_ms": rng.randint(120, 2600),
                "estado": rng.choices(["VALIDO", "REVISION", "INVALIDO"], weights=[93, 5, 2], k=1)[0],
                "timestamp_voto": (datetime(2026, 4, 15, 9, 0) + timedelta(seconds=index * 2)).isoformat(),
            }
        )

    logs_sample = []
    for index in range(min(total_logs, MAX_SAMPLE_ROWS)):
        nivel = rng.choices(["INFO", "WARNING", "ERROR"], weights=[78, 16, 6], k=1)[0]
        logs_sample.append(
            {
                "id_log": f"L-{index + 1:07d}",
                "nivel": nivel,
                "servicio": rng.choice(["streamlit", "simulador", "etl", "auditoria"]),
                "evento": rng.choice(["registro", "validacion", "voto_emitido", "agregacion", "reporte"]),
                "duracion_ms": rng.randint(15, 900),
                "timestamp": (datetime(2026, 4, 15, 8, 0) + timedelta(seconds=index)).isoformat(),
            }
        )

    ciudadanos_validados = int(total_ciudadanos * tasa_validacion)
    return {
        "resumen": {
            "ciudadanos_registrados": total_ciudadanos,
            "ciudadanos_validados": ciudadanos_validados,
            "votos_emitidos": total_votos,
            "logs": total_logs,
            "errores": logs_por_nivel.get("ERROR", 0),
            "tasa_participacion": round((total_votos / max(total_ciudadanos, 1)) * 100, 2),
        },
        "ciudadanos_por_comuna": ciudadanos_rows,
        "votos_por_comuna_proyecto": comuna_project_rows,
        "votos_por_proyecto": proyecto_rows,
        "votos_por_canal": canal_rows,
        "votos_por_dependencia": sorted(dependencia_rows, key=lambda item: item["total"], reverse=True),
        "logs_por_nivel": [{"nivel": key, "total": value} for key, value in logs_por_nivel.items()],
        "ciudadanos_sample": ciudadanos_sample,
        "votos_sample": votos_sample,
        "logs_sample": logs_sample,
    }


def project_dependency(project_name: str) -> str:
    return DEPENDENCIA_BY_PROYECTO.get(project_name, "Sin dependencia")


def to_frame(records: Optional[Iterable[dict[str, Any]]]) -> pd.DataFrame:
    return pd.DataFrame(records or [])


def metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label, value if value is not None else 0, help=help_text)


def render_bar_chart(df: pd.DataFrame, label_column: str, value_column: str) -> None:
    if df.empty or label_column not in df or value_column not in df:
        st.info("Sin datos disponibles.")
        return
    chart_df = df[[label_column, value_column]].dropna().set_index(label_column)
    st.bar_chart(chart_df)


def html_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def render_console(lines: list[str], progress: float | None = None):
    bar = ""
    if progress is not None:
        filled = int(round(max(0.0, min(progress, 1.0)) * 32))
        bar = f"\n[{'#' * filled}{'.' * (32 - filled)}] {progress * 100:5.1f}%"
    html = (
        "<div style='background:#07110c;border:1px solid #1f3b2b;border-radius:8px;"
        "padding:14px 16px;color:#72f59b;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;"
        "font-size:13px;line-height:1.45;white-space:pre-wrap;min-height:230px;'>"
        f"{html_escape(chr(10).join(lines) + bar)}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def simulation_terminal(total_ciudadanos: int, total_votos: int, total_logs: int, sample_rows: int) -> list[str]:
    payload = {
        "total_ciudadanos": total_ciudadanos,
        "total_votos": total_votos,
        "total_logs": total_logs,
        "modo": "streamlit_local",
    }
    return [
        "$ streamlit run dashboard/streamlit_app.py",
        f"[payload] {json.dumps(payload)}",
        "[1/5] generar muestra sintetica en memoria",
        f"[2/5] agregar {total_ciudadanos:,} ciudadanos por comuna",
        f"[3/5] agregar {total_votos:,} votos por proyecto, canal y territorio",
        f"[4/5] simular {total_logs:,} eventos de auditoria",
        f"[5/5] publicar {sample_rows:,} filas de ejemplo para exploracion",
        "[ok] no se uso PostgreSQL, FastAPI ni servicios externos de base de datos",
    ]


def datalake_terminal(resumen: dict) -> list[str]:
    rows = resumen["ciudadanos_registrados"] + resumen["votos_emitidos"] + resumen["logs"]
    return [
        "$ flujo conceptual de Data Lake",
        f"[input] filas simuladas={rows:,}",
        "[raw] conservar datos originales tal como llegaron",
        "[bronze] tipar columnas y validar esquema",
        "[silver] limpiar nulos, deduplicar votos, normalizar comunas",
        "[gold] calcular KPIs por proyecto, comuna, canal y dependencia",
        "[serving] exponer tablas agregadas al dashboard Streamlit",
        "[nota] en clase puedes reemplazar pandas por Spark sin cambiar el concepto",
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def load_communes_geojson():
    request = Request(COMMUNES_GEOJSON_URL, headers={"Accept": "application/json"})
    # Timeout reducido a 5s para no bloquear a estudiantes con internet lento
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_polygon_rings(geometry: dict[str, Any]) -> Iterable[list]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        if coordinates and coordinates[0]:
            yield coordinates[0]
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if polygon and polygon[0]:
                yield polygon[0]


def interpolate_color(start: list[int], end: list[int], factor: float) -> list[int]:
    factor = max(0, min(float(factor), 1))
    return [round(start[index] + (end[index] - start[index]) * factor) for index in range(3)]


def heat_color(value: float, min_value: float, max_value: float) -> list[int]:
    if max_value <= min_value:
        return [46, 125, 50, 172] if value > 0 else [148, 163, 184, 92]
    normalized = (float(value) - float(min_value)) / (float(max_value) - float(min_value))
    red = [198, 40, 40]
    yellow = [251, 192, 45]
    green = [46, 125, 50]
    if normalized <= 0.5:
        color = interpolate_color(red, yellow, normalized * 2)
    else:
        color = interpolate_color(yellow, green, (normalized - 0.5) * 2)
    return [*color, 172]


def build_territory_frame(dataset: dict[str, Any]) -> pd.DataFrame:
    ciudadanos = to_frame(dataset["ciudadanos_por_comuna"]).rename(columns={"total": "ciudadanos"})
    votos = to_frame(dataset["votos_por_comuna_proyecto"])
    if votos.empty:
        votos_total = pd.DataFrame({"comuna": ciudadanos["comuna"], "votos": 0})
        top_projects = pd.DataFrame(
            {"comuna": ciudadanos["comuna"], "proyecto_lider": "Sin votos", "votos_proyecto_lider": 0}
        )
    else:
        votos_total = votos.groupby("comuna", as_index=False)["total"].sum().rename(columns={"total": "votos"})
        top_projects = (
            votos.sort_values("total", ascending=False)
            .groupby("comuna", as_index=False)
            .first()[["comuna", "proyecto", "total"]]
            .rename(columns={"proyecto": "proyecto_lider", "total": "votos_proyecto_lider"})
        )
    territory = ciudadanos.merge(votos_total, on="comuna", how="outer").merge(top_projects, on="comuna", how="left")
    territory = territory.fillna({"ciudadanos": 0, "votos": 0, "proyecto_lider": "Sin votos", "votos_proyecto_lider": 0})
    territory["ciudadanos"] = territory["ciudadanos"].astype(int)
    territory["votos"] = territory["votos"].astype(int)
    territory["votos_proyecto_lider"] = territory["votos_proyecto_lider"].astype(int)
    territory["comuna_id"] = territory["comuna"].map(lambda alias: comuna_from_alias(alias)["id"])
    territory["map_label"] = territory["comuna"].map(lambda alias: comuna_from_alias(alias)["nombre"])
    territory["lon"] = territory["comuna"].map(lambda alias: comuna_from_alias(alias)["coord"][0])
    territory["lat"] = territory["comuna"].map(lambda alias: comuna_from_alias(alias)["coord"][1])
    territory["participacion"] = territory.apply(
        lambda row: round((row["votos"] / row["ciudadanos"]) * 100, 2) if row["ciudadanos"] else 0,
        axis=1,
    )
    total_votos = max(int(territory["votos"].sum()), 1)
    territory["peso_votos"] = territory["votos"].apply(lambda value: round((int(value) / total_votos) * 100, 2))
    return territory


def build_communes_layer(territory: pd.DataFrame, color_metric: str):
    try:
        geojson = load_communes_geojson()
    # Capturamos Exception genérica para cubrir errores de certificados SSL
    # comunes en redes de universidades y asegurar el fallback del mapa.
    except Exception as e:
        print(f"Error al cargar GeoJSON: {e}")
        return None

    summary = territory.set_index("comuna_id").to_dict("index")
    metric_column = "participacion" if color_metric == "participacion" else "votos"
    min_metric = territory[metric_column].min()
    max_metric = territory[metric_column].max()
    polygon_rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        comuna_id = int(props.get("COMUNA") or 0)
        row = summary.get(comuna_id, {})
        if not row:
            continue
        metric_value = row.get(metric_column, 0)
        for ring in iter_polygon_rings(feature.get("geometry") or {}):
            polygon_rows.append(
                {
                    "polygon": ring,
                    "map_label": row["map_label"],
                    "ciudadanos": int(row["ciudadanos"]),
                    "votos": int(row["votos"]),
                    "peso_votos": float(row["peso_votos"]),
                    "participacion": float(row["participacion"]),
                    "proyecto_lider": row["proyecto_lider"],
                    "fill_color": heat_color(metric_value, min_metric, max_metric),
                    "outline_color": [15, 23, 42, 230],
                }
            )
    if not polygon_rows:
        return None
    return pdk.Layer(
        "PolygonLayer",
        data=polygon_rows,
        get_polygon="polygon",
        pickable=True,
        stroked=True,
        filled=True,
        auto_highlight=True,
        get_fill_color="fill_color",
        get_line_color="outline_color",
        line_width_min_pixels=2,
    )


def render_territory_map(territory: pd.DataFrame, color_metric: str):
    communes_layer = build_communes_layer(territory, color_metric)
    if communes_layer:
        layer = communes_layer
    else:
        st.info("No se pudo cargar la capa oficial de comunas; se muestra una vista por puntos.")
        metric_column = "participacion" if color_metric == "participacion" else "votos"
        min_metric = territory[metric_column].min()
        max_metric = territory[metric_column].max()
        point_data = territory.copy()
        point_data["color"] = point_data[metric_column].apply(lambda value: heat_color(value, min_metric, max_metric))
        point_data["radius"] = point_data["votos"].apply(lambda value: 700 + int(value) / max(int(point_data["votos"].max()), 1) * 3600)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=point_data,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )

    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=6.2518, longitude=-75.5636, zoom=10.65, pitch=0),
            width="100%",
            height=620,
            layers=[layer],
            tooltip={
                "html": (
                    "<b>{map_label}</b><br/>"
                    "Ciudadanos: {ciudadanos}<br/>"
                    "Votos: {votos}<br/>"
                    "Peso sobre total: {peso_votos}%<br/>"
                    "Participacion: {participacion}%<br/>"
                    "Proyecto lider: {proyecto_lider}"
                ),
                "style": {"backgroundColor": "#111827", "color": "white"},
            },
        ),
        use_container_width=True,
    )


with st.sidebar:
    st.header("Simulacion")
    total_ciudadanos = st.number_input(
        "Ciudadanos",
        min_value=MIN_SIMULATION_ROWS,
        max_value=MAX_CIUDADANOS_SIMULACION,
        value=max(settings.total_ciudadanos, MIN_SIMULATION_ROWS),
        step=50_000,
    )
    total_votos = st.number_input(
        "Votos",
        min_value=MIN_SIMULATION_ROWS,
        max_value=MAX_EVENTOS_SIMULACION,
        value=max(settings.total_votos, MIN_SIMULATION_ROWS),
        step=50_000,
    )
    total_logs = st.number_input(
        "Logs",
        min_value=MIN_SIMULATION_ROWS,
        max_value=MAX_EVENTOS_SIMULACION,
        value=max(settings.total_logs, MIN_SIMULATION_ROWS),
        step=100_000,
    )
    tasa_validacion = st.slider("Tasa de validacion", 0.0, 1.0, 0.75, 0.05)
    alpha_comunas = st.slider(
        "Varianza comunas",
        min_value=0.05,
        max_value=5.0,
        value=0.8,
        step=0.05,
        help="Menor valor produce una distribucion territorial mas desigual.",
    )
    seed = st.number_input("Semilla", min_value=1, max_value=9999, value=42, step=1)
    st.divider()
    st.caption(
        "⚠️ **Regla de Volumetría Estricta:**\n\n"
        f"Para garantizar un escenario tipo Big Data, el sistema exige un procesamiento "
        f"mínimo de **{MIN_SIMULATION_ROWS:,} registros** por cada fuente de datos (Ciudadanos, Votos y Logs)."
    )


if total_ciudadanos < MIN_SIMULATION_ROWS or total_votos < MIN_SIMULATION_ROWS or total_logs < MIN_SIMULATION_ROWS:
    st.error(
        f"⚠️ **Error de Volumetría:** Se requiere una cantidad mínima de {MIN_SIMULATION_ROWS:,} "
        f"registros por fuente de datos para realizar el análisis."
    )
    st.stop()

dataset = build_dataset(
    int(total_ciudadanos),
    int(total_votos),
    int(total_logs),
    float(tasa_validacion),
    float(alpha_comunas),
    int(seed),
)
resumen = dataset["resumen"]

st.title("Sistema de Votacion Educativo")
st.caption("Simulador local para aprender Big Data, Machine Learning, datos sinteticos, Data Lake y visualizacion territorial.")

top = st.columns(5)
with top[0]:
    metric_card("Modo", "LOCAL")
with top[1]:
    metric_card("Ciudadanos", f"{resumen['ciudadanos_registrados']:,}")
with top[2]:
    metric_card("Validados", f"{resumen['ciudadanos_validados']:,}")
with top[3]:
    metric_card("Votos", f"{resumen['votos_emitidos']:,}")
with top[4]:
    metric_card("Participacion", f"{resumen['tasa_participacion']}%")

tabs = st.tabs(["Aprender", "Simulador", "Data Lake", "Resultados", "Territorio", "Mapa", "Registros", "Auditoria"])

with tabs[0]:
    st.subheader("Como funciona")
    st.markdown(
        """
        Este proyecto quedo como una experiencia educativa autocontenida. Streamlit recibe parametros,
        genera una muestra sintetica reproducible y calcula agregados como si fueran salidas de un
        pipeline Big Data y Machine Learning.

        1. **Ingesta:** ciudadanos, votos y logs simulados.
        2. **Preparacion:** validacion, normalizacion y agrupacion con pandas.
        3. **Analitica:** KPIs por comuna, proyecto, canal y dependencia.
        4. **Machine Learning:** variables para segmentar participacion, detectar anomalias y priorizar territorios.
        5. **Visualizacion:** tablas, graficas y mapa territorial.

        La version anterior modelaba PostgreSQL como sistema transaccional. Aqui se retiro para que
        cualquiera pueda ejecutar la demo con un solo comando y concentrarse en el aprendizaje.
        """
    )
    st.info(
        f"📌 **Restricción de Volumen:** El procesamiento mínimo está bloqueado estrictamente en "
        f"**{MIN_SIMULATION_ROWS:,} registros** para obligar al sistema a simular cargas de trabajo masivas."
    )
    st.code("streamlit run dashboard/streamlit_app.py", language="bash")
    st.dataframe(
        pd.DataFrame(
            [
                {"concepto": "PostgreSQL", "en esta version": "Retirado", "proposito educativo": "Evitar infraestructura extra"},
                {"concepto": "FastAPI", "en esta version": "No requerido", "proposito educativo": "Menos piezas para explicar"},
                {"concepto": "Streamlit", "en esta version": "Interfaz principal", "proposito educativo": "Explorar datos rapido"},
                {"concepto": "Data Lake", "en esta version": "Conceptual", "proposito educativo": "Entender capas raw/bronze/silver/gold"},
                {"concepto": "Machine Learning", "en esta version": "Conceptual", "proposito educativo": "Preparar variables y casos de uso"},
                {"concepto": "Spark", "en esta version": "Opcional", "proposito educativo": "Puede agregarse cuando el grupo domine pandas"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Generador de registros")
    st.caption("Ajusta los parametros en la barra lateral. La simulacion se recalcula automaticamente.")
    render_console(
        simulation_terminal(
            resumen["ciudadanos_registrados"],
            resumen["votos_emitidos"],
            resumen["logs"],
            len(dataset["ciudadanos_sample"]) + len(dataset["votos_sample"]) + len(dataset["logs_sample"]),
        ),
        1.0,
    )
    st.markdown("#### Escenarios sugeridos")
    st.dataframe(
        pd.DataFrame(
            [
                {"escenario": "Minimo Big Data", "ciudadanos": 500_000, "votos": 500_000, "logs": 500_000},
                {"escenario": "Jornada municipal", "ciudadanos": 1_000_000, "votos": 650_000, "logs": 1_500_000},
                {"escenario": "Alta concurrencia", "ciudadanos": 1_600_000, "votos": 1_100_000, "logs": 3_000_000},
                {"escenario": "Tope poblacional", "ciudadanos": 2_200_166, "votos": 1_800_000, "logs": 5_000_000},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    st.subheader("Data Lake por capas")
    render_console(datalake_terminal(resumen), 1.0)
    st.dataframe(
        pd.DataFrame(
            [
                {"zona": "raw", "pregunta": "Que llego?", "salida": "ciudadanos.csv, votos.csv, logs.csv"},
                {"zona": "bronze", "pregunta": "Tiene estructura?", "salida": "tablas tipadas y versionadas"},
                {"zona": "silver", "pregunta": "Esta limpio?", "salida": "votos validos, comunas normalizadas"},
                {"zona": "gold", "pregunta": "Que decision apoya?", "salida": "KPIs por territorio y proyecto"},
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tabs[3]:
    col_a, col_b = st.columns(2)
    votos_proyecto = to_frame(dataset["votos_por_proyecto"])
    votos_canal = to_frame(dataset["votos_por_canal"])
    votos_dependencia = to_frame(dataset["votos_por_dependencia"])

    with col_a:
        st.subheader("Votos por proyecto")
        render_bar_chart(votos_proyecto, "proyecto", "total")
        st.dataframe(votos_proyecto, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Votos por canal")
        render_bar_chart(votos_canal, "canal", "total")
        st.dataframe(votos_canal, use_container_width=True, hide_index=True)

    st.subheader("Votos por dependencia responsable")
    render_bar_chart(votos_dependencia, "dependencia", "total")
    st.dataframe(votos_dependencia, use_container_width=True, hide_index=True)

with tabs[4]:
    col_a, col_b = st.columns(2)
    ciudadanos_comuna = to_frame(dataset["ciudadanos_por_comuna"])
    comuna_proyecto = to_frame(dataset["votos_por_comuna_proyecto"])

    with col_a:
        st.subheader("Ciudadanos por comuna")
        render_bar_chart(ciudadanos_comuna, "comuna", "total")
        st.dataframe(ciudadanos_comuna, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("Votos por comuna y proyecto")
        st.dataframe(comuna_proyecto.sort_values("total", ascending=False), use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Mapa de participacion en Medellin")
    territory = build_territory_frame(dataset)
    color_option = st.radio(
        "Escala de color",
        options=["votos", "participacion"],
        format_func=lambda value: "Cantidad de votos por comuna" if value == "votos" else "Participacion electoral",
        horizontal=True,
    )
    render_territory_map(territory, color_option)
    st.caption("Rojo indica menor valor observado y verde mayor valor observado para la metrica seleccionada.")
    sort_column = "votos" if color_option == "votos" else "participacion"
    st.dataframe(territory.sort_values(sort_column, ascending=False), use_container_width=True, hide_index=True)

with tabs[6]:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Ciudadanos de muestra")
        st.dataframe(to_frame(dataset["ciudadanos_sample"]), use_container_width=True, hide_index=True)
    with col_b:
        st.subheader("Votos de muestra")
        st.dataframe(to_frame(dataset["votos_sample"]), use_container_width=True, hide_index=True)

with tabs[7]:
    logs = to_frame(dataset["logs_sample"])
    errores = int((logs["nivel"] == "ERROR").sum()) if not logs.empty else 0
    st.subheader("Logs del sistema")
    st.metric("Errores en muestra", f"{errores:,}")
    render_bar_chart(to_frame(dataset["logs_por_nivel"]), "nivel", "total")
    st.dataframe(logs, use_container_width=True, hide_index=True)
