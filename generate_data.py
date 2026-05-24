"""Genera los datasets sinteticos del Sistema de Votacion para el Taller Final.

Se exportan cuatro CSV en la carpeta data/:
    - ciudadanos.csv
    - votos.csv
    - logs.csv
    - comunas_agregado.csv

La logica es coherente con el dashboard Streamlit, pero amplia ligeramente la
estructura de las muestras para que los modelos de Machine Learning encuentren
patrones reales (latencia dependiente del canal, estado del voto influido por
latencia y validacion del ciudadano correlacionada con la edad).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
N_CIUDADANOS = 8_000
N_VOTOS = 7_500
N_LOGS = 6_000

COMUNAS = [
    {"id": 1, "alias": "Popular", "perfil": "ladera_norte"},
    {"id": 2, "alias": "Santa Cruz", "perfil": "ladera_norte"},
    {"id": 3, "alias": "Manrique", "perfil": "ladera_norte"},
    {"id": 4, "alias": "Aranjuez", "perfil": "valle"},
    {"id": 5, "alias": "Castilla", "perfil": "valle"},
    {"id": 6, "alias": "Doce de Octubre", "perfil": "ladera_noroccidente"},
    {"id": 7, "alias": "Robledo", "perfil": "ladera_noroccidente"},
    {"id": 8, "alias": "Villa Hermosa", "perfil": "ladera_oriente"},
    {"id": 9, "alias": "Buenos Aires", "perfil": "ladera_oriente"},
    {"id": 10, "alias": "La Candelaria", "perfil": "centro"},
    {"id": 11, "alias": "Laureles-Estadio", "perfil": "centro_sur"},
    {"id": 12, "alias": "La America", "perfil": "centro_sur"},
    {"id": 13, "alias": "San Javier", "perfil": "ladera_occidente"},
    {"id": 14, "alias": "El Poblado", "perfil": "sur"},
    {"id": 15, "alias": "Guayabal", "perfil": "sur"},
    {"id": 16, "alias": "Belen", "perfil": "sur"},
    {"id": 50, "alias": "San Sebastian de Palmitas", "perfil": "rural"},
    {"id": 60, "alias": "San Cristobal", "perfil": "rural"},
    {"id": 70, "alias": "Altavista", "perfil": "rural"},
    {"id": 80, "alias": "San Antonio de Prado", "perfil": "rural"},
    {"id": 90, "alias": "Santa Elena", "perfil": "rural"},
]

# Probabilidad de validacion segun el perfil del territorio. Imita el hecho
# de que las comunas del centro y zonas con mayor cobertura digital tienden
# a registrar tasas mas altas de ciudadanos validados.
PESO_PARTICIPACION = {
    "centro": 0.85,
    "centro_sur": 0.82,
    "sur": 0.78,
    "valle": 0.72,
    "ladera_noroccidente": 0.66,
    "ladera_occidente": 0.64,
    "ladera_oriente": 0.62,
    "ladera_norte": 0.58,
    "rural": 0.48,
}

PROYECTOS = [
    {"id": "PP-101", "proyecto": "Bibliotecas barriales", "dependencia": "Cultura"},
    {"id": "PP-205", "proyecto": "Huertas urbanas", "dependencia": "Medio Ambiente"},
    {"id": "PP-318", "proyecto": "Ciclorutas seguras", "dependencia": "Movilidad"},
    {"id": "PP-422", "proyecto": "Escuelas deportivas", "dependencia": "INDER"},
    {"id": "PP-517", "proyecto": "Cuidado adulto mayor", "dependencia": "Inclusion Social"},
    {"id": "PP-633", "proyecto": "Aulas digitales", "dependencia": "Educacion"},
    {"id": "PP-704", "proyecto": "Mejoramiento de parques", "dependencia": "Infraestructura"},
]

CANALES = ["web", "movil", "punto_fisico", "kiosko"]
DISPOSITIVOS = ["android", "ios", "desktop", "tablet", "terminal"]

# Latencia base (ms) tipica por canal. Punto fisico y kiosko reportan
# tiempos mas altos porque dependen de hardware del puesto de votacion.
LATENCIA_BASE = {
    "web": 380,
    "movil": 460,
    "punto_fisico": 1_350,
    "kiosko": 1_700,
}

# Sesgo de canal por perfil territorial (probabilidades relativas).
SESGO_CANAL = {
    "rural": [0.18, 0.22, 0.45, 0.15],
    "centro": [0.42, 0.36, 0.14, 0.08],
    "centro_sur": [0.40, 0.40, 0.12, 0.08],
    "sur": [0.45, 0.38, 0.10, 0.07],
    "valle": [0.34, 0.36, 0.20, 0.10],
    "ladera_norte": [0.22, 0.30, 0.32, 0.16],
    "ladera_noroccidente": [0.25, 0.32, 0.28, 0.15],
    "ladera_oriente": [0.24, 0.30, 0.30, 0.16],
    "ladera_occidente": [0.27, 0.32, 0.26, 0.15],
}


def generar_ciudadanos(rng: np.random.Generator) -> pd.DataFrame:
    """Crea la tabla de ciudadanos habilitados para votar."""
    comuna_ids = rng.choice(
        [c["id"] for c in COMUNAS],
        size=N_CIUDADANOS,
        p=_distribuir_pesos([3.0 if c["perfil"] != "rural" else 0.8 for c in COMUNAS]),
    )
    edades = rng.integers(18, 85, size=N_CIUDADANOS)
    generos = rng.choice(["F", "M", "No reporta"], size=N_CIUDADANOS, p=[0.51, 0.46, 0.03])

    perfiles = np.array([_comuna_perfil(int(c)) for c in comuna_ids])
    p_base = np.array([PESO_PARTICIPACION[p] for p in perfiles])
    # La probabilidad de validacion baja en mayores de 70 (brecha digital)
    # y sube ligeramente entre 25 y 55 anios.
    ajuste_edad = np.where(edades >= 70, -0.18, np.where((edades >= 25) & (edades <= 55), 0.05, 0.0))
    p_validacion = np.clip(p_base + ajuste_edad + rng.normal(0, 0.04, N_CIUDADANOS), 0.05, 0.97)
    validados = rng.random(N_CIUDADANOS) < p_validacion

    # Voto efectivo: depende de la validacion y de la afinidad territorial.
    p_voto = np.where(validados, np.clip(p_base + rng.normal(0, 0.05, N_CIUDADANOS), 0.1, 0.95), 0.05)
    voto_emitido = rng.random(N_CIUDADANOS) < p_voto

    inicio = datetime(2026, 4, 15, 7, 30)
    fechas = [inicio + timedelta(seconds=int(rng.integers(0, 18_000))) for _ in range(N_CIUDADANOS)]

    df = pd.DataFrame(
        {
            "id_ciudadano": [f"C-{i + 1:07d}" for i in range(N_CIUDADANOS)],
            "edad": edades,
            "genero": generos,
            "comuna_id": comuna_ids,
            "comuna": [_comuna_alias(int(c)) for c in comuna_ids],
            "perfil_territorial": perfiles,
            "validado": validados,
            "voto_emitido": voto_emitido,
            "fecha_registro": [f.isoformat() for f in fechas],
        }
    )
    # Inyectamos un 0.6% de nulos en genero para que el notebook practique
    # tratamiento de valores faltantes.
    mascara_null = rng.random(N_CIUDADANOS) < 0.006
    df.loc[mascara_null, "genero"] = np.nan
    return df


def generar_votos(rng: np.random.Generator) -> pd.DataFrame:
    """Crea la tabla de votos emitidos, con latencia y estado dependientes."""
    comuna_ids = rng.choice(
        [c["id"] for c in COMUNAS],
        size=N_VOTOS,
        p=_distribuir_pesos([3.0 if c["perfil"] != "rural" else 0.8 for c in COMUNAS]),
    )
    perfiles = np.array([_comuna_perfil(int(c)) for c in comuna_ids])
    canales = np.array(
        [rng.choice(CANALES, p=SESGO_CANAL[p]) for p in perfiles]
    )
    dispositivos = np.where(
        canales == "movil",
        rng.choice(["android", "ios"], size=N_VOTOS, p=[0.78, 0.22]),
        np.where(
            canales == "web",
            rng.choice(["desktop", "tablet"], size=N_VOTOS, p=[0.74, 0.26]),
            "terminal",
        ),
    )

    base = np.array([LATENCIA_BASE[c] for c in canales])
    ruido = rng.normal(0, 180, size=N_VOTOS)
    # Picos puntuales de latencia: 4% de votos con saturacion del puesto.
    pico = rng.random(N_VOTOS) < 0.04
    latencia = base + ruido + np.where(pico, rng.integers(1_500, 3_500, size=N_VOTOS), 0)
    latencia = np.clip(latencia, 90, 6_000).astype(int)

    # Estado del voto: la latencia y el canal influyen en la probabilidad
    # de REVISION / INVALIDO. Mantiene la mayoria VALIDO como en la realidad.
    p_valido = np.clip(0.97 - (latencia - 400) / 18_000, 0.65, 0.985)
    estados = np.empty(N_VOTOS, dtype=object)
    aleatorio = rng.random(N_VOTOS)
    estados[aleatorio < p_valido] = "VALIDO"
    falta = aleatorio >= p_valido
    estados[falta & (aleatorio < p_valido + 0.65 * (1 - p_valido))] = "REVISION"
    estados[estados == None] = "INVALIDO"  # noqa: E711

    proyectos = rng.choice([p["proyecto"] for p in PROYECTOS], size=N_VOTOS,
                          p=[0.21, 0.13, 0.18, 0.14, 0.09, 0.15, 0.10])
    dep_map = {p["proyecto"]: p["dependencia"] for p in PROYECTOS}
    id_map = {p["proyecto"]: p["id"] for p in PROYECTOS}

    inicio = datetime(2026, 4, 15, 9, 0)
    timestamps = [inicio + timedelta(seconds=int(rng.integers(0, 28_800))) for _ in range(N_VOTOS)]

    df = pd.DataFrame(
        {
            "id_voto": [f"V-{i + 1:07d}" for i in range(N_VOTOS)],
            "id_proyecto": [id_map[p] for p in proyectos],
            "proyecto": proyectos,
            "dependencia": [dep_map[p] for p in proyectos],
            "comuna_id": comuna_ids,
            "comuna": [_comuna_alias(int(c)) for c in comuna_ids],
            "perfil_territorial": perfiles,
            "canal": canales,
            "dispositivo": dispositivos,
            "latencia_ms": latencia,
            "estado": estados,
            "timestamp_voto": [t.isoformat() for t in timestamps],
        }
    )
    # Algunos duplicados controlados para practicar deduplicacion (0.3%).
    duplicados = df.sample(n=int(N_VOTOS * 0.003), random_state=SEED)
    df = pd.concat([df, duplicados], ignore_index=True)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def generar_logs(rng: np.random.Generator) -> pd.DataFrame:
    """Crea la tabla de eventos de auditoria del sistema."""
    servicios = rng.choice(
        ["streamlit", "simulador", "etl", "auditoria", "validador"],
        size=N_LOGS,
        p=[0.30, 0.20, 0.22, 0.15, 0.13],
    )
    # Los servicios etl y validador concentran mas errores y latencias altas.
    p_error = np.where(
        np.isin(servicios, ["etl", "validador"]),
        rng.normal(0.11, 0.02, N_LOGS),
        rng.normal(0.04, 0.015, N_LOGS),
    )
    p_warning = np.clip(rng.normal(0.16, 0.03, N_LOGS), 0.05, 0.30)
    aleatorio = rng.random(N_LOGS)
    nivel = np.empty(N_LOGS, dtype=object)
    nivel[aleatorio < p_error] = "ERROR"
    nivel[(aleatorio >= p_error) & (aleatorio < p_error + p_warning)] = "WARNING"
    nivel[nivel == None] = "INFO"  # noqa: E711

    duraciones = np.where(
        nivel == "ERROR",
        rng.integers(400, 2_500, N_LOGS),
        np.where(nivel == "WARNING", rng.integers(120, 800, N_LOGS), rng.integers(15, 320, N_LOGS)),
    )

    eventos = rng.choice(
        ["registro", "validacion", "voto_emitido", "agregacion", "reporte", "auditoria_diaria"],
        size=N_LOGS,
    )

    inicio = datetime(2026, 4, 15, 7, 0)
    timestamps = [inicio + timedelta(seconds=int(rng.integers(0, 36_000))) for _ in range(N_LOGS)]

    return pd.DataFrame(
        {
            "id_log": [f"L-{i + 1:07d}" for i in range(N_LOGS)],
            "nivel": nivel,
            "servicio": servicios,
            "evento": eventos,
            "duracion_ms": duraciones,
            "timestamp": [t.isoformat() for t in timestamps],
        }
    )


def construir_comunas_agregado(ciudadanos: pd.DataFrame, votos: pd.DataFrame) -> pd.DataFrame:
    """Une totales por comuna a partir de ciudadanos y votos."""
    base_ciudadanos = (
        ciudadanos.groupby("comuna", as_index=False)
        .agg(
            ciudadanos=("id_ciudadano", "count"),
            validados=("validado", "sum"),
            edad_promedio=("edad", "mean"),
            voto_efectivo=("voto_emitido", "sum"),
        )
    )
    base_votos = (
        votos.drop_duplicates(subset=["id_voto"])
        .groupby("comuna", as_index=False)
        .agg(
            votos=("id_voto", "count"),
            latencia_media=("latencia_ms", "mean"),
            invalidos=("estado", lambda x: (x == "INVALIDO").sum()),
            revisiones=("estado", lambda x: (x == "REVISION").sum()),
        )
    )
    proyecto_lider = (
        votos.drop_duplicates(subset=["id_voto"])
        .groupby(["comuna", "proyecto"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .groupby("comuna", as_index=False)
        .first()
        .rename(columns={"proyecto": "proyecto_lider", "size": "votos_proyecto_lider"})
    )
    df = base_ciudadanos.merge(base_votos, on="comuna", how="left").merge(proyecto_lider, on="comuna", how="left")
    df["tasa_participacion"] = (df["votos"] / df["ciudadanos"]).round(4)
    df["tasa_validacion"] = (df["validados"] / df["ciudadanos"]).round(4)
    df["tasa_invalidos"] = (df["invalidos"] / df["votos"].clip(lower=1)).round(4)
    df["edad_promedio"] = df["edad_promedio"].round(1)
    df["latencia_media"] = df["latencia_media"].round(1)
    df["perfil_territorial"] = df["comuna"].map(
        lambda alias: _comuna_perfil_por_alias(alias)
    )
    return df.sort_values("votos", ascending=False).reset_index(drop=True)


def _distribuir_pesos(pesos: list[float]) -> np.ndarray:
    arr = np.array(pesos, dtype=float)
    return arr / arr.sum()


def _comuna_alias(comuna_id: int) -> str:
    for c in COMUNAS:
        if c["id"] == comuna_id:
            return c["alias"]
    return "Desconocida"


def _comuna_perfil(comuna_id: int) -> str:
    for c in COMUNAS:
        if c["id"] == comuna_id:
            return c["perfil"]
    return "rural"


def _comuna_perfil_por_alias(alias: str) -> str:
    for c in COMUNAS:
        if c["alias"] == alias:
            return c["perfil"]
    return "rural"


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(">> Generando ciudadanos...")
    ciudadanos = generar_ciudadanos(rng)
    print(">> Generando votos...")
    votos = generar_votos(rng)
    print(">> Generando logs...")
    logs = generar_logs(rng)
    print(">> Construyendo agregado por comuna...")
    comunas = construir_comunas_agregado(ciudadanos, votos)

    ciudadanos.to_csv(DATA_DIR / "ciudadanos.csv", index=False)
    votos.to_csv(DATA_DIR / "votos.csv", index=False)
    logs.to_csv(DATA_DIR / "logs.csv", index=False)
    comunas.to_csv(DATA_DIR / "comunas_agregado.csv", index=False)

    print()
    print("Archivos exportados en:", DATA_DIR)
    print(f"   ciudadanos.csv        -> {len(ciudadanos):>6} filas")
    print(f"   votos.csv             -> {len(votos):>6} filas")
    print(f"   logs.csv              -> {len(logs):>6} filas")
    print(f"   comunas_agregado.csv  -> {len(comunas):>6} filas")


if __name__ == "__main__":
    main()
