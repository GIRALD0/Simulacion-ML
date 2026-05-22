"""Configuracion simple para la version educativa Streamlit."""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    """Valores usados por Streamlit, Docker y scripts auxiliares."""

    streamlit_port: int = int(os.getenv("STREAMLIT_PORT", 8501))
    streamlit_address: str = os.getenv("STREAMLIT_ADDRESS", "0.0.0.0")

    spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
    spark_memory: str = os.getenv("SPARK_MEMORY", "1G")

    total_ciudadanos: int = int(os.getenv("TOTAL_CIUDADANOS", 1_000_000))
    total_votos: int = int(os.getenv("TOTAL_VOTOS", 650_000))
    total_logs: int = int(os.getenv("TOTAL_LOGS", 1_500_000))
    fecha_jornada: str = os.getenv("FECHA_JORNADA", "2026-04-15")
    max_ciudadanos_simulacion: int = int(os.getenv("MAX_CIUDADANOS_SIMULACION", 2_200_166))

    environment: str = os.getenv("ENVIRONMENT", "education")


@lru_cache()
def get_settings() -> Settings:
    """Retorna configuracion cacheada sin depender de base de datos."""
    return Settings()
