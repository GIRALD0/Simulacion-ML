"""CLI minimal para la version educativa del Sistema de Votacion."""

import subprocess
import sys
from pathlib import Path
from typing import Callable

from config.settings import get_settings


ROOT = Path(__file__).resolve().parent
settings = get_settings()


def cmd_dashboard() -> None:
    """Ejecuta el dashboard Streamlit local."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "dashboard" / "streamlit_app.py"),
            "--server.address",
            settings.streamlit_address,
            "--server.port",
            str(settings.streamlit_port),
        ],
        check=True,
    )


def cmd_resumen() -> None:
    """Muestra el alcance educativo actual."""
    print("Sistema de Votacion Educativo")
    print(f"- Streamlit: http://localhost:{settings.streamlit_port}")
    print("- PostgreSQL: retirado")
    print("- FastAPI: no requerido")
    print("- Datos: sinteticos, reproducibles y generados en memoria")
    print("- Enfoque: Big Data, Machine Learning conceptual, Data Lake y mapa")


def ayuda() -> None:
    """Muestra ayuda de uso."""
    print(
        """
Sistema de Votacion Educativo

Uso:
  python cli.py dashboard   Abre el dashboard Streamlit
  python cli.py resumen     Explica la version actual
  python cli.py ayuda       Muestra esta ayuda

Comando directo equivalente:
  streamlit run dashboard/streamlit_app.py
        """.strip()
    )


if __name__ == "__main__":
    comandos: dict[str, Callable[[], None]] = {
        "dashboard": cmd_dashboard,
        "resumen": cmd_resumen,
        "ayuda": ayuda,
    }

    if len(sys.argv) < 2:
        ayuda()
        sys.exit(0)

    comando = sys.argv[1]
    if comando not in comandos:
        print(f"Comando no reconocido: {comando}")
        ayuda()
        sys.exit(1)

    comandos[comando]()
