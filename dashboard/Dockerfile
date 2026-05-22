# Usa una imagen base oficial de Python.
# La etiqueta 'slim' es una version ligera, ideal para produccion.
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copia primero el archivo de requerimientos para aprovechar el cache de Docker.
# Si este archivo no cambia, Docker no reinstalara las dependencias en builds futuros.
COPY dashboard/requirements.txt .

# Instala las dependencias de Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del codigo de la aplicacion al directorio de trabajo.
COPY . .

# Expone el puerto que Streamlit usa por defecto.
EXPOSE 8501

# El comando para ejecutar la aplicacion cuando el contenedor inicie.
# --server.address=0.0.0.0 es CRUCIAL para que la app sea accesible desde fuera del contenedor.
ENTRYPOINT ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]