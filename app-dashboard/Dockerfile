FROM python:3.10-slim-bullseye

WORKDIR /app

# Copia dependencias de Python
COPY requirements.txt .

# Instala dependencias del sistema necesarias y Python
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Copia todo el código
COPY . .

# Expone puerto de Streamlit
EXPOSE 8501

# Comando por defecto
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
