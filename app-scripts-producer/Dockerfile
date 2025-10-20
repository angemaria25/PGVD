# Usar una versión más reciente de Python basada en Debian Bullseye
FROM python:3.9-slim-bullseye

WORKDIR /app

# Instala dependencias del sistema y Java 11
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Copia dependencias de Python
COPY common_requirements.txt .

# Instala dependencias de Python
RUN pip install --no-cache-dir -r common_requirements.txt

# Descarga recursos de NLTK básicos
RUN python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

# Copia los scripts de la aplicación
COPY . .

# Comando por defecto
CMD ["python", "advanced_producer.py"]
