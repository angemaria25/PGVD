# Usamos la imagen oficial como base
FROM apache/spark:3.5.0

# Cambiamos a root para poder instalar cosas
USER root

# Instalamos pip y las librerías de IA
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip install --no-cache-dir \
    scikit-learn \
    pandas \
    numpy

# Volvemos al usuario spark por seguridad
USER spark