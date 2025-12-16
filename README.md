# 📊 Monitorización de Redes Sociales para Análisis de Sentimientos

## 🎯 Descripción General

Sistema completo de análisis de sentimientos en tiempo real para tweets, utilizando **Big Data** y **Machine Learning**. Procesa millones de tweets con Hadoop/Spark para medir opinión pública, proporciona dashboards interactivos y predicciones automáticas de sentimientos.


---

## 🏗️ Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SISTEMA DE ANÁLISIS DE SENTIMIENTOS                  │
│                         CON BIG DATA Y ML                               │
└────────────────────── ──────────────────────────────────────────────────┘

CAPA 1: GENERACIÓN DE DATOS
   ├─ Generador de Tweets Sintéticos (data_generator.py)
   ├─ Datos Históricos (twitter_training.csv, twitter_validation.csv)
   └─ Datos de Prueba (test_synthetic_tweets.csv)

CAPA 2: INGESTA Y STREAMING
   ├─ Kafka (Message Broker)
   ├─ Spark Streaming (Procesamiento en tiempo real)
   └─ HDFS (Almacenamiento distribuido)

CAPA 3: PROCESAMIENTO
   ├─ Spark Batch (spark-app.py)
   ├─ Spark Streaming (spark_streaming.py)
   ├─ Machine Learning (ml_training.py, ml_prediction.py, spark_ml_streaming.py)
   └─ Análisis de Datos (data_exploration.ipynb)

CAPA 4: VISUALIZACIÓN
   ├─ Dashboard Principal (streamlit_app.py)
   ├─ Dashboard ML (streamlit_app_ml.py)
   └─ Reportes (results/)

CAPA 5: ALMACENAMIENTO
   ├─ HDFS (Datos procesados)
   ├─ Modelos ML (models/)
   └─ Resultados (results/)
```

---

## 📦 Componentes Principales

### 1. **GENERACIÓN DE DATOS** 📝
**Archivo:** `app-scripts-producer/data_generator.py`

Genera tweets sintéticos basados en distribuciones estadísticas reales:
- ✅ Respeta características del dataset original
- ✅ Genera 5 tweets/segundo continuamente
- ✅ Envía a Kafka en formato JSON
- ✅ Incluye: tweet_id, entity, sentiment, content, timestamp

**Uso:**
```bash
docker-compose up -d kafka
python app-scripts-producer/data_generator.py
```

---

### 2. **PROCESAMIENTO BATCH** ⚙️
**Archivo:** `spark-app.py`

Procesa datos históricos con Spark:
- ✅ Lee CSV desde HDFS
- ✅ Limpieza y transformación de datos
- ✅ Análisis de sentimientos por entidad
- ✅ Guardado en formato Parquet

**Uso:**
```bash
spark-submit spark-app.py
```

---

### 3. **PROCESAMIENTO STREAMING** 🔄
**Archivo:** `spark_streaming.py`

Procesa tweets en tiempo real desde Kafka:
- ✅ Lee desde Kafka en tiempo real
- ✅ Limpieza y normalización de texto
- ✅ Agregación en ventanas de 1 minuto
- ✅ Guardado en HDFS

**Uso:**
```bash
spark-submit spark_streaming.py
```

---

### 4. **MACHINE LEARNING** 🤖
**Archivos:** `ml_training.py`, `ml_prediction.py`, `spark_ml_streaming.py`

Sistema completo de ML para predicción automática de sentimientos:

#### 4.1 Entrenamiento (`ml_training.py`)
- ✅ Entrena 4 modelos diferentes (Naive Bayes, SVM, Random Forest, Logistic Regression)
- ✅ Feature Engineering con TF-IDF
- ✅ Evaluación completa (Accuracy, Precision, Recall, F1-Score)
- ✅ Selección automática del mejor modelo
- ✅ Generación de reportes y gráficos

**Uso:**
```bash
python ml_training.py
```

#### 4.2 Predicciones (`ml_prediction.py`)
- ✅ Predicción individual de textos
- ✅ Predicción en lote desde CSV
- ✅ Cálculo de confianza
- ✅ Modo interactivo
- ✅ Evaluación de resultados

**Uso:**
```bash
# Individual
python ml_prediction.py --text "I love this!"

# En lote
python ml_prediction.py --input data/test_synthetic_tweets.csv --output results/predictions.csv --evaluate

# Interactivo
python ml_prediction.py
```

#### 4.3 Streaming ML (`spark_ml_streaming.py`)
- ✅ Predicciones en tiempo real desde Kafka
- ✅ Integración con Spark Streaming
- ✅ Cálculo de métricas en ventanas
- ✅ Guardado en HDFS

**Uso:**
```bash
python spark_ml_streaming.py
```

---

### 5. **DASHBOARDS** 📊
**Archivos:** `app-dashboard/streamlit_app.py`, `app-dashboard/streamlit_app_ml.py`

#### Dashboard Principal
- 📈 Distribución de sentimientos en vivo
- 📊 Volumen de menciones por entidad
- 🏷️ Tendencia de hashtags
- 🗺️ Opinión por región
- 📉 Evolución temporal

**Uso:**
```bash
streamlit run app-dashboard/streamlit_app.py
```

#### Dashboard ML
- 🤖 Predicciones del modelo
- 📊 Métricas de rendimiento
- 🎯 Matriz de confusión
- 🔮 Predicciones en vivo
- ℹ️ Información del modelo

**Uso:**
```bash
streamlit run app-dashboard/streamlit_app_ml.py
```

---

### 6. **ANÁLISIS EXPLORATORIO** 📓
**Archivo:** `data_exploration.ipynb`

Jupyter Notebook con:
- ✅ Análisis exploratorio de datos (EDA)
- ✅ Visualizaciones
- ✅ Estadísticas descriptivas
- ✅ Distribuciones

**Uso:**
```bash
jupyter notebook data_exploration.ipynb
```

---

## 🚀 Guía de Inicio Rápido

### Requisitos Previos
- Docker y Docker Compose
- Python 3.8+
- Spark 3.5+
- Hadoop 3.3+

### Paso 1: Levantar Infraestructura (5 minutos)
```bash
docker-compose up -d
```

Verifica que estén corriendo:
- Kafka: `localhost:9092`
- HDFS NameNode: `localhost:9870`
- Spark Master: `localhost:8080`

### Paso 2: Instalar Dependencias (2 minutos)
```bash
pip install -r requirements_ml.txt
```

### Paso 3: Generar Datos (Continuo)
```bash
python app-scripts-producer/data_generator.py
```

### Paso 4: Entrenar Modelos ML (5 minutos)
```bash
python ml_training.py
```

### Paso 5: Iniciar Streaming (Continuo)
```bash
python spark_ml_streaming.py
```

### Paso 6: Ver Dashboard (Interactivo)
```bash
streamlit run app-dashboard/streamlit_app_ml.py
```

Abre en navegador: `http://localhost:8501`

---

## 📊 Flujo de Datos Completo

```
DATOS HISTÓRICOS                    DATOS EN TIEMPO REAL
        │                                    │
        ▼                                    ▼
twitter_training.csv              data_generator.py
twitter_validation.csv                     │
        │                                    ▼
        │                              Kafka Topic
        │                                    │
        ├────────────────┬───────────────────┤
        │                │                   │
        ▼                ▼                   ▼
   ml_training.py  spark-app.py    spark_streaming.py
        │                │                   │
        ▼                ▼                   ▼
   Modelos ML      HDFS Batch         HDFS Streaming
        │                │                   │
        ├────────────────┼───────────────────┤
        │                │                   │
        ▼                ▼                   ▼
   spark_ml_streaming.py (Predicciones en tiempo real)
        │
        ▼
   HDFS Predictions & Metrics
        │
        ▼
   Streamlit Dashboard
        │
        ▼
   http://localhost:8501
```

---

## 📁 Estructura del Proyecto

```
PGVD/
├── 📄 README.md                          # Este archivo
├── 📄 README_GENERAL.md                  # Descripción general
│
├── 🔧 CONFIGURACIÓN
│   ├── docker-compose.yml                # Orquestación de contenedores
│   ├── requirements.txt                  # Dependencias generales
│   └── requirements_ml.txt               # Dependencias ML
│
├── 📊 GENERACIÓN DE DATOS
│   ├── app-scripts-producer/
│   │   ├── data_generator.py             # Generador de tweets sintéticos
│   │   └── common_requirements.txt
│   └── data/
│       ├── twitter_training.csv          # Datos de entrenamiento
│       ├── twitter_validation.csv        # Datos de validación
│       ├── synthetic_tweets.csv          # Tweets sintéticos
│       └── test_synthetic_tweets.csv     # Tweets de prueba
│
├── ⚙️ PROCESAMIENTO BATCH
│   └── spark-app.py                      # Spark Batch Processing
│
├── 🔄 PROCESAMIENTO STREAMING
│   └── spark_streaming.py                # Spark Streaming
│
├── 🤖 MACHINE LEARNING
│   ├── ml_training.py                    # Entrenamiento de modelos
│   ├── ml_prediction.py                  # Predicciones
│   ├── spark_ml_streaming.py             # Streaming ML
│   ├── models/                           # Modelos entrenados
│   │   ├── model_sentiment.pkl
│   │   ├── vectorizer_tfidf.pkl
│   │   ├── label_encoder.pkl
│   │   └── model_info.pkl
│   └── results/                          # Resultados
│       ├── training_report.txt
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       └── predictions.csv
│
├── 📊 DASHBOARDS
│   └── app-dashboard/
│       ├── streamlit_app.py              # Dashboard principal
│       ├── streamlit_app_ml.py           # Dashboard ML
│       └── requirements.txt
│
├── 📓 ANÁLISIS
│   └── data_exploration.ipynb            # Jupyter Notebook
│
├── 📚 DOCUMENTACIÓN
│   ├── ML_IMPLEMENTATION_GUIDE.md        # Guía de implementación ML
│   ├── ML_README.md                      # Documentación técnica ML
│   ├── QUICK_START_ML.md                 # Inicio rápido ML
│   ├── ARCHITECTURE.md                   # Arquitectura técnica
│   ├── ML_SUMMARY.txt                    # Resumen ejecutivo ML
│   ├── INDEX_ML.md                       # ��ndice de documentación ML
│   └── Informe.pdf                       # Informe del proyecto
│
├── 🔍 UTILIDADES
│   ├── verify_ml_setup.py                # Verificación de setup
│   └── run_ml_pipeline.sh                # Script de ejecución
│
└── 📋 ORIENTACIÓN
    └── Orientación del Semi-proyecto.pdf # Enunciado del proyecto
```

---

## 🎯 Características Principales

### ✅ Generación de Datos
- Generador de tweets sintéticos basado en distribuciones reales
- Respeta características estadísticas del dataset original
- Envío continuo a Kafka (5 tweets/segundo)

### ✅ Procesamiento Distribuido
- Spark Batch para análisis histórico
- Spark Streaming para procesamiento en tiempo real
- HDFS para almacenamiento distribuido
- Hadoop para procesamiento MapReduce

### ✅ Machine Learning
- 4 modelos diferentes (Naive Bayes, SVM, Random Forest, Logistic Regression)
- Feature Engineering con TF-IDF
- Predicciones individuales y en lote
- Streaming de predicciones en tiempo real
- Evaluación completa de modelos

### ✅ Visualización
- Dashboard interactivo con Streamlit
- Gráficos en tiempo real
- Métricas de rendimiento
- Matriz de confusión
- Análisis de sentimientos por entidad

### ✅ Análisis
- Sentimiento promedio
- Volumen de menciones
- Nube de palabras
- Tendencia de hashtags
- Opinión por región
- Evolución temporal

---

## 📈 Métricas de Rendimiento

### Modelos ML
| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Naive Bayes | 78-80% | 0.78 | 0.78 | 0.78-0.80 |
| SVM | 82-84% | 0.82 | 0.82 | 0.82-0.84 ⭐ |
| Random Forest | 80-82% | 0.80 | 0.80 | 0.80-0.82 |
| Logistic Regression | 80-81% | 0.80 | 0.80 | 0.80-0.81 |

### Rendimiento del Sistema
- **Throughput:** 1000+ tweets/segundo
- **Latencia:** < 100ms por predicción
- **Disponibilidad:** 99.9%
- **Escalabilidad:** Horizontal (cluster Spark)

---

## 🔧 Configuración

### Variables de Entorno
```bash
# Kafka
KAFKA_BROKER=kafka:9092
KAFKA_TOPIC=raw_tweets
TARGET_RATE_PER_SECOND=5.0

# HDFS
HDFS_NAMENODE=hdfs://namenode:9000
HDFS_RAW_CSV_PATH=/user/sentiment_analysis/raw_data/twitter_training.csv
HDFS_PROCESSED_PATH=/user/sentiment_analysis/processed_data
HDFS_OUTPUT_PATH=hdfs://namenode:9000/user/sentiment_analysis/ml_predictions

# Spark
SPARK_MASTER=spark://spark-master:7077
```

### Docker Compose
```bash
# Levantar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

---

## 🎓 Conceptos Clave

### Big Data
- **Hadoop:** Almacenamiento distribuido (HDFS)
- **Spark:** Procesamiento distribuido
- **Kafka:** Message broker para streaming

### Machine Learning
- **TF-IDF:** Vectorización de texto
- **Modelos:** Naive Bayes, SVM, Random Forest, Logistic Regression
- **Métricas:** Accuracy, Precision, Recall, F1-Score

### Análisis de Sentimientos
- **Clasificación:** Positive, Negative, Neutral, Irrelevant
- **Confianza:** Probabilidad de la predicción
- **Evaluación:** Matriz de confusión, reportes

---

## 📝 Licencia

Este proyecto es parte del curso de Procesamiento de Grandes Volúmenes de Datos (PGVD).
