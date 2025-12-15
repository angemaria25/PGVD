# 🏗️ Arquitectura de Machine Learning para Análisis de Sentimientos

## 📐 Diagrama General de la Solución

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SISTEMA DE ANÁLISIS DE SENTIMIENTOS                 │
│                              CON MACHINE LEARNING                           │
└─────────────────────────────────────────────────────────────────────────────┘

                                    DATOS HISTÓRICOS
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
            twitter_training.csv  twitter_validation.csv  synthetic_tweets.csv
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────┐
                        │   ml_training.py               │
                        │  (Entrenamiento de Modelos)    │
                        │                                │
                        │  • Exploración de datos        │
                        │  • Feature Engineering (TF-IDF)│
                        │  • Entrenamiento (4 modelos)   │
                        │  • Evaluación y selección      │
                        │  • Generación de reportes      │
                        └────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
            model_sentiment.pkl  vectorizer_tfidf.pkl  label_encoder.pkl
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
        ┌──────────────────────────┐          ┌──────────────────────────┐
        │  ml_prediction.py        │          │ spark_ml_streaming.py    │
        │ (Predicciones en Lote)   │          │ (Streaming en Tiempo Real)
        │                          │          │                          │
        │ • Predicción individual  │          │ • Lee desde Kafka        │
        │ • Predicción en CSV      │          │ • Aplica modelo ML       │
        │ • Modo interactivo       │          │ • Calcula métricas       │
        │ • Evaluación             │          │ • Guarda en HDFS         │
        └──────────────────────────┘          └──────────────────────────┘
                    │                                         │
                    ▼                                         ▼
            results/predictions.csv              HDFS/ml_predictions/
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                        ┌────────────────────────────────┐
                        │ streamlit_app_ml.py            │
                        │ (Dashboard Interactivo)        │
                        │                                │
                        │ • Dashboard Principal          │
                        │ • Predicciones en Vivo         │
                        │ • Métricas de Rendimiento      │
                        │ • Matriz de Confusión          │
                        │ • Información del Modelo       │
                        └────────────────────────────────┘
                                         │
                                         ▼
                                  http://localhost:8501
```

---

## 🔄 Flujo de Datos Detallado

### FASE 1: ENTRENAMIENTO

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 1: ENTRENAMIENTO                        │
└─────────────────────────────────────────────────────────────────┘

1. CARGA DE DATOS
   ┌──────────────────────────────────────────┐
   │ twitter_training.csv (1000 registros)    │
   │ twitter_validation.csv (500 registros)   │
   │ Total: 1500 tweets con sentimientos      │
   └──────────────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────┐
   │ Limpieza de Datos                        │
   │ • Eliminar duplicados                    │
   │ • Eliminar valores nulos                 │
   │ • Normalizar texto                       │
   └──────────────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────┐
   │ Análisis Exploratorio (EDA)              │
   │ • Distribución de sentimientos           │
   │ • Estadísticas de texto                  │
   │ • Palabras más frecuentes                │
   └──────────────────────────────────────────┘

2. FEATURE ENGINEERING
   ┌──────────────────────────────────────────┐
   │ TF-IDF Vectorization                     │
   │ • Convierte texto en números             │
   │ • 5000 features máximo                   │
   │ • Unigramas y bigramas                   │
   │ • Elimina stop words                     │
   └──────────────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────┐
   │ Matriz de Features (1500 x 5000)         │
   │ • Cada fila: un tweet                    │
   │ • Cada columna: un feature               │
   │ • Valores: TF-IDF scores                 │
   └──────────────────────────────────────────┘

3. DIVISIÓN DE DATOS
   ┌──────────────────────────────────────────┐
   │ Train/Test Split (80/20)                 │
   │ • Training: 1200 muestras                │
   │ • Testing: 300 muestras                  │
   │ • Stratificado por sentimiento           │
   └─────────────────────────────────────── ──┘

4. ENTRENAMIENTO DE MODELOS
   ┌──────────────────────────────────────────┐
   │ Modelo 1: Naive Bayes                    │
   │ Modelo 2: SVM (Linear)                   │
   │ Modelo 3: Random Forest                  │
   │ Modelo 4: Logistic Regression            │
   └──────────────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────┐
   │ Evaluación en Conjunto de Prueba         │
   │ • Accuracy                               │
   │ • Precision                              │
   │ • Recall                                 │
   │ • F1-Score                               │
   │ • Matriz de Confusión                    │
   └──────────────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────┐
   │ Selección del Mejor Modelo               │
   │ Criterio: F1-Score más alto              │
   │ Típicamente: SVM (F1 ≈ 0.82)             │
   └──────────────────────────────────────────┘
                    │
                    ▼
   ┌──────────────────────────────────────────┐
   │ Guardado de Modelos                      │
   │ • model_sentiment.pkl                    │
   │ • vectorizer_tfidf.pkl                   │
   │ • label_encoder.pkl                      │
   │ • model_info.pkl                         │
   └──────────────────────────────────────────┘
```

---

### FASE 2: PREDICCIÓN EN LOTE

```
┌─────────────────────────────────────────────────────────────────┐
│              FASE 2: PREDICCIÓN EN LOTE (CSV)                   │
└─────────────────────────────────────────────────────────────────┘

1. ENTRADA
   ┌──────────────────────────────────────────┐
   │ test_synthetic_tweets.csv                │
   │ • 1000 tweets sin etiquetar              │
   │ • Columnas: tweet_id, entity, content    │
   └──────────────────────────────────────────┘
                    │
                    ▼
2. CARGA DE MODELOS
   ┌──────────────────────────────────────────┐
   │ Cargar desde disco:                      │
   │ • model_sentiment.pkl                    │
   │ • vectorizer_tfidf.pkl                   │
   │ • label_encoder.pkl                      │
   └──────────────────────────────────────────┘
                    │
                    ▼
3. VECTORIZACIÓN
   ┌──────────────────────────────────────────┐
   │ Aplicar TF-IDF a nuevos textos           │
   │ • Usar mismo vectorizador                │
   │ • Generar matriz (1000 x 5000)           │
   └──────────────────────────────────────────┘
                    │
                    ▼
4. PREDICCIÓN
   ┌──────────────────────────────────────────┐
   │ Para cada tweet:                         │
   │ • Aplicar modelo ML                      │
   │ • Obtener predicción                     │
   │ • Calcular confianza                     │
   │ • Decodificar etiqueta                   │
   └──────────────────────────────────────────┘
                    │
                    ▼
5. SALIDA
   ┌──────────────────────────────────────────┐
   │ results/predictions.csv                  │
   │ Columnas:                                │
   │ • tweet_id                               │
   │ • entity                                 │
   │ • tweet_content                          │
   │ • prediction (Positive/Negative/Neutral) │
   │ • confidence (0.0 - 1.0)                 │
   │ • confidence_pct (0% - 100%)             │
   └──────────────────────────────────────────┘
```

---

### FASE 3: STREAMING EN TIEMPO REAL

```
┌─────────────────────────────────────────────────────────────────┐
│           FASE 3: STREAMING EN TIEMPO REAL (SPARK)              │
└─────────────────────────────────────────────────────────────────┘

1. FUENTE DE DATOS
   ┌──────────────────────────────────────────┐
   │ Kafka Topic: raw_tweets                  │
   │ • Tweets en tiempo real                  │
   │ • Formato: JSON                          │
   │ • Tasa: 5 tweets/segundo                 │
   └──────────────────────────────────────────┘
                    │
                    ▼
2. LECTURA CON SPARK
   ┌──────────────────────────────────────────┐
   │ spark.readStream                         │
   │ • Conectar a Kafka                       │
   │ • Leer en tiempo real                    │
   │ • Parsear JSON                           │
   └──────────────────────────────────────────┘
                    │
                    ▼
3. LIMPIEZA Y TRANSFORMACIÓN
   ┌──────────────────────────────────────────┐
   │ • Convertir a minúsculas                 │
   │ • Eliminar caracteres especiales         │
   │ • Normalizar texto                       │
   │ • Crear timestamp                        │
   └──────────────────────────────────────────┘
                    │
                    ▼
4. APLICAR MODELO ML
   ┌──────────────────────────────────────────┐
   │ UDF (User Defined Function)              │
   │ • Vectorizar con TF-IDF                  │
   │ • Aplicar modelo ML                      │
   │ • Obtener predicción y confianza         │
   │ • Comparar con sentimiento real          │
   └──────────────────────────────────────────┘
                    │
                    ▼
5. AGREGACIÓN EN VENTANAS
   ┌──────────────────────────────────────────┐
   │ Ventanas de 1 minuto:                    │
   │ • Contar predicciones por tipo           │
   │ • Calcular accuracy                      │
   │ • Calcular confianza promedio            │
   │ • Generar métricas                       │
   └──────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   HDFS Predictions        HDFS Metrics
   /predictions/           /metrics/
   • Predicciones          • Accuracy
   • Confianza             • Confianza
   • Comparación           • Conteos
```

---

### FASE 4: VISUALIZACIÓN EN DASHBOARD

```
┌───────────────────────────────────────────────���─────────────────┐
│              FASE 4: VISUALIZACIÓN (STREAMLIT)                  │
└─────────────────────────────────────────────────────────────────┘

1. LECTURA DE DATOS
   ┌──────────────────────────────────────────┐
   │ Fuentes:                                 │
   │ • Kafka (tweets en vivo)                 │
   │ • HDFS (predicciones guardadas)          │
   │ • Modelos (información)                  │
   └──────────────────────────────────────────┘
                    │
                    ▼
2. PROCESAMIENTO
   ┌──────────────────────────────────────────┐
   │ • Convertir a DataFrames                 │
   │ • Calcular estadísticas                  │
   │ • Preparar datos para gráficos           │
   │ • Generar métricas                       │
   └──────────────────────────────────────────┘
                    │
                    ▼
3. VISUALIZACIÓN
   ┌──────────────────────────────────────────┐
   │ Tab 1: Dashboard Principal               │
   │ • Distribución de predicciones           │
   │ • Confianza por predicción               │
   │ • Tabla de predicciones recientes        │
   │                                          │
   │ Tab 2: Predicciones en Vivo              │
   │ • Clasificar texto individual            │
   │ • Ver tweets en vivo desde Kafka         │
   │                                          │
   │ Tab 3: Métricas de Rendimiento           │
   │ • Accuracy, Precision, Recall, F1        │
   │ • Gráficos de comparación                │
   │                                          │
   │ Tab 4: Matriz de Confusión               │
   │ • Heatmap de predicciones                │
   │ • Análisis de errores                    │
   │                                          │
   │ Tab 5: Información del Modelo            │
   │ • Detalles técnicos                      │
   │ • Guía de uso                            │
   └──────────────────────────────────────────┘
                    │
                    ▼
4. SALIDA
   ┌──────────────────────────────────────────┐
   │ http://localhost:8501                    │
   │ Dashboard interactivo en navegador       │
   └──────────────────────────────────────────┘
```

---

## 🔌 Componentes Técnicos

### MODELOS DE MACHINE LEARNING

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODELOS DISPONIBLES                          │
└─────────────────────────────────────────────────────────────────┘

1. NAIVE BAYES (Multinomial)
   ├─ Algoritmo: Probabilístico
   ├─ Ventajas: Rápido, bueno para texto
   ├─ Desventajas: Menos preciso
   ├─ Tiempo: ~1 segundo
   └─ F1-Score típico: 0.78-0.80

2. SVM (Support Vector Machine)
   ├─ Algoritmo: Kernel Linear
   ├─ Ventajas: Muy preciso, robusto
   ├─ Desventajas: Más lento
   ├─ Tiempo: ~5 segundos
   └─ F1-Score típico: 0.82-0.84 ⭐ MEJOR

3. RANDOM FOREST
   ├─ Algoritmo: Ensemble de árboles
   ├─ Ventajas: Robusto, interpretable
   ├─ Desventajas: Requiere más memoria
   ├─ Tiempo: ~10 segundos
   └─ F1-Score típico: 0.80-0.82

4. LOGISTIC REGRESSION
   ├─ Algoritmo: Regresión lineal
   ├─ Ventajas: Rápido, interpretable
   ├─ Desventajas: Menos preciso
   ├─ Tiempo: ~2 segundos
   └─ F1-Score típico: 0.80-0.81
```

---

### PIPELINE DE FEATURES

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE FEATURES                         │
└─────────────────────────────────────────────────────────────────┘

ENTRADA: Texto sin procesar
   │
   ▼
LIMPIEZA
   • Convertir a minúsculas
   • Eliminar URLs
   • Eliminar menciones (@usuario)
   • Eliminar hashtags (#tema)
   • Eliminar caracteres especiales
   │
   ���
TOKENIZACIÓN
   • Dividir en palabras
   • Eliminar puntuación
   │
   ▼
ELIMINACIÓN DE STOP WORDS
   • Remover palabras comunes (the, a, is, etc.)
   • Mantener palabras significativas
   │
   ▼
TF-IDF VECTORIZATION
   • Calcular Term Frequency (TF)
   • Calcular Inverse Document Frequency (IDF)
   • Generar vector numérico (5000 dimensiones)
   │
   ▼
NORMALIZACIÓN
   • Escalar valores a [0, 1]
   • Aplicar L2 normalization
   │
   ▼
SALIDA: Vector numérico (5000 features)
```

---

## 📊 Flujo de Datos en Tiempo Real

```
┌─────────────────────────────────────────────────────────────────┐
│              FLUJO DE DATOS EN TIEMPO REAL                      │
└─────────────────────────────────────────────────────────────────┘

GENERADOR DE DATOS
   │ (5 tweets/segundo)
   ▼
KAFKA BROKER
   │ (Topic: raw_tweets)
   ▼
SPARK STREAMING
   ├─ Lectura desde Kafka
   ├─ Limpieza de datos
   ├─ Aplicar modelo ML
   ├─ Calcular métricas
   │
   ├─ Predicciones → HDFS (/predictions/)
   ├─ Métricas → HDFS (/metrics/)
   │
   └─ Logs → Consola
   │
   ▼
DASHBOARD (Streamlit)
   ├─ Lee desde HDFS
   ├─ Lee desde Kafka
   ├─ Visualiza en tiempo real
   │
   └─ http://localhost:8501
```

---

## 🔐 Seguridad y Confiabilidad

```
┌─────────────────────────────────────────────────────────────────┐
│              SEGURIDAD Y CONFIABILIDAD                          │
└─────────────────────────────────────────────────────────────────┘

VALIDACIÓN DE DATOS
   • Verificar formato JSON
   • Validar campos requeridos
   • Detectar valores nulos
   • Manejo de excepciones

MANEJO DE ERRORES
   • Try-catch en predicciones
   • Fallback a predicción por defecto
   • Logging de errores
   • Reintentos automáticos

MONITOREO
   • Tracking de accuracy
   • Alertas de baja confianza
   • Logs de predicciones
   • Métricas en tiempo real

PERSISTENCIA
   • Guardado en HDFS
   • Checkpoints en Spark
   • Backup de modelos
   • Versionado de modelos
```

---

## ���� Casos de Uso

```
┌─────────────────────────────────────────────────────────────────┐
│                    CASOS DE USO                                 │
└─────────────────────────────────────────────────────────────────┘

CASO 1: ANÁLISIS HISTÓRICO
   • Cargar tweets históricos
   • Aplicar modelo ML
   • Generar reportes
   • Identificar tendencias

CASO 2: MONITOREO EN TIEMPO REAL
   • Recibir tweets en vivo
   • Clasificar automáticamente
   • Alertar sobre sentimientos negativos
   • Actualizar dashboard

CASO 3: ANÁLISIS POR ENTIDAD
   • Filtrar tweets por empresa/tema
   • Calcular sentimiento promedio
   • Comparar con competencia
   • Generar insights

CASO 4: DETECCIÓN DE CRISIS
   • Monitorear picos de negatividad
   • Alertar a equipo de crisis
   • Generar reportes automáticos
   • Sugerir acciones
```

---

## 📈 Escalabilidad

```
┌──────────────────────────────────────────────────────���──────────┐
│                    ESCALABILIDAD                                │
└─────────────────────────────────────────────────────────────────┘

ACTUAL
   • 5 tweets/segundo
   • 1 modelo
   • 1 máquina

ESCALABLE A
   • 1000+ tweets/segundo
   • Múltiples modelos
   • Cluster de Spark
   • Múltiples particiones
   • Caché distribuido

OPTIMIZACIONES
   • Paralelización en Spark
   • Caché de modelos
   • Batch processing
   • Compresión de datos
   • Índices en HDFS
```

---

**Arquitectura diseñada para ser escalable, confiable y fácil de mantener.** 🚀
