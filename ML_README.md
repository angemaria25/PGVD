# 🤖 Machine Learning para Análisis de Sentimientos

## 📋 Descripción

Este módulo implementa Machine Learning para clasificación automática de sentimientos en tweets. Incluye entrenamiento de modelos, predicciones en lote, streaming en tiempo real e integración con Spark.

---

## 🎯 Características

✅ **Múltiples Modelos:** Naive Bayes, SVM, Random Forest, Logistic Regression  
✅ **Feature Engineering:** TF-IDF vectorization  
✅ **Evaluación Completa:** Accuracy, Precision, Recall, F1-Score  
✅ **Predicciones en Tiempo Real:** Integración con Spark Streaming  
✅ **Dashboard Interactivo:** Visualización con Streamlit  
✅ **Predicciones en Lote:** Procesamiento de archivos CSV  

---

## 📦 Instalación

### 1. Instalar Dependencias

```bash
pip install -r requirements_ml.txt
```

O instalar manualmente:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn plotly joblib nltk scipy
```

### 2. Verificar Instalación

```bash
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
```

---

## 🚀 Guía de Uso Paso a Paso

### PASO 1: Entrenar el Modelo (Una sola vez)

```bash
python ml_training.py
```

**Qué hace:**
- Carga datos de entrenamiento y validación
- Realiza análisis exploratorio
- Entrena 4 modelos diferentes
- Evalúa y selecciona el mejor
- Guarda modelos en `models/`

**Salida esperada:**
```
✓ Modelo guardado: models/model_sentiment.pkl
✓ Vectorizador guardado: models/vectorizer_tfidf.pkl
✓ Codificador guardado: models/label_encoder.pkl
✓ Información del modelo guardada: models/model_info.pkl
```

**Archivos generados:**
- `models/model_sentiment.pkl` - Modelo entrenado
- `models/vectorizer_tfidf.pkl` - Vectorizador TF-IDF
- `models/label_encoder.pkl` - Codificador de etiquetas
- `models/model_info.pkl` - Información del modelo
- `results/training_report.txt` - Reporte de entrenamiento
- `results/model_comparison.png` - Gráfico de comparación
- `results/confusion_matrix.png` - Matriz de confusión

---

### PASO 2: Hacer Predicciones

#### Opción A: Predicción Individual

```bash
python ml_prediction.py --text "Este es un tweet excelente"
```

**Salida:**
```
Predicción: Positive
Confianza: 92.45%
```

#### Opción B: Predicción en Lote (CSV)

```bash
python ml_prediction.py --input data/test_synthetic_tweets.csv --output results/predictions.csv --evaluate
```

**Parámetros:**
- `--input` - Archivo CSV de entrada
- `--output` - Archivo CSV de salida (opcional)
- `--evaluate` - Evaluar contra sentimientos reales (opcional)

**Salida:**
```
Cargados 1000 registros
Realizando predicciones...
✓ Resultados guardados: results/predictions.csv

=== MÉTRICAS DE EVALUACIÓN ===
Accuracy:  0.8234
Precision: 0.8156
Recall:    0.8234
F1-Score:  0.8190
Predicciones correctas: 823/1000
```

#### Opción C: Modo Interactivo

```bash
python ml_prediction.py
```

Ingresa textos para clasificar en tiempo real.

---

### PASO 3: Streaming en Tiempo Real

```bash
python spark_ml_streaming.py
```

**Qué hace:**
- Lee tweets desde Kafka
- Aplica modelo ML para predecir sentimiento
- Compara predicción con sentimiento real
- Calcula métricas en tiempo real
- Guarda resultados en HDFS

**Requisitos:**
- Kafka corriendo
- Spark corriendo
- HDFS disponible
- Generador de datos enviando tweets

---

### PASO 4: Ver Dashboard

```bash
streamlit run app-dashboard/streamlit_app_ml.py
```

Abre en navegador: `http://localhost:8501`

**Características del Dashboard:**
- 📊 Dashboard Principal - Distribución de predicciones
- 🔮 Predicciones en Vivo - Clasificar textos individuales
- 📈 Métricas de Rendimiento - Accuracy, Precision, Recall, F1
- 🎯 Matriz de Confusión - Análisis de errores
- ℹ️ Información del Modelo - Detalles técnicos

---

## 📊 Estructura de Archivos

```
PGVD/
├── ml_training.py              # Entrenamiento de modelos
├── ml_prediction.py            # Predicciones
├── spark_ml_streaming.py       # Streaming con Spark
├── requirements_ml.txt         # Dependencias
├── ML_IMPLEMENTATION_GUIDE.md  # Guía detallada
├── ML_README.md               # Este archivo
├── models/                     # Modelos entrenados
│   ├── model_sentiment.pkl
│   ├── vectorizer_tfidf.pkl
│   ├── label_encoder.pkl
│   └── model_info.pkl
├── results/                    # Resultados
│   ├── training_report.txt
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── predictions.csv
└── app-dashboard/
    └── streamlit_app_ml.py     # Dashboard
```

---

## 🔧 Configuración Avanzada

### Ajustar Hiperparámetros

Edita `ml_training.py`:

```python
# Vectorizador TF-IDF
self.vectorizer = TfidfVectorizer(
    max_features=5000,      # Aumentar para más features
    min_df=2,               # Mínimo de documentos
    max_df=0.8,             # Máximo de documentos
    ngram_range=(1, 2),     # Unigramas y bigramas
    lowercase=True,
    stop_words='english'
)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,       # Número de árboles
    random_state=42,
    n_jobs=-1
)
```

### Agregar Nuevos Modelos

```python
# En ml_training.py, método train_models()

# Ejemplo: Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(self.X_train, self.y_train)
self.models['Gradient Boosting'] = gb_model
```

---

## 📈 Interpretación de Métricas

### Accuracy (Precisión General)
- Porcentaje de predicciones correctas
- **Bueno:** > 0.75
- **Excelente:** > 0.85

### Precision (Precisión por Clase)
- De las predicciones positivas, cuántas son correctas
- **Fórmula:** TP / (TP + FP)

### Recall (Cobertura)
- De los casos reales positivos, cuántos se detectaron
- **Fórmula:** TP / (TP + FN)

### F1-Score (Media Armónica)
- Balance entre Precision y Recall
- **Fórmula:** 2 * (Precision * Recall) / (Precision + Recall)

### Matriz de Confusión
```
                Predicho
              Pos  Neg  Neu
Real Pos      TP   FN   FN
     Neg      FP   TN   FP
     Neu      FP   FP   TN
```

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'sklearn'"

```bash
pip install scikit-learn
```

### Error: "Modelo no encontrado"

Ejecuta primero:
```bash
python ml_training.py
```

### Error: "Baja precisión del modelo"

**Soluciones:**
1. Aumentar datos de entrenamiento
2. Ajustar hiperparámetros
3. Usar feature engineering más avanzado
4. Probar con diferentes modelos
5. Limpiar datos de entrada

### Error: "Kafka connection refused"

Verifica que Kafka esté corriendo:
```bash
docker-compose ps
```

### Error: "HDFS connection refused"

Verifica que HDFS esté disponible:
```bash
hdfs dfs -ls /
```

---

## 📚 Ejemplos de Uso

### Ejemplo 1: Predicción Simple

```python
from ml_prediction import SentimentPredictor

predictor = SentimentPredictor()
result = predictor.predict_single("I love this product!")
print(f"Predicción: {result['prediction']}")
print(f"Confianza: {result['confidence_pct']}")
```

### Ejemplo 2: Predicción en Lote

```python
from ml_prediction import SentimentPredictor

predictor = SentimentPredictor()
df = predictor.predict_from_csv(
    'data/tweets.csv',
    'results/predictions.csv'
)
print(df.head())
```

### Ejemplo 3: Evaluación

```python
from ml_prediction import SentimentPredictor

predictor = SentimentPredictor()
df = predictor.predict_from_csv('data/tweets.csv')
metrics = predictor.evaluate_predictions(df, 'sentiment')
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

## 🎓 Conceptos Clave

### TF-IDF (Term Frequency-Inverse Document Frequency)
- Convierte texto en números
- Pondera palabras por importancia
- Reduce dimensionalidad

### Validación Cruzada
- Divide datos en múltiples folds
- Entrena y evalúa múltiples veces
- Reduce overfitting

### Matriz de Confusión
- Muestra errores de clasificación
- Identifica clases problemáticas
- Ayuda a mejorar el modelo

---

## 🚀 Próximos Pasos

1. **Mejorar Datos:** Recolectar más tweets de entrenamiento
2. **Feature Engineering:** Agregar features como emojis, URLs, menciones
3. **Modelos Avanzados:** Probar BERT, GPT, transformers
4. **Ensemble:** Combinar múltiples modelos
5. **Monitoreo:** Rastrear rendimiento en producción

---

## 📞 Soporte

Para problemas o preguntas:
1. Revisa `ML_IMPLEMENTATION_GUIDE.md`
2. Consulta logs en `results/training_report.txt`
3. Verifica que todos los archivos de datos existan

---

## 📄 Licencia

Este proyecto es parte del curso de Procesamiento de Grandes Volúmenes de Datos (PGVD).

---

**Última actualización:** 2024  
**Versión:** 1.0
