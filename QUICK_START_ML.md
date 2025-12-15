# ⚡ Quick Start - Machine Learning para Análisis de Sentimientos

## 🎯 Resumen Ejecutivo

Has recibido un **sistema completo de Machine Learning** para clasificación automática de sentimientos en tweets. Este documento te guía en 5 minutos para empezar.

---

## 📦 Qué Incluye

| Archivo | Descripción |
|---------|-------------|
| `ml_training.py` | Entrena 4 modelos diferentes y selecciona el mejor |
| `ml_prediction.py` | Realiza predicciones individuales o en lote |
| `spark_ml_streaming.py` | Integración con Spark para predicciones en tiempo real |
| `app-dashboard/streamlit_app_ml.py` | Dashboard interactivo con visualizaciones |
| `ML_IMPLEMENTATION_GUIDE.md` | Guía detallada (70+ páginas) |
| `ML_README.md` | Documentación completa |
| `verify_ml_setup.py` | Script para verificar que todo está configurado |

---

## 🚀 Inicio Rápido (5 minutos)

### Paso 1: Verificar Setup (1 minuto)

```bash
python verify_ml_setup.py
```

Esto verifica que todas las dependencias estén instaladas.

### Paso 2: Instalar Dependencias (2 minutos)

```bash
pip install -r requirements_ml.txt
```

### Paso 3: Entrenar Modelos (2 minutos)

```bash
python ml_training.py
```

**Salida esperada:**
```
✓ Modelo guardado: models/model_sentiment.pkl
✓ Vectorizador guardado: models/vectorizer_tfidf.pkl
✓ Codificador guardado: models/label_encoder.pkl
✓ MEJOR MODELO: SVM
  F1-Score: 0.8234
```

---

## 💡 Casos de Uso

### Caso 1: Clasificar un Tweet Individual

```bash
python ml_prediction.py --text "I love this product!"
```

**Salida:**
```
Predicción: Positive
Confianza: 92.45%
```

### Caso 2: Clasificar 1000 Tweets desde CSV

```bash
python ml_prediction.py --input data/test_synthetic_tweets.csv --output results/predictions.csv --evaluate
```

**Salida:**
```
Cargados 1000 registros
Realizando predicciones...
✓ Resultados guardados: results/predictions.csv

Accuracy:  0.8234
Precision: 0.8156
Recall:    0.8234
F1-Score:  0.8190
```

### Caso 3: Predicciones en Tiempo Real (Streaming)

```bash
# Terminal 1: Iniciar generador de datos
python app-scripts-producer/data_generator.py

# Terminal 2: Iniciar Spark ML Streaming
python spark_ml_streaming.py

# Terminal 3: Ver Dashboard
streamlit run app-dashboard/streamlit_app_ml.py
```

---

## 📊 Modelos Disponibles

El sistema entrena y compara automáticamente:

| Modelo | Ventajas | Desventajas |
|--------|----------|-------------|
| **Naive Bayes** | Rápido, bueno para texto | Menos preciso |
| **SVM** | Muy preciso, robusto | Más lento |
| **Random Forest** | Robusto, interpretable | Requiere más memoria |
| **Logistic Regression** | Rápido, interpretable | Menos preciso |

**El sistema selecciona automáticamente el mejor modelo basado en F1-Score.**

---

## 📈 Métricas de Rendimiento

Después del entrenamiento, verás:

```
=== RESULTADOS DE MODELOS ===

Naive Bayes:
  Accuracy:  0.7856
  Precision: 0.7834
  Recall:    0.7856
  F1-Score:  0.7845

SVM:
  Accuracy:  0.8234  ← MEJOR
  Precision: 0.8156
  Recall:    0.8234
  F1-Score:  0.8190  ← MEJOR

Random Forest:
  Accuracy:  0.8012
  Precision: 0.7945
  Recall:    0.8012
  F1-Score:  0.7978

Logistic Regression:
  Accuracy:  0.8089
  Precision: 0.8023
  Recall:    0.8089
  F1-Score:  0.8056
```

---

## 🎨 Dashboard Interactivo

Abre el dashboard con:

```bash
streamlit run app-dashboard/streamlit_app_ml.py
```

**Características:**
- 📊 Dashboard Principal - Distribución de predicciones
- 🔮 Predicciones en Vivo - Clasifica textos individuales
- 📈 Métricas - Accuracy, Precision, Recall, F1-Score
- 🎯 Matriz de Confusión - Análisis de errores
- ℹ️ Información - Detalles técnicos del modelo

---

## 🔧 Configuración Personalizada

### Cambiar Modelo

En `ml_training.py`, comenta los modelos que no quieras:

```python
# Comentar para no entrenar
# nb_model = MultinomialNB()
# nb_model.fit(self.X_train, self.y_train)
# self.models['Naive Bayes'] = nb_model
```

### Ajustar Hiperparámetros

En `ml_training.py`:

```python
# Aumentar features
self.vectorizer = TfidfVectorizer(
    max_features=10000,  # Aumentado de 5000
    ngram_range=(1, 3)   # Ahora incluye trigramas
)

# Más árboles en Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,    # Aumentado de 100
    max_depth=20         # Nuevo parámetro
)
```

---

## 📁 Estructura de Archivos Generados

Después de ejecutar `ml_training.py`:

```
models/
├── model_sentiment.pkl          # Modelo entrenado
├── vectorizer_tfidf.pkl         # Vectorizador
├── label_encoder.pkl            # Codificador
└── model_info.pkl               # Información

results/
├── training_report.txt          # Reporte de texto
├── model_comparison.png         # Gráfico de comparación
├── confusion_matrix.png         # Matriz de confusión
└── predictions.csv              # Predicciones (si se ejecutó)
```

---

## 🐛 Solución Rápida de Problemas

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| `Modelo no encontrado` | Ejecuta `python ml_training.py` |
| `Baja precisión` | Aumenta datos de entrenamiento |
| `Kafka connection refused` | Verifica que Kafka esté corriendo |
| `HDFS connection refused` | Verifica que HDFS esté disponible |

---

## 📚 Documentación Completa

Para más detalles, consulta:

1. **ML_IMPLEMENTATION_GUIDE.md** - Guía paso a paso completa
2. **ML_README.md** - Documentación técnica detallada
3. **Código comentado** - Cada archivo tiene comentarios explicativos

---

## 🎓 Conceptos Clave (30 segundos)

- **TF-IDF:** Convierte texto en números ponderados
- **Accuracy:** Porcentaje de predicciones correctas
- **F1-Score:** Balance entre precisión y cobertura
- **Matriz de Confusión:** Muestra dónde falla el modelo
- **Validación Cruzada:** Prueba el modelo múltiples veces

---

## ✅ Checklist de Implementación

- [ ] Ejecutar `verify_ml_setup.py`
- [ ] Instalar dependencias: `pip install -r requirements_ml.txt`
- [ ] Entrenar modelos: `python ml_training.py`
- [ ] Hacer predicción de prueba: `python ml_prediction.py --text "test"`
- [ ] Ver resultados en `results/`
- [ ] Abrir dashboard: `streamlit run app-dashboard/streamlit_app_ml.py`
- [ ] Leer `ML_README.md` para casos avanzados

---

## 🚀 Próximos Pasos

1. **Entrenamiento:** `python ml_training.py`
2. **Predicciones:** `python ml_prediction.py --input data/test_synthetic_tweets.csv`
3. **Streaming:** `python spark_ml_streaming.py`
4. **Dashboard:** `streamlit run app-dashboard/streamlit_app_ml.py`

---

## 📞 Ayuda Rápida

```bash
# Ver ayuda de predicción
python ml_prediction.py --help

# Modo interactivo
python ml_prediction.py

# Verificar setup
python verify_ml_setup.py

# Ver logs de entrenamiento
cat results/training_report.txt
```

---

## 🎯 Objetivos Alcanzados

✅ Modelos de ML entrenados y evaluados  
✅ Predicciones individuales y en lote  
✅ Integración con Spark Streaming  
✅ Dashboard interactivo  
✅ Documentación completa  
✅ Scripts de verificación  

---

**¡Listo para empezar! Ejecuta `python ml_training.py` ahora mismo.** 🚀

---

*Última actualización: 2024*  
*Versión: 1.0*
