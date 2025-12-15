# 🤖 Guía de Implementación de Machine Learning para Análisis de Sentimientos

## 📋 Descripción General

Este documento proporciona un plan paso a paso para implementar Machine Learning en tu proyecto de análisis de sentimientos de tweets. El objetivo es crear modelos predictivos que clasifiquen automáticamente el sentimiento de nuevos tweets.

---

## 🎯 Objetivos

1. **Entrenar modelos de ML** usando los datos históricos (twitter_training.csv)
2. **Validar modelos** con datos de validación (twitter_validation.csv)
3. **Hacer predicciones** en tiempo real sobre nuevos tweets
4. **Integrar predicciones** en el pipeline de Spark Streaming
5. **Visualizar resultados** en el dashboard

---

## 📊 Arquitectura de la Solución

```
┌─────────────────────────────────────────────────────────────┐
│                    DATOS HISTÓRICOS                         │
│         (twitter_training.csv + twitter_validation.csv)     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   1. ml_training.py                │
        │   - Exploración de datos           │
        │   - Feature Engineering            │
        │   - Entrenamiento de modelos       │
        │   - Validación y evaluación        │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   2. Modelos Guardados             │
        │   - model_sentiment.pkl            │
        │   - vectorizer_tfidf.pkl           │
        │   - label_encoder.pkl              │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │   3. ml_prediction.py              │
        │   - Cargar modelos                 │
        │   - Hacer predicciones             │
        │   - Calcular confianza             │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   4. spark_ml_streaming.py         │
        │   - Integración con Spark          │
        │   - Predicciones en tiempo real    │
        │   - Guardado en HDFS               │
        └────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │   5. Dashboard Actualizado         │
        │   - Predicciones vs Reales         │
        │   - Confianza del modelo           │
        │   - Métricas de rendimiento        │
        └────────────────────��───────────────┘
```

---

## 🔧 Paso a Paso de Implementación

### **PASO 1: Exploración y Preparación de Datos**

**Archivo:** `ml_training.py`

**Qué hace:**
- Carga los datos de entrenamiento y validación
- Realiza análisis exploratorio (EDA)
- Limpia y prepara los datos
- Crea features (TF-IDF)
- Entrena múltiples modelos (Naive Bayes, SVM, Random Forest, Logistic Regression)
- Evalúa y compara modelos
- Guarda el mejor modelo

**Modelos a entrenar:**
1. **Naive Bayes** - Rápido, bueno para texto
2. **SVM (Support Vector Machine)** - Muy preciso
3. **Random Forest** - Robusto
4. **Logistic Regression** - Interpretable

**Métricas de evaluación:**
- Accuracy (Precisión general)
- Precision (Verdaderos positivos / Predichos positivos)
- Recall (Verdaderos positivos / Reales positivos)
- F1-Score (Media armónica de Precision y Recall)
- Matriz de confusión

---

### **PASO 2: Predicción en Lote**

**Archivo:** `ml_prediction.py`

**Qué hace:**
- Carga el modelo entrenado
- Realiza predicciones sobre nuevos tweets
- Calcula la confianza de cada predicción
- Exporta resultados

---

### **PASO 3: Integración con Spark Streaming**

**Archivo:** `spark_ml_streaming.py`

**Qué hace:**
- Lee tweets desde Kafka
- Aplica el modelo ML para predecir sentimiento
- Compara predicción con sentimiento real (si está disponible)
- Calcula métricas de rendimiento en tiempo real
- Guarda resultados en HDFS

---

### **PASO 4: Actualización del Dashboard**

**Archivo:** `app-dashboard/streamlit_app_ml.py`

**Qué hace:**
- Visualiza predicciones vs sentimientos reales
- Muestra confianza del modelo
- Gráficos de rendimiento
- Matriz de confusión
- Evolución temporal de precisión

---

## 📦 Dependencias Requeridas

```bash
# Agregar a requirements.txt
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
joblib==1.3.1
nltk==3.8.1
```

---

## 🚀 Instrucciones de Ejecución

### **1. Entrenar el Modelo (Una sola vez)**

```bash
cd d:\Documentos\3RO\1ER Semestre\PGVD\PGVD
python ml_training.py
```

**Salida esperada:**
- `models/model_sentiment.pkl` - Modelo entrenado
- `models/vectorizer_tfidf.pkl` - Vectorizador TF-IDF
- `models/label_encoder.pkl` - Codificador de etiquetas
- Reporte de evaluación en consola

---

### **2. Hacer Predicciones en Lote**

```bash
python ml_prediction.py --input data/test_synthetic_tweets.csv --output results/predictions.csv
```

---

### **3. Streaming con ML (Tiempo Real)**

```bash
python spark_ml_streaming.py
```

---

### **4. Ver Dashboard con Predicciones**

```bash
streamlit run app-dashboard/streamlit_app_ml.py
```

---

## 📈 Métricas de Éxito

- **Accuracy > 75%** - Precisión general del modelo
- **F1-Score > 0.70** - Balance entre precision y recall
- **Latencia < 100ms** - Tiempo de predicción por tweet
- **Throughput > 1000 tweets/segundo** - Capacidad de procesamiento

---

## 🔍 Interpretación de Resultados

### **Matriz de Confusión**
```
                Predicho
              Pos  Neg  Neu
Real Pos      TP   FN   FN
     Neg      FP   TN   FP
     Neu      FP   FP   TN
```

### **Métricas Clave**
- **TP (True Positive):** Predicción correcta positiva
- **TN (True Negative):** Predicción correcta negativa
- **FP (False Positive):** Predicción incorrecta positiva
- **FN (False Negative):** Predicción incorrecta negativa

---

## 🛠️ Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'sklearn'"
**Solución:** Instalar scikit-learn
```bash
pip install scikit-learn
```

### Problema: "Modelo no encontrado"
**Solución:** Ejecutar primero `ml_training.py` para entrenar el modelo

### Problema: "Baja precisión del modelo"
**Solución:** 
- Aumentar datos de entrenamiento
- Ajustar hiperparámetros
- Usar feature engineering más avanzado
- Probar con diferentes modelos

---

## 📚 Recursos Adicionales

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Sentiment Analysis Best Practices](https://huggingface.co/tasks/text-classification)

---

## ✅ Checklist de Implementación

- [ ] Instalar dependencias
- [ ] Ejecutar `ml_training.py`
- [ ] Verificar modelos guardados en `models/`
- [ ] Ejecutar `ml_prediction.py` con datos de prueba
- [ ] Ejecutar `spark_ml_streaming.py`
- [ ] Abrir dashboard con predicciones
- [ ] Validar métricas de rendimiento
- [ ] Documentar resultados

---

**Autor:** Sistema de Análisis de Sentimientos  
**Fecha:** 2024  
**Versión:** 1.0
