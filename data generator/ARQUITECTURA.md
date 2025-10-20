# Arquitectura del Generador de Datos Sintéticos

## 1. Componentes del Sistema

### 1.1 DataExplorer

**Responsabilidad**: Analizar el dataset original y extraer distribuciones estadísticas.

**Métodos**:
- `get_sentiment_distribution()`: Retorna proporciones de sentimientos
- `get_entity_distribution()`: Retorna proporciones de entidades
- `get_text_length_stats()`: Retorna estadísticas de longitud
- `get_top_words(n)`: Retorna las n palabras más frecuentes
- `get_statistics_summary()`: Retorna resumen completo

**Entrada**: Rutas a archivos CSV (training y validation)
**Salida**: Diccionario con estadísticas

### 1.2 SyntheticDataGenerator

**Responsabilidad**: Generar tweets sintéticos respetando distribuciones estadísticas.

**Métodos**:
- `generate_tweet()`: Genera un tweet individual
- `generate_batch(n)`: Genera n tweets
- `generate_stream(n, delay)`: Genera stream con delay entre tweets

**Distribuciones Utilizadas**:
- Sentimientos: Distribución categórica con probabilidades reales
- Entidades: Distribución categórica con probabilidades reales
- Longitud: Distribución normal truncada (μ=107.63, σ=76.47)
- Palabras: Selección empírica de las 50 más frecuentes

**Entrada**: Estadísticas del dataset original
**Salida**: Lista de diccionarios con tweets

### 1.3 DataExporter

**Responsabilidad**: Exportar datos a diferentes formatos.

**Métodos**:
- `to_csv(tweets, path)`: Exporta a CSV
- `to_json(tweets, path)`: Exporta a JSON
- `to_kafka(tweets, topic, servers)`: Envía a Kafka

**Entrada**: Lista de tweets
**Salida**: Datos en formato especificado

## 2. Flujo de Datos

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset Original                         │
│              (twitter_training.csv +                        │
│              twitter_validation.csv)                        │
│                   12,447 tweets                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────���───────────────────────┐
│                   DataExplorer                              │
│  • Carga y limpia datos                                     │
│  • Extrae distribuciones                                    │
│  • Calcula estadísticas                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Estadísticas Extraídas                         │
│  • Distribución de sentimientos                             │
│  • Distribución de entidades                                │
│  • Estadísticas de longitud                                 │
│  • Palabras frecuentes                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            SyntheticDataGenerator                           │
│  • Selecciona sentimiento (distribución categórica)         │
│  • Selecciona entidad (distribución categórica)             │
│  • Genera longitud (distribución normal truncada)           │
│  • Genera contenido (palabras frecuentes + sentimiento)     │
│  • Genera 1,000 tweets                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Tweets Sintéticos                              │
│                  1,000 tweets                               │
│  • Tweet ID: 10001-11000                                    │
│  • Entity: Distribuida proporcionalmente                    │
│  • Sentiment: Distribuido proporcionalmente                 │
│  • Content: Coherente y realista                            │
│  • Timestamp: ISO format                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  DataExporter                               │
│  • Serializa a JSON                                         │
│  • Envía a Kafka                                            │
└────────────────────────┬──────────────────────────���─────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Kafka Topic                                │
│              (tweets - streaming)                           │
└─────────────────────────────────────────────────────────────┘
```

## 3. Distribuciones Estadísticas

### 3.1 Distribución Categórica (Sentimientos y Entidades)

```python
np.random.choice(categories, p=probabilities)
```

**Ventajas**:
- Respeta proporciones reales del dataset
- Garantiza representatividad
- Evita sesgos de random simple

**Ejemplo**:
```python
sentiments = ['Negative', 'Positive', 'Neutral', 'Irrelevant']
probabilities = [0.3018, 0.2789, 0.2453, 0.1739]
sentiment = np.random.choice(sentiments, p=probabilities)
```

### 3.2 Distribución Normal Truncada (Longitud)

```python
length = np.random.normal(mean=107.63, std=76.47)
length = max(1, min(352, length))
```

**Ventajas**:
- Genera longitudes realistas
- Mantiene media y desviación estándar
- Respeta límites válidos

### 3.3 Distribución Empírica (Palabras)

```python
word = np.random.choice(top_50_words)
```

**Ventajas**:
- Utiliza vocabulario real
- Mantiene coherencia semántica
- Genera tweets realistas

## 4. Integración con Kafka

### 4.1 Configuración

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
```

### 4.2 Envío de Datos

```python
for tweet in tweets:
    producer.send('tweets', value=tweet)
producer.flush()
producer.close()
```

### 4.3 Formato de Mensaje

```json
{
  "Tweet ID": 10001,
  "Entity": "Borderlands",
  "Sentiment": "Positive",
  "Tweet content": "love great awesome...",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

## 5. Características de Diseño

### 5.1 Escalabilidad

- Genera 1,000 tweets en < 1 segundo
- Puede escalar a millones de tweets
- Bajo consumo de memoria

### 5.2 Confiabilidad

- Valida distribuciones automáticamente
- Maneja errores de conexión a Kafka
- Logging detallado

### 5.3 Flexibilidad

- Soporta múltiples formatos (CSV, JSON, Kafka)
- Configurable (número de tweets, servidor Kafka, etc.)
- Extensible (agregar nuevas distribuciones)

## 6. Validación de Calidad

El sistema valida automáticamente:

1. **Distribuciones**: Compara con dataset original
2. **Coherencia**: Verifica que tweets sean válidos
3. **Integridad**: Valida que no hay valores nulos
4. **Unicidad**: Verifica que Tweet IDs son únicos

## 7. Dependencias

- `pandas`: Manipulación de datos
- `numpy`: Operaciones numéricas y distribuciones
- `kafka-python`: Integración con Kafka
- `json`: Serialización de datos
- `logging`: Registro de eventos
