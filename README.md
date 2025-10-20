# Generador de Datos Sintéticos - PGVD

## Descripción

Generador de datos sintéticos que respeta distribuciones estadísticas del dataset original de Twitter. Genera 1,000 tweets para simular streaming en tiempo real mediante Kafka.

## Instalación

```bash
pip install pandas numpy kafka-python
```

## Uso

```bash
python data_generator.py
```

## Uso en Código

```python
from data_generator import DataExplorer, SyntheticDataGenerator, DataExporter

explorer = DataExplorer("data/twitter_training.csv", "data/twitter_validation.csv")
stats = explorer.get_statistics_summary()

generator = SyntheticDataGenerator(stats)
tweets = generator.generate_batch(n=500)

exporter = DataExporter()
exporter.to_kafka(tweets, topic='tweets', bootstrap_servers='localhost:9092')
```

## Resultados de Validación

### Sentimientos

| Sentimiento | Original | Sintético | Diferencia |
|------------|----------|-----------|-----------|
| Negative | 30.18% | 29.60% | 0.58% |
| Positive | 27.89% | 30.00% | 2.11% |
| Neutral | 24.53% | 24.00% | 0.53% |
| Irrelevant | 17.39% | 16.40% | 0.99% |

### Longitud de Texto

| Métrica | Original | Sintético | Diferencia |
|--------|----------|-----------|-----------|
| Media | 107.63 | 108.28 | 0.60% |
| Std Dev | 76.47 | 71.83 | 6.07% |

### Integridad

- Tweets generados: 1,000
- Tweets únicos: 1,000
- Valores nulos: 0

## Estructura de Datos

```json
{
  "Tweet ID": 10001,
  "Entity": "Borderlands",
  "Sentiment": "Positive",
  "Tweet content": "love great awesome...",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

## Distribuciones Utilizadas

1. **Categórica** (Sentimientos y Entidades): `np.random.choice()` con probabilidades reales
2. **Normal Truncada** (Longitud): μ=107.63, σ=76.47, rango 1-352
3. **Empírica** (Palabras): Top 50 palabras más frecuentes

## Archivos

- `data_generator.py` - Generador principal
- `test_validation.py` - Script de validación
- `data/synthetic_tweets.csv` - 1,000 tweets
- `data/synthetic_tweets.json` - 1,000 tweets (JSON)

## Dependencias

- pandas >= 1.0
- numpy >= 1.18
- kafka-python >= 2.0
