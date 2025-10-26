"""
Data Generator - Generador de tweets sintéticos basado en distribuciones estadísticas reales
Respeta las características estadísticas del dataset original (twitter_training.csv y twitter_validation.csv)
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
import json
import time
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple
import logging
import os # Importar os para variables de entorno
from kafka import KafkaProducer
from scipy.stats import truncnorm
# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de Kafka desde variables de entorno
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
TARGET_RATE_PER_SECOND = float(os.getenv("TARGET_RATE_PER_SECOND", "5.0")) # Para controlar la velocidad
DATA_DIR = os.getenv("DATA_DIR", "./data") # Directorio para el CSV de entrenamiento/validación


class DataExplorer:
    """Analiza las distribuciones estadísticas del dataset original"""
    
    def __init__(self, train_path: str, valid_path: str):
        """
        Inicializa el explorador de datos
        
        Args:
            train_path: Ruta al archivo de entrenamiento
            valid_path: Ruta al archivo de validación
        """
        def _safe_read(path, fallback_header=None):
            if not os.path.exists(path):
                logger.warning(f"Archivo no encontrado: {path}. No se cargará.")
                return pd.DataFrame() # Retornar DataFrame vacío si el archivo no existe
            try:
                # Intentar leer con header=True
                df = pd.read_csv(path, sep=',', encoding='utf-8', na_filter=False, dtype=str)
                # Si las columnas son numéricas (0, 1, 2...) y la primera fila es de datos, puede que no haya header
                if all(isinstance(c, (int, np.integer)) for c in df.columns) and df.shape[0] > 0 and \
                   any(pd.isna(df.iloc[0]).sum() > len(df.columns) / 2 for _ in [0]): # heurística para detectar posible header faltante
                    logger.warning(f"CSV {path} parece no tener encabezado, releyendo.")
                    df = pd.read_csv(path, header=None, sep=',', encoding='utf-8', na_filter=False, dtype=str)
                    df.columns = [f'col_{i}' for i in range(len(df.columns))] # Renombrar columnas a genéricas
                return df
            except Exception as e:
                logger.error(f"Error al leer {path}: {e}. Intentando sin header.")
                return pd.read_csv(path, header=None, sep=',', encoding='utf-8', na_filter=False, dtype=str)


        # --- Leer ambos CSV como strings para evitar inferencias raras ---
        self.train = _safe_read(train_path)
        self.valid = _safe_read(valid_path)
        
        # Combinar datasets (si alguno está vacío, concat lo manejará)
        self.df = pd.concat([self.train, self.valid], ignore_index=True)
        if self.df.empty:
            raise ValueError(f"No se pudieron cargar datos desde {train_path} o {valid_path}. Asegúrate de que los archivos existen y tienen contenido.")

        logger.info(f"Leídas {len(self.train)} filas (train) + {len(self.valid)} filas (valid) = {len(self.df)} filas totales (sin limpiar).")
        
        # Estándar esperado
        expected_cols = ['Tweet ID', 'Entity', 'Sentiment', 'Tweet content']
        
        # Heurística para renombrar columnas basado en patrones de nombres o contenido
        # Esto es más robusto si los headers son inconsistentes o no existen
        current_cols = [str(c).strip() for c in self.df.columns]
        self.df.columns = current_cols # Aplicar strip a columnas antes de inferencia

        detected_mapping = {}
        # Priorizar por nombres si son claros
        for current_col in current_cols:
            lower_current_col = current_col.lower()
            if 'id' in lower_current_col and 'tweet' in lower_current_col or lower_current_col == 'tweet id':
                if 'Tweet ID' not in detected_mapping.values(): detected_mapping['Tweet ID'] = current_col
            elif 'entity' in lower_current_col or lower_current_col == 'entity':
                if 'Entity' not in detected_mapping.values(): detected_mapping['Entity'] = current_col
            elif 'sentiment' in lower_current_col or lower_current_col == 'sentiment':
                if 'Sentiment' not in detected_mapping.values(): detected_mapping['Sentiment'] = current_col
            elif ('content' in lower_current_col and 'tweet' in lower_current_col) or lower_current_col == 'tweet content':
                if 'Tweet content' not in detected_mapping.values(): detected_mapping['Tweet content'] = current_col
            # Caso genérico para "col_0", "col_1", etc. si no hay header
            elif lower_current_col == 'col_0' or lower_current_col == '0': # asumiendo que el ID es la primera columna
                if 'Tweet ID' not in detected_mapping.values(): detected_mapping['Tweet ID'] = current_col
            elif lower_current_col == 'col_1' or lower_current_col == '1': # asumiendo que la entidad es la segunda
                if 'Entity' not in detected_mapping.values(): detected_mapping['Entity'] = current_col
            elif lower_current_col == 'col_2' or lower_current_col == '2': # asumiendo que el sentimiento es la tercera
                if 'Sentiment' not in detected_mapping.values(): detected_mapping['Sentiment'] = current_col
            elif lower_current_col == 'col_3' or lower_current_col == '3': # asumiendo que el contenido es la cuarta
                if 'Tweet content' not in detected_mapping.values(): detected_mapping['Tweet content'] = current_col

        # Invertir el mapeo para aplicar a df.rename
        rename_dict = {v: k for k, v in detected_mapping.items()}
        self.df = self.df.rename(columns=rename_dict)
        logger.info(f"Columnas después de renombrado inicial: {list(self.df.columns)}")

        # Verificación final y manejo de errores si aún faltan columnas cruciales
        for col_name in expected_cols:
            if col_name not in self.df.columns:
                raise ValueError(
                    f"Columna '{col_name}' no encontrada en el dataset después de la inferencia. "
                    f"Asegúrate de que el CSV tiene las columnas esperadas (Tweet ID, Entity, Sentiment, Tweet content) "
                    f"o que el formato permite su detección. Columnas actuales: {list(self.df.columns)}"
                )
        
        # Eliminar duplicados por Tweet ID
        self.df = self.df.drop_duplicates(subset=["Tweet ID"], keep="first").reset_index(drop=True)
        
        # Asegurarse de que las columnas clave sean de tipo string
        self.df['Sentiment'] = self.df['Sentiment'].astype(str)
        self.df['Entity'] = self.df['Entity'].astype(str)
        self.df['Tweet content'] = self.df['Tweet content'].astype(str)

        logger.info(f"Dataset cargado y limpiado: {len(self.df)} tweets únicos")
        logger.info(f"Columnas finales en DataExplorer: {list(self.df.columns)}")
    
    def get_sentiment_distribution(self) -> Dict[str, float]:
        """Obtiene la distribución de sentimientos"""
        return self.df['Sentiment'].value_counts(normalize=True).to_dict()
    
    def get_entity_distribution(self) -> Dict[str, float]:
        """Obtiene la distribución de entidades"""
        return self.df['Entity'].value_counts(normalize=True).to_dict()
    
    def get_text_length_stats(self) -> Dict[str, float]:
        """Obtiene estadísticas de longitud de texto"""
        text_lengths = self.df['Tweet content'].astype(str).apply(len)
        return {
            'mean': float(text_lengths.mean()),
            'std': float(text_lengths.std()),
            'min': float(text_lengths.min()),
            'max': float(text_lengths.max()),
            'median': float(text_lengths.median()),
            'q25': float(text_lengths.quantile(0.25)),
            'q75': float(text_lengths.quantile(0.75))
        }
    
    def get_top_words(self, n: int = 50) -> List[Tuple[str, int]]:
        """Obtiene las palabras más frecuentes"""
        def tokenize(text):
            text = re.sub(r"http\S+|@\S+|#\S+", "", str(text).lower())
            words = re.findall(r'\b[a-záéíóúñü]+', text)
            return words
        
        all_words = [w for text in self.df['Tweet content'] for w in tokenize(text)]
        counter = Counter(all_words)
        return counter.most_common(n)
    
    def get_statistics_summary(self) -> Dict:
        """Obtiene un resumen completo de estadísticas"""
        return {
            'total_tweets': len(self.df),
            'sentiment_distribution': self.get_sentiment_distribution(),
            'entity_distribution': self.get_entity_distribution(),
            'text_length_stats': self.get_text_length_stats(),
            'top_words': self.get_top_words(30)
        }



class SyntheticDataGenerator:
    """Genera tweets sintéticos basados en distribuciones estadísticas reales"""

    def __init__(self, stats: Dict):
        """
        Inicializa el generador de datos sintéticos.

        Args:
            stats: Diccionario de estadísticas calculadas por DataExplorer.get_statistics_summary()
        """
        self.stats = stats

        # Distribuciones
        self.sentiment_dist = stats['sentiment_distribution']
        self.entity_dist = stats['entity_distribution']
        self.text_length_stats = stats['text_length_stats']

        # Top palabras (si no hay, usar alguna por defecto)
        self.top_words = [word for word, _ in stats.get('top_words', [])] or ["tweet", "data", "content"]

        # Palabras por sentimiento
        self.positive_words = ['love', 'great', 'awesome', 'excellent', 'amazing', 'fantastic',
                               'wonderful', 'perfect', 'best', 'good', 'nice', 'beautiful', 'happy', 'enjoy']
        self.negative_words = ['hate', 'bad', 'terrible', 'awful', 'horrible', 'worst',
                               'poor', 'disappointing', 'useless', 'broken', 'sucks', 'sad', 'frustrating']
        self.neutral_words = ['is', 'the', 'a', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'from']

        # Contador para IDs nuevos
        self.tweet_id_counter = 10000

        logger.info("Generador de datos sintéticos inicializado correctamente usando estadísticas del dataset")

    def _generate_text_length(self) -> int:
        """Genera una longitud de texto usando distribución normal truncada"""
        mean = self.text_length_stats['mean']
        std = self.text_length_stats['std']
        min_len = int(self.text_length_stats['min'])
        max_len = int(self.text_length_stats['max'])
        a, b = (min_len - mean)/std, (max_len - mean)/std
        return int(truncnorm.rvs(a, b, loc=mean, scale=std))

    def _generate_tweet_content(self, sentiment: str, length: int) -> str:
        """Genera contenido de tweet coherente con el sentimiento"""
        words = []

        # Seleccionar palabras según sentimiento
        if sentiment == 'Positive':
            sentiment_words = self.positive_words
        elif sentiment == 'Negative':
            sentiment_words = self.negative_words
        else:  # Neutral e Irrelevant
            sentiment_words = self.neutral_words

        # Todas las palabras disponibles
        all_available_words = list(set(self.top_words + sentiment_words + self.neutral_words))
        if not all_available_words:
            all_available_words = ["default", "tweet", "content"]

        # Construir tweet mezclando palabras frecuentes y de sentimiento
        while len(' '.join(words)) < length:
            if random.random() < 0.3 and sentiment_words:
                words.append(random.choice(sentiment_words))
            else:
                words.append(random.choice(all_available_words))

            if len(words) > 100:  # límite para evitar bucles infinitos
                break

        tweet = ' '.join(words)[:length].strip()
        return tweet + "."

    def _select_sentiment(self) -> str:
        """Selecciona un sentimiento según la distribución real"""
        sentiments = list(self.sentiment_dist.keys())
        probabilities = list(self.sentiment_dist.values())
        return np.random.choice(sentiments, p=probabilities)

    def _select_entity(self) -> str:
        """Selecciona una entidad según la distribución real"""
        entities = list(self.entity_dist.keys())
        probabilities = list(self.entity_dist.values())
        return np.random.choice(entities, p=probabilities)

    def generate_tweet(self) -> Dict:
        """Genera un tweet sintético"""
        self.tweet_id_counter += 1
        sentiment = self._select_sentiment()
        entity = self._select_entity()
        text_length = self._generate_text_length()
        content = self._generate_tweet_content(sentiment, text_length)

        return {
            'tweet_id': str(self.tweet_id_counter),
            'original_tweet_id': str(self.tweet_id_counter),
            'entity': entity,
            'sentiment': sentiment,
            'tweet_content': content,
            'timestamp': datetime.now().isoformat()
        }

    def generate_batch(self, n: int = 100) -> List[Dict]:
        """Genera un lote de tweets sintéticos"""
        return [self.generate_tweet() for _ in range(n)]

    def generate_stream(self, n: int = 100, delay: float = 0.1):
        """Genera tweets en streaming (simula tiempo real)"""
        import time
        for _ in range(n):
            yield self.generate_tweet()
            time.sleep(delay)



class DataExporter:
    """Exporta datos generados a diferentes formatos (solo Kafka es relevante aquí)"""
    
    @staticmethod
    def to_kafka(tweet: Dict, topic: str, bootstrap_servers: str):
        """Envía un único tweet a Kafka"""
        try:
            from kafka import KafkaProducer
            
            # El productor se inicializa y se cierra por cada mensaje, lo cual es ineficiente.
            # Mejor pasar un productor ya inicializado o inicializarlo una vez en main.
            # Para este main, el productor ya está inicializado.
            pass # No hace nada aquí, la lógica de envío está en main
        except ImportError:
            logger.error("kafka-python no está instalado. Instala con: pip install kafka-python")


def main():
    """Genera tweets sintéticos continuamente y los envía a Kafka"""
    
    train_path = os.path.join(DATA_DIR, "twitter_training.csv")
    valid_path = os.path.join(DATA_DIR, "twitter_validation.csv")

    kafka_broker = os.getenv("KAFKA_BROKER", "kafka:9092")
    kafka_topic = os.getenv("KAFKA_TOPIC", "raw_tweets")

    logger.info("Iniciando generación CONTINUA de datos sintéticos")
    
    # Analizar dataset
    # Asegúrate de que los archivos existan en DATA_DIR antes de esto.
    explorer = DataExplorer(train_path, valid_path)
    stats = explorer.get_statistics_summary()
    generator = SyntheticDataGenerator(stats)

    # Productor Kafka persistente (se inicializa una vez)
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_broker,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Conectado a Kafka en {kafka_broker}")
    except Exception as e:
        logger.error(f"ERROR: No se pudo conectar a Kafka en {kafka_broker}. Asegúrate de que Kafka esté corriendo. Error: {e}")
        time.sleep(10) # Esperar un poco antes de salir, por si kafka está iniciando
        exit(1)


    # Control de la tasa de envío
    delay_per_tweet = 1.0 / TARGET_RATE_PER_SECOND
    logger.info(f"Enviando tweets a una tasa de {TARGET_RATE_PER_SECOND:.2f} tweets/segundo (delay: {delay_per_tweet:.2f}s por tweet)")

    # 🔥 Streaming continuo
    while True:
        try:
            tweet = generator.generate_tweet()
            producer.send(kafka_topic, value=tweet)
            logger.info(f"Tweet enviado: {tweet['tweet_id']} -> {tweet['sentiment']} ({tweet['entity']})")
            time.sleep(delay_per_tweet)
        except Exception as e:
            logger.error(f"Error al enviar tweet a Kafka: {e}. Reintentando en 5 segundos.")
            time.sleep(5) # Esperar antes de reintentar
        except KeyboardInterrupt:
            logger.info("Generación de tweets interrumpida.")
            break
    
    producer.flush()
    producer.close()
    logger.info("Productor Kafka cerrado.")


if __name__ == "__main__":
    main()