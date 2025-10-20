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

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataExplorer:
    """Analiza las distribuciones estadísticas del dataset original"""
    
    def __init__(self, train_path: str, valid_path: str):
        """
        Inicializa el explorador de datos
        
        Args:
            train_path: Ruta al archivo de entrenamiento
            valid_path: Ruta al archivo de validación
        """
        self.train = pd.read_csv(train_path)
        self.valid = pd.read_csv(valid_path)
        
        # Limpiar nombres de columnas
        self.train.columns = self.train.columns.str.strip()
        self.valid.columns = self.valid.columns.str.strip()
        
        # Combinar datasets
        self.df = pd.concat([self.train, self.valid], ignore_index=True)
        
        # Eliminar duplicados por Tweet ID
        self.df = self.df.drop_duplicates(subset=["Tweet ID"], keep="first").reset_index(drop=True)
        
        logger.info(f"Dataset cargado: {len(self.df)} tweets únicos")
    
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
        Inicializa el generador
        
        Args:
            stats: Diccionario con estadísticas del dataset original
        """
        self.stats = stats
        self.sentiment_dist = stats['sentiment_distribution']
        self.entity_dist = stats['entity_distribution']
        self.text_length_stats = stats['text_length_stats']
        self.top_words = [word for word, _ in stats['top_words']]
        
        # Palabras adicionales para generar variedad
        self.positive_words = ['love', 'great', 'awesome', 'excellent', 'amazing', 'fantastic', 
                                'wonderful', 'perfect', 'best', 'good', 'nice', 'beautiful']
        self.negative_words = ['hate', 'bad', 'terrible', 'awful', 'horrible', 'worst', 
                                'poor', 'disappointing', 'useless', 'broken', 'sucks']
        self.neutral_words = ['is', 'the', 'a', 'and', 'or', 'but', 'in', 'on', 'at', 'to']
        
        self.tweet_id_counter = 10000
        
        logger.info("Generador de datos sintéticos inicializado")
    
    def _generate_text_length(self) -> int:
        """Genera una longitud de texto usando distribución normal truncada"""
        mean = self.text_length_stats['mean']
        std = self.text_length_stats['std']
        min_len = int(self.text_length_stats['min'])
        max_len = int(self.text_length_stats['max'])
        
        # Usar distribución normal truncada
        length = int(np.random.normal(mean, std))
        return max(min_len, min(max_len, length))
    
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
        
        # Mezclar palabras frecuentes con palabras de sentimiento
        while len(' '.join(words)) < length:
            if random.random() < 0.3:  # 30% palabras de sentimiento
                words.append(random.choice(sentiment_words))
            else:  # 70% palabras frecuentes
                words.append(random.choice(self.top_words))
        
        tweet = ' '.join(words)
        return tweet[:length]  # Truncar a la longitud deseada
    
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
            'Tweet ID': self.tweet_id_counter,
            'Entity': entity,
            'Sentiment': sentiment,
            'Tweet content': content,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_batch(self, n: int = 100) -> List[Dict]:
        """Genera un lote de tweets sintéticos"""
        return [self.generate_tweet() for _ in range(n)]
    
    def generate_stream(self, n: int = 100, delay: float = 0.1):
        """Genera tweets en streaming (simula tiempo real)"""
        for i in range(n):
            tweet = self.generate_tweet()
            yield tweet
            time.sleep(delay)


class DataExporter:
    """Exporta datos generados a diferentes formatos"""
    
    @staticmethod
    def to_csv(tweets: List[Dict], output_path: str):
        """Exporta tweets a CSV"""
        df = pd.DataFrame(tweets)
        df.to_csv(output_path, index=False)
        logger.info(f"Datos exportados a {output_path}")
    
    @staticmethod
    def to_json(tweets: List[Dict], output_path: str):
        """Exporta tweets a JSON"""
        with open(output_path, 'w') as f:
            json.dump(tweets, f, indent=2)
        logger.info(f"Datos exportados a {output_path}")
    
    @staticmethod
    def to_kafka(tweets: List[Dict], topic: str, bootstrap_servers: str = 'localhost:9092'):
        """Exporta tweets a Kafka (requiere kafka-python)"""
        try:
            from kafka import KafkaProducer
            
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            for tweet in tweets:
                producer.send(topic, value=tweet)
            
            producer.flush()
            producer.close()
            logger.info(f"Datos enviados a Kafka topic: {topic}")
        except ImportError:
            logger.error("kafka-python no está instalado. Instala con: pip install kafka-python")


def main():
    """Genera datos sintéticos y los envía a Kafka"""
    
    train_path = "data/twitter_training.csv"
    valid_path = "data/twitter_validation.csv"
    
    logger.info("Iniciando generación de datos sintéticos")
    explorer = DataExplorer(train_path, valid_path)
    stats = explorer.get_statistics_summary()
    
    generator = SyntheticDataGenerator(stats)
    synthetic_tweets = generator.generate_batch(n=1000)
    
    logger.info(f"Se generaron {len(synthetic_tweets)} tweets sintéticos")
    
    exporter = DataExporter()
    exporter.to_kafka(synthetic_tweets, topic='tweets', bootstrap_servers='localhost:9092')
    logger.info("Datos enviados a Kafka exitosamente")


if __name__ == "__main__":
    main()
