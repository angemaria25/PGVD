import pandas as pd
import time
import random
import json
from kafka import KafkaProducer
from uuid import uuid4 # Para generar IDs únicos
import re # Para limpieza de texto

# Kafka configuration
KAFKA_BROKER = "localhost:9093"
KAFKA_TOPIC = "raw_tweets"

# Tasa objetivo: 1000 tweets por minuto
TARGET_RATE_PER_MINUTE = 1000
TIME_BETWEEN_TWEETS_SEC = 60 / TARGET_RATE_PER_MINUTE

# Diccionario de sinónimos simples para aumentar la variabilidad
SYNONYM_MAP = {
    "great": ["fantastic", "awesome", "superb"],
    "good": ["nice", "positive", "fine"],
    "bad": ["terrible", "horrible", "awful"],
    "happy": ["joyful", "delighted", "glad"],
    "sad": ["unhappy", "gloomy", "depressed"],
    "love": ["adore", "cherish", "like"],
    "hate": ["detest", "dislike", "loathe"]
}

def clean_text_for_augmentation(text):
    """Limpia el texto para facilitar la sustitución de palabras."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Quita puntuación
    return text

def augment_tweet_content(content):
    """Aplica tácticas de aumento al contenido del tweet."""
    if not isinstance(content, str):
        return content

    words = content.split()
    augmented_words = []

    for word in words:
        # 1. Sustitución de sinónimos
        found_synonym = False
        for key, synonyms in SYNONYM_MAP.items():
            if word == key and random.random() < 0.3: # 30% de probabilidad de reemplazar
                augmented_words.append(random.choice(synonyms))
                found_synonym = True
                break
        if not found_synonym:
            augmented_words.append(word)
    
    # 2. Reordenamiento ligero de palabras (para tweets más largos)
    if len(augmented_words) > 5 and random.random() < 0.1: # 10% de probabilidad
        idx1, idx2 = random.sample(range(len(augmented_words)), 2)
        augmented_words[idx1], augmented_words[idx2] = augmented_words[idx2], augmented_words[idx1]

    # 3. Añadir un ruido mínimo (ej. una letra aleatoria o un error de typo simple)
    if random.random() < 0.05 and augmented_words: # 5% de probabilidad
        idx = random.randint(0, len(augmented_words) - 1)
        original_word = augmented_words[idx]
        if len(original_word) > 2: # Solo si la palabra es lo suficientemente larga
            pos = random.randint(0, len(original_word) - 1)
            augmented_words[idx] = original_word[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + original_word[pos+1:]


    return " ".join(augmented_words)

def generate_and_send_tweet(producer, df_original):
    """Genera un tweet aumentado y lo envía a Kafka."""
    # Muestra una fila original
    original_row = df_original.sample(1).iloc[0]

    # Augmenta el contenido del tweet
    augmented_content = augment_tweet_content(original_row["Tweet content"])

    tweet = {
        "tweet_id": str(uuid4()), # Generar un ID único cada vez
        "original_tweet_id": str(original_row["Tweet ID"]), # Referencia al ID original para auditoría
        "entity": original_row["entity"],
        "sentiment": original_row["sentiment"],
        "tweet_content": augmented_content,
        "timestamp": time.time()
    }
    
    producer.send(KAFKA_TOPIC, value=tweet)
    return tweet # Retorna el tweet generado para estadísticas

def serialize_tweet(tweet):
    return json.dumps(tweet).encode('utf-8')

if __name__ == "__main__":
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=serialize_tweet
        )
        print(f"Conectado a Kafka en {KAFKA_BROKER}")

        try:
            df = pd.read_csv("data/twitter_training.csv")
            df.columns = ["Tweet ID", "entity", "sentiment", "Tweet content"]
            df = df.dropna(subset=["entity", "sentiment", "Tweet content"])
            print("Dataset original cargado para simular la producción.")
        except FileNotFoundError:
            print("Error: twitter_training.csv no encontrado para simular la producción.")
            exit()

        print(f"Produciendo mensajes en el topic '{KAFKA_TOPIC}' a una tasa de {TARGET_RATE_PER_MINUTE} por minuto (presiona Ctrl+C para detener)...")
        
        while True:
            tweet = generate_and_send_tweet(producer, df)
            # print(f"Enviado: {json.dumps(tweet)}") # Comenta esto si la consola se vuelve ilegible
            time.sleep(random.uniform(TIME_BETWEEN_TWEETS_SEC * 0.9, TIME_BETWEEN_TWEETS_SEC * 1.1)) # Pequeña variación

    except Exception as e:
        print(f"Error al conectar con Kafka o durante la producción: {e}")