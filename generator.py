import pandas as pd
import time
import random
import json

def generate_tweet(df):
    """Genera un tweet aleatorio del DataFrame."""
    random_row = df.sample(1).iloc[0]
    tweet = {
        "tweet_id": str(random_row["Tweet ID"]),
        "entity": random_row["entity"],
        "sentiment": random_row["sentiment"],
        "tweet_content": random_row["Tweet content"],
        "timestamp": time.time()
    }
    return tweet

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/twitter_training.csv")
        df.columns = ["Tweet ID", "entity", "sentiment", "Tweet content"] # Asegura nombres de columnas consistentes
        df = df.dropna(subset=["entity", "sentiment", "Tweet content"]) # Eliminar filas con valores nulos en columnas clave
        print("Dataset cargado y preprocesado.")
    except FileNotFoundError:
        print("Error: twitter_training.csv no encontrado. Asegúrate de que esté en la carpeta 'data/'.")
        exit()

    print("Generando tweets (presiona Ctrl+C para detener)...")
    while True:
        tweet = generate_tweet(df)
        print(f"Tweet generado: {json.dumps(tweet, indent=2)}")
        time.sleep(random.uniform(1, 3)) # Espera un tiempo aleatorio entre 1 y 3 segundos