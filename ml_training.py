import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configuración
DATA_PATH = "data/twitter_training.csv"
MODELS_DIR = "models"

def train():
    print("🚀 Iniciando entrenamiento del modelo...")
    
    # 1. Cargar Datos
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: No encuentro {DATA_PATH}")
        return

    # Leemos asumiendo que no tiene header, ajusta si el tuyo tiene
    df = pd.read_csv(DATA_PATH, names=["tweet_id", "entity", "sentiment", "content"])
    df = df.dropna(subset=["content", "sentiment"])
    
    # Solo tomamos una muestra para que no tarde mucho en tu laptop (opcional)
    # df = df.sample(n=20000, random_state=42) 

    print(f"📊 Entrenando con {len(df)} tweets...")

    # 2. Preprocesamiento
    X = df['content'].astype(str)
    y = df['sentiment']

    # Codificar etiquetas (Positive -> 0, Negative -> 1...)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Vectorizar Texto (TF-IDF)
    print("🔠 Vectorizando texto...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    # 3. Entrenar Modelo
    print("🧠 Ajustando Regresión Logística...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y_encoded)

    # 4. Guardar Modelos
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print("💾 Guardando archivos .pkl...")
    with open(f'{MODELS_DIR}/model_sentiment.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{MODELS_DIR}/vectorizer_tfidf.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(f'{MODELS_DIR}/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    # Guardar info extra para el dashboard
    model_info = {
        'model_name': 'Logistic Regression TF-IDF',
        'classes': le.classes_.tolist(),
        'metrics': {'accuracy': model.score(X_vec, y_encoded)} # Simplificado
    }
    with open(f'{MODELS_DIR}/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    print("✅ Entrenamiento completado exitosamente.")

if __name__ == "__main__":
    train()