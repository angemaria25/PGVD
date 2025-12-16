import os
import pickle
import numpy as np

# Configuración de rutas (ajustada para que funcione en Docker)
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")

class SentimentPredictor:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir if models_dir else MODELS_DIR
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self._load_models()

    def _load_models(self):
        try:
            with open(os.path.join(self.models_dir, 'model_sentiment.pkl'), 'rb') as f:
                self.model = pickle.load(f)
            with open(os.path.join(self.models_dir, 'vectorizer_tfidf.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
        except FileNotFoundError:
            print(f"⚠️ ADVERTENCIA: No se encontraron modelos en {self.models_dir}. Ejecuta ml_training.py primero.")

    def predict_single(self, text):
        if not self.model:
            return {"prediction": "Error", "confidence": 0.0}
        
        # 1. Vectorizar
        text_vec = self.vectorizer.transform([text])
        
        # 2. Predecir
        pred_idx = self.model.predict(text_vec)[0]
        prediction = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # 3. Calcular Confianza
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            confidence = np.max(self.model.predict_proba(text_vec))
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "confidence_pct": f"{float(confidence)*100:.1f}%"
        }