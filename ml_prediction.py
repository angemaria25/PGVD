"""
ML Prediction Module - Realiza predicciones de sentimientos usando modelos entrenados
Soporta predicciones en lote y en tiempo real
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de rutas
MODELS_DIR = "models"
RESULTS_DIR = "results"


class SentimentPredictor:
    """Clase para realizar predicciones de sentimientos"""
    
    def __init__(self, models_dir: str = MODELS_DIR):
        """
        Inicializa el predictor cargando los modelos entrenados
        
        Args:
            models_dir: Directorio donde están guardados los modelos
        """
        self.models_dir = models_dir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.model_info = None
        
        self._load_models()
        logger.info("SentimentPredictor inicializado")
    
    def _load_models(self):
        """Carga los modelos entrenados desde disco"""
        logger.info("Cargando modelos entrenados...")
        
        # Cargar modelo
        model_path = os.path.join(self.models_dir, 'model_sentiment.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"✓ Modelo cargado: {model_path}")
        
        # Cargar vectorizador
        vectorizer_path = os.path.join(self.models_dir, 'vectorizer_tfidf.pkl')
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizador no encontrado: {vectorizer_path}")
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        logger.info(f"✓ Vectorizador cargado: {vectorizer_path}")
        
        # Cargar codificador de etiquetas
        encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Codificador no encontrado: {encoder_path}")
        
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        logger.info(f"✓ Codificador cargado: {encoder_path}")
        
        # Cargar información del modelo
        info_path = os.path.join(self.models_dir, 'model_info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                self.model_info = pickle.load(f)
            logger.info(f"✓ Información del modelo cargada")
            logger.info(f"  Modelo: {self.model_info['model_name']}")
            logger.info(f"  F1-Score: {self.model_info['metrics']['f1_score']:.4f}")
    
    def predict_single(self, text: str) -> Dict:
        """
        Realiza predicción para un único texto
        
        Args:
            text: Texto a clasificar
            
        Returns:
            Diccionario con predicción y confianza
        """
        # Vectorizar texto
        X = self.vectorizer.transform([text])
        
        # Predicción
        y_pred_encoded = self.model.predict(X)[0]
        y_pred = self.label_encoder.inverse_transform([y_pred_encoded])[0]
        
        # Confianza (probabilidad)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
        elif hasattr(self.model, 'decision_function'):
            # Para SVM
            decision = self.model.decision_function(X)[0]
            confidence = float(1.0 / (1.0 + np.exp(-decision)))
        else:
            confidence = 1.0
        
        return {
            'text': text,
            'prediction': y_pred,
            'confidence': confidence,
            'confidence_pct': f"{confidence*100:.2f}%"
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Realiza predicciones para múltiples textos
        
        Args:
            texts: Lista de textos a clasificar
            
        Returns:
            Lista de diccionarios con predicciones
        """
        results = []
        
        # Vectorizar todos los textos
        X = self.vectorizer.transform(texts)
        
        # Predicciones
        y_pred_encoded = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Confianzas
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            confidences = np.max(probabilities, axis=1)
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(X)
            confidences = 1.0 / (1.0 + np.exp(-decision))
        else:
            confidences = np.ones(len(texts))
        
        # Construir resultados
        for text, pred, conf in zip(texts, y_pred, confidences):
            results.append({
                'text': text,
                'prediction': pred,
                'confidence': float(conf),
                'confidence_pct': f"{float(conf)*100:.2f}%"
            })
        
        return results
    
    def predict_from_csv(self, input_path: str, output_path: str = None, 
                        text_column: str = None) -> pd.DataFrame:
        """
        Realiza predicciones desde un archivo CSV
        
        Args:
            input_path: Ruta al archivo CSV de entrada
            output_path: Ruta para guardar resultados (opcional)
            text_column: Nombre de la columna con el texto (auto-detecta si no se especifica)
            
        Returns:
            DataFrame con predicciones
        """
        logger.info(f"Cargando datos desde: {input_path}")
        
        # Cargar CSV
        df = pd.read_csv(input_path, dtype=str)
        logger.info(f"Cargados {len(df)} registros")
        
        # Detectar columna de texto
        if text_column is None:
            cols = df.columns.tolist()
            for col in cols:
                col_lower = col.lower()
                if 'content' in col_lower or 'tweet' in col_lower:
                    text_column = col
                    break
            
            if text_column is None:
                text_column = cols[-1]  # Última columna por defecto
        
        logger.info(f"Usando columna de texto: {text_column}")
        
        # Realizar predicciones
        logger.info("Realizando predicciones...")
        texts = df[text_column].astype(str).tolist()
        predictions = self.predict_batch(texts)
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame(predictions)
        
        # Combinar con datos originales
        results_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        
        # Guardar resultados si se especifica
        if output_path:
            results_df.to_csv(output_path, index=False)
            logger.info(f"✓ Resultados guardados: {output_path}")
        
        return results_df
    
    def evaluate_predictions(self, df: pd.DataFrame, 
                            real_sentiment_column: str = None) -> Dict:
        """
        Evalúa las predicciones comparándolas con sentimientos reales
        
        Args:
            df: DataFrame con predicciones
            real_sentiment_column: Nombre de la columna con sentimientos reales
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if real_sentiment_column is None:
            # Intentar detectar columna de sentimiento real
            cols = df.columns.tolist()
            for col in cols:
                col_lower = col.lower()
                if 'sentiment' in col_lower and col != 'prediction':
                    real_sentiment_column = col
                    break
        
        if real_sentiment_column is None:
            logger.warning("No se encontró columna de sentimiento real")
            return {}
        
        logger.info(f"Evaluando predicciones contra: {real_sentiment_column}")
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_true = df[real_sentiment_column].values
        y_pred = df['prediction'].values
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_predictions': len(df),
            'correct_predictions': int((y_true == y_pred).sum()),
            'incorrect_predictions': int((y_true != y_pred).sum())
        }
        
        logger.info(f"\n=== MÉTRICAS DE EVALUACIÓN ===")
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        logger.info(f"Predicciones correctas: {metrics['correct_predictions']}/{metrics['total_predictions']}")
        
        return metrics
    
    def get_model_info(self) -> Dict:
        """Retorna información del modelo"""
        if self.model_info:
            return self.model_info
        
        return {
            'model_type': type(self.model).__name__,
            'vectorizer_type': type(self.vectorizer).__name__,
            'classes': self.label_encoder.classes_.tolist(),
            'n_features': len(self.vectorizer.get_feature_names_out())
        }


def main():
    """Función principal con argumentos de línea de comandos"""
    
    parser = argparse.ArgumentParser(
        description='Realiza predicciones de sentimientos usando modelos entrenados'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Archivo CSV de entrada para predicciones'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Archivo CSV de salida con predicciones'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Texto individual para clasificar'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default=MODELS_DIR,
        help='Directorio con los modelos entrenados'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluar predicciones contra sentimientos reales'
    )
    
    args = parser.parse_args()
    
    # Crear predictor
    try:
        predictor = SentimentPredictor(args.models_dir)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Asegúrate de ejecutar ml_training.py primero para entrenar los modelos")
        return
    
    # Mostrar información del modelo
    logger.info("\n=== INFORMACIÓN DEL MODELO ===")
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        logger.info(f"{key}: {value}")
    
    # Predicción de texto individual
    if args.text:
        logger.info(f"\nClasificando texto: '{args.text}'")
        result = predictor.predict_single(args.text)
        logger.info(f"Predicción: {result['prediction']}")
        logger.info(f"Confianza: {result['confidence_pct']}")
    
    # Predicción desde CSV
    elif args.input:
        logger.info(f"\nRealizando predicciones desde: {args.input}")
        
        output_path = args.output or os.path.join(RESULTS_DIR, 'predictions.csv')
        Path(RESULTS_DIR).mkdir(exist_ok=True)
        
        df_results = predictor.predict_from_csv(args.input, output_path)
        
        logger.info(f"\nPrimeras predicciones:")
        logger.info(df_results[['prediction', 'confidence_pct']].head(10))
        
        # Evaluar si hay sentimientos reales
        if args.evaluate:
            metrics = predictor.evaluate_predictions(df_results)
    
    else:
        # Modo interactivo
        logger.info("\n=== MODO INTERACTIVO ===")
        logger.info("Ingresa textos para clasificar (escribe 'salir' para terminar)")
        
        while True:
            text = input("\nIngresa un texto: ").strip()
            
            if text.lower() == 'salir':
                break
            
            if not text:
                continue
            
            result = predictor.predict_single(text)
            logger.info(f"Predicción: {result['prediction']}")
            logger.info(f"Confianza: {result['confidence_pct']}")


if __name__ == "__main__":
    main()
