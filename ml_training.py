"""
ML Training Module - Entrenamiento de modelos de clasificación de sentimientos
Entrena múltiples modelos y selecciona el mejor basado en F1-Score
"""

import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime
from pathlib import Path

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de rutas
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Crear directorios si no existen
Path(MODELS_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)


class SentimentMLTrainer:
    """Clase para entrenar modelos de clasificación de sentimientos"""
    
    def __init__(self, train_path: str, valid_path: str):
        """
        Inicializa el entrenador
        
        Args:
            train_path: Ruta al archivo de entrenamiento
            valid_path: Ruta al archivo de validación
        """
        self.train_path = train_path
        self.valid_path = valid_path
        self.df_train = None
        self.df_valid = None
        self.df_combined = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.label_encoder = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        logger.info("SentimentMLTrainer inicializado")
    
    def load_data(self):
        """Carga los datos de entrenamiento y validación"""
        logger.info(f"Cargando datos de entrenamiento desde: {self.train_path}")
        
        try:
            # Cargar datos
            self.df_train = pd.read_csv(self.train_path, dtype=str)
            self.df_valid = pd.read_csv(self.valid_path, dtype=str)
            
            logger.info(f"Datos de entrenamiento: {len(self.df_train)} registros")
            logger.info(f"Datos de validación: {len(self.df_valid)} registros")
            
            # Combinar datasets
            self.df_combined = pd.concat([self.df_train, self.df_valid], ignore_index=True)
            logger.info(f"Dataset combinado: {len(self.df_combined)} registros")
            
            # Mostrar información del dataset
            logger.info(f"Columnas: {list(self.df_combined.columns)}")
            logger.info(f"Primeras filas:\n{self.df_combined.head()}")
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def explore_data(self):
        """Realiza análisis exploratorio de datos"""
        logger.info("=== ANÁLISIS EXPLORATORIO DE DATOS ===")
        
        # Detectar columnas
        cols = self.df_combined.columns.tolist()
        logger.info(f"Columnas detectadas: {cols}")
        
        # Buscar columnas de sentimiento y contenido
        sentiment_col = None
        content_col = None
        
        for col in cols:
            col_lower = col.lower()
            if 'sentiment' in col_lower:
                sentiment_col = col
            if 'content' in col_lower or 'tweet' in col_lower:
                content_col = col
        
        if not sentiment_col or not content_col:
            # Intentar con nombres genéricos
            if len(cols) >= 4:
                sentiment_col = cols[2]  # Tercera columna
                content_col = cols[3]    # Cuarta columna
        
        logger.info(f"Columna de sentimiento: {sentiment_col}")
        logger.info(f"Columna de contenido: {content_col}")
        
        # Distribución de sentimientos
        if sentiment_col:
            logger.info("\n=== Distribución de Sentimientos ===")
            sentiment_dist = self.df_combined[sentiment_col].value_counts()
            logger.info(f"\n{sentiment_dist}")
            logger.info(f"\nProporción:\n{self.df_combined[sentiment_col].value_counts(normalize=True)}")
        
        # Estadísticas de longitud de texto
        if content_col:
            logger.info("\n=== Estadísticas de Longitud de Texto ===")
            text_lengths = self.df_combined[content_col].astype(str).apply(len)
            logger.info(f"Media: {text_lengths.mean():.2f}")
            logger.info(f"Mediana: {text_lengths.median():.2f}")
            logger.info(f"Min: {text_lengths.min()}")
            logger.info(f"Max: {text_lengths.max()}")
            logger.info(f"Std: {text_lengths.std():.2f}")
    
    def prepare_data(self):
        """Prepara los datos para el entrenamiento"""
        logger.info("=== PREPARACIÓN DE DATOS ===")
        
        # Detectar columnas
        cols = self.df_combined.columns.tolist()
        sentiment_col = None
        content_col = None
        
        for col in cols:
            col_lower = col.lower()
            if 'sentiment' in col_lower:
                sentiment_col = col
            if 'content' in col_lower or 'tweet' in col_lower:
                content_col = col
        
        if not sentiment_col or not content_col:
            if len(cols) >= 4:
                sentiment_col = cols[2]
                content_col = cols[3]
        
        logger.info(f"Usando columna de sentimiento: {sentiment_col}")
        logger.info(f"Usando columna de contenido: {content_col}")
        
        # Limpiar datos
        df = self.df_combined.copy()
        df = df.dropna(subset=[sentiment_col, content_col])
        df = df[df[content_col].astype(str).str.len() > 0]
        
        logger.info(f"Registros después de limpieza: {len(df)}")
        
        # Preparar X (contenido) e y (sentimiento)
        X = df[content_col].astype(str).values
        y = df[sentiment_col].astype(str).values
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"Clases detectadas: {self.label_encoder.classes_}")
        logger.info(f"Número de clases: {len(self.label_encoder.classes_)}")
        
        # Vectorizar texto usando TF-IDF
        logger.info("Vectorizando texto con TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english'
        )
        X_vectorized = self.vectorizer.fit_transform(X)
        
        logger.info(f"Dimensiones de features: {X_vectorized.shape}")
        logger.info(f"Número de features: {len(self.vectorizer.get_feature_names_out())}")
        
        # Dividir en entrenamiento y prueba (80-20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_vectorized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Conjunto de entrenamiento: {self.X_train.shape[0]} muestras")
        logger.info(f"Conjunto de prueba: {self.X_test.shape[0]} muestras")
    
    def train_models(self):
        """Entrena múltiples modelos"""
        logger.info("=== ENTRENAMIENTO DE MODELOS ===")
        
        # 1. Naive Bayes
        logger.info("\n1. Entrenando Naive Bayes...")
        nb_model = MultinomialNB()
        nb_model.fit(self.X_train, self.y_train)
        self.models['Naive Bayes'] = nb_model
        logger.info("✓ Naive Bayes entrenado")
        
        # 2. SVM (Linear)
        logger.info("\n2. Entrenando SVM (Linear)...")
        svm_model = LinearSVC(max_iter=2000, random_state=42)
        svm_model.fit(self.X_train, self.y_train)
        self.models['SVM'] = svm_model
        logger.info("✓ SVM entrenado")
        
        # 3. Random Forest
        logger.info("\n3. Entrenando Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        logger.info("✓ Random Forest entrenado")
        
        # 4. Logistic Regression
        logger.info("\n4. Entrenando Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr_model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr_model
        logger.info("✓ Logistic Regression entrenado")
        
        logger.info(f"\n✓ Total de modelos entrenados: {len(self.models)}")
    
    def evaluate_models(self):
        """Evalúa todos los modelos"""
        logger.info("=== EVALUACIÓN DE MODELOS ===")
        
        for model_name, model in self.models.items():
            logger.info(f"\n--- Evaluando {model_name} ---")
            
            # Predicciones
            y_pred = model.predict(self.X_test)
            
            # Métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_pred': y_pred,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            logger.info(f"Accuracy:  {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall:    {recall:.4f}")
            logger.info(f"F1-Score:  {f1:.4f}")
            
            # Reporte de clasificación
            logger.info(f"\nReporte de Clasificación:\n{classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_)}")
        
        # Seleccionar mejor modelo
        best_f1 = max([r['f1_score'] for r in self.results.values()])
        self.best_model_name = [name for name, r in self.results.items() if r['f1_score'] == best_f1][0]
        self.best_model = self.models[self.best_model_name]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"✓ MEJOR MODELO: {self.best_model_name}")
        logger.info(f"  F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}")
        logger.info(f"{'='*50}")
    
    def save_models(self):
        """Guarda los modelos entrenados"""
        logger.info("=== GUARDANDO MODELOS ===")
        
        # Guardar mejor modelo
        model_path = os.path.join(MODELS_DIR, 'model_sentiment.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        logger.info(f"✓ Modelo guardado: {model_path}")
        
        # Guardar vectorizador
        vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer_tfidf.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"✓ Vectorizador guardado: {vectorizer_path}")
        
        # Guardar codificador de etiquetas
        encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"✓ Codificador de etiquetas guardado: {encoder_path}")
        
        # Guardar información del modelo
        model_info = {
            'model_name': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'metrics': self.results[self.best_model_name],
            'classes': self.label_encoder.classes_.tolist(),
            'n_features': len(self.vectorizer.get_feature_names_out())
        }
        
        info_path = os.path.join(MODELS_DIR, 'model_info.pkl')
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        logger.info(f"✓ Información del modelo guardada: {info_path}")
    
    def generate_report(self):
        """Genera un reporte de entrenamiento"""
        logger.info("=== GENERANDO REPORTE ===")
        
        report_path = os.path.join(RESULTS_DIR, 'training_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE DE ENTRENAMIENTO DE MODELOS DE SENTIMIENTOS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Datos de entrenamiento: {len(self.df_train)} registros\n")
            f.write(f"Datos de validación: {len(self.df_valid)} registros\n")
            f.write(f"Total: {len(self.df_combined)} registros\n\n")
            
            f.write("CLASES DETECTADAS:\n")
            for i, cls in enumerate(self.label_encoder.classes_):
                f.write(f"  {i}: {cls}\n")
            f.write("\n")
            
            f.write("RESULTADOS DE MODELOS:\n")
            f.write("-"*70 + "\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
                f.write(f"  Precision: {results['precision']:.4f}\n")
                f.write(f"  Recall:    {results['recall']:.4f}\n")
                f.write(f"  F1-Score:  {results['f1_score']:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write(f"MEJOR MODELO: {self.best_model_name}\n")
            f.write(f"F1-Score: {self.results[self.best_model_name]['f1_score']:.4f}\n")
            f.write("="*70 + "\n")
        
        logger.info(f"✓ Reporte guardado: {report_path}")
    
    def plot_results(self):
        """Genera gráficos de resultados"""
        logger.info("=== GENERANDO GRÁFICOS ===")
        
        # Comparación de modelos
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparación de Modelos de Sentimientos', fontsize=16, fontweight='bold')
        
        model_names = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        precisions = [self.results[m]['precision'] for m in model_names]
        recalls = [self.results[m]['recall'] for m in model_names]
        f1_scores = [self.results[m]['f1_score'] for m in model_names]
        
        # Accuracy
        axes[0, 0].bar(model_names, accuracies, color='skyblue')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Precision
        axes[0, 1].bar(model_names, precisions, color='lightgreen')
        axes[0, 1].set_title('Precision')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim([0, 1])
        for i, v in enumerate(precisions):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Recall
        axes[1, 0].bar(model_names, recalls, color='lightcoral')
        axes[1, 0].set_title('Recall')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim([0, 1])
        for i, v in enumerate(recalls):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # F1-Score
        axes[1, 1].bar(model_names, f1_scores, color='gold')
        axes[1, 1].set_title('F1-Score')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim([0, 1])
        for i, v in enumerate(f1_scores):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_path = os.path.join(RESULTS_DIR, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Gráfico de comparación guardado: {plot_path}")
        
        # Matriz de confusión del mejor modelo
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = self.results[self.best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    ax=ax)
        ax.set_title(f'Matriz de Confusión - {self.best_model_name}')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
        
        cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Matriz de confusión guardada: {cm_path}")
        
        plt.close('all')
    
    def run_training_pipeline(self):
        """Ejecuta el pipeline completo de entrenamiento"""
        logger.info("="*70)
        logger.info("INICIANDO PIPELINE DE ENTRENAMIENTO")
        logger.info("="*70)
        
        try:
            self.load_data()
            self.explore_data()
            self.prepare_data()
            self.train_models()
            self.evaluate_models()
            self.save_models()
            self.generate_report()
            self.plot_results()
            
            logger.info("\n" + "="*70)
            logger.info("✓ PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"Error en el pipeline: {e}")
            raise


def main():
    """Función principal"""
    
    # Rutas de datos
    train_path = os.path.join(DATA_DIR, "twitter_training.csv")
    valid_path = os.path.join(DATA_DIR, "twitter_validation.csv")
    
    # Verificar que los archivos existan
    if not os.path.exists(train_path):
        logger.error(f"Archivo no encontrado: {train_path}")
        return
    
    if not os.path.exists(valid_path):
        logger.error(f"Archivo no encontrado: {valid_path}")
        return
    
    # Crear y ejecutar entrenador
    trainer = SentimentMLTrainer(train_path, valid_path)
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()
