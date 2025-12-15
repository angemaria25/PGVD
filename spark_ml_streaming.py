"""
Spark ML Streaming - Integración de Machine Learning con Spark Streaming
Realiza predicciones de sentimientos en tiempo real sobre tweets desde Kafka
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, window, from_json, to_timestamp, 
    count, when, udf, struct, to_json
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import os
import pickle
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuración de Kafka y HDFS ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
HDFS_OUTPUT_PATH = os.getenv("HDFS_OUTPUT_PATH", "hdfs://namenode:9000/user/sentiment_analysis/ml_predictions")
MODELS_DIR = "models"


class MLPredictor:
    """Clase para realizar predicciones con modelos entrenados"""
    
    def __init__(self, models_dir: str = MODELS_DIR):
        """Carga los modelos entrenados"""
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
        try:
            # Cargar modelo
            model_path = os.path.join(models_dir, 'model_sentiment.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✓ Modelo cargado: {model_path}")
            
            # Cargar vectorizador
            vectorizer_path = os.path.join(models_dir, 'vectorizer_tfidf.pkl')
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"✓ Vectorizador cargado: {vectorizer_path}")
            
            # Cargar codificador
            encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"✓ Codificador cargado: {encoder_path}")
            
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
            raise
    
    def predict(self, text: str) -> tuple:
        """
        Realiza predicción para un texto
        
        Returns:
            (predicción, confianza)
        """
        try:
            # Vectorizar
            X = self.vectorizer.transform([text])
            
            # Predicción
            y_pred_encoded = self.model.predict(X)[0]
            y_pred = self.label_encoder.inverse_transform([y_pred_encoded])[0]
            
            # Confianza
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 1.0
            
            return (y_pred, confidence)
        
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return ("Unknown", 0.0)


def main():
    """Función principal"""
    
    logger.info("="*70)
    logger.info("INICIANDO SPARK ML STREAMING")
    logger.info("="*70)
    
    # --- Crear SparkSession ---
    spark = SparkSession.builder \
        .appName("TwitterSentimentMLStreaming") \
        .master("spark://spark-master:7077") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"SparkSession creada")
    
    # --- Cargar modelos ML ---
    logger.info("Cargando modelos ML...")
    try:
        predictor = MLPredictor(MODELS_DIR)
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        logger.error("Asegúrate de ejecutar ml_training.py primero")
        spark.stop()
        return
    
    # --- Leer desde Kafka ---
    logger.info(f"Conectando a Kafka: {KAFKA_BROKER}, topic: {KAFKA_TOPIC}")
    
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()
    
    # --- Esquema del JSON ---
    schema = StructType([
        StructField("tweet_id", StringType(), True),
        StructField("original_tweet_id", StringType(), True),
        StructField("entity", StringType(), True),
        StructField("sentiment", StringType(), True),
        StructField("tweet_content", StringType(), True),
        StructField("timestamp", StringType(), True)
    ])
    
    # --- Parsear JSON ---
    df_json = df_raw.selectExpr("CAST(value AS STRING) as json_str")
    df_parsed = df_json.select(from_json(col("json_str"), schema).alias("data")).select("data.*")
    
    # --- Convertir timestamp ---
    df_parsed = df_parsed.withColumn("timestamp", to_timestamp(col("timestamp")))
    
    # --- Limpiar texto ---
    df_clean = df_parsed.withColumn("tweet_content_clean", lower(col("tweet_content"))) \
                        .withColumn("tweet_content_clean", regexp_replace(col("tweet_content_clean"), "[^a-z\\s]", ""))
    
    # --- UDF para predicción ---
    def predict_sentiment(text):
        """UDF para predicción"""
        try:
            pred, conf = predictor.predict(text)
            return f"{pred}|{conf:.4f}"
        except:
            return "Unknown|0.0"
    
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    
    predict_udf = udf(predict_sentiment, StringType())
    
    # --- Aplicar predicción ---
    logger.info("Aplicando modelo ML a stream...")
    
    df_predictions = df_clean.withColumn("ml_prediction_raw", predict_udf(col("tweet_content_clean")))
    
    # Separar predicción y confianza
    df_predictions = df_predictions \
        .withColumn("ml_prediction", col("ml_prediction_raw").substr(1, col("ml_prediction_raw").indexOf("|") - 1)) \
        .withColumn("ml_confidence", col("ml_prediction_raw").substr(col("ml_prediction_raw").indexOf("|") + 1).cast(DoubleType()))
    
    # Comparar con sentimiento real
    df_predictions = df_predictions \
        .withColumn("prediction_correct", when(col("ml_prediction") == col("sentiment"), 1).otherwise(0))
    
    # --- Agregación en ventanas de 1 minuto ---
    agg = df_predictions \
        .withWatermark("timestamp", "2 minutes") \
        .groupBy(
            window(col("timestamp"), "1 minute"),
            col("ml_prediction")
        ) \
        .agg(
            count("*").alias("count"),
            (col("count") * col("prediction_correct") / col("count")).alias("accuracy")
        )
    
    # --- Agregación de métricas globales ---
    metrics = df_predictions \
        .withWatermark("timestamp", "2 minutes") \
        .groupBy(window(col("timestamp"), "1 minute")) \
        .agg(
            count("*").alias("total_tweets"),
            (count(when(col("prediction_correct") == 1, 1)) / count("*")).alias("overall_accuracy"),
            col("ml_confidence").cast(DoubleType()).alias("avg_confidence")
        )
    
    # --- Escritura en HDFS (Predicciones) ---
    logger.info(f"Guardando predicciones en HDFS: {HDFS_OUTPUT_PATH}/predictions")
    
    query_predictions = agg.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", f"{HDFS_OUTPUT_PATH}/predictions") \
        .option("checkpointLocation", "hdfs://namenode:9000/user/sentiment_analysis/checkpoints/predictions") \
        .start()
    
    # --- Escritura en HDFS (Métricas) ---
    logger.info(f"Guardando métricas en HDFS: {HDFS_OUTPUT_PATH}/metrics")
    
    query_metrics = metrics.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", f"{HDFS_OUTPUT_PATH}/metrics") \
        .option("checkpointLocation", "hdfs://namenode:9000/user/sentiment_analysis/checkpoints/metrics") \
        .start()
    
    # --- Escritura en consola para debugging ---
    logger.info("Iniciando streaming...")
    
    query_console = df_predictions \
        .select("tweet_id", "sentiment", "ml_prediction", "ml_confidence", "prediction_correct") \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .start()
    
    # Esperar a que terminen las queries
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Streaming interrumpido por el usuario")
    finally:
        logger.info("Deteniendo queries...")
        for query in spark.streams.active:
            query.stop()
        
        spark.stop()
        logger.info("SparkSession detenida")


if __name__ == "__main__":
    main()
