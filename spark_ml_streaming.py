from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, udf, struct, window, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import os

# Importamos nuestra clase compartida (se copiará al contenedor)
from ml_prediction import SentimentPredictor

# Configuración
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka-1:9092,kafka-2:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
MODELS_DIR = "/opt/spark/models"  # Ruta dentro del contenedor

# --- UDF Wrapper ---
# Inicializamos el predictor en el Driver, pero para los workers usamos una técnica Lazy
# o simplemente cargamos dentro de la función UDF para evitar problemas de serialización.
def predict_sentiment_udf_func(text):
    # Instanciar aquí asegura que cada Worker cargue su copia del modelo
    if not text:
        return "Unknown", 0.0
    
    try:
        # Singleton pattern simple para no recargar el modelo por cada fila
        if not hasattr(predict_sentiment_udf_func, "predictor"):
            predict_sentiment_udf_func.predictor = SentimentPredictor(models_dir=MODELS_DIR)
        
        result = predict_sentiment_udf_func.predictor.predict_single(str(text))
        return result["prediction"], result["confidence"]
    except Exception as e:
        return "Error", 0.0

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("TwitterMLStreaming") \
        .master("spark://spark-master:7077") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")

    # Registrar UDF
    # Definimos el esquema de salida de la UDF (Predicción, Confianza)
    schema_output = StructType([
        StructField("ml_prediction", StringType(), False),
        StructField("ml_confidence", DoubleType(), False)
    ])
    
    predict_udf = udf(predict_sentiment_udf_func, schema_output)

    # Leer Kafka
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .load()

    # Parsear JSON
    json_schema = StructType([
        StructField("tweet_content", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("sentiment", StringType(), True) # El original para comparar
    ])
    
    df_parsed = df_raw.selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), json_schema).alias("data")) \
        .select("data.*") \
        .withColumn("timestamp", to_timestamp(col("timestamp")))

    # --- APLICAR MACHINE LEARNING ---
    # Esto ejecuta el modelo de Python sobre cada tweet
    df_predicted = df_parsed.withColumn("ml_result", predict_udf(col("tweet_content"))) \
                            .select("*", "ml_result.*") # Aplanar resultado

    # Agregación por ventana (Predicción ML)
    df_agg = df_predicted \
        .withWatermark("timestamp", "2 minutes") \
        .groupBy(window(col("timestamp"), "1 minute"), col("ml_prediction")) \
        .count()

    # Guardar en HDFS
    query = df_agg.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", "/user/sentiment_analysis/ml_results") \
        .option("checkpointLocation", "/user/sentiment_analysis/checkpoints/ml") \
        .start()

    query.awaitTermination()