from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, window, from_json, to_timestamp, count
from pyspark.sql.types import StructType, StructField, StringType # Importar count

import os

# --- Configuración de Kafka y HDFS ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka-1:9092,kafka-2:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
HDFS_OUTPUT_PATH = os.getenv("HDFS_OUTPUT_PATH", "hdfs://namenode:9000/user/sentiment_analysis/streaming_results")

if __name__ == "__main__":
    # --- Sesión Spark ---
    spark = SparkSession.builder \
        .appName("TwitterSentimentStreaming") \
        .master("spark://spark-master:7077") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.hadoop.dfs.replication", "2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"[SparkStreaming] Leyendo desde Kafka: {KAFKA_BROKER}, topic={KAFKA_TOPIC}")

    # --- Lectura desde Kafka ---
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # --- Esquema del JSON (DEBE COINCIDIR EXACTAMENTE CON EL GENERADOR) ---
    schema = StructType([
        StructField("tweet_id", StringType(), True),
        StructField("original_tweet_id", StringType(), True), # Esta columna no se usa, pero es parte del JSON
        StructField("entity", StringType(), True),
        StructField("sentiment", StringType(), True),
        StructField("tweet_content", StringType(), True),
        StructField("timestamp", StringType(), True)  # El generador envía ISO format como string
    ])

    # --- Parsear JSON ---
    df_json = df_raw.selectExpr("CAST(value AS STRING) as json_str")
    df_parsed = df_json.select(from_json(col("json_str"), schema).alias("data")).select("data.*")

    # --- Convertir timestamp a tipo de fecha (Spark puede parsear ISO string a TimestampType) ---
    df_parsed = df_parsed.withColumn("timestamp", to_timestamp(col("timestamp")))

    # --- Limpieza de texto ---
    df_clean = df_parsed.withColumn("tweet_content", lower(col("tweet_content"))) \
                        .withColumn("tweet_content", regexp_replace(col("tweet_content"), "[^a-z\\s]", ""))

    # --- Agregación en ventanas de 1 minuto ---
    # Usar withWatermark es una buena práctica para streaming si se espera que los eventos lleguen desordenados.
    # En este caso, como los generamos en orden, no es estrictamente necesario pero no hace daño.
    agg = df_clean \
        .withWatermark("timestamp", "2 minutes") \
        .groupBy(
            window(col("timestamp"), "1 minute"),
            col("sentiment")
        ) \
        .count() # Renombra la columna a 'count' por defecto

    # --- Escritura continua en HDFS ---
    query = agg.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", "/user/sentiment_analysis/streaming_results/data") \
        .option("checkpointLocation", "/user/sentiment_analysis/checkpoints") \
        .start()


    print(f"[SparkStreaming] Guardando resultados en HDFS: {HDFS_OUTPUT_PATH}")
    query.awaitTermination()