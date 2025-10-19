from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, window, count
from pyspark.sql.types import StructType, StringType, DoubleType
import os

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
HDFS_OUTPUT_PATH = os.getenv("HDFS_OUTPUT_PATH", "hdfs://namenode:9000/user/sentiment_analysis/streaming_results")

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("TwitterSentimentStreaming") \
        .master("spark://spark-master:7077") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.sql.shuffle.partitions", "2") \
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

    from pyspark.sql.functions import from_json
    schema = StructType() \
        .add("tweet_id", StringType()) \
        .add("original_tweet_id", StringType()) \
        .add("entity", StringType()) \
        .add("sentiment", StringType()) \
        .add("tweet_content", StringType()) \
        .add("timestamp", DoubleType())

    df_json = df_raw.selectExpr("CAST(value AS STRING) as json_str")
    df_parsed = df_json.select(from_json(col("json_str"), schema).alias("data")).select("data.*")

    # --- Limpieza de texto ---
    df_clean = df_parsed.withColumn("tweet_content", lower(col("tweet_content"))) \
                        .withColumn("tweet_content", regexp_replace(col("tweet_content"), "[^a-z\\s]", ""))

    # --- Agregación en ventanas de 1 minuto ---
    agg = df_clean.groupBy(window(col("timestamp").cast("timestamp"), "1 minute"), col("sentiment")) \
                  .agg(count("*").alias("count"))

    # --- Escritura continua en HDFS ---
    query = agg.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", f"{HDFS_OUTPUT_PATH}/data") \
        .option("checkpointLocation", f"{HDFS_OUTPUT_PATH}/checkpoint") \
        .trigger(processingTime="30 seconds") \
        .start()

    print(f"[SparkStreaming] Guardando resultados en HDFS: {HDFS_OUTPUT_PATH}")
    query.awaitTermination()
