from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
import os

# Configuración de HDFS
HDFS_NAMENODE = "hdfs://namenode:9000"
HDFS_RAW_CSV_PATH = "/user/sentiment_analysis/raw_data/twitter_training.csv"
HDFS_PROCESSED_PATH = "/user/sentiment_analysis/processed_data"

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("SentimentAnalysisBatchProcessing") \
        .master("spark://spark-master:7077") \
        .config("spark.hadoop.fs.defaultFS", HDFS_NAMENODE) \
        .config("spark.hadoop.dfs.replication", "1") \
        .getOrCreate()

    print("SparkSession inicializada.")

    # --- Parte nueva: Cargar el CSV desde HDFS si ya existe ---
    # En un entorno real, el CSV ya estaría en HDFS, cargado por otra herramienta o proceso.
    # Aquí, asumimos que ya ha sido cargado una vez.
    
    # Para que esto funcione, *tú* debes primero copiar el twitter_training.csv a HDFS.
    # Esto se hará manualmente una vez después de levantar los contenedores.
    # Consulta las instrucciones de ejecución para el paso "Subir el CSV a HDFS".

    try:
        print(f"Intentando leer el CSV de HDFS: {HDFS_RAW_CSV_PATH}")
        df_raw = spark.read.csv(HDFS_RAW_CSV_PATH, header=True, inferSchema=True)
        print(f"Cargado {df_raw.count()} registros del CSV de HDFS.")

        # Renombrar columnas para consistencia y simplificación
        df_raw = df_raw.select(
            col("Tweet ID").alias("tweet_id"),
            col("entity").alias("entity"),
            col("sentiment").alias("sentiment"),
            col("Tweet content").alias("tweet_content")
        )

        # *** Ejecución de primeros procesos básicos de limpieza y transformación ***
        # 1. Eliminar filas con valores nulos en columnas importantes
        df_cleaned = df_raw.dropna(subset=["tweet_id", "entity", "sentiment", "tweet_content"])

        # 2. Convertir texto a minúsculas y eliminar caracteres especiales en el contenido del tweet
        df_transformed = df_cleaned.withColumn("tweet_content_cleaned", lower(col("tweet_content"))) \
                                   .withColumn("tweet_content_cleaned", regexp_replace(col("tweet_content_cleaned"), "[^a-z\\s]", ""))

        print("Schema después de la limpieza y transformación:")
        df_transformed.printSchema()
        print("Primeros 5 registros transformados:")
        df_transformed.show(5, truncate=False)

        # *** Pruebas iniciales de consultas ***
        print("Conteo de tweets por sentimiento:")
        df_transformed.groupBy("sentiment").count().show()

        print("Conteo de tweets por entidad (Top 10):")
        df_transformed.groupBy("entity").count().orderBy(col("count").desc()).limit(10).show(truncate=False)

        # Guardar el DataFrame procesado en HDFS
        output_hdfs_path = f"{HDFS_NAMENODE}{HDFS_PROCESSED_PATH}/batch_processed_tweets"
        print(f"Guardando datos procesados en HDFS en: {output_hdfs_path}")
        df_transformed.limit(1000).write \
            .mode("overwrite") \
            .parquet(output_hdfs_path)
        print("Datos guardados en HDFS.")

        # Verificar que los datos se hayan escrito en HDFS
        print("Verificando datos en HDFS:")
        df_hdfs = spark.read.parquet(output_hdfs_path)
        df_hdfs.show(5, truncate=False)
        print(f"Total de registros leídos de HDFS: {df_hdfs.count()}")

    except Exception as e:
        print(f"Error en el procesamiento Spark o HDFS: {e}")
    finally:
        spark.stop()
        print("SparkSession detenida.")