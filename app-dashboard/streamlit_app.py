import streamlit as st
import pandas as pd
import json
from kafka import KafkaConsumer
import plotly.express as px
import os
import sys

# Configuración de entorno
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
HDFS_OUTPUT_PATH = os.getenv("HDFS_OUTPUT_PATH", "hdfs://namenode:9000/user/sentiment_analysis/streaming_results")

st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")
st.title("📊 Twitter Entity Sentiment — Dashboard")

# --- SparkSession seguro ---
from pyspark.sql import SparkSession
from pyspark import SparkContext

def get_spark_session():
    """
    Devuelve un SparkSession seguro para Streamlit.
    Si hay un SparkContext detenido, lo reinicia.
    """
    try:
        sc = SparkContext.getOrCreate()
        if sc._jsc.sc().isStopped():
            sc.stop()
            sc = SparkContext()
        spark = SparkSession(sc)
        return spark
    except Exception:
        spark = (
            SparkSession.builder
            .appName("Streamlit-HDFS-Reader")
            .master("spark://spark-master:7077")
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
            .getOrCreate()
        )
        return spark

if "spark" not in st.session_state:
    st.session_state.spark = get_spark_session()

# --- Tabs ---
tabs = st.tabs(["Live (Kafka)", "Procesado (HDFS)", "Métricas"])

# Test gráfico simple
st.subheader("Test de Gráfico Plotly Simple")
test_df = pd.DataFrame({'x': ['A', 'B', 'C'], 'y': [10, 20, 15]})
fig_test = px.bar(test_df, x='x', y='y')
st.plotly_chart(fig_test)

# --- TAB 1: Live Kafka ---
with tabs[0]:
    st.header("Live desde Kafka (últimos mensajes)")
    refresh = st.button("Refrescar")
    if "live_data" not in st.session_state:
        st.session_state["live_data"] = []

    if refresh:
        st.info("Intentando conectar a Kafka y leer mensajes...")
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=2000,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            messages = []
            for msg in consumer:
                messages.append(msg.value)
                if len(messages) >= 100:
                    break
            consumer.close()
            
            if messages:
                st.session_state["live_data"].extend(messages)
                st.session_state["live_data"] = st.session_state["live_data"][-1000:]
                st.success(f"{len(messages)} mensajes leídos y añadidos.")
            else:
                st.info("No se encontraron nuevos mensajes en Kafka.")

        except Exception as e:
            error_msg = f"Error crítico al conectar o procesar Kafka: {e}"
            st.error(error_msg)
            print(f"DEBUG Streamlit ERROR: {error_msg}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    data = st.session_state["live_data"]
    if data:
        df = pd.DataFrame(data)
        st.write("DEBUG columnas live:", df.columns.tolist())
        st.dataframe(df[["tweet_content", "entity", "sentiment"]].tail(10))

        col1, col2 = st.columns(2)
        with col1:
            if 'sentiment' in df.columns and not df['sentiment'].empty:
                sent_counts = df["sentiment"].value_counts()
                sent = pd.DataFrame({'sentiment': sent_counts.index.astype(str), 'count': sent_counts.values})
                fig_sent = px.bar(sent, x="sentiment", y="count", title="Distribución de Sentimientos (Live)")
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.warning("No hay datos válidos para el sentimiento.")

        with col2:
            if 'entity' in df.columns and not df['entity'].empty:
                ents_counts = df["entity"].value_counts().head(10)
                ents = pd.DataFrame({'entity': ents_counts.index.astype(str), 'count': ents_counts.values})
                fig_ents = px.bar(ents, x="entity", y="count", title="Top 10 Entidades (Live)")
                st.plotly_chart(fig_ents, use_container_width=True)
            else:
                st.info("No hay datos para mostrar entidades todavía.")
    else:
        st.info("No hay mensajes aún. Inicia el producer para ver flujo en vivo.")

# --- TAB 2: Procesado desde HDFS ---
with tabs[1]:
    st.header("Datos procesados (Spark Streaming → HDFS)")
    spark = st.session_state.spark  # ✅ Usamos la misma SparkSession ya creada

    try:
        parquet_path = f"{HDFS_OUTPUT_PATH}/data/*.parquet"
        st.write(f"Leyendo desde: `{parquet_path}`")
        df_spark = spark.read.parquet(parquet_path)
        df = df_spark.select("entity", "sentiment", "count").toPandas()
        st.dataframe(df.head(20))

        fig = px.bar(
            df.groupby("sentiment")["count"].sum().reset_index(),
            x="sentiment", y="count", color="sentiment",
            title="Distribución acumulada por sentimiento (HDFS)"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"No se pudieron leer los datos todavía: {e}")

# --- TAB 3: Métricas ---
#with tabs[2]:
    #st.header("Métricas generales del sistema")
    #st.metric("Mensajes en buffer Kafka (live)", len(st.session_state["live_data"]))
    #st.write("- Spark UI → [http://localhost:8080](http://localhost:8080)")
    #st.write("- HDFS NameNode → [http://localhost:9870](http://localhost:9870)")
    #st.write("- Kafka Broker → `kafka:9092` (interno Docker)")
