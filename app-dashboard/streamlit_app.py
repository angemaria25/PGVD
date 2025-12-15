import streamlit as st
import pandas as pd
import json
from kafka import KafkaConsumer
import plotly.express as px
import os
import sys

# Configuración de entorno
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
env_brokers = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_BROKERS_LIST = env_brokers.split(',')  # <--- ESTO SOLUCIONA EL ERROR
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
tabs = st.tabs(["Live (Kafka)", "📈 Análisis del Generador", "Métricas"])


# Test gráfico simple
# st.subheader("Test de Gráfico Plotly Simple")
# test_df = pd.DataFrame({'x': ['A', 'B', 'C'], 'y': [10, 20, 15]})
# fig_test = px.bar(test_df, x='x', y='y')
# st.plotly_chart(fig_test)

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
                bootstrap_servers=KAFKA_BROKERS_LIST, 
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=2000,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            messages = []
            for msg in consumer:
                messages.append(msg.value)
                if len(messages) >= 50:
                    break
            consumer.close()
            
            if messages:
                st.session_state["live_data"].extend(messages)
                # st.session_state["live_data"] = st.session_state["live_data"][-1000:]
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
# with tabs[1]:
#     st.header("Datos procesados (Spark Streaming → HDFS)")
#     spark = st.session_state.spark  # ✅ Usamos la misma SparkSession ya creada

#     try:
#         parquet_path = f"{HDFS_OUTPUT_PATH}/data/*.parquet"
#         st.write(f"Leyendo desde: `{parquet_path}`")
#         df_spark = spark.read.parquet(parquet_path)
#         df = df_spark.select("entity", "sentiment", "count").toPandas()
#         st.dataframe(df.head(20))

#         fig = px.bar(
#             df.groupby("sentiment")["count"].sum().reset_index(),
#             x="sentiment", y="count", color="sentiment",
#             title="Distribución acumulada por sentimiento (HDFS)"
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     except Exception as e:
#         st.warning(f"No se pudieron leer los datos todavía: {e}")

# --- TAB 3: Métricas ---
with tabs[2]:
    st.header("Métricas generales del sistema")
    st.metric("Mensajes en buffer Kafka (live)", len(st.session_state["live_data"]))
    st.write("- Spark UI → [http://localhost:8080](http://localhost:8080)")
    st.write("- HDFS NameNode → [http://localhost:9870](http://localhost:9870)")
    st.write("- Kafka Broker → `kafka:9092` (interno Docker)")


import re
import pandas as pd
from collections import Counter
from scipy.stats import chisquare, ks_2samp, ttest_ind
import plotly.express as px
import math
from scipy.stats import chisquare, ks_2samp
import plotly.figure_factory as ff
from scipy.stats import chisquare, entropy
# 🔧 Funciones auxiliares --------------------------------------------------

def safe_chisquare(f_obs, f_exp):
    """Versión robusta del test Chi²: corrige diferencias de suma entre observados y esperados."""
    f_obs = f_obs.astype(float)
    f_exp = f_exp.astype(float)

    if f_obs.sum() != f_exp.sum():
        ratio = f_exp.sum() / f_obs.sum() if f_obs.sum() > 0 else 1.0
        f_obs *= ratio
        diff = f_exp.sum() - f_obs.sum()
        if abs(diff) > 0:
            f_obs.iloc[-1] += diff

    return chisquare(f_obs=f_obs, f_exp=f_exp)

def kl_divergence(p, q):
    """Calcula la Divergencia KL entre dos distribuciones discretas."""
    p = [v / sum(p) for v in p]
    q = [v / sum(q) for v in q]
    kl = sum(pi * math.log(pi / qi, 2) for pi, qi in zip(p, q) if pi > 0 and qi > 0)
    return kl


# 📊 --- NUEVA PESTAÑA DE ANÁLISIS --- -----------------------------------


with tabs[1]:
    st.header("📈 Análisis de la calidad del generador de tweets sintéticos")

    try:
        # --- 1️⃣ Cargar datasets ---
        orig_path = "./data/twitter_training.csv"
        df_orig = pd.read_csv(orig_path)
        df_orig.columns = [c.strip().lower().replace(" ", "_") for c in df_orig.columns]
        df_gen = pd.DataFrame(st.session_state["live_data"])  # de Kafka live

        if df_gen.empty:
            st.warning("Aún no hay tweets generados en vivo. Refresca la pestaña Live primero.")
        else:
            st.success(f"{len(df_gen)} tweets generados listos para análisis.")

            # --- 2️⃣ Normalizar columnas ---
            if "sentiment" not in df_orig.columns:
                df_orig.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]

            df_gen["tweet_length"] = df_gen["tweet_content"].astype(str).apply(len)
            df_orig["tweet_length"] = df_orig["tweet_content"].astype(str).apply(len)

            # --- 3️⃣ Distribuciones básicas ---
            col1, col2 = st.columns(2)
            with col1:
                orig_sent = df_orig["sentiment"].value_counts(normalize=True)
                gen_sent = df_gen["sentiment"].value_counts(normalize=True)
                df_sent_compare = pd.DataFrame({
                    "Sentimiento": orig_sent.index,
                    "Original": orig_sent.values,
                    "Generado": [gen_sent.get(k, 0) for k in orig_sent.index]
                })
                fig_sent_comp = px.bar(df_sent_compare, x="Sentimiento", y=["Original", "Generado"],
                                       barmode="group", title="Distribución de Sentimientos (Original vs Generado)")
                st.plotly_chart(fig_sent_comp, use_container_width=True)

            with col2:
                orig_ent = df_orig["entity"].value_counts(normalize=True).head(10)
                gen_ent = df_gen["entity"].value_counts(normalize=True).head(10)
                df_ent_compare = pd.DataFrame({
                    "Entidad": orig_ent.index,
                    "Original": orig_ent.values,
                    "Generado": [gen_ent.get(k, 0) for k in orig_ent.index]
                })
                fig_ent_comp = px.bar(df_ent_compare, x="Entidad", y=["Original", "Generado"],
                                      barmode="group", title="Top 10 Entidades (Original vs Generado)")
                st.plotly_chart(fig_ent_comp, use_container_width=True)

            # --- 4️⃣ Longitudes de texto ---
            st.subheader("📏 Distribución de longitudes de texto")
            fig_lengths = ff.create_distplot(
                [df_orig["tweet_length"], df_gen["tweet_length"]],
                group_labels=["Original", "Generado"],
                show_hist=True, bin_size=10
            )
            st.plotly_chart(fig_lengths, use_container_width=True)

            # --- 5️⃣ Frecuencia de palabras ---
            def top_words(df, n=15):
                words = []
                for t in df["tweet_content"]:
                    words += re.findall(r"[a-záéíóúñü]+", str(t).lower())
                return pd.DataFrame(Counter(words).most_common(n), columns=["Palabra", "Frecuencia"])

            st.subheader("💬 Palabras más frecuentes")
            top_orig = top_words(df_orig)
            top_gen = top_words(df_gen)
            col3, col4 = st.columns(2)
            with col3:
                fig_top_orig = px.bar(top_orig, x="Palabra", y="Frecuencia", title="Top palabras originales")
                st.plotly_chart(fig_top_orig, use_container_width=True)
            with col4:
                fig_top_gen = px.bar(top_gen, x="Palabra", y="Frecuencia", title="Top palabras generadas")
                st.plotly_chart(fig_top_gen, use_container_width=True)

            # --- 6️⃣ Correlaciones entre entidad y sentimiento ---
            # st.subheader("🔗 Correlación Entidad vs Sentimiento")
            # pivot_orig = df_orig.pivot_table(index="entity", columns="sentiment", aggfunc="size", fill_value=0)
            # pivot_gen = df_gen.pivot_table(index="entity", columns="sentiment", aggfunc="size", fill_value=0)
            # fig_heat_orig = px.imshow(pivot_orig.head(10), text_auto=True, title="Original")
            # fig_heat_gen = px.imshow(pivot_gen.head(10), text_auto=True, title="Generado")
            # col5, col6 = st.columns(2)
            
            # with col5: st.plotly_chart(fig_heat_orig, use_container_width=True)
            # with col6: st.plotly_chart(fig_heat_gen, use_container_width=True)

            # --- 7️⃣ Estadísticas descriptivas ---
            st.subheader("📊 Estadísticas descriptivas")
            stats_compare = pd.DataFrame({
                "Métrica": ["Media", "Desv.Std", "Min", "Max", "Mediana"],
                "Original": [
                    df_orig["tweet_length"].mean(),
                    df_orig["tweet_length"].std(),
                    df_orig["tweet_length"].min(),
                    df_orig["tweet_length"].max(),
                    df_orig["tweet_length"].median(),
                ],
                "Generado": [
                    df_gen["tweet_length"].mean(),
                    df_gen["tweet_length"].std(),
                    df_gen["tweet_length"].min(),
                    df_gen["tweet_length"].max(),
                    df_gen["tweet_length"].median(),
                ]
            })
            st.dataframe(stats_compare)

            # --- 8️⃣ Prueba Chi² + KL ---
            st.subheader("📈 Tests estadísticos")

            def safe_chi_square(obs, exp):
                obs_sum, exp_sum = sum(obs), sum(exp)
                if obs_sum != exp_sum:
                    exp = [e * (obs_sum / exp_sum) for e in exp]  # reescala
                from scipy.stats import chisquare
                chi2, p = chisquare(f_obs=obs, f_exp=exp)
                return chi2, p

            f_obs = [v * 1000 for v in orig_sent.values]
            f_exp = [v * 1000 for v in gen_sent.reindex(orig_sent.index, fill_value=0).values]
            chi2_stat, chi2_p = safe_chi_square(f_obs, f_exp)
            kl_div = entropy(orig_sent, gen_sent.reindex(orig_sent.index, fill_value=1e-8))
            
            st.markdown(f"**Chi²:** {chi2_stat:.4f} | **p-valor:** {chi2_p:.5f}")
            st.markdown(f"**Divergencia KL:** {kl_div:.5f} (0 = idénticas)")

    except Exception as e:
        st.error(f"Error en el análisis: {e}")