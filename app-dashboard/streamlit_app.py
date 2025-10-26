import streamlit as st
import pandas as pd
import json
from kafka import KafkaConsumer
import plotly.express as px
import os
import sys # Importar sys para imprimir a stderr, que a veces se ve mejor en Docker logs

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
HDFS_OUTPUT_PATH = os.getenv("HDFS_OUTPUT_PATH", "hdfs://namenode:9000/user/sentiment_analysis/streaming_results")

st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")
st.title("📊 Twitter Entity Sentiment — Dashboard")

tabs = st.tabs(["Live (Kafka)", "Procesado (HDFS)", "Métricas"])
# Al final del script, fuera de cualquier pestaña, o en una pestaña nueva
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
        # ... (código de consumo de Kafka) ...
        # Tu código actual para consumir mensajes está bien, no lo cambiamos aquí.
        # Solo asegúrate de que el bloque `if messages:` se ejecute si hay datos.

        try: # Asegúrate de que este bloque de try/except esté rodeando tu lógica de Kafka
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

        # --- Depuración adicional (déjalas, son útiles) ---
        st.write("DEBUG (gráfico): Contenido completo del DataFrame (df):")
        st.dataframe(df)
        st.write("DEBUG (gráfico): Información de columnas y tipos de datos en df:")
        df.info(buf=sys.stdout) # Usar buf=sys.stdout para que se imprima correctamente en el log
        st.write("DEBUG (gráfico): Columnas presentes en df:")
        st.write(df.columns.tolist())
        # --- Fin Depuración ---

        st.dataframe(df[["tweet_content", "entity", "sentiment"]].tail(10))
        col1, col2 = st.columns(2)
        with col1:
            if 'sentiment' in df.columns and not df['sentiment'].empty:
                sent_counts = df["sentiment"].value_counts()
                
                # *** CAMBIO CLAVE AQUÍ ***
                # Asegurarse de que el índice se convierte a string y el Series a DataFrame
                sent = pd.DataFrame({'sentiment': sent_counts.index.astype(str), 'count': sent_counts.values})
                # *** FIN CAMBIO CLAVE ***

                st.write("DEBUG (gráfico sent): DataFrame 'sent' antes de Plotly:")
                st.dataframe(sent)
                st.write(f"DEBUG (gráfico sent): Columnas en 'sent': {sent.columns.tolist()}")
                st.write(f"DEBUG (gráfico sent): Tipos de datos en 'sent':\n{sent.dtypes}") # Nuevo DEBUG para tipos

                if not sent.empty:
                    fig_sent = px.bar(sent, x="sentiment", y="count", color="sentiment", title="Distribución de Sentimientos (Live)")
                    # Opcional: intentar forzar el rango del eje Y si el problema persiste
                    # max_count = sent['count'].max()
                    # fig_sent.update_yaxes(range=[0, max_count * 1.1])
                    st.plotly_chart(fig_sent, use_container_width=True)
                else:
                    st.info("No hay datos para mostrar en la gráfica de sentimientos.")
            else:
                st.warning("No hay datos válidos para el sentimiento.")
        with col2:
            if 'entity' in df.columns and not df['entity'].empty:
                ents_counts = df["entity"].value_counts().head(10)

                # *** CAMBIO CLAVE AQUÍ ***
                # Asegurarse de que el índice se convierte a string y el Series a DataFrame
                ents = pd.DataFrame({'entity': ents_counts.index.astype(str), 'count': ents_counts.values})
                # *** FIN CAMBIO CLAVE ***

                st.write("DEBUG (gráfico ent): DataFrame 'ents' antes de Plotly:")
                st.dataframe(ents)
                st.write(f"DEBUG (gráfico ent): Columnas en 'ents': {ents.columns.tolist()}")
                st.write(f"DEBUG (gráfico ent): Tipos de datos en 'ents':\n{ents.dtypes}") # Nuevo DEBUG para tipos
                
                if not ents.empty:
                    fig_ents = px.bar(ents, x="entity", y="count", color="entity", title="Top 10 Entidades (Live)")
                    # Opcional: intentar forzar el rango del eje Y si el problema persiste
                    # max_count_ent = ents['count'].max()
                    # fig_ents.update_yaxes(range=[0, max_count_ent * 1.1])
                    st.plotly_chart(fig_ents, use_container_width=True)
                else:
                    st.info("No hay datos para mostrar en la gráfica de entidades.")
    else:
        st.info("No hay mensajes aún. Inicia el producer para ver flujo en vivo.")
# --- TAB 2: Procesado desde HDFS ---
with tabs[1]:
    st.header("Datos procesados (Spark Streaming → HDFS)")
    try:
        parquet_path = f"{HDFS_OUTPUT_PATH}/data"
        st.write(f"Leyendo desde: `{parquet_path}`")
        df = pd.read_parquet(parquet_path)
        st.dataframe(df.head(10))
        fig = px.bar(df.groupby("sentiment")["count"].sum().reset_index(),
                     x="sentiment", y="count", color="sentiment",
                     title="Distribución acumulada por sentimiento (HDFS)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudieron leer los datos de HDFS todavía: {e}")

# --- TAB 3: Métricas ---
with tabs[2]:
    st.header("Métricas generales del sistema")
    st.metric("Mensajes en buffer Kafka (live)", len(st.session_state["live_data"]))
    st.write("- Spark UI → [http://localhost:8080](http://localhost:8080)")
    st.write("- HDFS NameNode → [http://localhost:9870](http://localhost:9870)")
    st.write("- Kafka Broker → `kafka:9092` (interno Docker)")
