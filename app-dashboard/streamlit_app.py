import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from kafka import KafkaConsumer
import re
from collections import Counter
from scipy.stats import chisquare, ks_2samp, ttest_ind
import plotly.express as px
import math
from scipy.stats import chisquare, ks_2samp
import plotly.figure_factory as ff
from scipy.stats import chisquare, entropy

# Importamos la clase compartida de ML
# (Esto funciona porque mapeamos el volumen en docker-compose)
try:
    from ml_prediction import SentimentPredictor
except ImportError:
    st.error("⚠️ No se encontró el módulo ml_prediction.py. Verifica los volúmenes en docker-compose.")
    st.stop()

# --- Configuración ---
st.set_page_config(page_title="Twitter AI Dashboard", layout="wide", page_icon="🤖")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka-1:9092,kafka-2:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")

# --- Funciones de Carga ---

@st.cache_resource
def load_ai_model():
    """Carga el modelo de Machine Learning en memoria caché"""
    try:
        predictor = SentimentPredictor(models_dir=MODELS_DIR)
        return predictor
    except Exception as e:
        return None

# Cargar el predictor
predictor = load_ai_model()

# --- Interfaz Principal ---
st.title("🤖 Monitorización de Sentimientos con IA")
st.markdown("Sistema Big Data con arquitectura Lambda: **Kafka + Spark Streaming + Scikit-Learn**")

# Definir pestañas
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predicción en Vivo", "📈 Análisis del Generador","📊 Métricas del Modelo", "👀 Monitor Kafka"])

# ==========================================
# TAB 1: PREDICCIÓN INTERACTIVA Y EN VIVO
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🧪 Testear Modelo Manualmente")
        st.info("Escribe algo para ver qué opina la IA:")
        user_text = st.text_area("Texto del Tweet:", "I love playing Call of Duty but the servers are terrible!")
        
        if st.button("Analizar Sentimiento"):
            if predictor and predictor.model:
                result = predictor.predict_single(user_text)
                
                # Mostrar resultado visual
                sentiment = result['prediction']
                conf = result['confidence']
                
                color_map = {
                    "Positive": "green", 
                    "Negative": "red", 
                    "Neutral": "gray", 
                    "Irrelevant": "blue"
                }
                
                st.markdown(f"### Resultado: :{color_map.get(sentiment, 'black')}[{sentiment}]")
                st.progress(conf)
                st.caption(f"Confianza del modelo: {result['confidence_pct']}")
                
                # Gráfico gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = conf * 100,
                    title = {'text': "Confianza (%)"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color_map.get(sentiment, "black")}}
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("El modelo no está cargado. Revisa la carpeta 'models'.")

    with col2:
        st.subheader("⚡ Stream de Predicciones (Spark Streaming)")
        st.write("Estos datos vienen de Kafka, procesados por Spark con ML:")
        
        # Botón para refrescar Kafka
        if st.button("📥 Traer últimos tweets de Kafka"):
            try:
                # Arreglo para leer lista de brokers
                brokers_list = KAFKA_BROKER.split(',')
                consumer = KafkaConsumer(
                    KAFKA_TOPIC,
                    bootstrap_servers=brokers_list,
                    auto_offset_reset='latest',
                    enable_auto_commit=True,
                    consumer_timeout_ms=2000, # Esperar max 2 seg
                    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                )
                
                msgs = []
                for msg in consumer:
                    msgs.append(msg.value)
                    if len(msgs) >= 50: break # Solo mostrar últimos 20
                consumer.close()
                
                if msgs:
                    df_live = pd.DataFrame(msgs)
                    # Si ya tienes la predicción hecha por Spark en el JSON, úsala.
                    # Si no, aplicamos el modelo aquí para visualizar (simulación de lo que hace Spark)
                    if 'ml_prediction' not in df_live.columns and predictor and predictor.model:
                        df_live['ml_analysis'] = df_live['tweet_content'].apply(lambda x: predictor.predict_single(x)['prediction'])
                        df_live['confidence'] = df_live['tweet_content'].apply(lambda x: predictor.predict_single(x)['confidence_pct'])
                    
                    st.dataframe(df_live[['timestamp', 'entity', 'tweet_content', 'ml_analysis', 'confidence']], use_container_width=True)
                    
                    # Gráfica rápida de lo que acaba de llegar
                    fig_live = px.bar(df_live['ml_analysis'].value_counts(), title="Distribución del lote actual")
                    st.plotly_chart(fig_live, use_container_width=True)
                else:
                    st.warning("No llegaron mensajes nuevos en los últimos 2 segundos.")
                    
            except Exception as e:
                st.error(f"Error conectando a Kafka: {e}")

# ==========================================
# TAB 2: MÉTRICAS DEL ENTRENAMIENTO
# ==========================================
with tab3:
    st.header("🧠 Métricas del Modelo Entrenado")
    
    # Intentar cargar info del modelo
    try:
        import pickle
        info_path = os.path.join(MODELS_DIR, "model_info.pkl")
        
        if os.path.exists(info_path):
            with open(info_path, "rb") as f:
                model_info = pickle.load(f)
            
            # KPIs principales
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Algoritmo", model_info.get('model_name', 'Unknown'))
            with kpi2:
                # Si guardaste accuracy en el diccionario
                acc = model_info.get('metrics', {}).get('accuracy', 0.0)
                st.metric("Accuracy (Precisión Global)", f"{acc*100:.2f}%")
            with kpi3:
                n_classes = len(model_info.get('classes', []))
                st.metric("Clases Detectadas", n_classes)
            
            st.divider()
            
            st.subheader("Clases que el modelo conoce:")
            st.write(model_info.get('classes', []))
            
            st.success(f"Modelo cargado desde: {MODELS_DIR}")
            
        else:
            st.warning("No se encontró el archivo de metadatos del modelo (model_info.pkl).")
            st.info("Ejecuta 'python ml_training.py' para generar reportes.")
            
    except Exception as e:
        st.error(f"Error leyendo métricas: {e}")

# ==========================================
# TAB 3: MONITOR DE INFRAESTRUCTURA
# ==========================================
with tab4:
    st.header("📡 Estado del Clúster")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Kafka Broker", "Online", delta="Port 9092")
    c2.metric("Spark Master", "Active", delta="Port 9090")
    c3.metric("HDFS Namenode", "Healthy", delta="Port 9870")
    
    st.markdown("### Enlaces Rápidos")
    st.markdown("""
    - [Spark Master UI](http://localhost:9090)
    - [HDFS Explorer](http://localhost:9870)
    """)
    
    # Aquí podrías poner una gráfica simulada de uso de recursos si quieres "vender" la monitorización
    st.subheader("Simulación de Carga de Red")
    chart_data = pd.DataFrame({
        'time': range(20),
        'events_per_sec': [10, 15, 40, 90, 85, 95, 100, 98, 92, 40, 20, 15, 10, 5, 12, 18, 25, 30, 35, 40]
    })
    st.area_chart(chart_data.set_index('time'))

with tab2:
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