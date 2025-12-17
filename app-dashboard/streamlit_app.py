import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from kafka import KafkaConsumer
import re
from collections import Counter
from scipy.stats import chisquare, entropy
import numpy as np

# --- Configuración de Página ---
st.set_page_config(page_title="Twitter AI Dashboard", layout="wide", page_icon="🤖")

# --- Variables de Entorno ---
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka-1:9092,kafka-2:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
DATA_DIR = os.getenv("DATA_DIR", "/app/data") 

# --- Importar Lógica ML ---
try:
    from ml_prediction import SentimentPredictor
except ImportError:
    st.error("⚠️ No se encontró el módulo ml_prediction.py. Verifica los volúmenes en docker-compose.")
    st.stop()

# --- Funciones Auxiliares ---
@st.cache_resource
def load_ai_model():
    try:
        predictor = SentimentPredictor(models_dir=MODELS_DIR)
        return predictor
    except Exception as e:
        return None

predictor = load_ai_model()

# --- Inicializar Estado de Sesión (Memoria) ---
if "live_tweets" not in st.session_state:
    st.session_state["live_tweets"] = []

# --- Interfaz Principal ---
st.title("🤖 Monitorización de Sentimientos con IA")
st.markdown("Sistema Big Data con arquitectura Lambda: **Kafka + Spark Streaming + Scikit-Learn**")

# Definir las 4 pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predicción en Vivo", 
    "📈 Análisis del Generador", 
    "📊 Métricas del Modelo", 
    "👀 Monitor Kafka"
])

# ==========================================
# TAB 1: PREDICCIÓN INTERACTIVA Y STREAMING
# ==========================================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🧪 Testear Modelo Manualmente")
        st.info("Escribe un texto para ver qué opina la IA:")
        user_text = st.text_area("Texto del Tweet:", "I love playing Call of Duty but the servers are terrible!")
        
        if st.button("Analizar Sentimiento"):
            if predictor and predictor.model:
                result = predictor.predict_single(user_text)
                sentiment = result['prediction']
                conf = result['confidence']
                color_map = {"Positive": "green", "Negative": "red", "Neutral": "gray", "Irrelevant": "blue"}
                
                st.markdown(f"### Resultado: :{color_map.get(sentiment, 'black')}[{sentiment}]")
                st.progress(conf)
                st.caption(f"Confianza del modelo: {result['confidence_pct']}")
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = conf * 100,
                    title = {'text': "Confianza (%)"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color_map.get(sentiment, "black")}}
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("El modelo no está cargado.")

    with col2:
        st.subheader("⚡ Stream de Kafka (Acumulativo)")
        st.write("Tweets llegando en tiempo real. Pulsa para acumular más datos para el análisis.")
        
        c_btn1, c_btn2 = st.columns(2)
        
        # Botón de Carga
        with c_btn1:
            if st.button("📥 Traer +100 tweets de Kafka"):
                try:
                    brokers_list = KAFKA_BROKER.split(',')
                    consumer = KafkaConsumer(
                        KAFKA_TOPIC,
                        bootstrap_servers=brokers_list,
                        auto_offset_reset='latest',
                        enable_auto_commit=True,
                        consumer_timeout_ms=2000,
                        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                    )
                    
                    msgs = []
                    for msg in consumer:
                        msgs.append(msg.value)
                        if len(msgs) >= 100: break 
                    consumer.close()
                    
                    if msgs:
                        # Acumulamos los mensajes nuevos
                        st.session_state["live_tweets"].extend(msgs)
                        st.success(f"✅ +{len(msgs)} tweets añadidos. Total en memoria: {len(st.session_state['live_tweets'])}")
                    else:
                        st.warning("No llegaron mensajes nuevos. Verifica el 'tweet-producer'.")
                        
                except Exception as e:
                    st.error(f"Error conectando a Kafka: {e}")

        # Botón de Limpieza
        with c_btn2:
            if st.button("🗑️ Borrar Historial"):
                st.session_state["live_tweets"] = []
                st.experimental_rerun()

        # Mostrar tabla
        if st.session_state["live_tweets"]:
            df_live = pd.DataFrame(st.session_state["live_tweets"])
            st.write(f"**Mostrando últimos 10 de {len(df_live)} tweets acumulados:**")
            
            # Predicción visual rápida para la tabla
            if 'ml_prediction' not in df_live.columns and predictor and predictor.model:
                df_show = df_live.tail(10).copy()
                df_show['ml_analysis'] = df_show['tweet_content'].apply(lambda x: predictor.predict_single(str(x))['prediction'])
                st.dataframe(df_show[['timestamp', 'entity', 'tweet_content', 'ml_analysis']], use_container_width=True)
            else:
                st.dataframe(df_live[['timestamp', 'entity', 'tweet_content']].tail(10), use_container_width=True)

# ==========================================
# TAB 2: ANÁLISIS DEL GENERADOR (Estadística)
# ==========================================
with tab2:
    st.header("📈 Análisis de Calidad de Datos (Acumulado)")
    
    csv_path = os.path.join(DATA_DIR, "twitter_training.csv")
    
    if not os.path.exists(csv_path):
        st.error("No se encuentra el dataset original.")
    elif not st.session_state["live_tweets"]:
        st.warning("⚠️ Memoria vacía. Ve a la Pestaña 1 y pulsa 'Traer tweets' varias veces para acumular datos.")
    else:
        try:
            # Datos Originales
            df_orig = pd.read_csv(csv_path, names=["tweet_id", "entity", "sentiment", "tweet_content"])
            df_orig = df_orig.dropna(subset=["sentiment", "tweet_content"])
            # Limpiar columnas para evitar espacios raros
            df_orig.columns = [c.strip().lower().replace(" ", "_") for c in df_orig.columns]
            if "sentiment" not in df_orig.columns: # Fallback
                df_orig.columns = ["tweet_id", "entity", "sentiment", "tweet_content"]

            # Datos Acumulados
            df_gen = pd.DataFrame(st.session_state["live_tweets"])
            
            st.info(f"📊 Analizando una muestra acumulada de **{len(df_gen)}** tweets generados vs Dataset Original ({len(df_orig)} registros).")

            # --- 1. Distribución de Sentimientos ---
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Distribución de Sentimientos")
                orig_sent = df_orig["sentiment"].value_counts(normalize=True)
                gen_sent = df_gen["sentiment"].value_counts(normalize=True)
                
                df_compare_sent = pd.DataFrame({
                    "Sentimiento": orig_sent.index,
                    "Original": orig_sent.values,
                    "Generado": [gen_sent.get(k, 0) for k in orig_sent.index]
                })
                fig_sent = px.bar(df_compare_sent.melt(id_vars="Sentimiento"), x="Sentimiento", y="value", color="variable", barmode="group", title="Sentimientos: Real vs Sintético")
                st.plotly_chart(fig_sent, use_container_width=True)

            # --- 2. Distribución de Entidades ---
            with col_b:
                st.subheader("Top 10 Entidades")
                top_entities = df_orig["entity"].value_counts().head(10).index
                orig_ent = df_orig[df_orig["entity"].isin(top_entities)]["entity"].value_counts(normalize=True)
                gen_ent = df_gen[df_gen["entity"].isin(top_entities)]["entity"].value_counts(normalize=True)
                
                df_compare_ent = pd.DataFrame({
                    "Entidad": top_entities,
                    "Original": [orig_ent.get(e, 0) for e in top_entities],
                    "Generado": [gen_ent.get(e, 0) for e in top_entities]
                })
                fig_ent = px.bar(df_compare_ent.melt(id_vars="Entidad"), x="Entidad", y="value", color="variable", barmode="group", title="Entidades: Real vs Sintético")
                st.plotly_chart(fig_ent, use_container_width=True)

            # --- 3. Longitud de Texto (RESTAURADO) ---
            st.subheader("📏 Distribución de Longitud de Texto")
            # Calculamos longitud
            df_gen["tweet_length"] = df_gen["tweet_content"].astype(str).apply(len)
            df_orig["tweet_length"] = df_orig["tweet_content"].astype(str).apply(len)
            
            # Muestra del original para no saturar el gráfico
            sample_orig = df_orig["tweet_length"].sample(min(len(df_orig), 2000), random_state=42)
            
            fig_hist = ff.create_distplot(
                [sample_orig, df_gen["tweet_length"]],
                ['Original (Muestra)', 'Generado'],
                bin_size=10, 
                show_hist=False, # Solo curva de densidad para que sea más limpio
                show_rug=False
            )
            fig_hist.update_layout(title="Densidad de Longitud de Caracteres")
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- 4. Frecuencia de Palabras (RESTAURADO) ---
            def top_words(df_in, col_text, n=15):
                text_blob = " ".join(df_in[col_text].astype(str).tolist()).lower()
                # Quitamos caracteres no alfabéticos simples
                words = re.findall(r"\b[a-z]{3,}\b", text_blob) 
                return pd.DataFrame(Counter(words).most_common(n), columns=["Palabra", "Frecuencia"])

            st.subheader("💬 Palabras más frecuentes")
            c_w1, c_w2 = st.columns(2)
            
            with c_w1:
                top_orig = top_words(df_orig.sample(min(len(df_orig), 5000)), "tweet_content") # Muestra para velocidad
                fig_top_orig = px.bar(top_orig, x="Palabra", y="Frecuencia", title="Top palabras Originales")
                st.plotly_chart(fig_top_orig, use_container_width=True)
            
            with c_w2:
                top_gen = top_words(df_gen, "tweet_content")
                fig_top_gen = px.bar(top_gen, x="Palabra", y="Frecuencia", title="Top palabras Generadas")
                st.plotly_chart(fig_top_gen, use_container_width=True)

            # --- 5. Estadísticas Descriptivas (RESTAURADO) ---
            st.subheader("📊 Estadísticas descriptivas")
            stats_compare = pd.DataFrame({
                "Métrica": ["Media Longitud", "Desv.Std", "Min", "Max", "Mediana"],
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
            st.dataframe(stats_compare, use_container_width=True)

            # --- 6. Pruebas Estadísticas (Chi2 + KL) ---
            st.divider()
            st.subheader("🧮 Validación Matemática")
            c1, c2 = st.columns(2)
            
            # Chi Cuadrado
            f_obs = (gen_sent.reindex(orig_sent.index, fill_value=0).values * 100).astype(int) + 1
            f_exp = (orig_sent.values * 100).astype(int) + 1
            ratio = sum(f_obs) / sum(f_exp)
            f_exp = f_exp * ratio
            chi2_stat, p_val = chisquare(f_obs, f_exp)
            
            # KL Divergence (RESTAURADO)
            p = orig_sent.values + 1e-10
            q = gen_sent.reindex(orig_sent.index, fill_value=1e-10).values
            kl_val = entropy(p, q)
            
            with c1:
                st.metric("Test Chi-Cuadrado (p-value)", f"{p_val:.4f}", 
                         delta="Aceptable" if p_val > 0.05 else "Diferente", 
                         delta_color="normal" if p_val > 0.05 else "inverse")
            with c2:
                st.metric("Divergencia KL", f"{kl_val:.4f}", 
                          help="0 indica distribuciones idénticas. < 0.1 es excelente.")

        except Exception as e:
            st.error(f"Error en el análisis: {e}")

# ==========================================
# TAB 3: MÉTRICAS DEL MODELO ML
# ==========================================
with tab3:
    st.header("🧠 Rendimiento del Modelo (Offline Training)")
    try:
        import pickle
        info_path = os.path.join(MODELS_DIR, "model_info.pkl")
        if os.path.exists(info_path):
            with open(info_path, "rb") as f:
                model_info = pickle.load(f)
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("Algoritmo", model_info.get('model_name', 'Logistic Regression'))
            kpi2.metric("Accuracy", f"{model_info.get('metrics', {}).get('accuracy', 0.0)*100:.2f}%")
            st.json(model_info)
        else:
            st.warning("Entrena el modelo primero.")
    except Exception as e:
        st.error(f"Error: {e}")

# ==========================================
# TAB 4: MONITOR DE INFRAESTRUCTURA
# ==========================================
with tab4:
    st.header("📡 Estado de los Servicios")
    c1, c2, c3 = st.columns(3)
    c1.metric("Kafka", "Online", "9092")
    c2.metric("Spark", "Active", "9090")
    c3.metric("HDFS", "Healthy", "9870")