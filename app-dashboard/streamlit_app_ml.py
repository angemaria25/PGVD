"""
Dashboard Streamlit con Machine Learning
Visualiza predicciones de sentimientos, confianza del modelo y métricas de rendimiento
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from kafka import KafkaConsumer
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import pickle
from pathlib import Path

# Configuración de entorno
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw_tweets")
HDFS_OUTPUT_PATH = os.getenv("HDFS_OUTPUT_PATH", "hdfs://namenode:9000/user/sentiment_analysis/ml_predictions")
MODELS_DIR = "models"

st.set_page_config(page_title="ML Sentiment Dashboard", layout="wide")
st.title("🤖 Twitter Sentiment Analysis - ML Dashboard")

# --- SparkSession seguro ---
from pyspark.sql import SparkSession
from pyspark import SparkContext

def get_spark_session():
    """Devuelve un SparkSession seguro para Streamlit"""
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
            .appName("Streamlit-ML-Dashboard")
            .master("spark://spark-master:7077")
            .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
            .getOrCreate()
        )
        return spark

if "spark" not in st.session_state:
    st.session_state.spark = get_spark_session()

# --- Cargar información del modelo ---
@st.cache_resource
def load_model_info():
    """Carga información del modelo entrenado"""
    try:
        info_path = os.path.join(MODELS_DIR, 'model_info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return None

model_info = load_model_info()

# --- Tabs principales ---
tabs = st.tabs([
    "📊 Dashboard Principal",
    "🔮 Predicciones en Vivo",
    "📈 Métricas de Rendimiento",
    "🎯 Matriz de Confusión",
    "ℹ️ Información del Modelo"
])

# ============================================================================
# TAB 1: DASHBOARD PRINCIPAL
# ============================================================================
with tabs[0]:
    st.header("Dashboard Principal - Análisis de Sentimientos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Modelo", model_info['model_name'] if model_info else "N/A")
    
    with col2:
        if model_info:
            st.metric("F1-Score", f"{model_info['metrics']['f1_score']:.4f}")
        else:
            st.metric("F1-Score", "N/A")
    
    with col3:
        if model_info:
            st.metric("Accuracy", f"{model_info['metrics']['accuracy']:.4f}")
        else:
            st.metric("Accuracy", "N/A")
    
    with col4:
        if model_info:
            st.metric("Clases", len(model_info['classes']))
        else:
            st.metric("Clases", "N/A")
    
    st.divider()
    
    # Leer datos de HDFS
    spark = st.session_state.spark
    
    try:
        # Leer predicciones
        predictions_path = f"{HDFS_OUTPUT_PATH}/predictions/*.parquet"
        st.write(f"📂 Leyendo predicciones desde: `{predictions_path}`")
        
        df_spark = spark.read.parquet(predictions_path)
        df_predictions = df_spark.toPandas()
        
        if not df_predictions.empty:
            st.success(f"✓ Cargadas {len(df_predictions)} predicciones")
            
            # Gráfico de distribución de predicciones
            col1, col2 = st.columns(2)
            
            with col1:
                pred_counts = df_predictions['ml_prediction'].value_counts()
                fig_pred = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Distribución de Predicciones ML",
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Negative': '#e74c3c',
                        'Neutral': '#95a5a6',
                        'Irrelevant': '#34495e'
                    }
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                # Confianza promedio por predicción
                conf_by_pred = df_predictions.groupby('ml_prediction')['ml_confidence'].mean().sort_values(ascending=False)
                fig_conf = px.bar(
                    x=conf_by_pred.index,
                    y=conf_by_pred.values,
                    title="Confianza Promedio por Predicción",
                    labels={'x': 'Predicción', 'y': 'Confianza'},
                    color=conf_by_pred.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_conf, use_container_width=True)
            
            # Tabla de predicciones recientes
            st.subheader("Predicciones Recientes")
            st.dataframe(
                df_predictions[['ml_prediction', 'ml_confidence', 'prediction_correct']].tail(20),
                use_container_width=True
            )
        
        else:
            st.info("⏳ Aún no hay predicciones. Inicia el streaming para ver datos.")
    
    except Exception as e:
        st.warning(f"⚠️ No se pudieron leer predicciones: {e}")

# ============================================================================
# TAB 2: PREDICCIONES EN VIVO
# ============================================================================
with tabs[1]:
    st.header("Predicciones en Vivo desde Kafka")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Ingresa un texto para clasificar")
        user_text = st.text_area("Texto a clasificar:", height=100)
    
    with col2:
        st.write("")
        st.write("")
        predict_button = st.button("🔮 Predecir", use_container_width=True)
    
    if predict_button and user_text:
        try:
            # Cargar predictor
            from ml_prediction import SentimentPredictor
            predictor = SentimentPredictor(MODELS_DIR)
            
            # Realizar predicción
            result = predictor.predict_single(user_text)
            
            # Mostrar resultado
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicción", result['prediction'])
            
            with col2:
                st.metric("Confianza", result['confidence_pct'])
            
            with col3:
                confidence_value = result['confidence']
                color = '🟢' if confidence_value > 0.7 else '🟡' if confidence_value > 0.5 else '🔴'
                st.metric("Nivel", color)
            
            # Gráfico de confianza
            fig = go.Figure(data=[
                go.Bar(
                    x=['Confianza'],
                    y=[confidence_value],
                    marker_color='#3498db',
                    text=f'{confidence_value:.2%}',
                    textposition='auto',
                )
            ])
            fig.update_layout(
                yaxis_range=[0, 1],
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error en predicción: {e}")
    
    # Leer tweets en vivo desde Kafka
    st.divider()
    st.subheader("Tweets en Vivo desde Kafka")
    
    refresh_button = st.button("🔄 Refrescar Tweets", use_container_width=True)
    
    if "live_tweets" not in st.session_state:
        st.session_state["live_tweets"] = []
    
    if refresh_button:
        st.info("Conectando a Kafka...")
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=3000,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            messages = []
            for msg in consumer:
                messages.append(msg.value)
                if len(messages) >= 50:
                    break
            
            consumer.close()
            
            if messages:
                st.session_state["live_tweets"] = messages
                st.success(f"✓ {len(messages)} tweets cargados")
            else:
                st.info("No hay nuevos tweets")
        
        except Exception as e:
            st.error(f"Error conectando a Kafka: {e}")
    
    if st.session_state["live_tweets"]:
        df_live = pd.DataFrame(st.session_state["live_tweets"])
        
        # Mostrar tweets
        for idx, row in df_live.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{row.get('entity', 'N/A')}** - {row.get('sentiment', 'N/A')}")
                    st.write(row.get('tweet_content', 'N/A'))
                with col2:
                    st.write(f"🕐 {row.get('timestamp', 'N/A')[:10]}")
                st.divider()

# ============================================================================
# TAB 3: MÉTRICAS DE RENDIMIENTO
# ============================================================================
with tabs[2]:
    st.header("Métricas de Rendimiento del Modelo")
    
    if model_info:
        metrics = model_info['metrics']
        
        # Mostrar métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        
        st.divider()
        
        # Gráfico de métricas
        metrics_data = {
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Valor': [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score']
            ]
        }
        
        fig = px.bar(
            metrics_data,
            x='Métrica',
            y='Valor',
            title='Métricas de Rendimiento del Modelo',
            color='Valor',
            color_continuous_scale='Viridis',
            text='Valor'
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Información adicional
        st.subheader("Información del Modelo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Modelo:** {model_info['model_name']}")
            st.write(f"**Número de Features:** {model_info['n_features']}")
            st.write(f"**Clases:** {', '.join(model_info['classes'])}")
        
        with col2:
            st.write(f"**Fecha de Entrenamiento:** {model_info['training_date']}")
            st.write(f"**Número de Clases:** {len(model_info['classes'])}")
    
    else:
        st.warning("⚠️ Información del modelo no disponible")

# ============================================================================
# TAB 4: MATRIZ DE CONFUSIÓN
# ============================================================================
with tabs[3]:
    st.header("Matriz de Confusión")
    
    if model_info and 'confusion_matrix' in model_info['metrics']:
        cm = model_info['metrics']['confusion_matrix']
        classes = model_info['classes']
        
        # Crear heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Matriz de Confusión',
            xaxis_title='Predicción',
            yaxis_title='Real',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretación
        st.subheader("Interpretación")
        st.write("""
        - **Diagonal principal (azul oscuro):** Predicciones correctas
        - **Fuera de la diagonal:** Predicciones incorrectas
        - **Intensidad del color:** Número de muestras
        """)
    
    else:
        st.info("Matriz de confusión no disponible")

# ============================================================================
# TAB 5: INFORMACIÓN DEL MODELO
# ============================================================================
with tabs[4]:
    st.header("Información del Modelo")
    
    if model_info:
        st.subheader("Detalles del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tipo de Modelo:**", model_info['model_name'])
            st.write("**Número de Features:**", model_info['n_features'])
            st.write("**Número de Clases:**", len(model_info['classes']))
        
        with col2:
            st.write("**Fecha de Entrenamiento:**", model_info['training_date'])
            st.write("**Clases:**")
            for cls in model_info['classes']:
                st.write(f"  - {cls}")
        
        st.divider()
        
        st.subheader("Métricas de Evaluación")
        
        metrics = model_info['metrics']
        
        metrics_df = pd.DataFrame({
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Valor': [
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1_score']:.4f}"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        st.divider()
        
        st.subheader("Guía de Uso")
        
        st.write("""
        ### Cómo usar el modelo:
        
        1. **Predicción Individual:** Ve a la pestaña "Predicciones en Vivo" e ingresa un texto
        2. **Predicción en Lote:** Usa el script `ml_prediction.py` con un archivo CSV
        3. **Streaming en Tiempo Real:** Ejecuta `spark_ml_streaming.py` para procesar tweets en vivo
        
        ### Interpretación de Resultados:
        
        - **Confianza > 0.7:** Predicción muy confiable
        - **Confianza 0.5-0.7:** Predicción moderada
        - **Confianza < 0.5:** Predicción poco confiable
        
        ### Mejora del Modelo:
        
        - Recolectar más datos de entrenamiento
        - Ajustar hiperparámetros
        - Probar diferentes modelos
        - Usar feature engineering más avanzado
        """)
    
    else:
        st.warning("⚠️ Información del modelo no disponible. Ejecuta ml_training.py primero.")

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>🤖 Dashboard de Machine Learning para Análisis de Sentimientos</p>
    <p>Powered by Streamlit, Spark, Kafka & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
