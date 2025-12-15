"""
Script de verificación del setup de Machine Learning
Verifica que todas las dependencias y archivos necesarios estén disponibles
"""

import os
import sys
import importlib
from pathlib import Path

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def print_error(text):
    print(f"{Colors.RED}✗{Colors.END} {text}")

def print_warning(text):
    print(f"{Colors.YELLOW}!{Colors.END} {text}")

def check_python_version():
    """Verifica versión de Python"""
    print_header("Verificando Python")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} - Se requiere 3.8+")
        return False

def check_dependencies():
    """Verifica dependencias de Python"""
    print_header("Verificando Dependencias")
    
    required_packages = {
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'joblib': 'joblib',
        'nltk': 'nltk',
        'pyspark': 'pyspark',
        'kafka': 'kafka-python',
        'streamlit': 'streamlit'
    }
    
    all_ok = True
    
    for import_name, package_name in required_packages.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{package_name} ({version})")
        except ImportError:
            print_error(f"{package_name} - NO INSTALADO")
            all_ok = False
    
    return all_ok

def check_data_files():
    """Verifica archivos de datos"""
    print_header("Verificando Archivos de Datos")
    
    required_files = {
        'data/twitter_training.csv': 'Datos de entrenamiento',
        'data/twitter_validation.csv': 'Datos de validación',
        'data/synthetic_tweets.csv': 'Tweets sintéticos',
        'data/test_synthetic_tweets.csv': 'Tweets de prueba'
    }
    
    all_ok = True
    
    for filepath, description in required_files.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            print_success(f"{description} ({size_mb:.2f} MB)")
        else:
            print_warning(f"{description} - NO ENCONTRADO: {filepath}")
            all_ok = False
    
    return all_ok

def check_directories():
    """Verifica directorios necesarios"""
    print_header("Verificando Directorios")
    
    required_dirs = {
        'models': 'Modelos entrenados',
        'results': 'Resultados',
        'data': 'Datos',
        'app-dashboard': 'Dashboard'
    }
    
    all_ok = True
    
    for dirname, description in required_dirs.items():
        if os.path.isdir(dirname):
            print_success(f"{description} ({dirname})")
        else:
            print_warning(f"{description} - NO ENCONTRADO: {dirname}")
            all_ok = False
    
    return all_ok

def check_model_files():
    """Verifica archivos de modelos entrenados"""
    print_header("Verificando Modelos Entrenados")
    
    model_files = {
        'models/model_sentiment.pkl': 'Modelo de sentimientos',
        'models/vectorizer_tfidf.pkl': 'Vectorizador TF-IDF',
        'models/label_encoder.pkl': 'Codificador de etiquetas',
        'models/model_info.pkl': 'Información del modelo'
    }
    
    all_ok = True
    
    for filepath, description in model_files.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_kb = size / 1024
            print_success(f"{description} ({size_kb:.2f} KB)")
        else:
            print_warning(f"{description} - NO ENCONTRADO: {filepath}")
            all_ok = False
    
    if not all_ok:
        print_warning("Los modelos no están entrenados. Ejecuta: python ml_training.py")
    
    return all_ok

def check_python_scripts():
    """Verifica scripts de Python"""
    print_header("Verificando Scripts de Python")
    
    required_scripts = {
        'ml_training.py': 'Entrenamiento de modelos',
        'ml_prediction.py': 'Predicciones',
        'spark_ml_streaming.py': 'Spark ML Streaming',
        'app-dashboard/streamlit_app_ml.py': 'Dashboard ML'
    }
    
    all_ok = True
    
    for filepath, description in required_scripts.items():
        if os.path.exists(filepath):
            print_success(f"{description} ({filepath})")
        else:
            print_error(f"{description} - NO ENCONTRADO: {filepath}")
            all_ok = False
    
    return all_ok

def check_configuration():
    """Verifica configuración"""
    print_header("Verificando Configuración")
    
    # Verificar variables de entorno
    env_vars = {
        'KAFKA_BROKER': 'kafka:9092',
        'KAFKA_TOPIC': 'raw_tweets',
        'HDFS_OUTPUT_PATH': 'hdfs://namenode:9000/user/sentiment_analysis/ml_predictions'
    }
    
    for var, default in env_vars.items():
        value = os.getenv(var, default)
        print_success(f"{var} = {value}")

def generate_report():
    """Genera reporte de verificación"""
    print_header("Generando Reporte")
    
    results = {
        'Python': check_python_version(),
        'Dependencias': check_dependencies(),
        'Archivos de Datos': check_data_files(),
        'Directorios': check_directories(),
        'Scripts': check_python_scripts(),
        'Modelos': check_model_files()
    }
    
    check_configuration()
    
    # Resumen
    print_header("RESUMEN")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for check, result in results.items():
        status = f"{Colors.GREEN}✓ PASÓ{Colors.END}" if result else f"{Colors.RED}✗ FALLÓ{Colors.END}"
        print(f"{check}: {status}")
    
    print(f"\n{Colors.BLUE}Resultado: {passed}/{total} verificaciones pasadas{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{'='*60}")
        print("✓ SETUP COMPLETAMENTE VERIFICADO")
        print(f"{'='*60}{Colors.END}")
        print("\nPróximos pasos:")
        print("1. Entrenar modelos: python ml_training.py")
        print("2. Hacer predicciones: python ml_prediction.py --text 'Tu texto'")
        print("3. Iniciar streaming: python spark_ml_streaming.py")
        print("4. Abrir dashboard: streamlit run app-dashboard/streamlit_app_ml.py")
        return True
    else:
        print(f"\n{Colors.RED}{'='*60}")
        print("✗ ALGUNAS VERIFICACIONES FALLARON")
        print(f"{'='*60}{Colors.END}")
        print("\nAcciones recomendadas:")
        if not results['Dependencias']:
            print("- Instalar dependencias: pip install -r requirements_ml.txt")
        if not results['Archivos de Datos']:
            print("- Verificar que los archivos de datos existan en data/")
        if not results['Modelos']:
            print("- Entrenar modelos: python ml_training.py")
        return False

def main():
    """Función principal"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("🤖 VERIFICACIÓN DE SETUP DE MACHINE LEARNING")
    print(f"{'='*60}{Colors.END}\n")
    
    success = generate_report()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
