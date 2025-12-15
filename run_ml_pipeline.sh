#!/bin/bash

# Script para ejecutar el pipeline completo de Machine Learning
# Uso: bash run_ml_pipeline.sh [opción]

set -e

echo "=========================================="
echo "🤖 PIPELINE DE MACHINE LEARNING"
echo "=========================================="
echo ""

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con color
print_step() {
    echo -e "${BLUE}[PASO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Verificar que Python está instalado
if ! command -v python &> /dev/null; then
    print_error "Python no está instalado"
    exit 1
fi

print_success "Python encontrado: $(python --version)"

# Opción por defecto
OPTION=${1:-"all"}

case $OPTION in
    
    "install")
        print_step "Instalando dependencias..."
        pip install -r requirements_ml.txt
        print_success "Dependencias instaladas"
        ;;
    
    "train")
        print_step "Entrenando modelos..."
        python ml_training.py
        print_success "Modelos entrenados"
        ;;
    
    "predict")
        print_step "Realizando predicciones..."
        if [ -z "$2" ]; then
            print_warning "Uso: bash run_ml_pipeline.sh predict <archivo_csv>"
            echo "Ejemplo: bash run_ml_pipeline.sh predict data/test_synthetic_tweets.csv"
        else
            python ml_prediction.py --input "$2" --output results/predictions.csv --evaluate
            print_success "Predicciones completadas"
        fi
        ;;
    
    "stream")
        print_step "Iniciando Spark ML Streaming..."
        python spark_ml_streaming.py
        ;;
    
    "dashboard")
        print_step "Iniciando Dashboard..."
        streamlit run app-dashboard/streamlit_app_ml.py
        ;;
    
    "all")
        print_step "Ejecutando pipeline completo..."
        
        # 1. Instalar dependencias
        print_step "1/4 Instalando dependencias..."
        pip install -r requirements_ml.txt
        print_success "Dependencias instaladas"
        
        # 2. Entrenar modelos
        print_step "2/4 Entrenando modelos..."
        python ml_training.py
        print_success "Modelos entrenados"
        
        # 3. Hacer predicciones de prueba
        print_step "3/4 Realizando predicciones de prueba..."
        if [ -f "data/test_synthetic_tweets.csv" ]; then
            python ml_prediction.py --input data/test_synthetic_tweets.csv --output results/predictions.csv --evaluate
            print_success "Predicciones completadas"
        else
            print_warning "Archivo de prueba no encontrado: data/test_synthetic_tweets.csv"
        fi
        
        # 4. Información final
        print_step "4/4 Pipeline completado"
        echo ""
        echo -e "${GREEN}=========================================="
        echo "✓ PIPELINE COMPLETADO EXITOSAMENTE"
        echo "==========================================${NC}"
        echo ""
        echo "Próximos pasos:"
        echo "  1. Iniciar Spark ML Streaming:"
        echo "     python spark_ml_streaming.py"
        echo ""
        echo "  2. Abrir Dashboard:"
        echo "     streamlit run app-dashboard/streamlit_app_ml.py"
        echo ""
        echo "  3. Hacer predicciones individuales:"
        echo "     python ml_prediction.py --text 'Tu texto aquí'"
        echo ""
        ;;
    
    "help")
        echo "Uso: bash run_ml_pipeline.sh [opción]"
        echo ""
        echo "Opciones:"
        echo "  install    - Instalar dependencias"
        echo "  train      - Entrenar modelos"
        echo "  predict    - Realizar predicciones (requiere archivo CSV)"
        echo "  stream     - Iniciar Spark ML Streaming"
        echo "  dashboard  - Abrir Dashboard Streamlit"
        echo "  all        - Ejecutar pipeline completo"
        echo "  help       - Mostrar esta ayuda"
        echo ""
        echo "Ejemplos:"
        echo "  bash run_ml_pipeline.sh train"
        echo "  bash run_ml_pipeline.sh predict data/test_synthetic_tweets.csv"
        echo "  bash run_ml_pipeline.sh all"
        ;;
    
    *)
        print_error "Opción desconocida: $OPTION"
        echo "Usa 'bash run_ml_pipeline.sh help' para ver las opciones disponibles"
        exit 1
        ;;

esac

echo ""
