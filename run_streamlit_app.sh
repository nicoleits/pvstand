#!/bin/bash
# Script para ejecutar la aplicación Streamlit de curvas IV

echo "🚀 Iniciando aplicación Streamlit para curvas IV del PVStand..."
echo ""

# Verificar si streamlit está instalado
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit no está instalado"
    echo "📦 Instalando dependencias..."
    pip install -r requirements_streamlit.txt
    echo ""
fi

# Ejecutar la aplicación
echo "🌐 Iniciando servidor Streamlit..."
echo "📱 La aplicación estará disponible en: http://localhost:8501"
echo "🔗 Para compartir: usa ngrok o similar para exponer el puerto 8501"
echo ""

streamlit run streamlit_iv_app.py --server.port 8501 --server.address 0.0.0.0
