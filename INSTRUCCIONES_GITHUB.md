# 🚀 Instrucciones para subir a GitHub

## 📋 Pasos detallados:

### 1. **Crear repositorio en GitHub**
- Ve a: https://github.com
- Click en "New repository"
- Nombre: `pvstand-iv-analysis`
- Descripción: "Análisis interactivo de curvas IV del PVStand"
- ✅ Marcar como **PÚBLICO** (importante para Streamlit Cloud)
- ❌ NO marcar "Add a README file"
- Click "Create repository"

### 2. **Subir archivos desde terminal**
```bash
# En tu terminal, ejecuta estos comandos:
cd /home/nicole/atamo_pvstand

# Inicializar git
git init

# Agregar archivos
git add streamlit_iv_app.py
git add requirements_streamlit.txt
git add run_app_simple.py
git add README_streamlit.md

# Hacer commit
git commit -m "Aplicación Streamlit para análisis de curvas IV PVStand"

# Conectar con GitHub (reemplaza TU-USUARIO)
git remote add origin https://github.com/TU-USUARIO/pvstand-iv-analysis.git

# Subir archivos
git branch -M main
git push -u origin main
```

### 3. **Deploy en Streamlit Cloud**
- Ve a: https://share.streamlit.io
- Click "New app"
- Conecta tu repositorio GitHub
- Selecciona: `pvstand-iv-analysis`
- Main file path: `streamlit_iv_app.py`
- Click "Deploy"

### 4. **¡Listo!**
- Tu aplicación estará disponible en una URL como:
  `https://pvstand-iv-analysis.streamlit.app`
- Comparte esta URL con quien quieras

## 🎯 **Archivos que se suben:**
- `streamlit_iv_app.py` - Aplicación principal
- `requirements_streamlit.txt` - Dependencias
- `run_app_simple.py` - Script de ejecución local
- `README_streamlit.md` - Documentación

## ✅ **Ventajas del deploy:**
- 🌐 **URL pública** permanente
- 🔄 **Actualización automática** cuando cambies el código
- 📱 **Funciona en móviles**
- 🚀 **Sin configuración** del servidor
- 💰 **Gratis** para repositorios públicos
