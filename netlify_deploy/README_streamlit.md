# 🌐 Aplicación Streamlit - Análisis de Curvas IV PVStand

## 🚀 ¿Qué es Streamlit?

Streamlit es una herramienta que convierte scripts de Python en aplicaciones web interactivas. Es perfecto para compartir análisis de datos de forma profesional.

## ✨ Características de la aplicación

- **📊 Gráficos interactivos** con plotly
- **🔍 Filtros dinámicos** por tipo de módulo
- **📋 Tabla de datos** con ordenamiento
- **💾 Descarga de datos** en CSV
- **📱 Responsive** - funciona en móviles
- **🌐 Compartible** - URL pública

## 🛠️ Instalación y uso

### Opción 1: Ejecutar localmente
```bash
# Instalar dependencias
pip install -r requirements_streamlit.txt

# Ejecutar aplicación
streamlit run streamlit_iv_app.py
```

### Opción 2: Usar script automático
```bash
# Ejecutar script (instala dependencias automáticamente)
./run_streamlit_app.sh
```

## 🌍 Compartir la aplicación

### Opción 1: Streamlit Cloud (GRATIS)
1. Sube el código a GitHub
2. Ve a https://share.streamlit.io
3. Conecta tu repositorio
4. ¡Deploy automático!

### Opción 2: ngrok (RÁPIDO)
```bash
# Instalar ngrok
# Descargar de: https://ngrok.com/

# Ejecutar aplicación
streamlit run streamlit_iv_app.py

# En otra terminal, exponer puerto
ngrok http 8501
```

### Opción 3: Heroku/Railway
- Deploy automático desde GitHub
- URL pública permanente

## 📊 Funcionalidades

### 🎯 Gráficos interactivos
- **Curvas I-V y P-V** superpuestas
- **Colores diferenciados**: Azul (Risen), Rojo (Minimódulo)
- **Zoom y pan** interactivos
- **Hover** con información detallada

### 🔍 Filtros y análisis
- **Filtrar por tipo de módulo**
- **Ordenar por parámetros**
- **Métricas en tiempo real**
- **Descarga de datos**

### 📱 Interfaz responsive
- **Funciona en móviles**
- **Navegación intuitiva**
- **Carga rápida**

## 🎨 Ventajas sobre HTML estático

| Característica | HTML | Streamlit |
|----------------|------|-----------|
| **Interactividad** | Básica | Avanzada |
| **Filtros** | No | Sí |
| **Descarga datos** | No | Sí |
| **Responsive** | Limitado | Excelente |
| **Actualización** | Manual | Automática |
| **Compartir** | Archivo | URL |

## 🚀 Comandos útiles

```bash
# Ejecutar en puerto específico
streamlit run streamlit_iv_app.py --server.port 8080

# Ejecutar en modo desarrollo
streamlit run streamlit_iv_app.py --server.runOnSave true

# Ver logs detallados
streamlit run streamlit_iv_app.py --logger.level debug
```

## 📈 Próximos pasos

1. **Ejecutar localmente** para probar
2. **Subir a GitHub** para Streamlit Cloud
3. **Compartir URL** con colegas
4. **Personalizar** interfaz según necesidades

---
*¡Streamlit hace que compartir análisis de datos sea súper fácil!* 🎉
