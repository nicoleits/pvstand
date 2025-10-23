#!/usr/bin/env python3
"""
Aplicación Streamlit para visualización de curvas IV del PVStand
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Agregar el directorio raíz del proyecto al path de Python
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pvstand.analysis.pvstand_iv_processor import process_pvstand_iv_files

def load_iv_data():
    """Carga los datos de curvas IV"""
    try:
        # Procesar archivos si no existen los datos
        data_dir = os.path.join(project_root, "pvstand", "datos")
        output_dir = os.path.join(project_root, "pvstand", "datos_procesados_analisis_integrado_py", "iv_curves")
        
        if not os.path.exists(os.path.join(output_dir, "iv_analysis.csv")):
            st.info("Procesando datos de curvas IV...")
            results = process_pvstand_iv_files(data_dir=data_dir, output_dir=output_dir)
            if not results:
                st.error("Error procesando datos")
                return None
        
        # Cargar datos procesados
        analysis_file = os.path.join(output_dir, "iv_analysis.csv")
        df_analysis = pd.read_csv(analysis_file)
        
        return df_analysis
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

def load_real_iv_data():
    """Carga los datos reales de las curvas IV"""
    try:
        # Buscar archivos de datos procesados
        data_dir = os.path.join(project_root, "pvstand", "datos")
        output_dir = os.path.join(project_root, "pvstand", "datos_procesados_analisis_integrado_py", "iv_curves")
        
        # Procesar archivos si no existen
        if not os.path.exists(os.path.join(output_dir, "iv_analysis.csv")):
            from pvstand.analysis.pvstand_iv_processor import process_pvstand_iv_files
            results = process_pvstand_iv_files(data_dir=data_dir, output_dir=output_dir)
            if not results:
                return None
        
        # Cargar datos reales de curvas IV
        real_curves = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                try:
                    # Leer archivo real
                    with open(filepath, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
                    
                    # Extraer datos de curva IV (desde línea 24)
                    iv_data = []
                    for line in lines[23:]:  # Desde línea 24
                        if line.strip():
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                try:
                                    voltage = float(parts[0])
                                    current = float(parts[1])
                                    power = float(parts[2])
                                    iv_data.append([voltage, current, power])
                                except ValueError:
                                    continue
                    
                    if iv_data:
                        st.write(f"Archivo: {filename}")
                        st.write(f"Hora: {time_str} → {module_category}")
                        st.write(f"Puntos IV: {len(iv_data)}")

                        st.write(iv_data)

                        # Determinar tipo de módulo por hora
                        try:
                            st.write(f"Primera línea: {lines[1]}")
                            time_str = lines[1].split('\t')[1].strip()
                        except Exception as e:
                            st.warning(f"No se pudo extraer hora de {filename}: {e}")
                            time_str = ""
                        if time_str >= "14:30:00" and time_str <= "15:05:00":
                            module_category = "Minimódulo"
                            color = "red"
                        else:
                            module_category = "Módulo Risen"
                            color = "blue"
                        
                        real_curves.append({
                            'filename': filename,
                            'time': time_str,
                            'module_category': module_category,
                            'color': color,
                            'iv_data': np.array(iv_data)
                        })
                        
                except Exception as e:
                    continue
        
        return real_curves
    except Exception as e:
        st.error(f"Error cargando datos reales: {e}")
        return None

def create_interactive_plot(df_analysis):
    """Crea el gráfico interactivo con plotly usando datos reales"""
    
    # Cargar datos reales de curvas IV
    real_curves = load_real_iv_data()
    
    if not real_curves:
        st.error("No se pudieron cargar los datos reales de las curvas IV")
        return None
    
    # Separar curvas por tipo de módulo
    grouped_curves = {
        'Minimódulo': [],
        'Módulo Risen': []
    }
    for curve in real_curves:
        grouped_curves[curve['module_category']].append(curve)

    for module_type, curves in grouped_curves.items():
        if not curves:
            st.warning(f"No hay curvas para el tipo de módulo: {module_type}")
            continue

        st.subheader(f"Curvas {module_type}")

        # Crear subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Curvas I-V -{module_type}", f"Curvas P-V -{module_type}"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        for curve in curves:
            iv_data = curve['iv_data']
            voltage = iv_data[:, 0]
            current = iv_data[:, 1]
            power = iv_data[:, 2]
            name = f"{module_type} {curve['time']}"
            
            # Curva I-V
            fig.add_trace(
                go.Scatter(
                    x=voltage, y=current,
                    mode='lines',
                    name=name,
                    line=dict(color=curve['color'], width=2),
                    hovertemplate=f'<b>{name}</b><br>Voltaje: %{{x:.2f}} V<br>Corriente: %{{y:.2f}} A<extra></extra>'
                ),
                row=1, col=1
            )
        
            # Curva P-V
            fig.add_trace(
                go.Scatter(
                    x=voltage, y=power,
                    mode='lines',
                    name=name,
                    line=dict(color=curve['color'], width=2),
                    hovertemplate=f'<b>{name}</b><br>Voltaje: %{{x:.2f}} V<br>Potencia: %{{y:.2f}} W<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Actualizar layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text=f"Curvas IV/PV - PVStand - {module_type}",
        title_x=0.5,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="Voltaje [V]", row=1, col=1)
    fig.update_yaxes(title_text="Corriente [A]", row=1, col=1)
    fig.update_xaxes(title_text="Voltaje [V]", row=1, col=2)
    fig.update_yaxes(title_text="Potencia [W]", row=1, col=2)
    
    st.plotly_chart(fig, width="stretch")

def main():
    """Función principal de la aplicación"""
    
    # Configurar página
    st.set_page_config(
        page_title="Análisis PVStand - Curvas IV",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título principal
    st.title("☀️ Análisis de Curvas IV - PVStand")
    st.markdown("---")
    
    # Sidebar con información
    with st.sidebar:
        st.header("📊 Información del Análisis")
        st.markdown("""
        **Módulos analizados:**
        - 🔵 **Módulo Risen**: 13:30-14:05
        - 🔴 **Minimódulo**: 14:30-15:00
        
        **Parámetros calculados:**
        - Pmax, Vmp, Imp
        - Isc, Voc
        - Factor de llenado
        - Eficiencia
        """)
    
    # Cargar datos
    with st.spinner("Cargando datos de curvas IV..."):
        df_analysis = load_iv_data()
    
    if df_analysis is None:
        st.error("No se pudieron cargar los datos")
        return
    
    # Mostrar resumen de datos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Mediciones", len(df_analysis))
    
    with col2:
        risen_count = len(df_analysis[df_analysis['Module_Category'] == 'Módulo Risen'])
        st.metric("🔵 Módulo Risen", risen_count)
    
    with col3:
        minimodule_count = len(df_analysis[df_analysis['Module_Category'] == 'Minimódulo'])
        st.metric("🔴 Minimódulo", minimodule_count)
    
    with col4:
        avg_irradiance = df_analysis['Irradiance_W_m2'].mean()
        st.metric("☀️ Irradiación Promedio", f"{avg_irradiance:.0f} W/m²")
    
    st.markdown("---")
    
    # Gráfico interactivo
    st.header("📊 Curvas IV Interactivas (Datos Reales)")
    
    # Mostrar información sobre las curvas reales
    real_curves = load_real_iv_data()
    if real_curves:
        st.info(f"✅ Cargadas {len(real_curves)} curvas reales de los archivos de datos")
        
        # Mostrar resumen de curvas
        col1, col2 = st.columns(2)
        with col1:
            risen_count = len([c for c in real_curves if c['module_category'] == 'Módulo Risen'])
            st.metric("🔵 Curvas Risen", risen_count)
        with col2:
            minimodule_count = len([c for c in real_curves if c['module_category'] == 'Minimódulo'])
            st.metric("🔴 Curvas Minimódulo", minimodule_count)
    
    create_interactive_plot(df_analysis)

    
    # Tabla de datos
    st.header("📋 Datos del Análisis")
    
    # Filtros
    col1, col2 = st.columns(2)
    
    with col1:
        module_filter = st.selectbox(
            "Filtrar por tipo de módulo:",
            ["Todos"] + list(df_analysis['Module_Category'].unique())
        )
    
    with col2:
        sort_by = st.selectbox(
            "Ordenar por:",
            ["Time", "Pmax_W", "Efficiency_%", "Irradiance_W_m2"]
        )
    
    # Aplicar filtros
    filtered_df = df_analysis.copy()
    if module_filter != "Todos":
        filtered_df = filtered_df[filtered_df['Module_Category'] == module_filter]
    
    filtered_df = filtered_df.sort_values(by=sort_by)
    
    # Mostrar tabla
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Descargar datos
    st.header("💾 Descargar Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv_data,
            file_name="iv_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
        st.info("💡 Los datos incluyen todos los parámetros calculados de las curvas IV")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Generado automáticamente por el sistema de análisis PVStand</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
