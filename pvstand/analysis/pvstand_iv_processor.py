# analysis/pvstand_iv_processor.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pathlib import Path
import glob

# Agregar el directorio raíz del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.logging_config import logger
from config import paths, settings

def process_pvstand_iv_files(data_dir=None, output_dir=None):
    """
    Función principal para procesar archivos de curvas IV del PVStand.
    
    Args:
        data_dir: Directorio con archivos de datos IV (opcional)
        output_dir: Directorio de salida (opcional)
    
    Returns:
        dict: Resultados del procesamiento
    """
    logger.info("=== INICIO DEL PROCESAMIENTO DE CURVAS IV PVSTAND ===")
    
    # Configurar directorios
    if data_dir is None:
        data_dir = paths.BASE_INPUT_DIR
    if output_dir is None:
        output_dir = os.path.join(paths.BASE_OUTPUT_CSV_DIR, "iv_curves")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar archivos
    iv_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not iv_files:
        logger.error(f"No se encontraron archivos .txt en {data_dir}")
        return None
    
    logger.info(f"Encontrados {len(iv_files)} archivos de datos IV")
    
    # Procesar archivos
    processed_curves = []
    metadata_list = []
    
    for file_path in sorted(iv_files):
        logger.info(f"Procesando: {os.path.basename(file_path)}")
        
        result = parse_single_iv_file(file_path)
        if result is not None:
            processed_curves.append(result)
            metadata_list.append(result['metadata'])
    
    if not processed_curves:
        logger.error("No se pudieron procesar archivos de datos IV")
        return None
    
    # Generar análisis y reportes
    analysis_results = generate_iv_analysis(processed_curves, output_dir)
    generate_iv_plots(processed_curves, output_dir)
    generate_iv_reports(processed_curves, metadata_list, output_dir)
    
    logger.info("=== FIN DEL PROCESAMIENTO DE CURVAS IV PVSTAND ===")
    
    return {
        'curves': processed_curves,
        'metadata': metadata_list,
        'analysis': analysis_results
    }

def parse_single_iv_file(filepath):
    """
    Parsea un archivo individual de datos IV.
    """
    try:
        # Intentar diferentes codificaciones
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        lines = None
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        if lines is None:
            raise UnicodeDecodeError("No se pudo decodificar el archivo con ninguna codificación")
        
        # Extraer metadatos
        metadata = extract_metadata(lines)
        
        # Extraer datos de curva IV
        iv_data = extract_iv_data(lines)
        
        # Calcular parámetros característicos
        characteristics = calculate_iv_characteristics(iv_data, metadata)
        
        return {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'metadata': metadata,
            'iv_data': iv_data,
            'characteristics': characteristics
        }
        
    except Exception as e:
        logger.error(f"Error procesando archivo {filepath}: {e}")
        return None

def extract_metadata(lines):
    """
    Extrae metadatos del archivo.
    """
    metadata = {}
    
    try:
        # Información básica (línea 2)
        if len(lines) >= 2:
            date_time_line = lines[1].strip().split('\t')
            if len(date_time_line) >= 6:
                metadata.update({
                    'date': date_time_line[0],
                    'time': date_time_line[1],
                    'module': date_time_line[2],
                    'serial': date_time_line[3],
                    'module_type': date_time_line[4],
                    'area': float(date_time_line[5]) if date_time_line[5] else 0
                })
                
                # Identificar tipo de módulo basado en la hora
                time_str = date_time_line[1]
                if time_str >= "14:30:00" and time_str <= "15:00:00":
                    metadata['module_category'] = 'Minimódulo'
                    metadata['module_color'] = 'red'
                else:
                    metadata['module_category'] = 'Módulo Risen'
                    metadata['module_color'] = 'blue'
        
        # Irradiación (línea 5) - formato: COM1 301 Pyr 0.000009100 797.188462
        if len(lines) >= 5:
            irrad_line = lines[4].strip().split()
            if len(irrad_line) >= 4:
                # El valor de irradiación es el último número en la línea
                metadata['irradiance'] = float(irrad_line[-1])
        
        # Temperaturas de celdas monitor (líneas 9-10)
        for i in range(9, 11):
            if i < len(lines) and lines[i].strip():
                temp_line = lines[i].strip().split()
                if len(temp_line) >= 6:
                    cell_name = temp_line[1]
                    temp_value = float(temp_line[5])  # La temperatura está en la columna 5
                    metadata[f'temp_{cell_name}'] = temp_value
        
        # Parámetros principales (línea 22)
        if len(lines) >= 22:
            params_line = lines[21].strip().split('\t')
            if len(params_line) >= 17:
                metadata.update({
                    'Pmax_raw': float(params_line[0]),
                    'Imax_raw': float(params_line[1]),
                    'Umax_raw': float(params_line[2]),
                    'I_at_pmax': float(params_line[3]),
                    'U_at_pmax': float(params_line[4]),
                    'FF_raw': float(params_line[5]),
                    'Eta_raw': float(params_line[6]),
                    'E0_pyr': float(params_line[7]),
                    'MPPFit': float(params_line[8]),
                    'IscFit': float(params_line[9]),
                    'UocFit': float(params_line[10]),
                    'EtaFit': float(params_line[11]),
                    'ImppFit': float(params_line[12]),
                    'UmppFit': float(params_line[13]),
                    'FF_fit': float(params_line[14]),
                    'MSE_MPPFit': float(params_line[15]),
                    'Irradiation_Fluctuation': float(params_line[16])
                })
                
                # Temperaturas adicionales
                if len(params_line) >= 21:
                    metadata.update({
                        'temp_015_2017': float(params_line[18]),
                        'temp_019_2017': float(params_line[19]),
                        'temp_015_2017_actual': float(params_line[20]),
                        'temp_019_2017_actual': float(params_line[21])
                    })
        
    except Exception as e:
        logger.warning(f"Error extrayendo metadatos: {e}")
    
    return metadata

def extract_iv_data(lines):
    """
    Extrae datos de la curva IV.
    """
    iv_data = []
    
    try:
        # Los datos de curva IV empiezan en la línea 24 (índice 23)
        for i in range(23, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 3:
                    try:
                        voltage = float(parts[0])
                        current = float(parts[1])
                        power = float(parts[2])
                        iv_data.append([voltage, current, power])
                    except ValueError:
                        continue
        
        # Crear DataFrame
        if iv_data:
            df_iv = pd.DataFrame(iv_data, columns=['Voltage_V', 'Current_A', 'Power_W'])
            # Calcular resistencia (evitar división por cero)
            df_iv['Resistance_Ohm'] = np.where(
                df_iv['Current_A'] != 0,
                df_iv['Voltage_V'] / df_iv['Current_A'],
                np.nan
            )
        else:
            df_iv = pd.DataFrame(columns=['Voltage_V', 'Current_A', 'Power_W', 'Resistance_Ohm'])
        
    except Exception as e:
        logger.warning(f"Error extrayendo datos IV: {e}")
        df_iv = pd.DataFrame(columns=['Voltage_V', 'Current_A', 'Power_W', 'Resistance_Ohm'])
    
    return df_iv

def calculate_iv_characteristics(df_iv, metadata):
    """
    Calcula características de la curva IV.
    """
    characteristics = {}
    
    if df_iv.empty:
        return characteristics
    
    try:
        # Punto de máxima potencia
        max_power_idx = df_iv['Power_W'].idxmax()
        max_power_point = df_iv.loc[max_power_idx]
        
        characteristics.update({
            'Pmax': max_power_point['Power_W'],
            'Vmp': max_power_point['Voltage_V'],
            'Imp': max_power_point['Current_A']
        })
        
        # Corriente de cortocircuito (Isc) - corriente cuando V < 1V
        isc_data = df_iv[df_iv['Voltage_V'] < 1.0]
        isc = isc_data['Current_A'].max() if not isc_data.empty else np.nan
        characteristics['Isc'] = isc
        
        # Voltaje de circuito abierto (Voc) - voltaje cuando I < 0.1A
        voc_data = df_iv[df_iv['Current_A'] < 0.1]
        voc = voc_data['Voltage_V'].max() if not voc_data.empty else np.nan
        characteristics['Voc'] = voc
        
        # Factor de llenado
        if not np.isnan(isc) and not np.isnan(voc) and isc > 0 and voc > 0:
            ff = max_power_point['Power_W'] / (isc * voc)
        else:
            ff = np.nan
        characteristics['FF'] = ff
        
        # Eficiencia
        area = metadata.get('area', 0)
        irradiance = metadata.get('irradiance', 0)
        if area > 0 and irradiance > 0:
            efficiency = (max_power_point['Power_W'] / (area * irradiance)) * 100
        else:
            efficiency = np.nan
        characteristics['Efficiency'] = efficiency
        
        # Temperatura promedio
        temp_015 = metadata.get('temp_015_2017', np.nan)
        temp_019 = metadata.get('temp_019_2017', np.nan)
        if not np.isnan(temp_015) and not np.isnan(temp_019):
            characteristics['Avg_Temperature'] = (temp_015 + temp_019) / 2
        elif not np.isnan(temp_015):
            characteristics['Avg_Temperature'] = temp_015
        elif not np.isnan(temp_019):
            characteristics['Avg_Temperature'] = temp_019
        else:
            characteristics['Avg_Temperature'] = np.nan
        
    except Exception as e:
        logger.warning(f"Error calculando características: {e}")
    
    return characteristics

def generate_iv_analysis(processed_curves, output_dir):
    """
    Genera análisis de las curvas procesadas.
    """
    logger.info("Generando análisis de curvas IV...")
    
    # Crear DataFrame con análisis
    analysis_data = []
    for curve in processed_curves:
        row = {
            'Filename': curve['filename'],
            'Date': curve['metadata'].get('date', ''),
            'Time': curve['metadata'].get('time', ''),
            'Module': curve['metadata'].get('module', ''),
            'Module_Category': curve['metadata'].get('module_category', ''),
            'Irradiance_W_m2': curve['metadata'].get('irradiance', np.nan),
            'Temperature_C': curve['characteristics'].get('Avg_Temperature', np.nan),
            'Pmax_W': curve['characteristics'].get('Pmax', np.nan),
            'Vmp_V': curve['characteristics'].get('Vmp', np.nan),
            'Imp_A': curve['characteristics'].get('Imp', np.nan),
            'Isc_A': curve['characteristics'].get('Isc', np.nan),
            'Voc_V': curve['characteristics'].get('Voc', np.nan),
            'FF': curve['characteristics'].get('FF', np.nan),
            'Efficiency_%': curve['characteristics'].get('Efficiency', np.nan)
        }
        analysis_data.append(row)
    
    df_analysis = pd.DataFrame(analysis_data)
    
    # Guardar análisis
    analysis_file = os.path.join(output_dir, "iv_analysis.csv")
    df_analysis.to_csv(analysis_file, index=False)
    logger.info(f"Análisis guardado en: {analysis_file}")
    
    return df_analysis

def generate_iv_plots(processed_curves, output_dir):
    """
    Genera gráficos de las curvas IV.
    """
    logger.info("Generando gráficos de curvas IV...")
    
    if not processed_curves:
        logger.warning("No hay curvas procesadas para graficar")
        return
    
    # Separar curvas por tipo de módulo
    risen_curves = []
    minimodule_curves = []
    
    for curve in processed_curves:
        metadata = curve['metadata']
        if metadata.get('module_category') == 'Minimódulo':
            minimodule_curves.append(curve)
        else:
            risen_curves.append(curve)
    
    # Gráfico 1: Curvas I-V separadas
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Graficar curvas Risen
    for i, curve in enumerate(risen_curves):
        df_iv = curve['iv_data']
        metadata = curve['metadata']
        
        if df_iv.empty:
            continue
        
        time_str = metadata.get('time', '')
        irradiance = metadata.get('irradiance', 0)
        label = f"Risen {time_str} - {irradiance:.0f} W/m²"
        
        ax.plot(df_iv['Voltage_V'], df_iv['Current_A'], 
                color='blue', alpha=0.7, linewidth=2, label=label)
    
    # Graficar curvas Minimódulo
    for i, curve in enumerate(minimodule_curves):
        df_iv = curve['iv_data']
        metadata = curve['metadata']
        
        if df_iv.empty:
            continue
        
        time_str = metadata.get('time', '')
        irradiance = metadata.get('irradiance', 0)
        label = f"Minimódulo {time_str} - {irradiance:.0f} W/m²"
        
        ax.plot(df_iv['Voltage_V'], df_iv['Current_A'], 
                color='red', alpha=0.7, linewidth=2, label=label)
    
    ax.set_xlabel('Voltaje [V]', fontsize=12)
    ax.set_ylabel('Corriente [A]', fontsize=12)
    ax.set_title('Curvas I-V Comparativas', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Guardar gráfico I-V
    iv_plot_file = os.path.join(output_dir, "iv_curves_i_v.png")
    plt.savefig(iv_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico I-V guardado en: {iv_plot_file}")
    
    # Gráfico 2: Curvas P-V separadas
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Graficar curvas Risen
    for i, curve in enumerate(risen_curves):
        df_iv = curve['iv_data']
        metadata = curve['metadata']
        
        if df_iv.empty:
            continue
        
        time_str = metadata.get('time', '')
        irradiance = metadata.get('irradiance', 0)
        label = f"Risen {time_str} - {irradiance:.0f} W/m²"
        
        ax.plot(df_iv['Voltage_V'], df_iv['Power_W'], 
                color='blue', alpha=0.7, linewidth=2, label=label)
    
    # Graficar curvas Minimódulo
    for i, curve in enumerate(minimodule_curves):
        df_iv = curve['iv_data']
        metadata = curve['metadata']
        
        if df_iv.empty:
            continue
        
        time_str = metadata.get('time', '')
        irradiance = metadata.get('irradiance', 0)
        label = f"Minimódulo {time_str} - {irradiance:.0f} W/m²"
        
        ax.plot(df_iv['Voltage_V'], df_iv['Power_W'], 
                color='red', alpha=0.7, linewidth=2, label=label)
    
    ax.set_xlabel('Voltaje [V]', fontsize=12)
    ax.set_ylabel('Potencia [W]', fontsize=12)
    ax.set_title('Curvas P-V Comparativas', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Guardar gráfico P-V
    pv_plot_file = os.path.join(output_dir, "iv_curves_p_v.png")
    plt.savefig(pv_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico P-V guardado en: {pv_plot_file}")
    
    # Generar gráficos interactivos
    generate_interactive_plots(processed_curves, output_dir)

def generate_interactive_plots(processed_curves, output_dir):
    """
    Genera gráficos interactivos usando plotly.
    """
    logger.info("Generando gráficos interactivos...")
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
        
        # Separar curvas por tipo de módulo
        risen_curves = []
        minimodule_curves = []
        
        for curve in processed_curves:
            metadata = curve['metadata']
            if metadata.get('module_category') == 'Minimódulo':
                minimodule_curves.append(curve)
            else:
                risen_curves.append(curve)
        
        # Crear subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Curvas I-V Interactivas', 'Curvas P-V Interactivas'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Agregar curvas Risen
        for i, curve in enumerate(risen_curves):
            df_iv = curve['iv_data']
            metadata = curve['metadata']
            
            if df_iv.empty:
                continue
            
            time_str = metadata.get('time', '')
            irradiance = metadata.get('irradiance', 0)
            name = f"Risen {time_str} - {irradiance:.0f} W/m²"
            
            # Curva I-V
            fig.add_trace(
                go.Scatter(
                    x=df_iv['Voltage_V'], 
                    y=df_iv['Current_A'],
                    mode='lines',
                    name=name,
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Voltaje: %{x:.2f} V<br>' +
                                'Corriente: %{y:.2f} A<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Curva P-V
            fig.add_trace(
                go.Scatter(
                    x=df_iv['Voltage_V'], 
                    y=df_iv['Power_W'],
                    mode='lines',
                    name=name,
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Voltaje: %{x:.2f} V<br>' +
                                'Potencia: %{y:.2f} W<br>' +
                                '<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Agregar curvas Minimódulo
        for i, curve in enumerate(minimodule_curves):
            df_iv = curve['iv_data']
            metadata = curve['metadata']
            
            if df_iv.empty:
                continue
            
            time_str = metadata.get('time', '')
            irradiance = metadata.get('irradiance', 0)
            name = f"Minimódulo {time_str} - {irradiance:.0f} W/m²"
            
            # Curva I-V
            fig.add_trace(
                go.Scatter(
                    x=df_iv['Voltage_V'], 
                    y=df_iv['Current_A'],
                    mode='lines',
                    name=name,
                    line=dict(color='red', width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Voltaje: %{x:.2f} V<br>' +
                                'Corriente: %{y:.2f} A<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Curva P-V
            fig.add_trace(
                go.Scatter(
                    x=df_iv['Voltage_V'], 
                    y=df_iv['Power_W'],
                    mode='lines',
                    name=name,
                    line=dict(color='red', width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Voltaje: %{x:.2f} V<br>' +
                                'Potencia: %{y:.2f} W<br>' +
                                '<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Actualizar layout
        fig.update_layout(
            title_text="Curvas IV Interactivas - Comparación de Módulos",
            title_x=0.5,
            width=1400,
            height=600,
            showlegend=True,
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
        
        # Guardar gráfico interactivo
        interactive_file = os.path.join(output_dir, "iv_curves_interactive.html")
        pyo.plot(fig, filename=interactive_file, auto_open=False)
        logger.info(f"Gráfico interactivo guardado en: {interactive_file}")
        
    except ImportError:
        logger.warning("plotly no está disponible. No se pudo generar gráficos interactivos.")
        logger.info("Para instalar plotly: pip install plotly")
    except Exception as e:
        logger.error(f"Error generando gráficos interactivos: {e}")

def generate_iv_reports(processed_curves, metadata_list, output_dir):
    """
    Genera reportes consolidados.
    """
    logger.info("Generando reportes consolidados...")
    
    try:
        import openpyxl
        
        excel_file = os.path.join(output_dir, "iv_curves_report.xlsx")
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Hoja 1: Metadatos
            df_metadata = pd.DataFrame(metadata_list)
            df_metadata.to_excel(writer, sheet_name='Metadatos', index=False)
            
            # Hoja 2: Análisis de parámetros
            analysis_data = []
            for curve in processed_curves:
                row = {
                    'Archivo': curve['filename'],
                    'Fecha': curve['metadata'].get('date', ''),
                    'Hora': curve['metadata'].get('time', ''),
                    'Modulo': curve['metadata'].get('module', ''),
                    'Irradiacion_W_m2': curve['metadata'].get('irradiance', np.nan),
                    'Temperatura_C': curve['characteristics'].get('Avg_Temperature', np.nan),
                    'Pmax_W': curve['characteristics'].get('Pmax', np.nan),
                    'Vmp_V': curve['characteristics'].get('Vmp', np.nan),
                    'Imp_A': curve['characteristics'].get('Imp', np.nan),
                    'Isc_A': curve['characteristics'].get('Isc', np.nan),
                    'Voc_V': curve['characteristics'].get('Voc', np.nan),
                    'FF': curve['characteristics'].get('FF', np.nan),
                    'Eficiencia_%': curve['characteristics'].get('Efficiency', np.nan)
                }
                analysis_data.append(row)
            
            df_analysis = pd.DataFrame(analysis_data)
            df_analysis.to_excel(writer, sheet_name='Analisis_Parametros', index=False)
            
            # Hoja 3: Curvas IV individuales
            for curve in processed_curves:
                df_iv = curve['iv_data']
                if not df_iv.empty:
                    # Limpiar nombre de hoja (Excel tiene límite de 31 caracteres)
                    sheet_name = curve['filename'].replace('.txt', '')[:31]
                    df_iv.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Reporte consolidado guardado en: {excel_file}")
        
    except ImportError:
        logger.warning("openpyxl no está disponible. No se pudo generar el archivo Excel.")
    except Exception as e:
        logger.error(f"Error generando reporte consolidado: {e}")

def run_iv_curve_processing():
    """
    Función principal para ejecutar el procesamiento de curvas IV.
    """
    # Directorio de datos
    data_dir = paths.BASE_INPUT_DIR
    
    # Ejecutar procesamiento
    results = process_pvstand_iv_files(data_dir=data_dir)
    
    if results:
        logger.info("Procesamiento de curvas IV completado exitosamente")
        return True
    else:
        logger.error("Error en el procesamiento de curvas IV")
        return False

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando procesamiento de curvas IV del PVStand...")
    run_iv_curve_processing()