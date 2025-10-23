#!/usr/bin/env python3
"""
Script de preprocesamiento IV600
================================

Este script procesa el archivo processed_iv600_data.csv (que contiene curvas IV completas)
y genera raw_iv600_data.csv con parámetros calculados (pmp, isc, voc, imp, vmp).

Resuelve el problema de:
- Datos faltantes hasta mayo 2025
- Fechas desordenadas 
- Formato CSV corrupto en raw_iv600_data.csv

Autor: Sistema de Análisis de Soiling
Fecha: 2025
"""

import pandas as pd
import numpy as np
import os
import ast
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def calculate_iv_parameters(voltages: List[float], currents: List[float]) -> Tuple[float, float, float, float, float]:
    """
    Calcula parámetros de curva IV desde listas de voltajes y corrientes.
    
    Args:
        voltages: Lista de voltajes [V]
        currents: Lista de corrientes [A]
    
    Returns:
        Tuple con (pmp, isc, voc, imp, vmp)
    """
    try:
        # Validar que las listas tengan el mismo tamaño
        if len(voltages) != len(currents):
            logger.warning(f"Tamaños diferentes: {len(voltages)} voltajes vs {len(currents)} corrientes")
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Convertir a arrays numpy para cálculos
        V = np.array(voltages)
        I = np.array(currents)
        
        # Calcular potencia para cada punto
        P = V * I
        
        # Parámetros básicos
        pmp = np.max(P)  # Potencia máxima
        isc = np.max(I)  # Corriente de cortocircuito 
        voc = np.max(V)  # Voltaje de circuito abierto
        
        # Punto de máxima potencia
        idx_pmp = np.argmax(P)
        imp = I[idx_pmp]  # Corriente en Pmp
        vmp = V[idx_pmp]  # Voltaje en Pmp
        
        return pmp, isc, voc, imp, vmp
        
    except Exception as e:
        logger.error(f"Error calculando parámetros IV: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

def parse_array_string(array_str: str) -> List[float]:
    """
    Convierte string de array a lista de floats.
    
    Args:
        array_str: String que representa un array (ej: "[1.2, 3.4, 5.6]")
    
    Returns:
        Lista de floats
    """
    try:
        if pd.isna(array_str) or array_str == '':
            return []
        
        # Usar ast.literal_eval para parsear de forma segura
        parsed = ast.literal_eval(array_str)
        
        # Convertir a lista de floats
        if isinstance(parsed, list):
            return [float(x) for x in parsed if not pd.isna(x)]
        else:
            logger.warning(f"Formato inesperado: {type(parsed)}")
            return []
            
    except Exception as e:
        logger.error(f"Error parseando array: {e} - String: {array_str[:100]}...")
        return []

def expand_to_5min_intervals(df: pd.DataFrame, target_interval_minutes: int = 5) -> pd.DataFrame:
    """
    Expande datos esporádicos a intervalos regulares de 5 minutos.
    
    Args:
        df: DataFrame con índice de fecha y parámetros calculados
        target_interval_minutes: Intervalo objetivo en minutos
    
    Returns:
        DataFrame expandido con datos interpolados cada 5 minutos
    """
    logger.info(f"Expandiendo datos a intervalos de {target_interval_minutes} minutos...")
    
    try:
        # Ordenar por fecha
        df_sorted = df.sort_index()
        
        # Crear índice de tiempo regular cada 5 minutos
        start_time = df_sorted.index.min()
        end_time = df_sorted.index.max()
        
        # Redondear start_time al múltiplo de 5 minutos más cercano
        start_time = start_time.replace(minute=(start_time.minute // target_interval_minutes) * target_interval_minutes, second=0, microsecond=0)
        
        # Crear rango de fechas cada 5 minutos
        regular_index = pd.date_range(start=start_time, end=end_time, freq=f'{target_interval_minutes}min')
        
        # Reindexar y usar interpolación lineal para rellenar gaps
        df_expanded = df_sorted.reindex(regular_index)
        
        # Interpolar valores faltantes
        for col in df_expanded.columns:
            df_expanded[col] = df_expanded[col].interpolate(method='linear', limit_direction='both')
        
        # Rellenar valores al inicio y final si es necesario
        df_expanded = df_expanded.fillna(method='bfill').fillna(method='ffill')
        
        logger.info(f"Datos expandidos: {len(df_sorted)} → {len(df_expanded)} registros")
        return df_expanded
        
    except Exception as e:
        logger.error(f"Error expandiendo datos: {e}")
        return df

def process_iv600_data(
    input_file: str = "SOILING/datos/processed_iv600_data.csv",
    output_file: str = "SOILING/datos/raw_iv600_data.csv",
    expand_data: bool = False  # Cambiar a True si quieres expansión a 5min
) -> bool:
    """
    Procesa datos IV600 desde processed_iv600_data.csv y genera raw_iv600_data.csv.
    
    Args:
        input_file: Ruta al archivo de entrada (processed_iv600_data.csv)
        output_file: Ruta al archivo de salida (raw_iv600_data.csv)
        expand_data: Si expandir datos a intervalos de 5 minutos
    
    Returns:
        True si el procesamiento fue exitoso
    """
    logger.info("=== INICIANDO PROCESAMIENTO IV600 ===")
    logger.info(f"📁 Archivo de entrada: {input_file}")
    logger.info(f"📁 Archivo de salida: {output_file}")
    
    try:
        # 1. Verificar existencia del archivo de entrada
        if not os.path.exists(input_file):
            logger.error(f"❌ Archivo de entrada no encontrado: {input_file}")
            return False
        
        # 2. Cargar datos procesados
        logger.info("📖 Cargando datos procesados...")
        df_processed = pd.read_csv(input_file)
        logger.info(f"✅ Datos cargados: {len(df_processed)} registros")
        logger.info(f"📊 Columnas: {list(df_processed.columns)}")
        
        # 3. Verificar columnas necesarias
        required_columns = ['fecha', 'modulo', 'voltajes', 'corrientes']
        missing_columns = [col for col in required_columns if col not in df_processed.columns]
        
        if missing_columns:
            logger.error(f"❌ Columnas faltantes: {missing_columns}")
            return False
        
        # 4. Procesar fechas
        logger.info("🕒 Procesando fechas...")
        df_processed['fecha'] = pd.to_datetime(df_processed['fecha'])
        
        # Mostrar rango de fechas
        fecha_min = df_processed['fecha'].min()
        fecha_max = df_processed['fecha'].max()
        logger.info(f"📅 Rango de fechas: {fecha_min} → {fecha_max}")
        
        # 5. Procesar datos por módulo
        logger.info("🔧 Calculando parámetros IV...")
        results = []
        
        modulos_unicos = df_processed['modulo'].unique()
        logger.info(f"🔍 Módulos encontrados: {modulos_unicos}")
        
        for idx, row in df_processed.iterrows():
            if idx % 200 == 0:  # Log progreso cada 200 registros
                logger.info(f"   Procesando registro {idx+1}/{len(df_processed)}...")
            
            try:
                # Parsear arrays de voltajes y corrientes
                voltages = parse_array_string(row['voltajes'])
                currents = parse_array_string(row['corrientes'])
                
                if len(voltages) == 0 or len(currents) == 0:
                    logger.warning(f"Registro {idx}: Arrays vacíos")
                    continue
                
                # Calcular parámetros
                pmp, isc, voc, imp, vmp = calculate_iv_parameters(voltages, currents)
                
                # Crear registro resultado
                result_row = {
                    'timestamp': row['fecha'],
                    'module': row['modulo'],
                    'pmp': pmp,
                    'isc': isc,
                    'voc': voc,
                    'imp': imp,
                    'vmp': vmp
                }
                
                results.append(result_row)
                
            except Exception as e:
                logger.error(f"Error procesando registro {idx}: {e}")
                continue
        
        # 6. Crear DataFrame de resultados
        logger.info(f"📊 Creando DataFrame final...")
        df_results = pd.DataFrame(results)
        
        if df_results.empty:
            logger.error("❌ No se generaron datos válidos")
            return False
        
        # 7. Configurar índice temporal
        df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
        df_results = df_results.set_index('timestamp')
        
        # 8. Ordenar cronológicamente
        df_results = df_results.sort_index()
        
        # 9. Estadísticas de procesamiento
        logger.info("📈 Estadísticas de procesamiento:")
        logger.info(f"   • Registros de entrada: {len(df_processed)}")
        logger.info(f"   • Registros de salida: {len(df_results)}")
        logger.info(f"   • Módulos procesados: {df_results['module'].nunique()}")
        logger.info(f"   • Rango temporal: {df_results.index.min()} → {df_results.index.max()}")
        
        # Estadísticas por módulo
        for module in df_results['module'].unique():
            count = len(df_results[df_results['module'] == module])
            logger.info(f"   • {module}: {count} mediciones")
        
        # 10. Expansión a intervalos regulares (opcional)
        if expand_data:
            logger.info("🔄 Expandiendo datos a intervalos de 5 minutos...")
            # Procesar por módulo para mantener consistencia
            expanded_dfs = []
            
            for module in df_results['module'].unique():
                df_module = df_results[df_results['module'] == module].drop('module', axis=1)
                df_module_expanded = expand_to_5min_intervals(df_module)
                df_module_expanded['module'] = module
                expanded_dfs.append(df_module_expanded)
            
            df_results = pd.concat(expanded_dfs).sort_index()
            logger.info(f"✅ Datos expandidos: {len(df_results)} registros totales")
        
        # 11. Guardar resultados
        logger.info(f"💾 Guardando resultados en {output_file}...")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Guardar CSV
        df_results.to_csv(output_file)
        
        # 12. Verificación final
        file_size_mb = os.path.getsize(output_file) / (1024*1024)
        logger.info(f"✅ Archivo guardado exitosamente")
        logger.info(f"📁 Tamaño: {file_size_mb:.2f} MB")
        
        # Verificar carga del archivo generado
        df_verify = pd.read_csv(output_file, nrows=5)
        logger.info(f"🔍 Verificación - primeras 5 filas cargadas correctamente")
        logger.info(f"📊 Columnas finales: {list(df_verify.columns)}")
        
        logger.info("=== PROCESAMIENTO IV600 COMPLETADO EXITOSAMENTE ===")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en procesamiento IV600: {e}")
        import traceback
        logger.error(f"🔍 Detalles: {traceback.format_exc()}")
        return False

def main():
    """Función principal para ejecutar el script."""
    logger.info("🚀 INICIANDO SCRIPT DE PREPROCESAMIENTO IV600")
    
    # Rutas de archivos (relativas al directorio del proyecto)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(base_dir, "SOILING", "datos", "processed_iv600_data.csv")
    output_file = os.path.join(base_dir, "SOILING", "datos", "raw_iv600_data.csv")
    
    logger.info(f"📂 Directorio base: {base_dir}")
    logger.info(f"📥 Entrada: {input_file}")
    logger.info(f"📤 Salida: {output_file}")
    
    # Ejecutar procesamiento
    success = process_iv600_data(input_file, output_file, expand_data=False)
    
    if success:
        logger.info("🎉 ¡PREPROCESAMIENTO COMPLETADO EXITOSAMENTE!")
        logger.info("📋 PRÓXIMOS PASOS:")
        logger.info("   1. Ejecutar analisis_iv600_fixed.py")
        logger.info("   2. Los datos ahora incluyen todas las fechas hasta mayo 2025")
        logger.info("   3. Las fechas están ordenadas cronológicamente")
    else:
        logger.error("❌ PREPROCESAMIENTO FALLÓ")
        logger.error("🔧 Revisa los logs anteriores para más detalles")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 