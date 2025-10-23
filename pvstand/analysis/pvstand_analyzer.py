# analysis/pvstand_analyzer.py

import os
import sys
import matplotlib.pyplot as plt
import logging

# Agregar el directorio raíz del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import timezone, timedelta # Añadido desde el script temporal
# import pytz # Pytz ya no se usa explícitamente en la nueva lógica para la conversión de índice principal

from config.logging_config import logger
from config import paths, settings
from utils.helpers import normalize_series_from_date_pd
from utils.solar_time import UtilsMedioDiaSolar

def save_plot_matplotlib(fig, filename_base, output_dir, subfolder=None, dpi=300):
    """
    Guarda una figura de Matplotlib en el directorio especificado, opcionalmente en un subdirectorio.
    Cierra la figura después de guardarla.
    """
    full_output_dir = output_dir
    if subfolder:
        full_output_dir = os.path.join(output_dir, subfolder)
    os.makedirs(full_output_dir, exist_ok=True)

    filepath = os.path.join(full_output_dir, filename_base)
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logging.info(f"Gráfico guardado en: {filepath}")
    except Exception as e:
        logging.error(f"Error al guardar el gráfico en {filepath}: {e}")
    finally:
        plt.close(fig)

# La función _plot_series_temporales_pvstand original se eliminará,
# ya que la nueva lógica de analyze_pvstand_data incluye su propia función de graficado.

def analyze_pvstand_data(
    pv_iv_data_filepath: str, 
    temperature_data_filepath: str
) -> bool:
    """
    Analiza los datos de PVStand IV, los combina con datos de temperatura,
    realiza correcciones de temperatura, calcula Soiling Ratios (SR),
    y genera CSVs y gráficos de resultados.
    Esta función está basada en la lógica desarrollada y probada en el notebook analisis_soiling.ipynb
    y refinada en el script temporal temp_run_pvstand_analysis.py.

    Args:
        pv_iv_data_filepath: Ruta al archivo CSV de datos IV de PVStand 
                             (generalmente paths.PVSTAND_IV_RAW_DATA_FILE).
        temperature_data_filepath: Ruta al archivo CSV de datos de temperatura procesados y combinados
                                   (generalmente paths.TEMP_DATA_COMBINED_PROCESSED_CSV).

    Returns:
        True si el análisis fue exitoso, False en caso contrario.
    """
    logger.info("=== INICIO DEL ANÁLISIS PVSTAND ===")
    logger.info(f"Archivo de datos IV: {pv_iv_data_filepath}")
    logger.info(f"Archivo de datos de temperatura: {temperature_data_filepath}")
    
    # Verificar que los archivos existan
    if not os.path.exists(pv_iv_data_filepath):
        logger.error(f"El archivo de datos IV no existe: {pv_iv_data_filepath}")
        return False
        
    if not os.path.exists(temperature_data_filepath):
        logger.error(f"El archivo de datos de temperatura no existe: {temperature_data_filepath}")
        return False

    # Parámetros obtenidos de config.settings y config.paths
    # Paths de salida
    output_csv_dir = paths.PVSTAND_OUTPUT_SUBDIR_CSV
    output_graph_dir = paths.BASE_OUTPUT_GRAPH_DIR # La función de graficado creará un subdirectorio

    # Parámetros de filtrado y análisis
    filter_start_date_str = settings.PVSTAND_ANALYSIS_START_DATE_STR
    filter_end_date_str = settings.PVSTAND_ANALYSIS_END_DATE_STR
    filter_start_time_str = settings.PVSTAND_FILTER_START_TIME
    filter_end_time_str = settings.PVSTAND_FILTER_END_TIME
    
    pv_module_soiled_id = settings.PVSTAND_MODULE_SOILED_ID
    pv_module_reference_id = settings.PVSTAND_MODULE_REFERENCE_ID
    temp_sensor_soiled_col = settings.PVSTAND_TEMP_SENSOR_SOILED_COL
    temp_sensor_reference_col = settings.PVSTAND_TEMP_SENSOR_REFERENCE_COL
    
    alpha_isc_corr = settings.PVSTAND_ALPHA_ISC_CORR
    beta_pmax_corr = settings.PVSTAND_BETA_PMAX_CORR
    temp_ref_correction_c = settings.PVSTAND_TEMP_REF_CORRECTION_C
    
    normalize_sr_flag = settings.PVSTAND_NORMALIZE_SR_FLAG
    normalize_sr_ref_date_str = settings.PVSTAND_NORMALIZE_SR_REF_DATE_STR
    pmax_sr_offset = settings.PVSTAND_PMAX_SR_OFFSET
    
    sr_min_filter_threshold = settings.PVSTAND_SR_MIN_FILTER_THRESHOLD
    sr_max_filter_threshold = settings.PVSTAND_SR_MAX_FILTER_THRESHOLD
    
    save_figures_setting = settings.SAVE_FIGURES
    show_figures_setting = settings.SHOW_FIGURES
    
    resample_freq_minutes = settings.PVSTAND_RESAMPLE_FREQ_MINUTES
    graph_quantile = settings.PVSTAND_GRAPH_QUANTILE
    
    # Asegurar que directorios de salida específicos para PVStand existan
    os.makedirs(output_csv_dir, exist_ok=True)
    # El subdirectorio de gráficos específico se creará dentro de la función de ploteo

    try:
        start_datetime_filter_str = f"{filter_start_date_str} {filter_start_time_str}"
        end_datetime_filter_str = f"{filter_end_date_str} {filter_end_time_str}"
        
        start_date_dt = pd.Timestamp(start_datetime_filter_str, tz='UTC')
        end_date_dt = pd.Timestamp(end_datetime_filter_str, tz='UTC')
        
        logger.info(f"Periodo de filtro (fechas UTC): {start_date_dt} a {end_date_dt}")

    except Exception as e:
        logger.error(f"Error al parsear fechas/horas de filtro desde settings: {e}. Abortando análisis.")
        return False

    df_pvstand_raw_data = pd.DataFrame() # Renombrado para evitar confusión con df_pvstand_raw en la lógica interna
    if os.path.exists(pv_iv_data_filepath):
        logger.info("Cargando datos PVStand IV...")
        try:
            # Leer el archivo completo primero para detectar la estructura
            df_pvstand_raw_data = pd.read_csv(pv_iv_data_filepath)
            logger.info(f"Columnas encontradas en archivo IV: {df_pvstand_raw_data.columns.tolist()}")
            
            # Detectar columna de tiempo
            time_col_actual = None
            for col in df_pvstand_raw_data.columns:
                if col.lower() in ['_time', 'timestamp', 'time']:
                    time_col_actual = col
                    break
            
            if not time_col_actual:
                logger.error("No se encontró columna de tiempo en los datos IV de PVStand.")
                return False
            
            # Renombrar columna de tiempo a _time para consistencia
            if time_col_actual != '_time':
                df_pvstand_raw_data.rename(columns={time_col_actual: '_time'}, inplace=True)
                logger.info(f"Columna de tiempo '{time_col_actual}' renombrada a '_time'")

            df_pvstand_raw_data['_time'] = pd.to_datetime(
                df_pvstand_raw_data['_time'], 
                errors='coerce', 
                format=settings.PVSTAND_IV_DATA_TIME_FORMAT # Usar formato de settings
            )
            df_pvstand_raw_data.dropna(subset=['_time'], inplace=True)
            df_pvstand_raw_data.set_index('_time', inplace=True)
            logger.info(f"Datos PVStand IV cargados: {len(df_pvstand_raw_data)} filas iniciales.")

            if df_pvstand_raw_data.index.tz is None:
                logger.info("Localizando índice de PVStand IV a UTC (ambiguous='infer', nonexistent='shift_forward').")
                df_pvstand_raw_data.index = df_pvstand_raw_data.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            elif df_pvstand_raw_data.index.tz != timezone.utc: # Usar datetime.timezone.utc
                logger.info(f"Convirtiendo índice de PVStand IV de {df_pvstand_raw_data.index.tz} a UTC.")
                df_pvstand_raw_data.index = df_pvstand_raw_data.index.tz_convert('UTC')
            
            logger.info(f"Zona horaria del índice PVStand asegurada a UTC: {df_pvstand_raw_data.index.tz}")

            # Guardar copia de datos originales para gráfico de potencias brutas (antes de filtros)
            df_pvstand_original_unfiltered = df_pvstand_raw_data.copy()
            logger.info(f"Datos originales guardados para gráfico de potencias brutas: {len(df_pvstand_original_unfiltered)} filas.")

            if '_measurement' in df_pvstand_raw_data.columns:
                logger.info("Pivotando datos PVStand IV por '_measurement'...")
                values_to_pivot = [col for col in ['Imax', 'Pmax', 'Umax'] if col in df_pvstand_raw_data.columns]
                if not values_to_pivot:
                    logger.error("Ninguna de las columnas de valores (Imax, Pmax, Umax) encontradas para pivotar.")
                    return False # No se puede continuar sin estas columnas
                
                df_pvstand_pivot = df_pvstand_raw_data.pivot_table(
                    index=df_pvstand_raw_data.index, 
                    columns='_measurement', 
                    values=values_to_pivot
                )
                df_pvstand_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pvstand_pivot.columns]
                df_pvstand_raw_data = df_pvstand_pivot # Ahora df_pvstand_raw_data es el pivotado
                logger.info(f"Datos PVStand IV pivotados. {len(df_pvstand_raw_data)} filas. Columnas ejemplo: {df_pvstand_raw_data.columns[:5].tolist()}...")
                
                # También pivotar los datos originales para el gráfico de potencias brutas
                if '_measurement' in df_pvstand_original_unfiltered.columns:
                    df_pvstand_original_pivot = df_pvstand_original_unfiltered.pivot_table(
                        index=df_pvstand_original_unfiltered.index, 
                        columns='_measurement', 
                        values=values_to_pivot
                    )
                    df_pvstand_original_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pvstand_original_pivot.columns]
                    df_pvstand_original_unfiltered = df_pvstand_original_pivot
                    logger.info(f"Datos originales pivotados para gráfico de potencias brutas: {len(df_pvstand_original_unfiltered)} filas.")
            elif 'module' in df_pvstand_raw_data.columns:
                logger.info("Pivotando datos PVStand IV por columna 'module'...")
                # Formato: timestamp, module, pmax, imax, umax
                # Necesitamos pivotar por module
                values_to_pivot = [col for col in ['imax', 'pmax', 'umax'] if col in df_pvstand_raw_data.columns]
                if not values_to_pivot:
                    logger.error("Ninguna de las columnas de valores (imax, pmax, umax) encontradas para pivotar.")
                    return False
                
                df_pvstand_pivot = df_pvstand_raw_data.pivot_table(
                    index=df_pvstand_raw_data.index, 
                    columns='module', 
                    values=values_to_pivot
                )
                # Renombrar columnas para consistencia: module_metric -> module_Metric
                df_pvstand_pivot.columns = [f'{col[1]}_{col[0].capitalize()}' for col in df_pvstand_pivot.columns]
                df_pvstand_raw_data = df_pvstand_pivot
                logger.info(f"Datos PVStand IV pivotados por module. {len(df_pvstand_raw_data)} filas. Columnas: {df_pvstand_raw_data.columns.tolist()}")
                
                # También pivotar los datos originales para el gráfico de potencias brutas
                df_pvstand_original_pivot = df_pvstand_original_unfiltered.pivot_table(
                    index=df_pvstand_original_unfiltered.index, 
                    columns='module', 
                    values=values_to_pivot
                )
                df_pvstand_original_pivot.columns = [f'{col[1]}_{col[0].capitalize()}' for col in df_pvstand_original_pivot.columns]
                df_pvstand_original_unfiltered = df_pvstand_original_pivot
                logger.info(f"Datos originales pivotados para gráfico de potencias brutas: {len(df_pvstand_original_unfiltered)} filas.")
            else:
                logger.warning("No se encontró columna '_measurement' ni 'module' para pivotar. Se continuará con las columnas existentes.")

            # Filtrado por rango de fechas Y horas diarias (después de asegurar UTC)
            # Primero filtro por rango de fechas global
            start_date_only = pd.Timestamp(start_date_dt.date(), tz='UTC')
            end_date_only = pd.Timestamp(end_date_dt.date(), tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df_pvstand_raw_data = df_pvstand_raw_data[(df_pvstand_raw_data.index >= start_date_only) & (df_pvstand_raw_data.index <= end_date_only)]
            
            # Luego filtro por horas diarias (13:00-18:00 cada día)
            start_hour = int(filter_start_time_str.split(':')[0])
            start_minute = int(filter_start_time_str.split(':')[1])
            end_hour = int(filter_end_time_str.split(':')[0])
            end_minute = int(filter_end_time_str.split(':')[1])
            
            # Crear máscara para filtro horario diario
            time_mask = (
                (df_pvstand_raw_data.index.hour > start_hour) |
                ((df_pvstand_raw_data.index.hour == start_hour) & (df_pvstand_raw_data.index.minute >= start_minute))
            ) & (
                (df_pvstand_raw_data.index.hour < end_hour) |
                ((df_pvstand_raw_data.index.hour == end_hour) & (df_pvstand_raw_data.index.minute <= end_minute))
            )
            
            df_pvstand_raw_data = df_pvstand_raw_data[time_mask]
            logger.info(f"Datos PVStand IV filtrados por rango de fecha/hora (UTC) Y filtro horario diario ({filter_start_time_str}-{filter_end_time_str}): {len(df_pvstand_raw_data)} puntos.")
            if df_pvstand_raw_data.empty: 
                logger.warning("PVStand IV vacío después de filtro de fecha/hora. No se puede continuar.")
                return False # Si no hay datos IV, no se puede analizar
        except Exception as e:
            logger.error(f"Error cargando/preprocesando PVStand IV: {e}", exc_info=True)
            return False # Error en carga/preprocesamiento de IV es fatal
    else:
        logger.error(f"Archivo PVStand IV no encontrado: {pv_iv_data_filepath}")
        return False

    # Cargar datos de Temperatura preprocesados
    logger.info(f"Cargando datos de Temperatura preprocesados desde: {temperature_data_filepath}")
    df_temp_processed = pd.DataFrame() # Renombrado para evitar confusión
    if temperature_data_filepath and os.path.exists(temperature_data_filepath):
        try:
            # Leer archivo y convertir primera columna a DatetimeIndex explícitamente
            df_temp_intermediate = pd.read_csv(temperature_data_filepath)
            
            # La primera columna contiene las fechas, convertirla a DatetimeIndex
            time_col = df_temp_intermediate.columns[0]  # Primera columna (índice sin nombre)
            df_temp_intermediate[time_col] = pd.to_datetime(df_temp_intermediate[time_col], errors='coerce')
            df_temp_intermediate.set_index(time_col, inplace=True)
            
            # Eliminar columnas innecesarias de metadatos de InfluxDB
            cols_to_remove = ['_start', '_stop', '_measurement']
            cols_found_to_remove = [col for col in cols_to_remove if col in df_temp_intermediate.columns]
            if cols_found_to_remove:
                df_temp_intermediate.drop(columns=cols_found_to_remove, inplace=True)
                logger.info(f"Eliminadas columnas innecesarias: {cols_found_to_remove}")
            
            logger.info(f"Archivo de temperatura cargado con {len(df_temp_intermediate)} filas.")
            logger.info(f"Tipo de índice: {type(df_temp_intermediate.index)}")
            
            # Verificar si hay timestamps inválidos en el índice
            if df_temp_intermediate.index.isna().any():
                logger.warning(f"Eliminando {df_temp_intermediate.index.isna().sum()} filas con timestamps inválidos.")
                df_temp_intermediate = df_temp_intermediate[~df_temp_intermediate.index.isna()]

            if not df_temp_intermediate.empty:
                # Convertir columnas de temperatura a numérico
                sensor_cols = [col for col in df_temp_intermediate.columns if col.startswith('1TE') and col.endswith('(C)')]
                if sensor_cols:
                    logger.info(f"Convirtiendo columnas de sensores de temperatura a numérico: {sensor_cols}")
                    for col in sensor_cols:
                        df_temp_intermediate[col] = pd.to_numeric(df_temp_intermediate[col], errors='coerce')
                
                # El DataFrame ya tiene el índice como DatetimeIndex
                df_temp_processed = df_temp_intermediate
                
                # Asegurar zona horaria UTC solo si tenemos DatetimeIndex
                if isinstance(df_temp_processed.index, pd.DatetimeIndex):
                    if df_temp_processed.index.tz is None:
                        df_temp_processed.index = df_temp_processed.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
                        logger.info("Localizado índice de DataFrame de Temperatura a UTC.")
                    elif df_temp_processed.index.tz != timezone.utc:
                        df_temp_processed.index = df_temp_processed.index.tz_convert('UTC')
                        logger.info("Convertido índice de DataFrame de Temperatura a UTC.")
                    else:
                        logger.info("Índice de DataFrame de Temperatura ya está en UTC.")
                else:
                    logger.error(f"El índice no es DatetimeIndex: {type(df_temp_processed.index)}. No se puede procesar.")
                
                # Ordenar y filtrar por rango de fechas Y horas diarias
                df_temp_processed = df_temp_processed.sort_index()
                # Filtro por rango de fechas
                df_temp_processed = df_temp_processed[(df_temp_processed.index >= start_date_only) & (df_temp_processed.index <= end_date_only)]
                
                # Filtro por horas diarias (13:00-18:00 cada día)
                temp_time_mask = (
                    (df_temp_processed.index.hour > start_hour) |
                    ((df_temp_processed.index.hour == start_hour) & (df_temp_processed.index.minute >= start_minute))
                ) & (
                    (df_temp_processed.index.hour < end_hour) |
                    ((df_temp_processed.index.hour == end_hour) & (df_temp_processed.index.minute <= end_minute))
                )
                df_temp_processed = df_temp_processed[temp_time_mask]
                
                logger.info(f"Datos de Temperatura cargados, procesados y filtrados: {len(df_temp_processed)} filas. Zona horaria: {df_temp_processed.index.tz}")
                if not df_temp_processed.empty:
                    logger.info(f"Rango de tiempo datos de Temperatura (procesados y filtrados): {df_temp_processed.index.min()} a {df_temp_processed.index.max()}")
                    logger.info(f"Columnas de temperatura disponibles: {sensor_cols}")
                else:
                    logger.warning("DataFrame de Temperatura vacío después del filtro de fecha/hora.")
            else:
                logger.warning("El archivo de datos de Temperatura está vacío después de limpiar timestamps inválidos.")
        except Exception as e:
            logger.error(f"Error cargando/preprocesando datos de Temperatura: {e}", exc_info=True)
            # No retornar False aquí necesariamente, el análisis podría continuar sin corrección de temp si df_temp_processed está vacío.
    else:
        logger.warning(f"Archivo de datos de Temperatura no encontrado o no especificado: {temperature_data_filepath}")
        # df_temp_processed permanece vacío

    # Remuestreo (df_pvstand_raw_data y df_temp_processed son los DataFrames con índice DatetimeIndex UTC)
    df_pvstand_resampled = pd.DataFrame()
    if not df_pvstand_raw_data.empty:
        logger.info(f"Remuestreando datos PVStand IV a {resample_freq_minutes} minuto(s) (promedio)...")
        try:
            df_pvstand_resampled = df_pvstand_raw_data.resample(f'{resample_freq_minutes}min').mean()
            logger.info(f"PVStand IV remuestreado: {len(df_pvstand_resampled)} puntos.")
            if not df_pvstand_resampled.empty:
                logger.info(f"Rango de tiempo PVStand IV (remuestreado): {df_pvstand_resampled.index.min()} a {df_pvstand_resampled.index.max()}")
        except Exception as e_resample_pv:
            logger.error(f"Error remuestreando datos PVStand: {e_resample_pv}", exc_info=True)
            # Continuar sin datos PVStand resampleados
    else:
        logger.warning("Datos PVStand IV crudos (después de pivotar y filtrar fecha) están vacíos. No se puede remuestrear.")
        return False # Si no hay datos PV IV para remuestrear, no se puede continuar.

    df_temp_resampled = pd.DataFrame()
    if not df_temp_processed.empty:
        logger.info(f"Remuestreando datos de Temperatura a {resample_freq_minutes} minuto(s) (promedio)...")
        try:
            df_temp_resampled = df_temp_processed.resample(f'{resample_freq_minutes}min').mean()
            logger.info(f"Temperatura remuestreada: {len(df_temp_resampled)} puntos.")
            if not df_temp_resampled.empty:
                logger.info(f"Rango de tiempo Temperatura (remuestreada): {df_temp_resampled.index.min()} a {df_temp_resampled.index.max()}")
        except Exception as e_resample_temp:
            logger.error(f"Error remuestreando datos de Temperatura: {e_resample_temp}", exc_info=True)
            # Continuar sin datos de temp resampleados
    else:
        logger.warning("Datos de Temperatura procesados están vacíos. No se puede remuestrear.")

    # Definición de Columnas para SR
    col_isc_soiled = f'{pv_module_soiled_id}_Imax'
    col_pmax_soiled = f'{pv_module_soiled_id}_Pmax'
    col_isc_reference = f'{pv_module_reference_id}_Imax'
    col_pmax_reference = f'{pv_module_reference_id}_Pmax'

    logger.info(f"Columnas para SR: Soiled Isc='{col_isc_soiled}', Pmax='{col_pmax_soiled}'")
    logger.info(f"Columnas para SR: Reference Isc='{col_isc_reference}', Pmax='{col_pmax_reference}'")
    logger.info(f"Columnas de temp para corrección: Soiled='{temp_sensor_soiled_col}', Reference='{temp_sensor_reference_col}'")

    # Inicializar Series para SRs
    sr_isc_pvstand_raw_no_offset = pd.Series(dtype=float, name="SR_Isc_Uncorrected_Raw_NoOffset")
    sr_pmax_pvstand_raw_no_offset = pd.Series(dtype=float, name="SR_Pmax_Uncorrected_Raw_NoOffset")
    sr_isc_pvstand_corrected_raw_no_offset = pd.Series(dtype=float, name="SR_Isc_Corrected_Raw_NoOffset")
    sr_pmax_pvstand_corrected_raw_no_offset = pd.Series(dtype=float, name="SR_Pmax_Corrected_Raw_NoOffset")

    # --- Cálculo de SR SIN corrección de temperatura ---
    if not df_pvstand_resampled.empty:
        logger.info("Calculando SRs SIN corrección de temperatura...")
        if col_isc_soiled in df_pvstand_resampled.columns and col_isc_reference in df_pvstand_resampled.columns:
            # Filtrar valores anormalmente bajos en referencia que causan SRs extremos
            isc_reference_filtered = df_pvstand_resampled[col_isc_reference].copy()
            isc_soiled_filtered = df_pvstand_resampled[col_isc_soiled].copy()
            
            # Filtrar también por Pmax para mantener consistencia de datos
            if col_pmax_soiled in df_pvstand_resampled.columns and col_pmax_reference in df_pvstand_resampled.columns:
                pmax_outlier_threshold = 170.0  # Watts
                pmax_consistency_mask = (df_pvstand_resampled[col_pmax_soiled] >= pmax_outlier_threshold) & (df_pvstand_resampled[col_pmax_reference] >= pmax_outlier_threshold)
                isc_soiled_filtered = isc_soiled_filtered[pmax_consistency_mask]
                isc_reference_filtered = isc_reference_filtered[pmax_consistency_mask]
                logger.info(f"Aplicado filtro de consistencia Pmax (>={pmax_outlier_threshold}W) a datos Isc. Datos restantes: {len(isc_soiled_filtered)}")
            
            # Excluir casos donde la referencia es < 0.5A (anormalmente baja para Isc)
            valid_isc_reference_mask = isc_reference_filtered >= 0.5
            isc_reference_filtered = isc_reference_filtered[valid_isc_reference_mask]
            isc_soiled_filtered = isc_soiled_filtered[valid_isc_reference_mask]
            
            denom_isc = isc_reference_filtered.replace(0, np.nan)
            if denom_isc.count() > 0:
                sr_isc_calc_raw = isc_soiled_filtered.div(denom_isc)
                
                # Aplicar filtro variable por fecha: 101% para ago-sep, 100% para oct en adelante
                early_months_cutoff = pd.Timestamp('2024-10-01', tz='UTC')
                early_mask = sr_isc_calc_raw.index < early_months_cutoff
                late_mask = sr_isc_calc_raw.index >= early_months_cutoff
                
                # Filtro para primeros meses (ago-sep): hasta 101%
                sr_isc_early = sr_isc_calc_raw[early_mask]
                sr_isc_early_filtered = sr_isc_early[(sr_isc_early.notna()) & (sr_isc_early >= sr_min_filter_threshold) & (sr_isc_early <= 1.01)]
                
                # Filtro para meses posteriores (oct en adelante): hasta 100%
                sr_isc_late = sr_isc_calc_raw[late_mask]
                sr_isc_late_filtered = sr_isc_late[(sr_isc_late.notna()) & (sr_isc_late >= sr_min_filter_threshold) & (sr_isc_late <= 1.0)]
                
                # Combinar ambos filtros
                sr_isc_filtered = pd.concat([sr_isc_early_filtered, sr_isc_late_filtered]).sort_index()
                
                sr_isc_pvstand_raw_no_offset = (100 * sr_isc_filtered).rename("SR_Isc_Uncorrected_Raw_NoOffset")
                logger.info(f"SR Isc (sin corregir, raw) calculado: {len(sr_isc_pvstand_raw_no_offset)} puntos válidos. Filtro variable: ≤101% (ago-sep), ≤100% (oct+)")
        else: logger.warning(f"Columnas faltantes para SR Isc sin corregir: {col_isc_soiled} o {col_isc_reference}")

        if col_pmax_soiled in df_pvstand_resampled.columns and col_pmax_reference in df_pvstand_resampled.columns:
            # Filtrar valores anormalmente bajos en referencia que causan SRs extremos
            pmax_reference_filtered = df_pvstand_resampled[col_pmax_reference].copy()
            pmax_soiled_filtered = df_pvstand_resampled[col_pmax_soiled].copy()
            
            # Filtrar valores de Pmax menores a 170W (considerados outliers) - ANTES de otros filtros
            pmax_outlier_threshold = 170.0  # Watts
            pmax_outlier_mask = (pmax_soiled_filtered >= pmax_outlier_threshold) & (pmax_reference_filtered >= pmax_outlier_threshold)
            pmax_soiled_filtered = pmax_soiled_filtered[pmax_outlier_mask]
            pmax_reference_filtered = pmax_reference_filtered[pmax_outlier_mask]
            logger.info(f"Filtrados outliers de Pmax (<{pmax_outlier_threshold}W) en cálculo SIN corrección. Datos restantes: {len(pmax_soiled_filtered)}")
            
            # Excluir casos donde la referencia es < 200W (anormalmente baja para este sistema)
            valid_reference_mask = pmax_reference_filtered >= 200.0
            pmax_reference_filtered = pmax_reference_filtered[valid_reference_mask]
            pmax_soiled_filtered = pmax_soiled_filtered[valid_reference_mask]
            
            denom_pmax = pmax_reference_filtered.replace(0, np.nan)
            if denom_pmax.count() > 0:
                sr_pmax_calc_raw = pmax_soiled_filtered.div(denom_pmax)
                
                # Aplicar filtro variable por fecha: 101% para ago-sep, 100% para oct en adelante
                early_months_cutoff = pd.Timestamp('2024-10-01', tz='UTC')
                early_mask = sr_pmax_calc_raw.index < early_months_cutoff
                late_mask = sr_pmax_calc_raw.index >= early_months_cutoff
                
                # Filtro para primeros meses (ago-sep): hasta 101%
                sr_pmax_early = sr_pmax_calc_raw[early_mask]
                sr_pmax_early_filtered = sr_pmax_early[(sr_pmax_early.notna()) & (sr_pmax_early >= sr_min_filter_threshold) & (sr_pmax_early <= 1.01)]
                
                # Filtro para meses posteriores (oct en adelante): hasta 100%
                sr_pmax_late = sr_pmax_calc_raw[late_mask]
                sr_pmax_late_filtered = sr_pmax_late[(sr_pmax_late.notna()) & (sr_pmax_late >= sr_min_filter_threshold) & (sr_pmax_late <= 1.0)]
                
                # Combinar ambos filtros
                sr_pmax_filtered = pd.concat([sr_pmax_early_filtered, sr_pmax_late_filtered]).sort_index()
                
                sr_pmax_pvstand_raw_no_offset = (100 * sr_pmax_filtered).rename("SR_Pmax_Uncorrected_Raw_NoOffset")
                logger.info(f"SR Pmax (sin corregir, raw, ANTES de offset) calculado: {len(sr_pmax_pvstand_raw_no_offset)} puntos válidos. Filtro variable: ≤101% (ago-sep), ≤100% (oct+)")
        else: logger.warning(f"Columnas faltantes para SR Pmax sin corregir: {col_pmax_soiled} o {col_pmax_reference}")
    else:
        logger.warning("PVStand IV remuestreado está vacío. No se pueden calcular SRs sin corrección.")
        # Si no hay datos IV, no tiene sentido continuar con el resto que depende de ellos.
        return False


    # --- Merge y Cálculo de SR CON corrección de temperatura ---
    df_merged_for_corr = pd.DataFrame()
    if not df_pvstand_resampled.empty and not df_temp_resampled.empty:
        logger.info("Alineando datos PVStand remuestreados con Temperatura remuestreada usando merge_asof (nearest)...")
        df_merged_for_corr = pd.merge_asof(
            df_pvstand_resampled.sort_index(), 
            df_temp_resampled.sort_index(), 
            left_index=True, 
            right_index=True, 
            direction='nearest', 
            tolerance=pd.Timedelta(minutes=resample_freq_minutes)
        )
        logger.info(f"Datos PVStand y Temperatura alineados (df_merged_for_corr ANTES dropna temp cols): {len(df_merged_for_corr)} puntos.")
        if not df_merged_for_corr.empty:
            logger.info(f"Rango df_merged_for_corr (ANTES dropna temp cols): {df_merged_for_corr.index.min()} a {df_merged_for_corr.index.max()}")

        if temp_sensor_soiled_col in df_merged_for_corr.columns and temp_sensor_reference_col in df_merged_for_corr.columns:
             # Filtrar valores nulos Y valores cero anómalos en temperatura
             df_merged_for_corr.dropna(subset=[temp_sensor_soiled_col, temp_sensor_reference_col], how='any', inplace=True)
             
             # Filtrar temperaturas cero o anormalmente bajas (< 5°C) que causan correcciones extremas
             temp_valid_mask = (df_merged_for_corr[temp_sensor_soiled_col] > 5.0) & (df_merged_for_corr[temp_sensor_reference_col] > 5.0)
             df_merged_for_corr = df_merged_for_corr[temp_valid_mask]
             logger.info(f"Filtrados valores de temperatura anómalos (≤5°C). Datos restantes: {len(df_merged_for_corr)}")
             
             # Filtrar temperaturas extremadamente altas (>80°C) que pueden ser errores de sensor
             temp_high_valid_mask = (df_merged_for_corr[temp_sensor_soiled_col] <= 80.0) & (df_merged_for_corr[temp_sensor_reference_col] <= 80.0)
             df_merged_for_corr = df_merged_for_corr[temp_high_valid_mask]
             logger.info(f"Filtradas temperaturas extremadamente altas (>80°C). Datos restantes: {len(df_merged_for_corr)}")
             
             # Filtrar diferencias de temperatura extremas (>25°C) que causan correcciones problemáticas
             temp_diff = abs(df_merged_for_corr[temp_sensor_soiled_col] - df_merged_for_corr[temp_sensor_reference_col])
             temp_diff_valid_mask = temp_diff <= 25.0
             df_merged_for_corr = df_merged_for_corr[temp_diff_valid_mask]
             logger.info(f"Filtradas diferencias de temperatura extremas (>25°C). Datos restantes: {len(df_merged_for_corr)}")
             
             # Filtrar casos donde la corrección de temperatura podría causar divisiones problemáticas
             # Evitar que (1 + beta * (T - Tref)) se acerque mucho a cero
             temp_corr_factor_soiled = 1 + beta_pmax_corr * (df_merged_for_corr[temp_sensor_soiled_col] - temp_ref_correction_c)
             temp_corr_factor_ref = 1 + beta_pmax_corr * (df_merged_for_corr[temp_sensor_reference_col] - temp_ref_correction_c)
             temp_corr_valid_mask = (temp_corr_factor_soiled > 0.5) & (temp_corr_factor_soiled < 2.0) & (temp_corr_factor_ref > 0.5) & (temp_corr_factor_ref < 2.0)
             df_merged_for_corr = df_merged_for_corr[temp_corr_valid_mask]
             logger.info(f"Filtrados factores de corrección de temperatura extremos. Datos restantes: {len(df_merged_for_corr)}")
             
             # Filtrar outliers estadísticos en las temperaturas usando IQR
             for temp_col in [temp_sensor_soiled_col, temp_sensor_reference_col]:
                 Q1 = df_merged_for_corr[temp_col].quantile(0.25)
                 Q3 = df_merged_for_corr[temp_col].quantile(0.75)
                 IQR = Q3 - Q1
                 lower_bound = Q1 - 3.0 * IQR  # Usar 3.0 IQR en lugar de 1.5 para ser menos restrictivo
                 upper_bound = Q3 + 3.0 * IQR
                 outlier_mask = (df_merged_for_corr[temp_col] >= lower_bound) & (df_merged_for_corr[temp_col] <= upper_bound)
                 df_merged_for_corr = df_merged_for_corr[outlier_mask]
                 logger.info(f"Filtrados outliers estadísticos en {temp_col} (IQR). Datos restantes: {len(df_merged_for_corr)}")
             
             # Filtrar valores de Pmax menores a 170W (considerados outliers)
             pmax_outlier_threshold = 170.0  # Watts
             pmax_valid_mask = (df_merged_for_corr[col_pmax_soiled] >= pmax_outlier_threshold) & (df_merged_for_corr[col_pmax_reference] >= pmax_outlier_threshold)
             df_merged_for_corr = df_merged_for_corr[pmax_valid_mask]
             logger.info(f"Filtrados outliers de Pmax (<{pmax_outlier_threshold}W). Datos restantes: {len(df_merged_for_corr)}")
        else:
            logger.warning(f"Columnas de temperatura ({temp_sensor_soiled_col}, {temp_sensor_reference_col}) no encontradas en df_merged_for_corr. No se puede filtrar por ellas.")
        
        logger.info(f"Datos PVStand y Temperatura alineados (df_merged_for_corr DESPUÉS dropna temp cols): {len(df_merged_for_corr)} puntos.")
        if not df_merged_for_corr.empty:
            logger.info(f"Rango df_merged_for_corr (DESPUÉS dropna temp cols): {df_merged_for_corr.index.min()} a {df_merged_for_corr.index.max()}")
        
        if df_merged_for_corr.empty: 
            logger.warning("El DataFrame fusionado para corrección de temperatura está vacío después de dropna. No se calcularán SRs corregidos.")
        else:
            logger.info("Calculando SRs CON corrección de temperatura...")
            required_cols_corr = [col_isc_soiled, col_isc_reference, col_pmax_soiled, col_pmax_reference, temp_sensor_soiled_col, temp_sensor_reference_col]
            if all(c in df_merged_for_corr.columns for c in required_cols_corr):
                isc_soiled_corr_val = df_merged_for_corr[col_isc_soiled] / (1 + alpha_isc_corr * (df_merged_for_corr[temp_sensor_soiled_col] - temp_ref_correction_c))
                isc_ref_corr_val = df_merged_for_corr[col_isc_reference] / (1 + alpha_isc_corr * (df_merged_for_corr[temp_sensor_reference_col] - temp_ref_correction_c))
                
                # Filtrar valores anormalmente bajos en referencia corregida que causan SRs extremos
                valid_isc_reference_corr_mask = isc_ref_corr_val >= 0.5
                isc_soiled_corr_filtered = isc_soiled_corr_val[valid_isc_reference_corr_mask]
                isc_ref_corr_filtered = isc_ref_corr_val[valid_isc_reference_corr_mask]
                
                denom_isc_c = isc_ref_corr_filtered.replace(0, np.nan)
                if denom_isc_c.count() > 0:
                    sr_isc_c_calc_raw = isc_soiled_corr_filtered.div(denom_isc_c)
                    # Aplicar filtro variable por fecha: 101% para ago-sep, 100% para oct en adelante
                    early_months_cutoff = pd.Timestamp('2024-10-01', tz='UTC')
                    early_mask = sr_isc_c_calc_raw.index < early_months_cutoff
                    late_mask = sr_isc_c_calc_raw.index >= early_months_cutoff
                    
                    # Filtro para primeros meses (ago-sep): hasta 101%
                    sr_isc_c_early = sr_isc_c_calc_raw[early_mask]
                    sr_isc_c_early_filtered = sr_isc_c_early[(sr_isc_c_early.notna()) & (sr_isc_c_early >= sr_min_filter_threshold) & (sr_isc_c_early <= 1.01)]
                    
                    # Filtro para meses posteriores (oct en adelante): hasta 100%
                    sr_isc_c_late = sr_isc_c_calc_raw[late_mask]
                    sr_isc_c_late_filtered = sr_isc_c_late[(sr_isc_c_late.notna()) & (sr_isc_c_late >= sr_min_filter_threshold) & (sr_isc_c_late <= 1.0)]
                    
                    # Combinar ambos filtros
                    sr_isc_c_filtered = pd.concat([sr_isc_c_early_filtered, sr_isc_c_late_filtered]).sort_index()
                    
                    # Filtro adicional para detectar y eliminar picos anómalos usando rolling median
                    if len(sr_isc_c_filtered) > 50:  # Solo aplicar si hay suficientes datos
                        window_size = min(50, len(sr_isc_c_filtered) // 10)  # Ventana adaptativa
                        rolling_median = sr_isc_c_filtered.rolling(window=window_size, center=True, min_periods=window_size//2).median()
                        deviation_threshold = 0.05  # 5% de desviación permitida del median móvil
                        spike_mask = abs(sr_isc_c_filtered - rolling_median) <= deviation_threshold
                        sr_isc_c_filtered = sr_isc_c_filtered[spike_mask]
                        logger.info(f"Filtrados picos anómalos en SR Isc corregido usando rolling median. Datos restantes: {len(sr_isc_c_filtered)}")
                    
                    sr_isc_pvstand_corrected_raw_no_offset = (100 * sr_isc_c_filtered).rename("SR_Isc_Corrected_Raw_NoOffset")
                    logger.info(f"SR Isc (corregido, raw) calculado: {len(sr_isc_pvstand_corrected_raw_no_offset)} puntos válidos.")

                pmax_soiled_corr_val = df_merged_for_corr[col_pmax_soiled] / (1 + beta_pmax_corr * (df_merged_for_corr[temp_sensor_soiled_col] - temp_ref_correction_c))
                pmax_ref_corr_val = df_merged_for_corr[col_pmax_reference] / (1 + beta_pmax_corr * (df_merged_for_corr[temp_sensor_reference_col] - temp_ref_correction_c))
                
                # Filtrar valores anormalmente bajos en referencia corregida que causan SRs extremos
                valid_reference_corr_mask = pmax_ref_corr_val >= 200.0
                pmax_soiled_corr_filtered = pmax_soiled_corr_val[valid_reference_corr_mask]
                pmax_ref_corr_filtered = pmax_ref_corr_val[valid_reference_corr_mask]
                
                denom_pmax_c = pmax_ref_corr_filtered.replace(0, np.nan)
                if denom_pmax_c.count() > 0:
                    sr_pmax_c_calc_raw = pmax_soiled_corr_filtered.div(denom_pmax_c)
                    # Aplicar filtro variable por fecha: 101% para ago-sep, 100% para oct en adelante
                    early_months_cutoff = pd.Timestamp('2024-10-01', tz='UTC')
                    early_mask = sr_pmax_c_calc_raw.index < early_months_cutoff
                    late_mask = sr_pmax_c_calc_raw.index >= early_months_cutoff
                    
                    # Filtro para primeros meses (ago-sep): hasta 101%
                    sr_pmax_c_early = sr_pmax_c_calc_raw[early_mask]
                    sr_pmax_c_early_filtered = sr_pmax_c_early[(sr_pmax_c_early.notna()) & (sr_pmax_c_early >= sr_min_filter_threshold) & (sr_pmax_c_early <= 1.01)]
                    
                    # Filtro para meses posteriores (oct en adelante): hasta 100%
                    sr_pmax_c_late = sr_pmax_c_calc_raw[late_mask]
                    sr_pmax_c_late_filtered = sr_pmax_c_late[(sr_pmax_c_late.notna()) & (sr_pmax_c_late >= sr_min_filter_threshold) & (sr_pmax_c_late <= 1.0)]
                    
                    # Combinar ambos filtros
                    sr_pmax_c_filtered = pd.concat([sr_pmax_c_early_filtered, sr_pmax_c_late_filtered]).sort_index()
                    
                    # Filtro adicional para detectar y eliminar picos anómalos usando rolling median
                    if len(sr_pmax_c_filtered) > 50:  # Solo aplicar si hay suficientes datos
                        window_size = min(50, len(sr_pmax_c_filtered) // 10)  # Ventana adaptativa
                        rolling_median = sr_pmax_c_filtered.rolling(window=window_size, center=True, min_periods=window_size//2).median()
                        deviation_threshold = 0.05  # 5% de desviación permitida del median móvil
                        spike_mask = abs(sr_pmax_c_filtered - rolling_median) <= deviation_threshold
                        sr_pmax_c_filtered = sr_pmax_c_filtered[spike_mask]
                        logger.info(f"Filtrados picos anómalos en SR Pmax corregido usando rolling median. Datos restantes: {len(sr_pmax_c_filtered)}")
                    
                    sr_pmax_pvstand_corrected_raw_no_offset = (100 * sr_pmax_c_filtered).rename("SR_Pmax_Corrected_Raw_NoOffset")
                    logger.info(f"SR Pmax (corregido, raw, ANTES de offset) calculado: {len(sr_pmax_pvstand_corrected_raw_no_offset)} puntos válidos. Filtro variable: ≤101% (ago-sep), ≤100% (oct+)")
            else:
                missing_actual = [c for c in required_cols_corr if c not in df_merged_for_corr.columns]
                logger.warning(f"Columnas faltantes en df_merged_for_corr para SR corregido: {missing_actual}. No se calcularán SRs corregidos.")
    else:
        logger.warning("Uno o ambos DataFrames (PVStand IV remuestreado, Temperatura remuestreada) están vacíos. No se puede realizar merge ni calcular SRs corregidos.")

    # --- Preparación de SRs para guardado y graficado ---
    # Copiar _raw_no_offset a las series base que podrían recibir offset
    sr_isc_pvstand = sr_isc_pvstand_raw_no_offset.copy()
    sr_pmax_pvstand = sr_pmax_pvstand_raw_no_offset.copy()
    sr_isc_pvstand_corrected = sr_isc_pvstand_corrected_raw_no_offset.copy()
    sr_pmax_pvstand_corrected = sr_pmax_pvstand_corrected_raw_no_offset.copy()

    if pmax_sr_offset != 0:
        logger.info(f"Aplicando offset de {pmax_sr_offset}% a SR Pmax (no corregido y corregido si existen).")
        if not sr_pmax_pvstand.empty: sr_pmax_pvstand += pmax_sr_offset
        if not sr_pmax_pvstand_corrected.empty: sr_pmax_pvstand_corrected += pmax_sr_offset
    
    sr_isc_pvstand.name = "SR_Isc_Uncorrected"
    sr_pmax_pvstand.name = "SR_Pmax_Uncorrected_Offset"
    sr_isc_pvstand_corrected.name = "SR_Isc_Corrected"
    sr_pmax_pvstand_corrected.name = "SR_Pmax_Corrected_Offset"

    sr_isc_pvstand_no_norm = sr_isc_pvstand.copy().rename("SR_Isc_Uncorrected_NoNorm")
    sr_pmax_pvstand_no_norm = sr_pmax_pvstand.copy().rename("SR_Pmax_Uncorrected_Offset_NoNorm")
    sr_isc_pvstand_corrected_no_norm = sr_isc_pvstand_corrected.copy().rename("SR_Isc_Corrected_NoNorm")
    sr_pmax_pvstand_corrected_no_norm = sr_pmax_pvstand_corrected.copy().rename("SR_Pmax_Corrected_Offset_NoNorm")
    
    if normalize_sr_flag:
        logger.info(f"Normalizando SRs principales (con offset si aplica) usando fecha de referencia: {normalize_sr_ref_date_str}...")
        if not sr_isc_pvstand.empty: sr_isc_pvstand = normalize_series_from_date_pd(sr_isc_pvstand, normalize_sr_ref_date_str, sr_isc_pvstand.name, target_ref_value=100.0)
        if not sr_pmax_pvstand.empty: sr_pmax_pvstand = normalize_series_from_date_pd(sr_pmax_pvstand, normalize_sr_ref_date_str, sr_pmax_pvstand.name, target_ref_value=100.0)
        if not sr_isc_pvstand_corrected.empty: sr_isc_pvstand_corrected = normalize_series_from_date_pd(sr_isc_pvstand_corrected, normalize_sr_ref_date_str, sr_isc_pvstand_corrected.name, target_ref_value=100.0)
        if not sr_pmax_pvstand_corrected.empty: sr_pmax_pvstand_corrected = normalize_series_from_date_pd(sr_pmax_pvstand_corrected, normalize_sr_ref_date_str, sr_pmax_pvstand_corrected.name, target_ref_value=100.0)

    # --- Guardar CSVs ---
    pvstand_graph_subdir_name = "pvstand" # Subdirectorio para gráficos
    graph_base_plus_subdir = os.path.join(output_graph_dir, pvstand_graph_subdir_name)
    os.makedirs(graph_base_plus_subdir, exist_ok=True) # Asegurar que el subdirectorio de gráficos existe

    # --- Guardar datos utilizados para cálculos de SR (Temperatura + Potencias) ---
    if not df_merged_for_corr.empty:
        # Extraer datos de temperatura y potencias utilizados para los cálculos
        df_data_used_for_sr = df_merged_for_corr[[
            col_pmax_soiled, col_pmax_reference, col_isc_soiled, col_isc_reference,
            temp_sensor_soiled_col, temp_sensor_reference_col
        ]].copy()
        
        # Renombrar columnas para mayor claridad
        df_data_used_for_sr.rename(columns={
            col_pmax_soiled: 'Pmax_Soiled_Original_W',
            col_pmax_reference: 'Pmax_Reference_Original_W',
            col_isc_soiled: 'Isc_Soiled_Original_A',
            col_isc_reference: 'Isc_Reference_Original_A',
            temp_sensor_soiled_col: 'Temp_Modulo_Soiled_C',
            temp_sensor_reference_col: 'Temp_Modulo_Reference_C'
        }, inplace=True)
        
        # Calcular diferencia de temperatura
        df_data_used_for_sr['Diferencia_Temperatura_C'] = abs(
            df_data_used_for_sr['Temp_Modulo_Soiled_C'] - 
            df_data_used_for_sr['Temp_Modulo_Reference_C']
        )
        
        # Calcular factores de corrección de temperatura
        df_data_used_for_sr['Factor_Corr_Isc_Soiled'] = 1 + alpha_isc_corr * (df_data_used_for_sr['Temp_Modulo_Soiled_C'] - temp_ref_correction_c)
        df_data_used_for_sr['Factor_Corr_Isc_Reference'] = 1 + alpha_isc_corr * (df_data_used_for_sr['Temp_Modulo_Reference_C'] - temp_ref_correction_c)
        df_data_used_for_sr['Factor_Corr_Pmax_Soiled'] = 1 + beta_pmax_corr * (df_data_used_for_sr['Temp_Modulo_Soiled_C'] - temp_ref_correction_c)
        df_data_used_for_sr['Factor_Corr_Pmax_Reference'] = 1 + beta_pmax_corr * (df_data_used_for_sr['Temp_Modulo_Reference_C'] - temp_ref_correction_c)
        
        # Calcular potencias e corrientes corregidas por temperatura
        df_data_used_for_sr['Pmax_Soiled_Temp_Corrected_W'] = df_data_used_for_sr['Pmax_Soiled_Original_W'] / df_data_used_for_sr['Factor_Corr_Pmax_Soiled']
        df_data_used_for_sr['Pmax_Reference_Temp_Corrected_W'] = df_data_used_for_sr['Pmax_Reference_Original_W'] / df_data_used_for_sr['Factor_Corr_Pmax_Reference']
        df_data_used_for_sr['Isc_Soiled_Temp_Corrected_A'] = df_data_used_for_sr['Isc_Soiled_Original_A'] / df_data_used_for_sr['Factor_Corr_Isc_Soiled']
        df_data_used_for_sr['Isc_Reference_Temp_Corrected_A'] = df_data_used_for_sr['Isc_Reference_Original_A'] / df_data_used_for_sr['Factor_Corr_Isc_Reference']
        
        # Calcular SRs instantáneos para verificación
        df_data_used_for_sr['SR_Pmax_Original_Percent'] = 100 * (df_data_used_for_sr['Pmax_Soiled_Original_W'] / df_data_used_for_sr['Pmax_Reference_Original_W'])
        df_data_used_for_sr['SR_Pmax_Temp_Corrected_Percent'] = 100 * (df_data_used_for_sr['Pmax_Soiled_Temp_Corrected_W'] / df_data_used_for_sr['Pmax_Reference_Temp_Corrected_W'])
        df_data_used_for_sr['SR_Isc_Original_Percent'] = 100 * (df_data_used_for_sr['Isc_Soiled_Original_A'] / df_data_used_for_sr['Isc_Reference_Original_A'])
        df_data_used_for_sr['SR_Isc_Temp_Corrected_Percent'] = 100 * (df_data_used_for_sr['Isc_Soiled_Temp_Corrected_A'] / df_data_used_for_sr['Isc_Reference_Temp_Corrected_A'])
        
        csv_filename_sr_data = os.path.join(output_csv_dir, "pvstand_datos_completos_calculos_sr.csv")
        df_data_used_for_sr.to_csv(csv_filename_sr_data)
        logger.info(f"Datos completos utilizados para cálculos de SR guardados en: {csv_filename_sr_data}")

    df_sr_to_save_main = pd.DataFrame({
        sr_isc_pvstand.name: sr_isc_pvstand,
        sr_pmax_pvstand.name: sr_pmax_pvstand,
        sr_isc_pvstand_corrected.name: sr_isc_pvstand_corrected,
        sr_pmax_pvstand_corrected.name: sr_pmax_pvstand_corrected
    }).sort_index().dropna(how='all')
    
    if not df_sr_to_save_main.empty:
        norm_suffix = 'norm' if normalize_sr_flag else 'abs'
        csv_filename_main = os.path.join(output_csv_dir, f"pvstand_sr_main_{norm_suffix}.csv")
        df_sr_to_save_main.to_csv(csv_filename_main)
        logger.info(f"PVStand SRs (main) guardados en: {csv_filename_main}")

    df_sr_to_save_no_norm = pd.DataFrame({
        sr_isc_pvstand_no_norm.name: sr_isc_pvstand_no_norm,
        sr_pmax_pvstand_no_norm.name: sr_pmax_pvstand_no_norm,
        sr_isc_pvstand_corrected_no_norm.name: sr_isc_pvstand_corrected_no_norm,
        sr_pmax_pvstand_corrected_no_norm.name: sr_pmax_pvstand_corrected_no_norm
    }).sort_index().dropna(how='all')
    if not df_sr_to_save_no_norm.empty:
        csv_filename_no_norm = os.path.join(output_csv_dir, "pvstand_sr_no_norm_with_offset.csv")
        df_sr_to_save_no_norm.to_csv(csv_filename_no_norm)
        logger.info(f"PVStand SRs (no normalizados, con offset) guardados en: {csv_filename_no_norm}")

    df_sr_to_save_raw = pd.DataFrame({
        sr_isc_pvstand_raw_no_offset.name: sr_isc_pvstand_raw_no_offset,
        sr_pmax_pvstand_raw_no_offset.name: sr_pmax_pvstand_raw_no_offset,
        sr_isc_pvstand_corrected_raw_no_offset.name: sr_isc_pvstand_corrected_raw_no_offset,
        sr_pmax_pvstand_corrected_raw_no_offset.name: sr_pmax_pvstand_corrected_raw_no_offset
    }).sort_index().dropna(how='all')
    if not df_sr_to_save_raw.empty:
        csv_filename_raw = os.path.join(output_csv_dir, "pvstand_sr_raw_no_offset.csv")
        df_sr_to_save_raw.to_csv(csv_filename_raw)
        logger.info(f"PVStand SRs (raw, sin offset) guardados en: {csv_filename_raw}")

    # --- Gráfico de Potencias Brutas (Sin filtros ni promedios) ---
    def _plot_raw_power_data(df_raw_power_data, output_dir, save_figs, show_figs):
        """
        Genera gráfico de potencias brutas sin filtros ni promedios.
        Muestra los datos tal como están en el archivo original.
        """
        if df_raw_power_data.empty:
            logger.info("No hay datos de potencias brutas para graficar.")
            return
            
        logger.info("Generando gráfico de potencias brutas (sin filtros)...")
        
        # Identificar columnas de potencia
        pmax_cols = [col for col in df_raw_power_data.columns if 'Pmax' in col]
        if not pmax_cols:
            logger.warning("No se encontraron columnas de Pmax para el gráfico de potencias brutas.")
            return
            
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Colores para cada serie
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, col in enumerate(pmax_cols):
            data_series = df_raw_power_data[col].dropna()
            if not data_series.empty:
                ax.plot(data_series.index, data_series.values, 
                       color=colors[i % len(colors)], 
                       alpha=0.7, 
                       linewidth=0.5,
                       label=col.replace('_', ' '))
        
        ax.set_title('Potencias Brutas PVStand (Datos Originales - Sin Filtros)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Potencia [W]', fontsize=14)
        ax.set_xlabel('Fecha', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=12)
        
        # Formatear fechas en eje x
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='both', labelsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Ajustar límites del eje y para mejor visualización
        y_max = df_raw_power_data[pmax_cols].max().max()
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.1)
            
        plt.tight_layout()
        
        # Guardar gráfico
        plot_filename = "pvstand_potencias_brutas_sin_filtros.png"
        if save_figs:
            save_plot_matplotlib(fig, plot_filename, output_dir, subfolder=None)
            logger.info(f"Gráfico de potencias brutas guardado: {plot_filename}")
        
        if show_figs:
            plt.show(block=True)
            logger.info("Gráfico de potencias brutas mostrado")
        else:
            plt.close(fig)

    # --- Graficado (Lógica de _plot_sr_section incorporada) ---
    # Colores fijos por posición para todos los gráficos
    colores_fijos = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    def limpiar_leyenda(nombre):
        nombre = nombre.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').replace('SR ', 'SR ').strip()
        return nombre

    def _plot_sr_section_internal(df_to_plot, title_prefix, filename_suffix, is_normalized_section_flag_param):
        if df_to_plot.empty:
            logger.info(f"No hay datos para graficar en la sección: {title_prefix}")
            return

        logger.info(f"Generando gráficos para: {title_prefix}")
        num_series = len(df_to_plot.columns)
        if num_series == 0: return
            
        line_styles = ['-', '--', '-.', ':'] * (num_series // 4 + 1)
        base_markersize = 4 if 'Media Móvil' not in filename_suffix else 2
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_'] * (num_series // 16 + 1)

        plotting_configs_local = [
            ('1W', f'Semanal Q({graph_quantile*100:.0f}%)'), 
            ('3D', f'3 Días Q({graph_quantile*100:.0f}%)')
        ]
        
        for resample_rule_str, plot_desc_agg_str in plotting_configs_local:
            fig, ax = plt.subplots(figsize=(15, 8))
            has_data_for_this_agg_plot = False
            tendencias_info = []  # Para guardar info de tendencias
            for i, col_name_plot in enumerate(df_to_plot.columns):
                series_plot = df_to_plot[col_name_plot]
                if not isinstance(series_plot, pd.Series) or series_plot.dropna().empty or not isinstance(series_plot.index, pd.DatetimeIndex):
                    continue
                try:
                    data_agg = pd.Series(dtype=float)
                    if 'Media Móvil' in plot_desc_agg_str:
                        min_periods_val = 1
                        if len(series_plot.dropna()) >= 1:
                            data_agg = series_plot.dropna().rolling(window=resample_rule_str, center=True, min_periods=min_periods_val).quantile(graph_quantile).dropna()
                        else: 
                            data_agg = series_plot.resample('D').quantile(graph_quantile).dropna() 
                    else:
                        data_agg = series_plot.resample(resample_rule_str).quantile(graph_quantile).dropna()
                    if not data_agg.empty:
                        ax.plot(data_agg.index, data_agg.values, 
                                linestyle=line_styles[i % len(line_styles)], 
                                marker=markers[i % len(markers)] if 'Media Móvil' not in plot_desc_agg_str else None, 
                                markersize=base_markersize, alpha=0.8, label=limpiar_leyenda(col_name_plot), color=colores_fijos[i % len(colores_fijos)])
                        has_data_for_this_agg_plot = True
                        # --- Línea de tendencia global, solo para el gráfico semanal normalizado ---
                        if filename_suffix == 'norm' and 'semanal' in plot_desc_agg_str.lower():
                            import numpy as np
                            from sklearn.linear_model import LinearRegression
                            from sklearn.metrics import r2_score
                            x = np.arange(len(data_agg)).reshape(-1, 1)
                            y = data_agg.values.reshape(-1, 1)
                            model = LinearRegression().fit(x, y)
                            y_pred = model.predict(x)
                            pendiente = model.coef_[0][0]
                            r2 = r2_score(y, y_pred)
                            pendiente_semana = pendiente
                            # Línea de tendencia: recta, continua, opaca, sin punteo, mismo color
                            ax.plot(data_agg.index, y_pred.flatten(), '-', color=colores_fijos[i % len(colores_fijos)], alpha=0.5, linewidth=2, label=f"Tendencia SR: {pendiente_semana:.3f} [%/semana], R2={r2:.2f}")
                except Exception as e_plot:
                    logger.error(f"Error graficando '{col_name_plot}' para '{plot_desc_agg_str}' en '{title_prefix}': {e_plot}", exc_info=True)
            
            if has_data_for_this_agg_plot:
                # Título especial para el gráfico semanal normalizado
                if filename_suffix == 'norm' and 'semanal' in plot_desc_agg_str.lower():
                    current_plot_title = 'Soiling Ratio PVStand'
                else:
                    current_plot_title = f'{title_prefix} ({plot_desc_agg_str})'

                ax.set_title(current_plot_title, fontsize=20)
                ax.set_ylabel('Soiling Ratio Normalizado [%]' if is_normalized_section_flag_param and normalize_sr_flag else 'Soiling Ratio [%]', fontsize=18)
                ax.set_xlabel('Fecha', fontsize=18)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.legend(loc='best', fontsize=14)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='both', labelsize=14)
                plt.xticks(rotation=30, ha='right', fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                
                file_suffix_cleaned = filename_suffix.lower().replace(" ", "_").replace("(", "").replace(")", "")
                agg_cleaned = plot_desc_agg_str.lower().replace(' ', '_').replace('%', '').replace('(', '').replace(')', '').replace('.', '')
                plot_filename = f"pvstand_sr_{file_suffix_cleaned}_{agg_cleaned}.png"
                
                if save_figures_setting: 
                    save_plot_matplotlib(fig, plot_filename, graph_base_plus_subdir, subfolder=None)
                    logger.info(f"Gáfico guardado en: {os.path.join(graph_base_plus_subdir, plot_filename)}")
                
                if show_figures_setting: 
                    plt.show(block=True)  # Agregado block=True para asegurar que se muestre
                    logger.info("Gráfico mostrado")
                else:
                    plt.close(fig)
            else:
                logger.info(f"No hay datos para graficar para {title_prefix} ({plot_desc_agg_str}). Plot omitido.")
                if 'fig' in locals() and fig is not None and plt.fignum_exists(fig.number): plt.close(fig)

    _plot_sr_section_internal(df_sr_to_save_main, 
                     f"PVStand SRs ({'Normalizado' if normalize_sr_flag else 'Absoluto'})", 
                     f"{'norm' if normalize_sr_flag else 'abs'}", 
                     is_normalized_section_flag_param=True) 

    _plot_sr_section_internal(df_sr_to_save_no_norm, 
                     "PVStand SRs (No Normalizado, Con Offset Pmax)", 
                     "no_norm_with_offset", 
                     is_normalized_section_flag_param=False)

    _plot_sr_section_internal(df_sr_to_save_raw, 
                     "PVStand SRs (Raw, Sin Offset)", 
                     "raw_no_offset", 
                     is_normalized_section_flag_param=False)

    # --- Generar gráfico de potencias brutas ---
    if 'df_pvstand_original_unfiltered' in locals() and not df_pvstand_original_unfiltered.empty:
        logger.info("Generando gráfico de potencias brutas (datos originales sin filtros)...")
        _plot_raw_power_data(df_pvstand_original_unfiltered, graph_base_plus_subdir, save_figures_setting, show_figures_setting)
    else:
        logger.warning("No hay datos originales disponibles para el gráfico de potencias brutas.")

    # --- Generar Excel consolidado con datos procesados (mismo procesamiento que gráficos) ---
    logger.info("Generando Excel consolidado con datos agregados...")
    
    # --- Generar Excel consolidado con todas las tablas ---
    try:
        import openpyxl
        excel_filename = f"pvstand_datos_completos_agregados_Q{int(graph_quantile*100)}.xlsx"
        excel_filepath = os.path.join(output_csv_dir, excel_filename)
        
        with pd.ExcelWriter(excel_filepath, engine='openpyxl',
                           date_format='YYYY-MM-DD', 
                           datetime_format='YYYY-MM-DD HH:MM:SS') as writer:
            # Hoja con datos principales (normalizados o absolutos)
            if not df_sr_to_save_main.empty:
                # Datos semanales
                df_main_weekly = pd.DataFrame()
                for col_name in df_sr_to_save_main.columns:
                    series_data = df_sr_to_save_main[col_name]
                    if not series_data.dropna().empty:
                        data_agg = series_data.resample('1W').quantile(graph_quantile).dropna()
                        if not data_agg.empty:
                            clean_col_name = col_name.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').strip()
                            df_main_weekly[clean_col_name] = data_agg
                
                if not df_main_weekly.empty:
                    sheet_name = f"SR_{'Norm' if normalize_sr_flag else 'Abs'}_Semanal"
                    df_main_weekly.to_excel(writer, sheet_name=sheet_name)
                
                # Datos cada 3 días
                df_main_3d = pd.DataFrame()
                for col_name in df_sr_to_save_main.columns:
                    series_data = df_sr_to_save_main[col_name]
                    if not series_data.dropna().empty:
                        data_agg = series_data.resample('3D').quantile(graph_quantile).dropna()
                        if not data_agg.empty:
                            clean_col_name = col_name.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').strip()
                            df_main_3d[clean_col_name] = data_agg
                
                if not df_main_3d.empty:
                    sheet_name = f"SR_{'Norm' if normalize_sr_flag else 'Abs'}_3Dias"
                    df_main_3d.to_excel(writer, sheet_name=sheet_name)
            
            # Hoja con datos raw
            if not df_sr_to_save_raw.empty:
                df_raw_weekly = pd.DataFrame()
                for col_name in df_sr_to_save_raw.columns:
                    series_data = df_sr_to_save_raw[col_name]
                    if not series_data.dropna().empty:
                        data_agg = series_data.resample('1W').quantile(graph_quantile).dropna()
                        if not data_agg.empty:
                            clean_col_name = col_name.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').strip()
                            df_raw_weekly[clean_col_name] = data_agg
                
                if not df_raw_weekly.empty:
                    df_raw_weekly.to_excel(writer, sheet_name="SR_Raw_Semanal")
            
            # Hoja con estadísticas consolidadas
            if not df_sr_to_save_main.empty:
                df_main_for_stats = df_sr_to_save_main.copy()
                stats_consolidado = pd.DataFrame({
                    'Serie': df_main_for_stats.columns,
                    'Cantidad_Puntos': [df_main_for_stats[col].count() for col in df_main_for_stats.columns],
                    'Promedio': [df_main_for_stats[col].mean() for col in df_main_for_stats.columns],
                    'Mediana': [df_main_for_stats[col].median() for col in df_main_for_stats.columns],
                    'Desv_Std': [df_main_for_stats[col].std() for col in df_main_for_stats.columns],
                    'Valor_Min': [df_main_for_stats[col].min() for col in df_main_for_stats.columns],
                    'Valor_Max': [df_main_for_stats[col].max() for col in df_main_for_stats.columns],
                    'Rango_Fechas': [f"{df_main_for_stats[col].dropna().index.min().strftime('%Y-%m-%d')} a {df_main_for_stats[col].dropna().index.max().strftime('%Y-%m-%d')}" if df_main_for_stats[col].count() > 0 else "N/A" for col in df_main_for_stats.columns]
                })
                stats_consolidado.to_excel(writer, sheet_name="Estadisticas", index=False)
                
            # Hoja con estadísticas de validez de resamples semanales
            if not df_sr_to_save_main.empty:
                logger.info("Generando estadísticas de validez de resamples semanales...")
                
                # Crear DataFrame para estadísticas de validez semanal
                validez_data = []
                
                for col_name in df_sr_to_save_main.columns:
                    series_original = df_sr_to_save_main[col_name].dropna()
                    if series_original.empty:
                        continue
                        
                    # Agrupar por semanas
                    semanas_grouped = series_original.groupby(pd.Grouper(freq='1W'))
                    
                    for semana_inicio, datos_semana in semanas_grouped:
                        if datos_semana.empty:
                            continue
                            
                        # Calcular estadísticas de validez para esta semana
                        num_puntos = len(datos_semana)
                        
                        # Calcular días teóricos en la semana (máximo 7)
                        semana_fin = semana_inicio + pd.Timedelta(days=6)
                        dias_teoricos = min(7, (min(semana_fin, series_original.index.max()) - max(semana_inicio, series_original.index.min())).days + 1)
                        
                                                # Calcular días con datos reales
                        if hasattr(datos_semana.index, 'date'):
                            dias_con_datos = len(np.unique(datos_semana.index.date))
                        else:
                            dias_con_datos = datos_semana.resample('D').count().astype(bool).sum()
                        
                        cobertura_dias = (dias_con_datos / dias_teoricos * 100) if dias_teoricos > 0 else 0
                        
                        # Estadísticas de dispersión
                        promedio = datos_semana.mean()
                        mediana = datos_semana.median()
                        desv_std = datos_semana.std()
                        coef_variacion = (desv_std / promedio * 100) if promedio != 0 else 0
                        rango = datos_semana.max() - datos_semana.min()
                        
                        # Cuantil usado para el resample
                        valor_resample = datos_semana.quantile(graph_quantile)
                        
                        # Diferencia entre cuantil y promedio (para evaluar sesgo)
                        sesgo_cuantil = abs(valor_resample - promedio)
                        sesgo_cuantil_pct = (sesgo_cuantil / promedio * 100) if promedio != 0 else 0
                        
                        # Evaluación de representatividad
                        if cobertura_dias >= 80 and num_puntos >= 50 and coef_variacion <= 10:
                            representatividad = "Excelente"
                        elif cobertura_dias >= 60 and num_puntos >= 30 and coef_variacion <= 20:
                            representatividad = "Buena"
                        elif cobertura_dias >= 40 and num_puntos >= 15:
                            representatividad = "Aceptable"
                        else:
                            representatividad = "Limitada"
                        
                        validez_data.append({
                            'Serie': col_name.replace('Raw', '').replace('NoOffset', '').replace('_', ' ').strip(),
                            'Semana_Inicio': semana_inicio.strftime('%Y-%m-%d'),
                            'Num_Puntos_Originales': num_puntos,
                            'Dias_Con_Datos': dias_con_datos,
                            'Dias_Teoricos': dias_teoricos,
                            'Cobertura_Dias_Pct': round(cobertura_dias, 1),
                            'Promedio_Original': round(promedio, 2),
                            'Mediana_Original': round(mediana, 2),
                            'Valor_Resample_Q' + str(int(graph_quantile*100)): round(valor_resample, 2),
                            'Desv_Std': round(desv_std, 2),
                            'Coef_Variacion_Pct': round(coef_variacion, 1),
                            'Rango_Min_Max': round(rango, 2),
                            'Sesgo_Cuantil_vs_Promedio': round(sesgo_cuantil, 2),
                            'Sesgo_Cuantil_Pct': round(sesgo_cuantil_pct, 1),
                            'Representatividad': representatividad
                        })
                
                if validez_data:
                    df_validez = pd.DataFrame(validez_data)
                    df_validez.to_excel(writer, sheet_name="Validez_Resamples_Semanales", index=False)
                    logger.info("Hoja de validez de resamples semanales agregada al Excel")
                    
                    # Crear resumen de representatividad por serie
                    resumen_repr = df_validez.groupby('Serie')['Representatividad'].value_counts().unstack(fill_value=0)
                    resumen_repr['Total_Semanas'] = resumen_repr.sum(axis=1)
                    
                    # Calcular porcentajes
                    for col in ['Excelente', 'Buena', 'Aceptable', 'Limitada']:
                        if col in resumen_repr.columns:
                            resumen_repr[f'Pct_{col}'] = round(resumen_repr[col] / resumen_repr['Total_Semanas'] * 100, 1)
                    
                    resumen_repr.to_excel(writer, sheet_name="Resumen_Representatividad")
                    logger.info("Hoja de resumen de representatividad agregada al Excel")
        
        logger.info(f"Excel consolidado con datos agregados guardado: {excel_filepath}")
        
    except ImportError:
        logger.warning("openpyxl no está disponible. No se pudo generar el archivo Excel consolidado.")
    except Exception as e:
        logger.error(f"Error generando Excel consolidado: {e}")

    logger.info("=== FIN DEL ANÁLISIS PVSTAND ===")
    return True

# Fin de la función analyze_pvstand_data 

def run_analysis():
    """
    Función estándar para ejecutar el análisis de PVStand.
    Usa la configuración centralizada para rutas y parámetros.
    """
    pv_iv_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_IV_DATA_FILENAME)
    temperature_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_TEMP_DATA_FILENAME)
    return analyze_pvstand_data(pv_iv_data_filepath, temperature_data_filepath)

if __name__ == "__main__":
    # Solo se ejecuta cuando el archivo se ejecuta directamente
    print("[INFO] Ejecutando análisis de PVStand...")
    pv_iv_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_IV_DATA_FILENAME)
    temperature_data_filepath = os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_TEMP_DATA_FILENAME)
    analyze_pvstand_data(pv_iv_data_filepath, temperature_data_filepath) 