# utils/helpers.py

import os
import matplotlib.pyplot as plt
from config.logging_config import logger # Importar el logger configurado
from config.settings import DPI_FIGURES # Importar DPI desde settings
import pandas as pd # Añadido para normalize_series_from_date_pd
import numpy as np # Añadido para normalize_series_from_date_pd
from config import paths
import polars as pl # Añadido para save_df_csv

def save_plot_matplotlib(fig, output_filename, output_dir, dpi=300, bbox_inches='tight'):
    """
    Guarda un gráfico de matplotlib en un archivo.
    
    Args:
        fig: Figura de matplotlib
        output_filename: Nombre del archivo de salida
        output_dir: Directorio de salida
        dpi: Resolución del gráfico
        bbox_inches: Ajuste del borde
    """
    try:
        # Asegurar que el directorio existe
        full_output_dir = os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Construir la ruta completa
        full_path = os.path.join(full_output_dir, output_filename)
        
        # Guardar el gráfico
        fig.savefig(full_path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
        
        logger.info(f"Gráfico guardado exitosamente en: {full_path}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar gráfico {output_filename}: {e}")
        return False

def normalize_series_from_date_pd(series: pd.Series, ref_date_str: str, column_name: str = "", target_ref_value: float = 100.0) -> pd.Series:
    """
    Normaliza una serie de Pandas (pd.Series) con DatetimeIndex para que el valor en o 
    después de ref_date_str sea target_ref_value (por defecto 100).
    Maneja series con y sin timezone.
    Args:
        series (pd.Series): La serie a normalizar. Debe tener un DatetimeIndex.
        ref_date_str (str): La fecha de referencia en formato string (YYYY-MM-DD o similar parseable por pd.Timestamp).
        column_name (str, optional): Nombre de la serie, usado para logging. Por defecto "".
        target_ref_value (float, optional): Valor al que se normalizará la fecha de referencia. Por defecto 100.0.
    Returns:
        pd.Series: La serie normalizada, o la original si ocurre un error o no se puede normalizar.
    """
    if not isinstance(series, pd.Series) or series.empty:
        logger.warning(f"La entrada para normalización '{column_name}' no es una Serie válida o está vacía. No se puede normalizar.")
        return series
    
    if not isinstance(series.index, pd.DatetimeIndex):
        logger.warning(f"Índice de la serie '{column_name}' no es DatetimeIndex. No se puede normalizar por fecha.")
        return series

    try:
        ref_date_input = pd.Timestamp(ref_date_str) 
    except Exception as e:
        logger.error(f"Error convirtiendo fecha de referencia '{ref_date_str}' para normalización de '{column_name}': {e}")
        return series

    series_tz = series.index.tz
    ref_date_aware = None

    if series_tz is not None: 
        if ref_date_input.tz is None: 
            try:
                ref_date_aware = ref_date_input.tz_localize(series_tz) 
            except Exception as e_localize:
                logger.warning(f"No se pudo localizar la fecha de referencia '{ref_date_str}' a la zona horaria de la serie '{column_name}' ({series_tz}): {e_localize}. Intentando con UTC.")
                try:
                    ref_date_aware = ref_date_input.tz_localize('UTC').tz_convert(series_tz)
                except Exception as e_convert:
                    logger.error(f"Falló la conversión de fecha de referencia a UTC y luego a {series_tz} para '{column_name}': {e_convert}")
                    return series
        else: 
            ref_date_aware = ref_date_input.tz_convert(series_tz) 
    else: 
        if ref_date_input.tz is not None: 
            ref_date_aware = ref_date_input.tz_localize(None) 
        else: 
            ref_date_aware = ref_date_input
    
    if ref_date_aware is None: # Fallback si algo salió mal
        logger.error(f"La fecha de referencia '{ref_date_str}' no pudo ser procesada correctamente para la serie '{column_name}'.")
        return series

    series_sorted = series.sort_index()
    
    ref_value_series_subset = series_sorted.loc[series_sorted.index >= ref_date_aware]

    if ref_value_series_subset.empty:
        date_str_formatted = ref_date_aware.strftime('%Y-%m-%d %H:%M:%S%z') if ref_date_aware.tzinfo else ref_date_aware.strftime('%Y-%m-%d %H:%M:%S')
        logger.warning(f"No se encontró un valor de referencia en o después de '{date_str_formatted}' para la serie '{column_name}'. No se normalizará.")
        return series
    
    ref_value_actual = ref_value_series_subset.iloc[0]
    actual_ref_date_used = ref_value_series_subset.index[0]

    if pd.isna(ref_value_actual) or ref_value_actual == 0:
        date_str_formatted_actual = actual_ref_date_used.strftime('%Y-%m-%d %H:%M:%S%z') if actual_ref_date_used.tzinfo else actual_ref_date_used.strftime('%Y-%m-%d %H:%M:%S')
        logger.warning(f"Valor de referencia para '{column_name}' en/después de '{ref_date_str}' (usando fecha '{date_str_formatted_actual}') es NaN o cero ({ref_value_actual}). No se puede normalizar.")
        return series
    
    date_str_formatted_actual = actual_ref_date_used.strftime('%Y-%m-%d %H:%M:%S%z') if actual_ref_date_used.tzinfo else actual_ref_date_used.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Normalizando serie '{column_name}' usando fecha de referencia efectiva '{date_str_formatted_actual}'. Valor original en esa fecha: {ref_value_actual:.2f}")
    return (series / ref_value_actual) * target_ref_value

# La función anterior normalize_series_from_date_pd que estaba aquí ha sido reemplazada por la de arriba.

# Aquí se pueden añadir más funciones auxiliares que se identifiquen como comunes.
# Por ejemplo, funciones para manejo de DataFrames, conversiones de tipos, etc. 

def save_df_csv(
    df: pl.DataFrame,
    filename_base: str,
    output_dir: str,
    date_format: str = "%Y-%m-%dT%H:%M:%S",
    processing_id_prefix: str = None
) -> bool:
    """
    Guarda un DataFrame de Polars en formato CSV.
    
    Args:
        df: DataFrame de Polars a guardar
        filename_base: Nombre base del archivo
        output_dir: Directorio de salida
        date_format: Formato de fecha para columnas datetime
        processing_id_prefix: Prefijo opcional para el nombre del archivo
    
    Returns:
        bool: True si se guardó exitosamente, False en caso contrario
    """
    if df is None or df.is_empty():
        logger.warning(f"DataFrame para guardar ('{filename_base}') está vacío o no existe.")
        return False
        
    try:
        # Construir el nombre del archivo
        if processing_id_prefix:
            filename = f"{processing_id_prefix}_{filename_base}.csv"
        else:
            filename = f"{filename_base}.csv"
            
        # Construir la ruta completa
        full_path = os.path.join(output_dir, filename)
        
        # Asegurar que el directorio existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar el DataFrame
        df.write_csv(full_path, datetime_format=date_format)
        
        logger.info(f"DataFrame guardado exitosamente en: {full_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error al guardar DataFrame {filename_base}: {e}")
        return False 