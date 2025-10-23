# data_processing/preprocessing.py

import pandas as pd
import os
from config.logging_config import logger
from config import paths
import numpy as np
from io import StringIO
from .fix_temperature_data import fix_temperature_data_format

def preprocess_original_temp_data(input_filepath: str, 
                                  output_filepath: str, 
                                  filter_start_date: str) -> bool:
    """
    Preprocesa el archivo de temperatura original (similar a la celda 'Celda de Preprocesamiento de Datos de Temperatura').
    Lee 'datos/data_tm_fix.txt', lo filtra y guarda como 'datos/temp_data_processed.csv'.
    
    Args:
        input_filepath: Ruta al archivo de entrada (ej. 'datos/data_tm_fix.txt').
        output_filepath: Ruta donde se guardará el CSV procesado (ej. 'datos/temp_data_processed.csv').
        filter_start_date: Fecha de inicio para el filtrado (YYYY-MM-DD).

    Returns:
        True si el preprocesamiento fue exitoso y se guardó el archivo, False en caso contrario.
    """
    logger.info(f"Iniciando preprocesamiento de datos de temperatura desde: {input_filepath}")
    logger.info(f"Se aplicará un filtro para guardar datos desde: {filter_start_date}")

    processed_chunks = []
    chunk_size = 100000 

    # Define all 21 column names as expected from the file structure
    # (1 timestamp string + 20 data columns like in the original notebook)
    col_names_for_reading = ['Timestamp_str'] + [f'1TE{401+i:03d}(C)' for i in range(20)]

    # Define the actual data columns we want to process and keep from the 20 data columns
    # Incluimos las primeras cuatro PLUS las necesarias para PVStand (1TE416, 1TE418)
    target_temp_data_cols = ['1TE401(C)', '1TE402(C)', '1TE403(C)', '1TE404(C)', '1TE416(C)', '1TE418(C)']
    
    # Define the final output columns for the CSV and DataFrame
    output_cols = ['Timestamp'] + target_temp_data_cols

    try:
        for chunk in pd.read_csv(
            input_filepath, 
            sep=r'\s+',
            header=None, 
            skiprows=4, 
            comment='#', 
            names=col_names_for_reading, # Provide all 21 expected column names
            # usecols is removed for now; names should guide the parsing of all columns found
            na_values=["NAN", "Nan", "nan"],
            encoding='utf-8',
            chunksize=chunk_size,
            low_memory=False # Added to see if it helps with potential mixed types across many columns if some are empty
        ):
            logger.info(f"Procesando chunk de {len(chunk)} filas...")

            # --- Start Diagnostic Logging ---
            if not chunk.empty:
                logger.info(f"Chunk {processed_chunks.__len__()+1} - Muestra de 'Timestamp_str' (antes de to_datetime, unique[:5]): {chunk['Timestamp_str'].unique()[:5]}")
                if '1TE401(C)' in chunk.columns:
                    logger.info(f"Chunk {processed_chunks.__len__()+1} - Muestra de '1TE401(C)' (antes de to_numeric, unique[:5]): {chunk['1TE401(C)'].unique()[:5]}")
                else:
                    logger.warning(f"Chunk {processed_chunks.__len__()+1} - Columna '1TE401(C)' no encontrada para logging de muestra.")
            # --- End Diagnostic Logging ---

            chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp_str'], errors='coerce')
            
            # --- Start Diagnostic Logging for Timestamp after conversion ---
            if not chunk.empty:
                valid_timestamps = chunk['Timestamp'].dropna()
                if not valid_timestamps.empty:
                    logger.info(f"Chunk {processed_chunks.__len__()} - Muestra de 'Timestamp' (después de to_datetime, dropna, unique[:5]): {valid_timestamps.unique()[:5]}")
                else:
                    logger.info(f"Chunk {processed_chunks.__len__()} - No hay Timestamps válidos después de to_datetime y dropna.")
            # --- End Diagnostic Logging ---

            chunk.dropna(subset=['Timestamp'], inplace=True)
            if chunk.empty:
                logger.info("Chunk vacío después de parsear Timestamp, saltando.")
                continue

            # Convert only the target temperature columns to numeric
            for col in target_temp_data_cols:
                if col in chunk.columns: # Ensure column exists before trying to convert
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                else:
                    logger.warning(f"Columna {col} no encontrada en el chunk durante la conversión numérica.")

            if filter_start_date:
                start_datetime = pd.to_datetime(filter_start_date)
                # Ensure Timestamp column is datetime before comparison
                if pd.api.types.is_datetime64_any_dtype(chunk['Timestamp']):
                    chunk = chunk[chunk['Timestamp'] >= start_datetime]
                else:
                    logger.warning("Columna Timestamp no es de tipo datetime antes de filtrar por fecha. Saltando filtro de fecha para este chunk.")
            
            if not chunk.empty:
                # Select only the desired output_cols before appending
                processed_chunks.append(chunk[output_cols])
            else:
                logger.info("Chunk vacío después de aplicar filtros, saltando.")

        if not processed_chunks:
            logger.warning("No se procesaron datos válidos de temperatura (preprocess_original_temp_data).")
            pd.DataFrame(columns=output_cols).to_csv(output_filepath, index=False)
            return True 

        df_temp_processed = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"Total de filas procesadas (preprocess_original_temp_data): {len(df_temp_processed)}")

        if not df_temp_processed.empty:
            df_temp_processed.to_csv(output_filepath, index=False) # Guardar sin el índice de pandas
            logger.info(f"Datos de temperatura preprocesados y guardados en: {output_filepath}")
            logger.info(f"Primeras filas del DataFrame guardado (filtrado):\n{df_temp_processed.head()}")
            return True
        else:
            logger.warning(f"El DataFrame de temperatura está vacío después del preprocesamiento. No se guardó ningún archivo en {output_filepath}.")
            return True # Retorna True porque el proceso se completó, aunque no haya datos que guardar.

    except FileNotFoundError:
        logger.error(f"Archivo de temperatura no encontrado: {input_filepath}")
        # Guardar un CSV vacío si el archivo de entrada no existe
        pd.DataFrame(columns=output_cols).to_csv(output_filepath, index=False)
        logger.info(f"Se creó un archivo CSV vacío en {output_filepath} porque el archivo de entrada no se encontró.")
        return False
    except Exception as e:
        logger.error(f"Error durante el preprocesamiento de datos de temperatura ({input_filepath}): {e}", exc_info=True)
        # Guardar un CSV vacío en caso de cualquier otro error
        pd.DataFrame(columns=output_cols).to_csv(output_filepath, index=False)
        logger.info(f"Se creó un archivo CSV vacío en {output_filepath} debido a un error de preprocesamiento.")
        return False

def combinar_y_preprocesar_temperaturas(
    original_temp_filepath: str, 
    additional_temp_filepath: str, 
    combined_output_filepath: str, 
    filter_start_date_str: str = None, 
    filter_end_date_str: str = None
) -> bool:
    """
    Combina dos archivos de datos de temperatura (CSV), priorizando el adicional en solapamientos.
    Asegura zona horaria UTC, interpola NaNs por columna y opcionalmente filtra por fechas.
    Guarda el resultado en un nuevo archivo CSV.

    Args:
        original_temp_filepath: Path al CSV de temperaturas original.
        additional_temp_filepath: Path al CSV de temperaturas adicional.
        combined_output_filepath: Path donde se guardará el CSV combinado y procesado.
        filter_start_date_str: Fecha de inicio para filtrar (YYYY-MM-DD), opcional.
        filter_end_date_str: Fecha de fin para filtrar (YYYY-MM-DD), opcional.

    Returns:
        True si la combinación y guardado fueron exitosos, False en caso contrario.
    """
    logger.info(f"Iniciando combinación de archivos de temperatura:")
    logger.info(f"  Original: {original_temp_filepath}")
    logger.info(f"  Adicional: {additional_temp_filepath}")
    logger.info(f"  Salida Combinada: {combined_output_filepath}")

    output_dir = os.path.dirname(combined_output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Directorio de salida creado: {output_dir}")

    try:
        df_original = None
        df_additional = None
        success_original = False
        success_additional = False

        # Cargar el archivo original (salida de preprocess_original_temp_data)
        try:
            df_original = pd.read_csv(original_temp_filepath, parse_dates=['Timestamp'])
            if not df_original.empty and 'Timestamp' in df_original.columns:
                df_original.set_index('Timestamp', inplace=True)
                df_original.sort_index(inplace=True)
                logger.info(f"Cargado archivo original: {original_temp_filepath}, {len(df_original)} filas.")
                success_original = True
            elif df_original.empty:
                logger.warning(f"Archivo original {original_temp_filepath} está vacío.")
                # Considerar éxito si está vacío, ya que el preprocesamiento anterior lo manejo
                success_original = True 
                df_original = pd.DataFrame() # Asegurar que sea un DF vacío para el merge
            else:
                logger.error(f"Archivo original {original_temp_filepath} no tiene columna 'Timestamp' después de cargar.")
                df_original = pd.DataFrame()

        except FileNotFoundError:
            logger.error(f"Archivo original no encontrado: {original_temp_filepath}")
            df_original = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error al cargar archivo original {original_temp_filepath}: {e}")
            df_original = pd.DataFrame()

        # Cargar el archivo adicional (temp_mod_fixed_data.csv)
        # Primero, verificar y corregir formato deformado del archivo si es necesario
        logger.info(f"Verificando integridad del archivo de temperatura: {additional_temp_filepath}")
        temp_fix_success = fix_temperature_data_format(additional_temp_filepath)
        if not temp_fix_success:
            logger.warning("Falló la corrección del formato de temperatura. Continuando con archivo original.")
        else:
            logger.info("Verificación/corrección de formato de temperatura completada.")
        
        try:
            # Intento 1: Como en el notebook, con index_col='Timestamp'
            df_additional = pd.read_csv(
                additional_temp_filepath, 
                sep=',', 
                parse_dates=['_time'], 
                index_col='_time',
                dayfirst=True
            )
            df_additional.sort_index(inplace=True)
            logger.info(f"Cargado archivo adicional (intento 1 '{additional_temp_filepath}'): {len(df_additional)} filas.")
            success_additional = True
        except (ValueError, KeyError) as e_ts_col:
            logger.warning(f"Error al cargar {additional_temp_filepath} con index_col='Timestamp' (puede que la columna no exista o no sea parseable como header): {e_ts_col}. Intentando con index_col=0.")
            try:
                # Intento 2: Asumiendo que el timestamp es la primera columna.
                # temp_mod_fixed_data.csv was saved in notebook with columns: '1TE401(C)', '1TE402(C)', 'Amb_Temp', 'Irradiance'
                # and 'Timestamp' as index.
                # No depender de paths.EXPECTED_COLUMNS_TEMP_MOD_FIXED
                
                df_additional = pd.read_csv(
                    additional_temp_filepath, 
                    sep=',', 
                    header=0, # Asumir que la primera línea es un encabezado
                    parse_dates=[0], # Parsear la primera columna como fecha
                    index_col=0,     # Usar la primera columna como índice
                    dayfirst=True
                )
                # Renombrar la columna de índice a 'Timestamp' para consistencia
                logger.info(f"El índice del archivo adicional se llama '{df_additional.index.name}', renombrando a 'Timestamp'.")
                df_additional.index.name = 'Timestamp'
                
                df_additional.sort_index(inplace=True)
                logger.info(f"Cargado archivo adicional (intento 2 modificado '{additional_temp_filepath}' con index_col=0 y parse_dates=[0]): {len(df_additional)} filas.")
                success_additional = True

            except Exception as e_col0:
                logger.error(f"Error al cargar archivo adicional {additional_temp_filepath} incluso con index_col=0: {e_col0}")
                df_additional = pd.DataFrame()
        except FileNotFoundError:
            logger.error(f"Archivo adicional no encontrado: {additional_temp_filepath}")
            df_additional = pd.DataFrame()
        except Exception as e_other:
            logger.error(f"Otro error al cargar archivo adicional {additional_temp_filepath}: {e_other}")
            df_additional = pd.DataFrame()

        # Si df_original o df_additional no son DataFrames (por ej. None), inicializar como vacíos
        if df_original is None: df_original = pd.DataFrame()
        if df_additional is None: df_additional = pd.DataFrame()

        # --- Start Diagnostic Logging for df_additional state ---
        logger.info("Estado de df_additional ANTES de la lógica de combinación:")
        if df_additional is not None:
            logger.info(f"  df_additional.empty: {df_additional.empty}")
            logger.info(f"  len(df_additional): {len(df_additional)}")
            logger.info(f"  df_additional.head():\n{df_additional.head()}")
            # Create a buffer to capture df_additional.info()
            buffer = StringIO()
            df_additional.info(buf=buffer)
            logger.info(f"  df_additional.info():\n{buffer.getvalue()}")
        else:
            logger.info("  df_additional es None.")
        # --- End Diagnostic Logging ---

        # Verificar si ambos están vacíos antes de intentar el merge
        if df_original.empty and df_additional.empty:
            logger.warning("Ambos DataFrames de temperatura están vacíos. No se puede combinar.")
            pd.DataFrame().to_csv(combined_output_filepath)
            logger.info(f"Archivo combinado vacío creado en {combined_output_filepath} por DataFrames vacíos.")
            return False

        # --- Consistencia de Zona Horaria (UTC) ---
        if not df_original.empty and df_original.index.tz is None:
            df_original = df_original.tz_localize('UTC')
            logger.info("Localizado índice de DataFrame original a UTC.")
        elif not df_original.empty:
            df_original = df_original.tz_convert('UTC')
            logger.info("Convertido índice de DataFrame original a UTC.")

        if not df_additional.empty and isinstance(df_additional.index, pd.DatetimeIndex):
            if df_additional.index.tz is None:
                df_additional = df_additional.tz_localize('UTC')
                logger.info("Localizado índice de DataFrame adicional a UTC.")
            else:
                df_additional = df_additional.tz_convert('UTC')
                logger.info("Convertido índice de DataFrame adicional a UTC.")
        elif not df_additional.empty:
            logger.warning("Índice de DataFrame adicional no es DatetimeIndex, intentando conversión...")
            try:
                df_additional.index = pd.to_datetime(df_additional.index, utc=True)
                logger.info("Índice de DataFrame adicional convertido a DatetimeIndex UTC.")
            except Exception as e:
                logger.error(f"Error al convertir índice de DataFrame adicional: {e}")

        # Combinar DataFrames: df_additional tiene prioridad en solapamientos
        # Usamos combine_first después de asegurar que los índices están alineados y ordenados
        # y que las columnas son las mismas o un subconjunto/superconjunto manejable.
        
        # Concatenar y luego eliminar duplicados del índice, manteniendo el último (`additional`)
        if df_original.empty and df_additional.empty:
            logger.warning("Ambos DataFrames de temperatura están vacíos. No se puede combinar.")
            pd.DataFrame().to_csv(combined_output_filepath)
            logger.info(f"Archivo combinado vacío creado en {combined_output_filepath} por DataFrames vacíos.")
            return False
        elif df_original.empty:
            df_combined_temp = df_additional.copy()
            logger.info("Usando solo DataFrame adicional ya que el original está vacío.")
        elif df_additional.empty:
            df_combined_temp = df_original.copy()
            logger.info("Usando solo DataFrame original ya que el adicional está vacío.")
        else:
            # Asegurarse que el índice es DatetimeIndex antes de ordenar
            if not isinstance(df_original.index, pd.DatetimeIndex):
                logger.warning("Índice de df_original no es DatetimeIndex. Intentando conversión.")
                try: df_original.index = pd.to_datetime(df_original.index, utc=True)
                except: logger.error("Fallo al convertir índice de df_original a DatetimeIndex."); # Continuar, puede fallar luego
            
            if not isinstance(df_additional.index, pd.DatetimeIndex):
                logger.warning("Índice de df_additional no es DatetimeIndex. Intentando conversión.")
                try: df_additional.index = pd.to_datetime(df_additional.index, utc=True)
                except: logger.error("Fallo al convertir índice de df_additional a DatetimeIndex."); # Continuar

            df_combined_temp = pd.concat([df_original, df_additional])
            logger.info(f"DataFrames concatenados. Filas antes de eliminar duplicados: {len(df_combined_temp)}")
            
            # Ordenar por índice (timestamp) - crucial para la lógica de duplicados
            # El error TypeError: '<' not supported between instances of 'str' and 'Timestamp' 
            # sugiere que el índice no es puramente Timestamp. Esto puede pasar si uno de los CSVs
            # no se parsea bien o tiene strings en la columna de índice.
            # La carga de CSV con index_col y parse_dates debería manejarlo, pero verificamos.
            if not isinstance(df_combined_temp.index, pd.DatetimeIndex):
                logger.error("El índice del DataFrame combinado NO es DatetimeIndex ANTES de sort_index. Intentando forzar conversión.")
                try:
                    df_combined_temp.index = pd.to_datetime(df_combined_temp.index, errors='coerce', utc=True)
                    df_combined_temp.dropna(axis=0, how='all', subset=df_combined_temp.columns, inplace=True) # Eliminar filas si el índice se volvió NaT
                    logger.info("Índice forzado a DatetimeIndex. Filas después de coerción y dropna de índice: {len(df_combined_temp)}")
                except Exception as e_conv_idx:
                    logger.error(f"Fallo crítico al forzar índice a DatetimeIndex: {e_conv_idx}")
                    # No se puede continuar si el índice no es ordenable
                    pd.DataFrame().to_csv(combined_output_filepath)
                    logger.info(f"Archivo combinado vacío creado en {combined_output_filepath} por error de índice.")
                    return False
            
            if not df_combined_temp.empty and isinstance(df_combined_temp.index, pd.DatetimeIndex):
                df_combined_temp.sort_index(inplace=True)
                logger.info("DataFrame combinado ordenado por índice.")
                
                # Eliminar duplicados en el índice, manteniendo la última ocurrencia (prioriza df_additional)
                df_combined_temp = df_combined_temp[~df_combined_temp.index.duplicated(keep='last')]
                logger.info(f"Duplicados eliminados (manteniendo última ocurrencia). Filas restantes: {len(df_combined_temp)}")
            elif df_combined_temp.empty:
                 logger.warning("DataFrame combinado vacío después de la concatenación (o forzado de índice). No se pueden eliminar duplicados.")
            else:
                 logger.error("No se pudieron ordenar o eliminar duplicados porque el índice no es DatetimeIndex o está vacío.")

        # Interpolar NaNs por columna (linealmente)
        if not df_combined_temp.empty:
            # Seleccionar solo columnas numéricas para interpolar
            numeric_cols = df_combined_temp.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                df_combined_temp[numeric_cols] = df_combined_temp[numeric_cols].interpolate(method='linear', axis=0)
                logger.info("Interpolación lineal de NaNs completada para columnas numéricas.")
            else:
                logger.info("No se encontraron columnas numéricas para interpolar.")

            # Filtrado por fecha (opcional)
            if filter_start_date_str:
                try:
                    start_date = pd.to_datetime(filter_start_date_str).tz_localize('UTC')
                    df_combined_temp = df_combined_temp[df_combined_temp.index >= start_date]
                    logger.info(f"Filtrado por fecha de inicio: >= {filter_start_date_str}. Filas restantes: {len(df_combined_temp)}")
                except Exception as e_date_filter_start:
                    logger.error(f"Error al aplicar filtro de fecha de inicio ({filter_start_date_str}): {e_date_filter_start}")
            
            if filter_end_date_str:
                try:
                    end_date = pd.to_datetime(filter_end_date_str).tz_localize('UTC')
                    df_combined_temp = df_combined_temp[df_combined_temp.index <= end_date]
                    logger.info(f"Filtrado por fecha de fin: <= {filter_end_date_str}. Filas restantes: {len(df_combined_temp)}")
                except Exception as e_date_filter_end:
                    logger.error(f"Error al aplicar filtro de fecha de fin ({filter_end_date_str}): {e_date_filter_end}")
        
        # Guardar el DataFrame combinado
        if not df_combined_temp.empty:
            df_combined_temp.to_csv(combined_output_filepath)
            logger.info(f"DataFrame de temperaturas combinado y procesado guardado en: {combined_output_filepath}")
            logger.info(f"Primeras filas del DataFrame combinado guardado:\n{df_combined_temp.head()}")
            return True
        else:
            logger.warning("El DataFrame combinado final está vacío. No se guardó ningún archivo.")
            pd.DataFrame().to_csv(combined_output_filepath)
            logger.info(f"Archivo combinado vacío creado en {combined_output_filepath} porque el resultado final estaba vacío.")
            return False

    except Exception as e:
        logger.error(f"Error general durante la combinación y preprocesamiento de temperaturas: {e}", exc_info=True)
        pd.DataFrame().to_csv(combined_output_filepath)
        logger.info(f"Archivo combinado vacío creado en {combined_output_filepath} debido a un error general.")
        return False

# Si es necesario, se pueden añadir más funciones de preprocesamiento aquí. 