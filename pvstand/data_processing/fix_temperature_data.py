#!/usr/bin/env python3
"""
Script para corregir el formato deformado del archivo temp_mod_fixed_data.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar el directorio raíz del proyecto al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.logging_config import logger
from config import paths

def fix_temperature_data_format(input_filepath: str, output_filepath: str = None) -> bool:
    """
    Corrige el formato deformado del archivo de datos de temperatura.
    """
    
    if not os.path.exists(input_filepath):
        logger.error(f"Archivo de entrada no encontrado: {input_filepath}")
        return False
    
    if output_filepath is None:
        # Crear archivo de respaldo y usar el original como salida
        backup_filepath = input_filepath.replace('.csv', '_backup.csv')
        if not os.path.exists(backup_filepath):
            logger.info(f"Creando respaldo: {backup_filepath}")
            os.rename(input_filepath, backup_filepath)
            input_filepath = backup_filepath
        output_filepath = input_filepath.replace('_backup.csv', '.csv')
    
    logger.info(f"=== INICIANDO CORRECCIÓN DE DATOS DE TEMPERATURA ===")
    logger.info(f"Archivo entrada: {input_filepath}")
    logger.info(f"Archivo salida: {output_filepath}")
    
    try:
        # Definir el punto de quiebre donde empezó el problema
        breakpoint_datetime = pd.Timestamp('2024-12-12 17:41:58', tz='UTC')
        logger.info(f"Punto de quiebre identificado: {breakpoint_datetime}")
        
        # Leer el archivo completo
        logger.info("Leyendo archivo de entrada...")
        df_all = pd.read_csv(input_filepath)
        
        # Convertir columna de tiempo
        df_all['_time'] = pd.to_datetime(df_all['_time'], errors='coerce')
        df_all = df_all.dropna(subset=['_time'])
        
        logger.info(f"Total de filas leídas: {len(df_all)}")
        
        # Separar datos buenos y deformados
        df_good = df_all[df_all['_time'] < breakpoint_datetime].copy()
        df_bad = df_all[df_all['_time'] >= breakpoint_datetime].copy()
        
        logger.info(f"Datos con formato correcto: {len(df_good)} filas")
        logger.info(f"Datos con formato deformado: {len(df_bad)} filas")
        
        if len(df_bad) == 0:
            logger.info("No se encontraron datos deformados. Archivo ya está correcto.")
            return True
        
        # Procesar datos deformados
        logger.info("Procesando datos deformados...")
        df_bad_filtered = df_bad[df_bad['_measurement'] == 'fixed_plant_atamo_1'].copy()
        
        # Obtener columnas de temperatura
        temp_columns = ['1TE416(C)', '1TE417(C)', '1TE418(C)', '1TE419(C)']
        
        # Agrupar por timestamp base
        df_bad_filtered['_time_rounded'] = df_bad_filtered['_time'].dt.floor('S')
        
        # Función para consolidar filas agrupadas
        def consolidate_temperature_rows(group):
            """Consolida múltiples filas de temperatura en una sola fila"""
            consolidated_row = group.iloc[0].copy()
            consolidated_row['_measurement'] = 'TempModFixed'
            
            # Consolidar valores de temperatura
            for col in temp_columns:
                non_null_values = group[col].dropna()
                if len(non_null_values) > 0:
                    consolidated_row[col] = non_null_values.iloc[0]
                else:
                    consolidated_row[col] = np.nan
            
            return consolidated_row
        
        # Agrupar y consolidar
        df_bad_consolidated = df_bad_filtered.groupby('_time_rounded').apply(
            consolidate_temperature_rows
        ).reset_index(drop=True)
        
        # Usar el timestamp redondeado como timestamp principal
        df_bad_consolidated['_time'] = df_bad_consolidated['_time_rounded']
        df_bad_consolidated = df_bad_consolidated.drop('_time_rounded', axis=1)
        
        logger.info(f"Datos deformados consolidados: {len(df_bad_consolidated)} filas")
        
        # Combinar datos buenos y corregidos
        df_fixed = pd.concat([df_good, df_bad_consolidated], ignore_index=True)
        df_fixed = df_fixed.sort_values('_time').reset_index(drop=True)
        
        logger.info(f"Total filas después de corrección: {len(df_fixed)}")
        
        # Guardar archivo corregido
        logger.info(f"Guardando archivo corregido: {output_filepath}")
        df_fixed.to_csv(output_filepath, index=False)
        
        if os.path.exists(output_filepath):
            file_size_mb = os.path.getsize(output_filepath) / (1024 * 1024)
            logger.info(f"Archivo corregido guardado exitosamente ({file_size_mb:.1f} MB)")
            return True
        else:
            logger.error("Error: No se pudo guardar el archivo corregido")
            return False
            
    except Exception as e:
        logger.error(f"Error durante la corrección de datos de temperatura: {e}", exc_info=True)
        return False

def main():
    """Función principal para ejecutar la corrección"""
    
    # Determinar rutas de archivos
    input_file = os.path.join(paths.BASE_INPUT_DIR, 'temp_mod_fixed_data.csv')
    
    if not os.path.exists(input_file):
        logger.error(f"Archivo de temperatura no encontrado: {input_file}")
        return False
    
    # Ejecutar corrección
    success = fix_temperature_data_format(input_file)
    
    if success:
        logger.info("✅ Corrección de datos de temperatura completada exitosamente")
    else:
        logger.error("❌ Error en la corrección de datos de temperatura")
    
    return success

if __name__ == "__main__":
    main() 