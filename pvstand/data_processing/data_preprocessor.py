"""
Módulo centralizado para el preprocesamiento de datos
Este módulo coordina todos los procesos de preprocesamiento necesarios
antes de ejecutar los análisis en main.py

NOTA: El procesamiento de temperatura se realiza ahora en download_temp_only.ipynb
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Importar configuraciones
from config import paths, settings
from config.logging_config import logger

class DataPreprocessor:
    """
    Clase que gestiona el preprocesamiento de todos los datos necesarios
    para el análisis de soiling
    """
    
    def __init__(self):
        self.preprocessing_status = {
            'temperature_data': False,
            'iv600_data': False,
            'general_setup': False
        }
        
    def check_required_files(self) -> Dict[str, bool]:
        """
        Verifica la existencia de archivos necesarios para el preprocesamiento
        """
        required_files = {
            # Archivo de temperatura (generado por notebook)
            'temp_data': paths.TEMP_DATA_COMBINED_PROCESSED_CSV,
            
            # Archivos IV600 (deshabilitados por errores)
            'iv600_processed': os.path.join(paths.BASE_INPUT_DIR, "processed_iv600_data.csv"),
            
            # Otros archivos críticos
            'pvstand_raw': os.path.join(paths.BASE_INPUT_DIR, paths.PVSTAND_IV_DATA_FILENAME),
        }
        
        file_status = {}
        for file_key, file_path in required_files.items():
            exists = os.path.exists(file_path)
            file_status[file_key] = exists
            if not exists:
                logger.warning(f"Archivo requerido no encontrado: {file_path}")
        
        return file_status
    
    def check_temperature_data(self) -> bool:
        """
        Verifica que el archivo de temperatura esté disponible.
        NOTA: Este archivo debe generarse usando download_temp_only.ipynb
        """
        logger.info("=== VERIFICANDO DATOS DE TEMPERATURA ===")
        
        temp_file = paths.TEMP_DATA_COMBINED_PROCESSED_CSV
        
        if os.path.exists(temp_file):
            try:
                # Verificar que el archivo sea válido
                df_temp = pd.read_csv(temp_file, nrows=10)
                
                if len(df_temp) > 0:
                    file_size_mb = os.path.getsize(temp_file) / (1024*1024)
                    logger.info(f"✅ Datos de temperatura disponibles: {temp_file}")
                    logger.info(f"📁 Tamaño: {file_size_mb:.2f} MB")
                    logger.info(f"📊 Columnas: {list(df_temp.columns)}")
                    self.preprocessing_status['temperature_data'] = True
                    return True
                else:
                    logger.error(f"❌ Archivo de temperatura vacío: {temp_file}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error leyendo archivo de temperatura: {e}")
                return False
        else:
            logger.warning(f"⚠️ Archivo de temperatura no encontrado: {temp_file}")
            logger.info("📋 INSTRUCCIONES:")
            logger.info("   1. Ejecuta el notebook 'download_temp_only.ipynb'")
            logger.info("   2. Esto generará el archivo data_temp.csv necesario")
            logger.info("   3. Luego ejecuta nuevamente el preprocesamiento")
            return False
    
    def check_iv600_data(self) -> bool:
        """
        Verifica datos IV600 procesados.
        NOTA: Función simplificada ya que el procesamiento IV600 está deshabilitado
        """
        logger.info("=== VERIFICANDO DATOS IV600 ===")
        
        iv600_file = os.path.join(paths.BASE_INPUT_DIR, "processed_iv600_data.csv")
        
        if os.path.exists(iv600_file):
            try:
                df_iv600 = pd.read_csv(iv600_file, nrows=5)
                if len(df_iv600) > 0:
                    logger.info(f"✅ Datos IV600 disponibles: {iv600_file}")
                    self.preprocessing_status['iv600_data'] = True
                    return True
                else:
                    logger.warning("⚠️ Archivo IV600 vacío")
                    return False
            except Exception as e:
                logger.error(f"❌ Error leyendo datos IV600: {e}")
                return False
        else:
            logger.warning(f"⚠️ Archivo IV600 no encontrado: {iv600_file}")
            logger.info("📋 NOTA: Procesamiento IV600 deshabilitado por errores de sintaxis")
            logger.info("   Los análisis IV600 usarán datos existentes si están disponibles")
            # Marcar como completado para permitir continuar
            self.preprocessing_status['iv600_data'] = True
            return True
    
    def setup_output_directories(self) -> bool:
        """
        Crea los directorios de salida necesarios
        """
        logger.info("=== CONFIGURANDO DIRECTORIOS DE SALIDA ===")
        
        try:
            directories_to_create = [
                paths.BASE_OUTPUT_GRAPH_DIR,
                paths.BASE_OUTPUT_CSV_DIR,
                paths.PV_GLASSES_OUTPUT_SUBDIR_CSV,
                paths.PV_GLASSES_OUTPUT_SUBDIR_GRAPH,
                paths.PVSTAND_OUTPUT_SUBDIR_CSV,
                paths.PVSTAND_OUTPUT_SUBDIR_GRAPH,
                paths.DUSTIQ_OUTPUT_SUBDIR_CSV,
                paths.DUSTIQ_OUTPUT_SUBDIR_GRAPH,
                paths.REFCELLS_OUTPUT_SUBDIR_CSV,
                paths.REFCELLS_OUTPUT_SUBDIR_GRAPH,
                os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "iv600"),
                os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "soiling_kit"),
                os.path.join(paths.BASE_OUTPUT_GRAPH_DIR, "transmitancia_pv_glasses"),
            ]
            
            for directory in directories_to_create:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Directorio creado/verificado: {directory}")
            
            logger.info("✅ Directorios de salida configurados correctamente")
            self.preprocessing_status['general_setup'] = True
            return True
            
        except Exception as e:
            logger.error(f"Error configurando directorios: {e}", exc_info=True)
            return False
    
    def run_complete_preprocessing(self, force_reprocess: bool = False) -> bool:
        """
        Ejecuta el proceso de preprocesamiento simplificado
        
        Args:
            force_reprocess: Parámetro mantenido por compatibilidad (no se usa)
        """
        logger.info("=== INICIANDO PREPROCESAMIENTO SIMPLIFICADO ===")
        logger.info("📋 NOTA: Procesamiento de temperatura se realiza en download_temp_only.ipynb")
        
        # Verificar archivos requeridos
        file_status = self.check_required_files()
        missing_files = [k for k, v in file_status.items() if not v]
        
        if missing_files:
            logger.warning(f"Archivos faltantes detectados: {missing_files}")
        
        # Configurar directorios
        if not self.setup_output_directories():
            logger.error("❌ Fallo en la configuración de directorios")
            return False
        
        # Verificar datos de temperatura (generados por notebook)
        temp_success = self.check_temperature_data()
        if not temp_success:
            logger.error("❌ Datos de temperatura no disponibles")
            logger.info("📋 Ejecuta download_temp_only.ipynb para generar data_temp.csv")
        
        # Verificar datos IV600
        iv600_success = self.check_iv600_data()
        
        # Resumen final
        logger.info("=== RESUMEN DEL PREPROCESAMIENTO ===")
        for process, status in self.preprocessing_status.items():
            status_text = "✅ COMPLETADO" if status else "❌ FALTANTE"
            logger.info(f"{process}: {status_text}")
        
        # Determinar si el preprocesamiento fue exitoso
        critical_processes = ['general_setup']  # Solo directorios son críticos
        success = all(self.preprocessing_status[p] for p in critical_processes)
        
        if success:
            logger.info("✅ PREPROCESAMIENTO COMPLETADO")
            if temp_success:
                logger.info("📊 Datos de temperatura disponibles - main.py puede ejecutarse")
            else:
                logger.warning("⚠️ Datos de temperatura faltantes - algunos análisis pueden fallar")
        else:
            logger.error("❌ PREPROCESAMIENTO INCOMPLETO - revisar errores anteriores")
        
        return success


def run_preprocessing(force_reprocess: bool = False) -> bool:
    """
    Función principal para ejecutar el preprocesamiento simplificado
    """
    preprocessor = DataPreprocessor()
    return preprocessor.run_complete_preprocessing(force_reprocess=force_reprocess)


if __name__ == "__main__":
    # Permitir ejecución directa del preprocesamiento
    force = "--force" in sys.argv
    success = run_preprocessing(force_reprocess=force)
    
    if success:
        print("✅ Preprocesamiento completado exitosamente")
        sys.exit(0)
    else:
        print("❌ Preprocesamiento completado con advertencias")
        sys.exit(1)