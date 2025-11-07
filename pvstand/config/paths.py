# config/paths.py

import os

# ============================================================================
# DIRECTORIOS BASE
# ============================================================================

# Directorio base del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Directorios principales
BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, "datos")
BASE_OUTPUT_GRAPH_DIR = os.path.join(PROJECT_ROOT, "graficos_analisis_integrado_py")
BASE_OUTPUT_CSV_DIR = os.path.join(PROJECT_ROOT, "datos_procesados_analisis_integrado_py")

# Alias para compatibilidad
DATA_DIR = BASE_INPUT_DIR
PROCESSED_DATA_DIR = BASE_OUTPUT_CSV_DIR

# ============================================================================
# ARCHIVOS DE DATOS CRUDOS
# ============================================================================

# Nombres de archivos
PVSTAND_IV_DATA_FILENAME = "raw_pvstand_iv_data.csv"
PVSTAND_TEMP_DATA_FILENAME = 'data_temp.csv'  # Generado por notebook
IV600_RAW_DATA_FILENAME = "iv600_risen.xlsx"

# Rutas completas de archivos de entrada
PVSTAND_IV_DATA_FILE = os.path.join(BASE_INPUT_DIR, PVSTAND_IV_DATA_FILENAME)
PVSTAND_TEMP_DATA_FILE = os.path.join(BASE_INPUT_DIR, PVSTAND_TEMP_DATA_FILENAME)
IV600_RAW_DATA_FILE = os.path.join(BASE_INPUT_DIR, IV600_RAW_DATA_FILENAME)

# ============================================================================
# ARCHIVOS DE TEMPERATURA (LEGACY + NUEVO SISTEMA)
# ============================================================================

# Archivos legacy (obsoletos - solo para compatibilidad)
ORIGINAL_TEMP_DATA_FILE = os.path.join(BASE_INPUT_DIR, "data_tm_fix.txt")
ADDITIONAL_TEMP_DATA_FILE = os.path.join(BASE_INPUT_DIR, "temp_mod_fixed_data.csv")
TEMP_DATA_PROCESSED_CSV = os.path.join(BASE_INPUT_DIR, "temp_data_processed.csv")

# Archivo principal de temperatura (generado por notebook)
TEMP_DATA_COMBINED_PROCESSED_CSV = os.path.join(BASE_INPUT_DIR, "data_temp.csv")
TEMP_DATA = TEMP_DATA_COMBINED_PROCESSED_CSV  # Alias principal

# ============================================================================
# ALIAS ADICIONALES PARA COMPATIBILIDAD
# ============================================================================

PVSTAND_IV_DATA = PVSTAND_IV_DATA_FILE
IV600_RAW_DATA = IV600_RAW_DATA_FILE

# ============================================================================
# DIRECTORIOS DE SALIDA POR MÓDULO
# ============================================================================

# CSV outputs
PVSTAND_OUTPUT_SUBDIR_CSV = os.path.join(BASE_OUTPUT_CSV_DIR, "pv_stand")
IV600_OUTPUT_SUBDIR_CSV = os.path.join(BASE_OUTPUT_CSV_DIR, "iv600")
# Graph outputs
PVSTAND_OUTPUT_SUBDIR_GRAPH = os.path.join(BASE_OUTPUT_GRAPH_DIR, "pv_stand")
IV600_OUTPUT_SUBDIR_GRAPH = os.path.join(BASE_OUTPUT_GRAPH_DIR, "iv600")