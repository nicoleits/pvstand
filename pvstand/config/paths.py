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
CALENDAR_RAW_DATA_FILENAME = 'calendario_muestras_seleccionado.csv'
PV_GLASSES_RAW_DATA_FILENAME = 'raw_pv_glasses_data.csv'
DUSTIQ_RAW_DATA_FILENAME = 'raw_dustiq_data.csv'
PVSTAND_IV_DATA_FILENAME = "raw_pvstand_iv_data.csv"
PVSTAND_TEMP_DATA_FILENAME = 'data_temp.csv'  # Generado por notebook
REFCELLS_RAW_DATA_FILENAME = "refcells_data.csv"
SOILING_KIT_RAW_DATA_FILENAME = "soiling_kit_raw_data.csv"

# Rutas completas de archivos de entrada
CALENDAR_RAW_DATA_FILE = os.path.join(BASE_INPUT_DIR, CALENDAR_RAW_DATA_FILENAME)
PV_GLASSES_RAW_DATA_FILE = os.path.join(BASE_INPUT_DIR, PV_GLASSES_RAW_DATA_FILENAME)
DUSTIQ_RAW_DATA_FILE = os.path.join(BASE_INPUT_DIR, DUSTIQ_RAW_DATA_FILENAME)
PVSTAND_IV_DATA_FILE = os.path.join(BASE_INPUT_DIR, PVSTAND_IV_DATA_FILENAME)
PVSTAND_TEMP_DATA_FILE = os.path.join(BASE_INPUT_DIR, PVSTAND_TEMP_DATA_FILENAME)
REFCELLS_RAW_DATA_FILE = os.path.join(BASE_INPUT_DIR, REFCELLS_RAW_DATA_FILENAME)
SOILING_KIT_RAW_DATA_FILE = os.path.join(BASE_INPUT_DIR, SOILING_KIT_RAW_DATA_FILENAME)

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

SOILING_KIT_RAW_DATA = SOILING_KIT_RAW_DATA_FILE
DUSTIQ_RAW_DATA = DUSTIQ_RAW_DATA_FILE
REFCELLS_DATA = REFCELLS_RAW_DATA_FILE
PVSTAND_IV_DATA = PVSTAND_IV_DATA_FILE
PV_GLASSES_RAW_DATA = PV_GLASSES_RAW_DATA_FILE
CALENDAR_FILE = CALENDAR_RAW_DATA_FILE

# ============================================================================
# DIRECTORIOS DE SALIDA POR MÓDULO
# ============================================================================

# CSV outputs
PV_GLASSES_OUTPUT_SUBDIR_CSV = os.path.join(BASE_OUTPUT_CSV_DIR, "pv_glasses")
PVSTAND_OUTPUT_SUBDIR_CSV = os.path.join(BASE_OUTPUT_CSV_DIR, "pv_stand")
DUSTIQ_OUTPUT_SUBDIR_CSV = os.path.join(BASE_OUTPUT_CSV_DIR, "dustiq")
REFCELLS_OUTPUT_SUBDIR_CSV = os.path.join(BASE_OUTPUT_CSV_DIR, "ref_cells")
SOILING_KIT_OUTPUT_SUBDIR_CSV = os.path.join(BASE_OUTPUT_CSV_DIR, "soiling_kit")
CALENDAR_OUTPUT_SUBDIR_CSV = os.path.join(BASE_OUTPUT_CSV_DIR, "calendario")

# Graph outputs
PV_GLASSES_OUTPUT_SUBDIR_GRAPH = os.path.join(BASE_OUTPUT_GRAPH_DIR, "pv_glasses")
PVSTAND_OUTPUT_SUBDIR_GRAPH = os.path.join(BASE_OUTPUT_GRAPH_DIR, "pv_stand")
DUSTIQ_OUTPUT_SUBDIR_GRAPH = os.path.join(BASE_OUTPUT_GRAPH_DIR, "dustiq")
REFCELLS_OUTPUT_SUBDIR_GRAPH = os.path.join(BASE_OUTPUT_GRAPH_DIR, "ref_cells")
SOILING_KIT_OUTPUT_SUBDIR_GRAPH = os.path.join(BASE_OUTPUT_GRAPH_DIR, "soiling_kit")
CALENDAR_OUTPUT_SUBDIR_GRAPH = os.path.join(BASE_OUTPUT_GRAPH_DIR, "calendario")