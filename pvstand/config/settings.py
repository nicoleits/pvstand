import pandas as pd

# --- Definición de Periodo de Análisis General (si aplica) ---
# Estas fechas pueden ser sobrescritas o usadas específicamente en cada sección si es necesario
ANALYSIS_START_DATE_GENERAL_STR = "2024-07-01" # YYYY-MM-DD
ANALYSIS_END_DATE_GENERAL_STR = "2025-12-31"   # YYYY-MM-DD

ANALYSIS_START_DATE_GENERAL = pd.to_datetime(ANALYSIS_START_DATE_GENERAL_STR, dayfirst=False)
ANALYSIS_END_DATE_GENERAL = pd.to_datetime(ANALYSIS_END_DATE_GENERAL_STR, dayfirst=False)

# --- Parámetros de PVStand (ejemplos, se completarán) ---
PVSTAND_FILTER_START_TIME = '13:00'
PVSTAND_FILTER_END_TIME = '18:00'
PVSTAND_MODULE_SOILED_ID = 'perc1fixed'
PVSTAND_MODULE_REFERENCE_ID = 'perc2fixed'
PVSTAND_TEMP_SENSOR_SOILED_COL = '1TE416(C)'
PVSTAND_TEMP_SENSOR_REFERENCE_COL = '1TE418(C)'
PVSTAND_ALPHA_ISC_CORR = 0.0004 
PVSTAND_BETA_PMAX_CORR = -0.0037 
PVSTAND_TEMP_REF_CORRECTION_C = 25.0 
PVSTAND_NORMALIZE_SR_FLAG = True 
PVSTAND_NORMALIZE_SR_REF_DATE_STR = '2024-08-01'
PVSTAND_PMAX_SR_OFFSET = 3.0 
PVSTAND_SR_MIN_FILTER_THRESHOLD = 0.7 
PVSTAND_SR_MAX_FILTER_THRESHOLD = 1.01

# --- Parámetros de DustIQ ---
DUSTIQ_SR_FILTER_THRESHOLD = 0
DUSTIQ_LOCAL_TIMEZONE_STR = "America/Santiago" # o la zona horaria local apropiada


# --- Otros parámetros globales que puedan surgir ---
SAVE_FIGURES = True
SHOW_FIGURES = True # Cambiado a True para mostrar los gráficos
DPI_FIGURES = 300 

# Configuraciones Generales de Análisis
ANALYSIS_START_DATE_GENERAL_STR = "2024-07-01" # YYYY-MM-DD
ANALYSIS_END_DATE_GENERAL_STR = "2025-12-31"   # YYYY-MM-DD

# Configuraciones Específicas para Módulos de Análisis

# --- Settings para Calendar Analyzer ---
CALENDAR_SHEET_NAME = 'Hoja1'
# ... (otras configuraciones específicas de calendario si las hubiera)

# --- Settings para DustIQ Analyzer ---
DUSTIQ_LOCAL_TIMEZONE_STR = "America/Santiago" # o la zona horaria local apropiada
DUSTIQ_SR_FILTER_THRESHOLD = 0 # Ejemplo, ajustar según necesidad
# ... (otras configuraciones específicas de DustIQ)

# --- Settings para PV Stand Analyzer ---
PVSTAND_LOCAL_TIMEZONE_STR = "America/Santiago"
PVSTAND_IV_DATA_TIME_COLUMN = "_time" # Nombre de la columna de tiempo en los datos IV
PVSTAND_IV_DATA_TIME_FORMAT = "%Y-%m-%d %H:%M:%S%z" # Formato de la columna de tiempo de datos IV, ej: 2024-07-01 00:00:03+00:00
PVSTAND_IV_DATA_DECIMAL_SEPARATOR = '.' # Separador decimal en los datos IV
PVSTAND_TEMP_DATA_TIME_COLUMN = "TIMESTAMP" # Nombre de la columna de tiempo en los datos de temperatura del piranómetro
PVSTAND_TEMP_DATA_TIME_FORMAT_INPUT = "%Y-%m-%d %H:%M:%S.%f" # Formato de tiempo de entrada para datos de temp. piranómetro
PVSTAND_GHI_COLUMN = "SolarIrradiance"
PVSTAND_POUT_COLUMN = "PotenciaMaxima"
PVSTAND_ISC_COLUMN = "CorrienteCortoCircuito"
PVSTAND_VOC_COLUMN = "TensionCircuitoAbierto"
PVSTAND_FILL_FACTOR_COLUMN = "FactorDeForma"
PVSTAND_TEMP_CELL_COLUMN = "CellTemperature"
PVSTAND_TEMP_MODULE_COLUMN = "ModuleTemperature"
PVSTAND_GHI_THRESHOLD_CLEANING = 700 # Umbral GHI para limpieza
PVSTAND_STC_GHI = 1000
PVSTAND_STC_TEMP = 25
# Nuevos parámetros basados en el notebook
PVSTAND_ANALYSIS_START_DATE_STR = '2024-08-01' # Fecha de inicio del análisis para PVStand
PVSTAND_ANALYSIS_END_DATE_STR = '2025-11-07'   # Fecha de fin del análisis para PVStand
PVSTAND_RESAMPLE_FREQ_MINUTES = 1             # Frecuencia de remuestreo en minutos para PVStand
PVSTAND_GRAPH_QUANTILE = 0.25                 # Cuantil para agregaciones en gráficos de PVStand
# Tomados de la sección superior de PVStand para consistencia, ya que la función del notebook los usa
PVSTAND_FILTER_START_TIME = '13:00'
PVSTAND_FILTER_END_TIME = '18:00'
PVSTAND_MODULE_SOILED_ID = 'perc1fixed'
PVSTAND_MODULE_REFERENCE_ID = 'perc2fixed'
PVSTAND_TEMP_SENSOR_SOILED_COL = '1TE418(C)'
PVSTAND_TEMP_SENSOR_REFERENCE_COL = '1TE419(C)'
PVSTAND_ALPHA_ISC_CORR = -0.0004 
PVSTAND_BETA_PMAX_CORR = +0.0037 
PVSTAND_TEMP_REF_CORRECTION_C = 25.0 
PVSTAND_NORMALIZE_SR_FLAG = True 
PVSTAND_NORMALIZE_SR_REF_DATE_STR = '2024-08-01' # Actualizado de '2024-08-01' a '2024-08-15' como en la celda 4, pero la celda 3 usa '2024-08-01' como default. Mantengo '2024-08-01' por ahora.
PVSTAND_PMAX_SR_OFFSET = 3.0 # El notebook usa 0.0 en la celda de llamada, pero 3.0 en la sección de arriba. Uso 3.0 por consistencia con lo existente.
PVSTAND_SR_MIN_FILTER_THRESHOLD = 0.7 
PVSTAND_SR_MAX_FILTER_THRESHOLD = 1.01
# ... (otras configuraciones)

# --- Settings para Ref Cells Analyzer ---
REFCELLS_LOCAL_TIMEZONE_STR = "America/Santiago"
REFCELLS_TIME_COLUMN = "timestamp" # Nombre de la columna de tiempo en los datos de celdas de referencia
REFCELLS_TIME_FORMAT = "%Y-%m-%d %H:%M:%S%z" # Formato de la columna de tiempo con zona horaria, ej: 2024-07-24 13:00:00+00:00
# Columnas de datos y configuraciones basadas en el notebook analisis_soiling.ipynb
REFCELLS_REFERENCE_COLUMN = '1RC412(w.m-2)' # Columna de referencia para el cálculo de SR
REFCELLS_SOILED_COLUMNS_TO_ANALYZE = ['1RC410(w.m-2)', '1RC411(w.m-2)'] # Columnas sucias a analizar
REFCELLS_IRRADIANCE_COLUMNS_TO_PLOT = ['1RC411(w.m-2)', '1RC412(w.m-2)'] # Columnas de irradiancia para graficar
REFCELLS_SR_MIN_FILTER = 0.80 # Límite inferior para el filtro de SR (80%)
REFCELLS_SR_MAX_FILTER = 1.05 # Límite superior para el filtro de SR (105%)
REFCELLS_ADJUST_TO_100_FLAG = True # Si es True, ajusta el inicio de las series de SR a 100

# Constantes para análisis de Soiling Ratio (SR) en PV Glasses
PV_GLASSES_SR_IRRADIANCE_THRESHOLD = 300
PV_GLASSES_SR_COLUMNS_TO_PLOT = ['SR_R_FC3', 'SR_R_FC4', 'SR_R_FC5']
PV_GLASSES_SR_TO_MASS_MAP = {
    'SR_R_FC3': 'Masa_C_Referencia',  # SR_R_FC3 se asocia con Masa C
    'SR_R_FC4': 'Masa_B_Referencia',  # SR_R_FC4 se asocia con Masa B
    'SR_R_FC5': 'Masa_A_Referencia'   # SR_R_FC5 se asocia con Masa A
}
PV_GLASSES_PERIOD_ORDER_FOR_BAR_CHART = ['semanal', '2 semanas', 'Mensual', 'Trimestral', 'Cuatrimestral', 'Semestral', '1 año']
PV_GLASSES_BAR_CHART_COLORS = ['#FFC300', '#FF5733', '#C70039', '#00505C', '#8E44AD'] # Amarillo-Naranja, Naranja-Rojo, Rojo-Carmesi, Azul verdoso oscuro, Morado
PV_GLASSES_DIRTY_COLUMNS_FOR_SR = ["R_FC3_Avg", "R_FC4_Avg", "R_FC5_Avg"] # Columnas de vidrios sucios para calcular SR
PV_GLASSES_REF_PREFIX = "R" # Prefijo para las columnas SR, ej. SR_R_FC3

# --- Configuraciones Geográficas del Sitio (NUEVO) ---
SITE_LATITUDE = -23.506  # Grados decimales (Ej: Atacama)
SITE_LONGITUDE = -69.079 # Grados decimales (Ej: Atacama)
SITE_ALTITUDE = 1380     # Metros sobre el nivel del mar (Ej: Atacama)

# Constantes relacionadas con el calendario para PV Glasses
CALENDAR_SELECTED_SAMPLES_FILENAME = "calendario_muestras_seleccionado.csv"
CALENDAR_STRUCTURE_FILTER_PV_GLASSES = "Fija a RC"
CALENDAR_MASS_COLUMNS_PV_GLASSES = ['Masa A', 'Masa B', 'Masa C'] # Nombres como aparecen en el CSV del calendario
PV_GLASSES_MASS_COLUMNS_IN_CALENDAR_MAP = { # Mapeo de nombre en CSV calendario a nombre en DataFrame final
    'Masa A': 'Masa_A_Referencia',
    'Masa B': 'Masa_B_Referencia',
    'Masa C': 'Masa_C_Referencia'
}

# Constantes para la selección de datos post-exposición para PV Glasses
PV_GLASSES_DAYS_POST_EXPOSURE = 5
PV_GLASSES_EXCEPTION_PERIOD = "Semestral"
PV_GLASSES_EXCEPTION_DATE = "2025-01-16"
PV_GLASSES_EXCEPTION_DAYS = 4

# Configuración de formatos de tiempo
DUSTIQ_TIME_COLUMN = 'Date&Time'
DUSTIQ_TIME_FORMAT = '%Y-%m-%d %H:%M:%S' # Ejemplo, ajustar si es diferente
PV_GLASSES_TIME_COLUMN = 'timestamp' # Columna de tiempo original en raw_pv_glasses_data.csv
PV_GLASSES_TIME_FORMAT = '%Y-%m-%d %H:%M:%S%z' # Formato de tiempo para PV Glasses
PV_GLASSES_TIME_COLUMN_PROCESSED = '_time_processed_utc_naive' # Nueva columna de tiempo procesada

# Patrones de columnas para PV Glasses
PV_GLASSES_COLUMN_PATTERN = r'R_FC\d_Avg' # Ejemplo: 'R_FC1_Avg', 'R_FC2_Avg', etc.
PV_GLASSES_REF_COL1_NAME = 'R_FC1_Avg'
PV_GLASSES_REF_COL2_NAME = 'R_FC2_Avg'

# Configuración de filtro de mediodía solar para PV Glasses
PV_GLASSES_FILTER_SOLAR_NOON = True
PV_GLASSES_USE_REAL_SOLAR_NOON = True # True para usar MedioDiaSolar, False para horario fijo
PV_GLASSES_SOLAR_NOON_START_HOUR = 10 # Hora de inicio para filtro fijo (si PV_GLASSES_USE_REAL_SOLAR_NOON = False)
PV_GLASSES_SOLAR_NOON_END_HOUR = 16   # Hora de fin para filtro fijo (si PV_GLASSES_USE_REAL_SOLAR_NOON = False)
PV_GLASSES_SOLAR_NOON_INTERVAL_MINUTES = 60 # Intervalo en minutos para MedioDiaSolar real

# Configuración de limpieza de datos para PV Glasses
PV_GLASSES_REMOVE_OUTLIERS_IQR = True

# Constantes para análisis de Soiling Ratio (SR) en PV Glasses
PV_GLASSES_SR_IRRADIANCE_THRESHOLD = 300
PV_GLASSES_SR_COLUMNS_TO_PLOT = ['SR_R_FC3', 'SR_R_FC4', 'SR_R_FC5']
PV_GLASSES_SR_TO_MASS_MAP = {
    'SR_R_FC3': 'Masa_C_Referencia',  # SR_R_FC3 se asocia con Masa C
    'SR_R_FC4': 'Masa_B_Referencia',  # SR_R_FC4 se asocia con Masa B
    'SR_R_FC5': 'Masa_A_Referencia'   # SR_R_FC5 se asocia con Masa A
}
PV_GLASSES_PERIOD_ORDER_FOR_BAR_CHART = ['semanal', '2 semanas', 'Mensual', 'Trimestral', 'Cuatrimestral', 'Semestral', '1 año']
PV_GLASSES_BAR_CHART_COLORS = ['#FFC300', '#FF5733', '#C70039', '#00505C', '#8E44AD'] # Amarillo-Naranja, Naranja-Rojo, Rojo-Carmesi, Azul verdoso oscuro, Morado
PV_GLASSES_DIRTY_COLUMNS_FOR_SR = ["R_FC3_Avg", "R_FC4_Avg", "R_FC5_Avg"] # Columnas de vidrios sucios para calcular SR
PV_GLASSES_REF_PREFIX = "R" # Prefijo para las columnas SR, ej. SR_R_FC3

# Constantes relacionadas con el calendario para PV Glasses
CALENDAR_SELECTED_SAMPLES_FILENAME = "calendario_muestras_seleccionado.csv"
CALENDAR_STRUCTURE_FILTER_PV_GLASSES = "Fija a RC"
CALENDAR_MASS_COLUMNS_PV_GLASSES = ['Masa A', 'Masa B', 'Masa C'] # Nombres como aparecen en el CSV del calendario
PV_GLASSES_MASS_COLUMNS_IN_CALENDAR_MAP = { # Mapeo de nombre en CSV calendario a nombre en DataFrame final
    'Masa A': 'Masa_A_Referencia',
    'Masa B': 'Masa_B_Referencia',
    'Masa C': 'Masa_C_Referencia'
}

# Constantes para la selección de datos post-exposición para PV Glasses
PV_GLASSES_DAYS_POST_EXPOSURE = 5
PV_GLASSES_EXCEPTION_PERIOD = "Semestral"
PV_GLASSES_EXCEPTION_DATE = "2025-01-16"
PV_GLASSES_EXCEPTION_DAYS = 4

# --- Configuraciones para Análisis de Desviaciones Estadísticas ---
STATISTICAL_DEVIATION_Z_SCORE_THRESHOLD = 3.0      # Umbral para detección Z-Score
STATISTICAL_DEVIATION_IQR_FACTOR = 1.5             # Factor multiplicador para IQR
STATISTICAL_DEVIATION_ISOLATION_CONTAMINATION = 0.1 # Proporción esperada de anomalías
STATISTICAL_DEVIATION_ROLLING_WINDOW = 24          # Ventana para media móvil (horas)
STATISTICAL_DEVIATION_ROLLING_THRESHOLD = 3.0      # Umbral para desviación móvil
STATISTICAL_DEVIATION_SEASONAL_PERIOD = 24         # Período estacional (horas)
STATISTICAL_DEVIATION_MIN_DATA_POINTS = 100        # Mínimo de datos para análisis
STATISTICAL_DEVIATION_DBSCAN_EPS = 0.5             # Parámetro eps para DBSCAN
STATISTICAL_DEVIATION_DBSCAN_MIN_SAMPLES = 5       # Mínimo de muestras para DBSCAN

# Configuración de formatos de tiempo
DUSTIQ_TIME_COLUMN = 'Date&Time'
DUSTIQ_TIME_FORMAT = '%Y-%m-%d %H:%M:%S' # Ejemplo, ajustar si es diferente
PV_GLASSES_TIME_COLUMN = '_time' # Columna de tiempo original en raw_pv_glasses_data.csv
PV_GLASSES_TIME_FORMAT = '%Y-%m-%d %H:%M:%S%z' # Formato de tiempo para PV Glasses
PV_GLASSES_TIME_COLUMN_PROCESSED = '_time_processed_utc_naive' # Nueva columna de tiempo procesada

# Puedes añadir más configuraciones globales o específicas aquí. 