
#!/usr/bin/env python3
"""
Procesamiento de curvas IV (PVStand .txt + IV600 .xlsx)
Versión integrada con parser robusto para IV600 (VOpc/IOpc/POpc), metadatos desde 'min',
cálculo consistente de Isc/Voc/FF, plots por categoría y reporte Excel reforzado.
"""

import os
import sys
import glob
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# === Integración con tu proyecto ===
# Agregar el directorio raíz del proyecto al path de Python si este archivo vive en pvstand/analysis
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
except Exception:
    pass

# === Logger ===
try:
    from config.logging_config import logger  # type: ignore
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("pvstand_iv_processor")

# === Rutas de configuración ===
try:
    from config import paths, settings  # type: ignore
except Exception:
    class _PathsFallback:
        BASE_INPUT_DIR = os.getcwd()
        BASE_OUTPUT_CSV_DIR = os.path.join(os.getcwd(), "outputs")
        IV600_RAW_DATA_DIR = os.getcwd()  # por defecto directorio
        IV600_OUTPUT_SUBDIR_CSV = os.path.join(os.getcwd(), "outputs_iv600")
    paths = _PathsFallback()
    settings = object()

# === Utils ===
import re

def _to_float_or_none(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        m = re.search(r"(-?\d+(?:[.,]\d+)?)", x)
        if not m:
            return None
        x = m.group(1).replace(",", ".")
    try:
        return float(x)
    except Exception:
        return None

def _is_valid_irr(x):
    try:
        v = float(x)
        return 50 <= v <= 1400   # ajusta si usas otro rango plausible
    except Exception:
        return False

def _get_full_triplet_from_xls(xls: pd.ExcelFile):
    """
    Devuelve dict con irradiancias {'D4': float|None, 'D20': ..., 'D36': ...}
    leyendo la hoja 'Full' del mismo workbook.
    """
    # Busca la hoja "Full" sin importar mayúsculas/acentos
    sheet = None
    for s in xls.sheet_names:
        if s.strip().lower() == "full":
            sheet = s
            break
    if not sheet:
        return {"D4": None, "D20": None, "D36": None}

    df = xls.parse(sheet, header=None)  # sin encabezados
    def cell(r,c):
        try: return df.iat[r,c]
        except Exception: return None

    d4  = _to_float_or_none(cell(3,3))     # D4  -> (3,3) 0-index
    d20 = _to_float_or_none(cell(19,3))    # D20 -> (19,3)
    d36 = _to_float_or_none(cell(35,3))    # D36 -> (35,3)

    # filtrar a rango plausible
    d4  = d4  if _is_valid_irr(d4)  else None
    d20 = d20 if _is_valid_irr(d20) else None
    d36 = d36 if _is_valid_irr(d36) else None
    return {"D4": d4, "D20": d20, "D36": d36}

def _pick_irradiance_for_time(time_str: str, cands: dict):
    """
    Regla por hora: <11 → D4; 11–14 → D20; >14 → D36.
    Si la hora no sirve o el candidato está vacío, cae al
    último válido disponible (D36→D20→D4).
    """
    hh = None
    try:
        hh = int(str(time_str).strip().split(":")[0])
    except Exception:
        pass

    if hh is not None:
        key = "D4" if hh < 11 else ("D20" if hh < 14 else "D36")
        if cands.get(key) is not None:
            return cands[key]
    # fallback
    for k in ("D36","D20","D4"):
        if cands.get(k) is not None:
            return cands[k]
    return None

def _get_sheet_case_insensitive(xls: pd.ExcelFile, name: str):
    for s in xls.sheet_names:
        if s.strip().lower() == name.strip().lower():
            return s
    return None

def _coerce_numeric(arr_like):
    return pd.to_numeric(pd.Series(arr_like, dtype=object), errors="coerce").to_numpy()

def _safe_attr(obj, name, default=None):
    return getattr(obj, name, default)

# =============================================================================
# Procesamiento PVStand (.txt)
# =============================================================================
def process_pvstand_iv_files(data_dir=None, output_dir=None):
    """
    Función principal para procesar archivos de curvas IV del PVStand (.txt).
    """
    logger.info("=== INICIO DEL PROCESAMIENTO DE CURVAS IV PVSTAND ===")

    # Configurar directorios
    if data_dir is None:
        data_dir = _safe_attr(paths, 'BASE_INPUT_DIR', os.getcwd())
    if output_dir is None:
        output_dir = os.path.join(_safe_attr(paths, 'BASE_OUTPUT_CSV_DIR', os.getcwd()), "iv_curves")

    os.makedirs(output_dir, exist_ok=True)

    # Buscar archivos
    iv_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not iv_files:
        logger.error(f"No se encontraron archivos .txt en {data_dir}")
        return None

    logger.info(f"Encontrados {len(iv_files)} archivos de datos IV (.txt)")

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
        logger.error("No se pudieron procesar archivos de datos IV (.txt)")
        return None

    # Generar análisis y reportes
    analysis_results = generate_iv_analysis(processed_curves, output_dir)
    generate_iv_plots(processed_curves, output_dir)
    generate_iv_reports(processed_curves, metadata_list, output_dir)

    logger.info("=== FIN DEL PROCESAMIENTO DE CURVAS IV PVSTAND ===")
    return {'curves': processed_curves, 'metadata': metadata_list, 'analysis': analysis_results}

# =============================================================================
# Procesamiento IV600 (.xlsx)
# =============================================================================
def process_IV600_iv_files(data_dir=None, output_dir=None):
    """
    Procesa archivos IV600 .xlsx con hoja 'samples' (VOpc/IOpc/POpc en filas) y metadatos en hoja 'min'.
    Acepta que 'data_dir' sea un directorio o un archivo .xlsx.
    """
    try:
        import openpyxl  # requerido por pandas para Excel
    except Exception:
        logger.error("openpyxl es requerido para leer Excel: pip install openpyxl")
        raise

    logger.info("=== INICIO DEL PROCESAMIENTO DE CURVAS IV IV600 ===")

    if data_dir is None:
        # preferir directorio si existe, sino archivo
        data_dir = _safe_attr(paths, "IV600_RAW_DATA_DIR", _safe_attr(paths, "IV600_RAW_DATA_FILE", os.getcwd()))

    if output_dir is None:
        base_out = _safe_attr(paths, "IV600_OUTPUT_SUBDIR_CSV", _safe_attr(paths, "BASE_OUTPUT_CSV_DIR", os.getcwd()))
        output_dir = os.path.join(base_out, "iv_curves")
    os.makedirs(output_dir, exist_ok=True)

    p = Path(str(data_dir))
    if p.is_file():
        iv_files = [str(p)]
    else:
        iv_files = [str(x) for x in Path(str(data_dir)).glob("*.xlsx")]

    if not iv_files:
        logger.error(f"No se encontraron archivos .xlsx en {data_dir}")
        return None

    processed_curves = []
    metadata_list = []

    for filepath in iv_files:
        try:
            logger.info(f"Procesando archivo IV600: {os.path.basename(filepath)}")
            xls = pd.ExcelFile(filepath)

            # NUEVO: leer triplete de irradiancias desde 'Full'
            full_cands = _get_full_triplet_from_xls(xls)
            logger.info(f"[IV600] Full D4/D20/D36 leídas: {full_cands}")

            samples_sheet = _get_sheet_case_insensitive(xls, "samples")
            if not samples_sheet:
                logger.warning(f"Hoja 'samples' no encontrada en {os.path.basename(filepath)}")
                continue
            df = xls.parse(samples_sheet, header=None)

            # Metadatos desde 'min' (opcional)
            irr, timestamp = np.nan, ""
            meta_sheet = _get_sheet_case_insensitive(xls, "min")
            if meta_sheet:
                meta_df = xls.parse(meta_sheet, header=None)
                try:
                    irr_row = meta_df[meta_df[0].astype(str).str.contains("Irradiaci", case=False, na=False)]
                    if not irr_row.empty:
                        irr = pd.to_numeric(irr_row.iloc[0, 2], errors="coerce")
                    ts_row = meta_df[meta_df[0].astype(str).str.contains("Fecha y hora", case=False, na=False)]
                    if not ts_row.empty:
                        timestamp = str(ts_row.iloc[0, 2])
                except Exception:
                    pass

            # detectar muestras: filas donde col0 == 'VOpc'
            starts = df.index[df[0].astype(str).str.upper().eq("VOPC")].tolist()

            for i, start in enumerate(starts):
                # tripleta VOpc/IOpc/POpc
                v_row = _coerce_numeric(df.iloc[start,   1:].to_list())
                i_row = _coerce_numeric(df.iloc[start+1, 1:].to_list())
                p_row = _coerce_numeric(df.iloc[start+2, 1:].to_list())

                n = min(len(v_row), len(i_row), len(p_row))
                v = v_row[:n]; i_ = i_row[:n]; pwr = p_row[:n]
                mask = ~(np.isnan(v) | np.isnan(i_) | np.isnan(pwr))
                v = v[mask]; i_ = i_[mask]; pwr = pwr[mask]
                if v.size == 0:
                    continue

                df_block = pd.DataFrame({
                    "Voltage_V": v,
                    "Current_A": i_,
                    "Power_W": pwr
                })
                df_block["Resistance_Ohm"] = np.where(df_block["Current_A"] != 0,
                                                      df_block["Voltage_V"]/df_block["Current_A"], np.nan)

                # características
                max_idx = int(np.nanargmax(pwr))
                max_p = float(pwr[max_idx])
                vmp   = float(v[max_idx])
                imp   = float(i_[max_idx])
                isc   = float(np.nanmax(i_))  # proxy robusto

                thr = 0.01 * isc if np.isfinite(isc) and isc > 0 else 0.1
                voc_candidates = v[i_ <= thr]
                voc = float(voc_candidates.max()) if voc_candidates.size > 0 else float(np.nanmax(v))
                ff = (max_p / (isc * voc)) if np.isfinite(isc) and np.isfinite(voc) and isc > 0 and voc > 0 else np.nan

                # metadata
                date_str, time_str = None, None
                if timestamp and len(timestamp.split()) >= 2:
                    date_str, time_str = timestamp.split()[0], timestamp.split()[1]

                # --- decidir irradiancia final ---
                irr_min = float(irr) if np.isfinite(irr) else np.nan  # lo que vino de 'min'
                irr_final = irr_min

                # si 'min' no trajo un valor válido, usar 'Full' según la hora
                if not _is_valid_irr(irr_final):
                    irr_from_full = _pick_irradiance_for_time(time_str or "", full_cands)
                    if irr_from_full is not None:
                        irr_final = float(irr_from_full)

                metadata = {
                    "date": date_str or datetime.today().strftime("%Y-%m-%d"),
                    "time": time_str or f"{9+i:02d}:00:00",
                    "module": f"IV600_sample_{i+1}",
                    "module_category": "IV600",
                    "irradiance": irr_final,   # ← YA QUEDA INYECTADA LA IRRADIANCIA CORRECTA
                    "area": np.nan,
                }


                efficiency = np.nan
                if np.isfinite(metadata["irradiance"]) and metadata["irradiance"] > 0 and np.isfinite(max_p):
                    if np.isfinite(metadata["area"]) and metadata["area"] > 0:
                        efficiency = (max_p / (metadata["area"] * metadata["irradiance"])) * 100

                characteristics = {
                    "Pmax": max_p, "Vmp": vmp, "Imp": imp,
                    "Isc": isc, "Voc": voc, "FF": ff,
                    "Efficiency": efficiency, "Avg_Temperature": np.nan
                }

                processed_curves.append({
                    "filename": f"{os.path.basename(filepath)}_sample{i+1}",
                    "filepath": filepath,
                    "metadata": metadata,
                    "iv_data": df_block,
                    "characteristics": characteristics
                })
                metadata_list.append(metadata)

        except Exception as e:
            logger.error(f"Error procesando archivo {filepath}: {e}")

    if not processed_curves:
        logger.error("No se pudieron procesar archivos de IV600")
        return None

    analysis_results = generate_iv_analysis(processed_curves, output_dir)
    generate_iv_plots(processed_curves, output_dir)
    generate_iv_reports(processed_curves, metadata_list, output_dir)

    logger.info("=== FIN DEL PROCESAMIENTO DE CURVAS IV IV600 ===")
    return {"curves": processed_curves, "metadata": metadata_list, "analysis": analysis_results}

# =============================================================================
# Parser PVStand .txt (sin cambios de formato, con robustez)
# =============================================================================
def parse_single_iv_file(filepath, file_type='pvstand'):
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
            raise UnicodeDecodeError("codec", b"", 0, 1, "No se pudo decodificar el archivo con ninguna codificación")

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
    Extrae metadatos del archivo PVStand .txt.
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
                    'area': float(date_time_line[5]) if date_time_line[5] else np.nan
                })
                # Categoría (solo para compatibilidad; plotting usa category explícita si existe)
                time_str = date_time_line[1]
                metadata['module_category'] = 'Minimódulo' if ("14:30:00" <= time_str <= "15:00:00") else 'Módulo Risen'

        # Irradiación (línea 5) - último campo numérico
        if len(lines) >= 5:
            irrad_line = lines[4].strip().split()
            try:
                metadata['irradiance'] = float(irrad_line[-1])
            except Exception:
                metadata['irradiance'] = np.nan

        # Temperaturas de celdas monitor (líneas 9-10)
        for i in range(9, 11):
            if i < len(lines) and lines[i].strip():
                temp_line = lines[i].strip().split()
                if len(temp_line) >= 6:
                    cell_name = temp_line[1]
                    try:
                        temp_value = float(temp_line[5])
                    except Exception:
                        temp_value = np.nan
                    metadata[f'temp_{cell_name}'] = temp_value

        # Parámetros principales (línea 22 aprox.)
        if len(lines) >= 22:
            params_line = lines[21].strip().split('\t')
            def _f(idx):
                try:
                    return float(params_line[idx])
                except Exception:
                    return np.nan
            if len(params_line) >= 17:
                metadata.update({
                    'Pmax_raw': _f(0), 'Imax_raw': _f(1), 'Umax_raw': _f(2),
                    'I_at_pmax': _f(3), 'U_at_pmax': _f(4),
                    'FF_raw': _f(5), 'Eta_raw': _f(6),
                    'E0_pyr': _f(7), 'MPPFit': _f(8), 'IscFit': _f(9),
                    'UocFit': _f(10), 'EtaFit': _f(11), 'ImppFit': _f(12),
                    'UmppFit': _f(13), 'FF_fit': _f(14), 'MSE_MPPFit': _f(15),
                    'Irradiation_Fluctuation': _f(16)
                })
            if len(params_line) >= 22:
                metadata.update({
                    'temp_015_2017': _f(18),
                    'temp_019_2017': _f(19),
                    'temp_015_2017_actual': _f(20),
                    'temp_019_2017_actual': _f(21)
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
        # Los datos de curva IV empiezan aprox. en la línea 24 (índice 23)
        for i in range(23, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 3:
                    try:
                        voltage = float(parts[0]); current = float(parts[1]); power = float(parts[2])
                        iv_data.append([voltage, current, power])
                    except ValueError:
                        continue
        if iv_data:
            df_iv = pd.DataFrame(iv_data, columns=['Voltage_V', 'Current_A', 'Power_W'])
            df_iv['Resistance_Ohm'] = np.where(
                df_iv['Current_A'] != 0, df_iv['Voltage_V'] / df_iv['Current_A'], np.nan
            )
        else:
            df_iv = pd.DataFrame(columns=['Voltage_V', 'Current_A', 'Power_W', 'Resistance_Ohm'])
    except Exception as e:
        logger.warning(f"Error extrayendo datos IV: {e}")
        df_iv = pd.DataFrame(columns=['Voltage_V', 'Current_A', 'Power_W', 'Resistance_Ohm'])
    return df_iv

def calculate_iv_characteristics(df_iv, metadata):
    """
    Calcula características de la curva IV (robusto en Isc/Voc/FF).
    """
    characteristics = {}
    if df_iv is None or df_iv.empty:
        return characteristics
    try:
        max_power_idx = int(df_iv['Power_W'].idxmax())
        max_power_point = df_iv.loc[max_power_idx]

        characteristics.update({
            'Pmax': float(max_power_point['Power_W']),
            'Vmp':  float(max_power_point['Voltage_V']),
            'Imp':  float(max_power_point['Current_A'])
        })

        # Isc como máximo de corriente (funciona para ambos layouts)
        isc = float(df_iv['Current_A'].max())
        characteristics['Isc'] = isc

        # Voc con umbral dinámico (1% de Isc; fallback 0.1 A)
        thr = 0.01 * isc if np.isfinite(isc) and isc > 0 else 0.1
        voc_candidates = df_iv.loc[df_iv['Current_A'] <= thr, 'Voltage_V']
        voc = float(voc_candidates.max()) if not voc_candidates.empty else float(df_iv['Voltage_V'].max())
        characteristics['Voc'] = voc

        if np.isfinite(isc) and np.isfinite(voc) and isc > 0 and voc > 0:
            ff = float(max_power_point['Power_W']) / (isc * voc)
        else:
            ff = np.nan
        characteristics['FF'] = ff

        area = metadata.get('area', np.nan)
        irradiance = metadata.get('irradiance', np.nan)
        if np.isfinite(area) and area > 0 and np.isfinite(irradiance) and irradiance > 0:
            efficiency = (float(max_power_point['Power_W']) / (area * irradiance)) * 100.0
        else:
            efficiency = np.nan
        characteristics['Efficiency'] = efficiency

        temp_015 = metadata.get('temp_015_2017', np.nan)
        temp_019 = metadata.get('temp_019_2017', np.nan)
        if np.isfinite(temp_015) and np.isfinite(temp_019):
            characteristics['Avg_Temperature'] = float((temp_015 + temp_019) / 2.0)
        elif np.isfinite(temp_015):
            characteristics['Avg_Temperature'] = float(temp_015)
        elif np.isfinite(temp_019):
            characteristics['Avg_Temperature'] = float(temp_019)
        else:
            characteristics['Avg_Temperature'] = np.nan

    except Exception as e:
        logger.warning(f"Error calculando características: {e}")
    return characteristics

# =============================================================================
# Análisis / Gráficos / Reportes
# =============================================================================
def generate_iv_analysis(processed_curves, output_dir):
    """
    Genera DataFrame con parámetros clave y guarda CSV.
    """
    logger.info("Generando análisis de curvas IV...")
    analysis_data = []
    for curve in processed_curves:
        analysis_data.append({
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
        })
    df_analysis = pd.DataFrame(analysis_data)
    os.makedirs(output_dir, exist_ok=True)
    analysis_file = os.path.join(output_dir, "iv_analysis.csv")
    df_analysis.to_csv(analysis_file, index=False)
    logger.info(f"Análisis guardado en: {analysis_file}")
    return df_analysis

def generate_iv_plots(processed_curves, output_dir):
    """
    Genera gráficos estáticos I-V y P-V por categoría.
    """
    logger.info("Generando gráficos de curvas IV...")

    if not processed_curves:
        logger.warning("No hay curvas procesadas para graficar")
        return

    # Agrupar por categoría
    buckets = {}
    for curve in processed_curves:
        cat = curve['metadata'].get('module_category', 'Desconocido')
        buckets.setdefault(cat, []).append(curve)

    COLOR_BY_CAT = {
        'Minimódulo': 'red',
        'Módulo Risen': 'blue',
        'IV600': 'green',
        'Desconocido': 'gray'
    }

    import matplotlib.pyplot as plt

    # I-V
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for cat, curves in buckets.items():
        for curve in curves:
            df_iv = curve['iv_data']
            if df_iv.empty: 
                continue
            md = curve['metadata']
            label = f"{cat} {md.get('time','')} - {md.get('irradiance',np.nan):.0f} W/m²"
            ax.plot(df_iv['Voltage_V'], df_iv['Current_A'],
                    color=COLOR_BY_CAT.get(cat, 'gray'), alpha=0.8, linewidth=1.8, label=label)
    ax.set_xlabel('Voltaje [V]', fontsize=12)
    ax.set_ylabel('Corriente [A]', fontsize=12)
    ax.set_title('Curvas I-V Comparativas', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    iv_plot_file = os.path.join(output_dir, "iv_curves_i_v.png")
    plt.savefig(iv_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico I-V guardado en: {iv_plot_file}")

    # P-V
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for cat, curves in buckets.items():
        for curve in curves:
            df_iv = curve['iv_data']
            if df_iv.empty: 
                continue
            md = curve['metadata']
            label = f"{cat} {md.get('time','')} - {md.get('irradiance',np.nan):.0f} W/m²"
            ax.plot(df_iv['Voltage_V'], df_iv['Power_W'],
                    color=COLOR_BY_CAT.get(cat, 'gray'), alpha=0.8, linewidth=1.8, label=label)
    ax.set_xlabel('Voltaje [V]', fontsize=12)
    ax.set_ylabel('Potencia [W]', fontsize=12)
    ax.set_title('Curvas P-V Comparativas', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    pv_plot_file = os.path.join(output_dir, "iv_curves_p_v.png")
    plt.savefig(pv_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico P-V guardado en: {pv_plot_file}")

    # Interactivo
    generate_interactive_plots(processed_curves, output_dir)

def generate_interactive_plots(processed_curves, output_dir):
    """
    Genera gráficos interactivos (Plotly) si está disponible.
    """
    logger.info("Generando gráficos interactivos...")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
    except Exception:
        logger.warning("plotly no está disponible. Para instalar: pip install plotly")
        return

    COLOR_BY_CAT = {
        'Minimódulo': 'red',
        'Módulo Risen': 'blue',
        'IV600': 'green',
        'Desconocido': 'gray'
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Curvas I-V Interactivas', 'Curvas P-V Interactivas'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    for curve in processed_curves:
        df_iv = curve['iv_data']
        if df_iv.empty:
            continue
        md = curve['metadata']
        cat = md.get('module_category', 'Desconocido')
        name = f"{cat} {md.get('time','')} - {md.get('irradiance',np.nan):.0f} W/m²"
        color = COLOR_BY_CAT.get(cat, 'gray')

        fig.add_trace(
            go.Scatter(x=df_iv['Voltage_V'], y=df_iv['Current_A'], mode='lines', name=name,
                       line=dict(color=color, width=2),
                       hovertemplate='<b>%{fullData.name}</b><br>V: %{x:.2f} V<br>I: %{y:.2f} A<extra></extra>'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_iv['Voltage_V'], y=df_iv['Power_W'], mode='lines', name=name, showlegend=False,
                       line=dict(color=color, width=2),
                       hovertemplate='<b>%{fullData.name}</b><br>V: %{x:.2f} V<br>P: %{y:.2f} W<extra></extra>'),
            row=1, col=2
        )

    fig.update_layout(
        title_text="Curvas IV Interactivas - Comparación de Módulos",
        title_x=0.5, width=1400, height=600,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    fig.update_xaxes(title_text="Voltaje [V]", row=1, col=1)
    fig.update_yaxes(title_text="Corriente [A]", row=1, col=1)
    fig.update_xaxes(title_text="Voltaje [V]", row=1, col=2)
    fig.update_yaxes(title_text="Potencia [W]", row=1, col=2)

    interactive_file = os.path.join(output_dir, "iv_curves_interactive.html")
    pyo.plot(fig, filename=interactive_file, auto_open=False)
    logger.info(f"Gráfico interactivo guardado en: {interactive_file}")

def generate_iv_reports(processed_curves, metadata_list, output_dir):
    """
    Genera reporte Excel consolidado con Metadatos, Analisis_Parametros y hojas por curva.
    """
    logger.info("Generando reportes consolidados...")
    try:
        import openpyxl  # requerido por pandas para Excel
    except Exception:
        logger.warning("openpyxl no está disponible. No se pudo generar el archivo Excel.")
        return

    excel_file = os.path.join(output_dir, "iv_curves_report.xlsx")
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Hoja 1: Metadatos
        df_metadata = pd.DataFrame(metadata_list)
        df_metadata.to_excel(writer, sheet_name='Metadatos', index=False)

        # Hoja 2: Análisis de parámetros
        analysis_data = []
        for curve in processed_curves:
            analysis_data.append({
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
            })
        pd.DataFrame(analysis_data).to_excel(writer, sheet_name='Analisis_Parametros', index=False)

        # Hoja 3+: Curvas IV individuales
        used_names = set()
        for idx, curve in enumerate(processed_curves, start=1):
            df_iv = curve['iv_data']
            if df_iv.empty:
                continue
            base = Path(curve['filename']).stem  # quita extensión
            sheet_name = (base[:25] + f"_{idx}")[:31]
            while sheet_name in used_names or len(sheet_name) == 0:
                # evita duplicados/overflow
                sheet_name = (sheet_name[:28] + "..") if len(sheet_name) >= 30 else (sheet_name + "_")
            used_names.add(sheet_name)
            df_iv.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Reporte consolidado guardado en: {excel_file}")

# =============================================================================
# Runner
# =============================================================================
def run_iv_curve_processing(include_iv600=True):
    """
    Ejecuta PVStand (.txt) y, opcionalmente, IV600 (.xlsx).
    """
    ok = False
    res_txt = process_pvstand_iv_files(data_dir=getattr(paths, "BASE_INPUT_DIR", os.getcwd()))
    ok = ok or bool(res_txt)
    
    if include_iv600:
        data_dir = getattr(paths, "IV600_RAW_DATA_DIR", None)
        if not data_dir:
            data_dir = getattr(paths, "IV600_RAW_DATA_FILE", None)

        res_xlsx = process_IV600_iv_files(data_dir=data_dir)
        ok = ok or bool(res_xlsx)

    logger.info("Procesamiento completado." if ok else "No se procesó ningún archivo.")
    return ok

if __name__ == "__main__":
    print("[INFO] Ejecutando procesamiento de curvas IV (PVStand + IV600)...")
    run_iv_curve_processing(include_iv600=True)
