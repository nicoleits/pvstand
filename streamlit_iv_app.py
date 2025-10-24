#!/usr/bin/env python3
"""
Aplicación Streamlit para visualización de curvas IV del PVStand (+ IV600 opcional)
- Mantiene import original (solo process_pvstand_iv_files)
- Suma IV600 de forma perezosa: si hay resultados, los usa; si no, intenta procesar sin romper la app
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import glob

# Agregar el directorio raíz del proyecto al path de Python
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# === Import original que ya te funcionaba ===
from pvstand.analysis.pvstand_iv_processor import process_pvstand_iv_files
from pathlib import Path

# Candidatos donde podría estar el reporte del IV600
IV600_REPORT_CANDIDATES = [
    "/home/nicole/atamos_pvstand/pvstand/pvstand/datos_procesados_analisis_integrado_py/iv600/iv_curves/iv_curves_report.xlsx",
    os.path.join(project_root, "pvstand", "datos_procesados_analisis_integrado_py", "iv600", "iv_curves", "iv_curves_report.xlsx"),
    os.path.join(project_root, "pvstand", "pvstand", "datos_procesados_analisis_integrado_py", "iv600", "iv_curves", "iv_curves_report.xlsx"),
]

def _find_iv600_report():
    for p in IV600_REPORT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

def _load_iv600_analysis_from_report(report_path: str):
    """
    Lee 'Analisis_Parametros' de iv_curves_report.xlsx y devuelve un DataFrame
    homogeneizado con las columnas esperadas por la app.
    """
    try:
        xls = pd.ExcelFile(report_path)
    except Exception as e:
        st.warning(f"IV600: no se pudo abrir el reporte ({e}).")
        return None

    # Buscar hoja 'Analisis_Parametros' (tolerante a acentos/mayúsculas)
    sheet = None
    for s in xls.sheet_names:
        sl = s.strip().lower().replace("á","a")
        if sl == "analisis_parametros":
            sheet = s
            break
    if sheet is None:
        st.warning("IV600: hoja 'Analisis_Parametros' no encontrada en el Excel.")
        return None

    df = xls.parse(sheet)

    # Renombrar columnas a las que usa la app
    rename = {
        "Archivo": "Filename",
        "Fecha": "Date",
        "Hora": "Time",
        "Modulo": "Module",
        "Irradiacion_W_m2": "Irradiance_W_m2",
        "Temperatura_C": "Temperature_C",
        "Eficiencia_%": "Efficiency_%"
    }
    df = df.rename(columns=rename, errors="ignore")

    # Asegurar columnas esperadas por la app
    expected = ["Filename","Date","Time","Module","Module_Category",
                "Irradiance_W_m2","Temperature_C","Pmax_W","Vmp_V","Imp_A",
                "Isc_A","Voc_V","FF","Efficiency_%"]
    if "Module_Category" not in df.columns:
        df["Module_Category"] = "IV600"
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    return df[expected]

def _load_iv600_curves_from_report(report_path: str):
    """
    Lee las hojas por curva del Excel (todas excepto 'Metadatos' y 'Analisis_Parametros')
    y devuelve una lista de curvas para graficar (como las reales de PVStand).
    """
    try:
        xls = pd.ExcelFile(report_path)
    except Exception:
        return []

    skip = set()
    for s in xls.sheet_names:
        sl = s.strip().lower().replace("á","a")
        if sl in {"metadatos","analisis_parametros"}:
            skip.add(s)

    # Mapa opcional filename -> hora, para etiquetar
    time_map = {}
    try:
        df_params = _load_iv600_analysis_from_report(report_path)
        if df_params is not None:
            time_map = dict(zip(df_params["Filename"].astype(str), df_params["Time"].astype(str)))
    except Exception:
        pass

    curves = []
    for s in xls.sheet_names:
        if s in skip:
            continue
        try:
            df = xls.parse(s)
            needed = {"Voltage_V","Current_A","Power_W"}
            if not needed.issubset(set(df.columns)):
                continue
            arr = df[["Voltage_V","Current_A","Power_W"]].dropna().to_numpy()
            if arr.size == 0:
                continue
            # etiquetar con hora si la encontramos por filename base
            base = Path(s).stem
            label_time = ""
            # buscar coincidencia relajada
            for k,v in time_map.items():
                if base in k or k in base:
                    label_time = v
                    break
            curves.append({
                "filename": s,
                "time": label_time,
                "module_category": "IV600",
                "color": "green",
                "iv_data": arr
            })
        except Exception:
            continue
    return curves

# -----------------------------------------------------------------------------
# Helpers específicos para IV600 (opcionales y sin romper el arranque)
# -----------------------------------------------------------------------------
def _iv600_out_dir():
    # Mantiene tu convención de outputs
    return os.path.join(project_root, "pvstand", "datos_procesados_analisis_integrado_py", "iv600", "iv_curves")

def _iv600_in_dir_or_file():
    """
    Devuelve una ruta candidata para IV600: intenta usar config.paths.IV600_RAW_DATA_DIR o
    config.paths.IV600_RAW_DATA_FILE si existen; sino, cae en pvstand/datos.
    """
    try:
        from config import paths as _paths
        cand = getattr(_paths, "IV600_RAW_DATA_DIR", None) or getattr(_paths, "IV600_RAW_DATA_FILE", None)
        if cand:
            return cand
    except Exception:
        pass
    return os.path.join(project_root, "pvstand", "datos")

def _try_process_iv600():
    """
    Intenta procesar IV600 **sin romper la app**:
    - Importa process_IV600_iv_files dentro de un try/except
    - Si falta openpyxl o la función, solo avisa en UI y continúa
    """
    try:
        from pvstand.analysis.pvstand_iv_processor import process_IV600_iv_files  # import perezoso
    except Exception:
        st.info("IV600: módulo aún no disponible para procesamiento automático. "
                "Si ya generaste resultados, se cargarán desde el disco.")
        return False

    data_dir = _iv600_in_dir_or_file()
    out_dir = _iv600_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    try:
        res = process_IV600_iv_files(data_dir=data_dir, output_dir=out_dir)
        return bool(res)
    except Exception as e:
        st.warning(f"IV600: no se pudo procesar automáticamente ({e}). "
                   f"Si ya tienes el CSV en {_iv600_out_dir()}, lo cargaremos.")
        return False

# -----------------------------------------------------------------------------
# Tu código original (con mínimos cambios para combinar IV600)
# -----------------------------------------------------------------------------
def load_corrected_curves():
    """Carga las curvas corregidas a STC"""
    corrected_dir = os.path.join(project_root, "pvstand", "resultados_correccion")
    pattern = os.path.join(corrected_dir, "*_corregida.csv")
    corrected_files = glob.glob(pattern)

    curves = []
    for filepath in corrected_files:
        try:
            df = pd.read_csv(filepath)
            df["Archivo"] = os.path.basename(filepath)
            curves.append(df)
        except Exception:
            st.warning(f"Error leyendo archivo corregido: {filepath}")
            continue
    return curves

def _load_pvstand_analysis():
    """Carga (y si falta, procesa) el análisis PVStand (tu lógica original)."""
    data_dir = os.path.join(project_root, "pvstand", "datos")
    output_dir = os.path.join(project_root, "pvstand", "datos_procesados_analisis_integrado_py", "iv_curves")
    os.makedirs(output_dir, exist_ok=True)

    analysis_file = os.path.join(output_dir, "iv_analysis.csv")
    if not os.path.exists(analysis_file):
        st.info("Procesando datos de curvas IV (PVStand)...")
        results = process_pvstand_iv_files(data_dir=data_dir, output_dir=output_dir)
        if not results:
            st.error("Error procesando datos PVStand")
            return None

    try:
        return pd.read_csv(analysis_file)
    except Exception as e:
        st.error(f"Error cargando análisis PVStand: {e}")
        return None

def _load_iv600_analysis_if_available():
    """
    Carga análisis de IV600 si ya existe; si no existe, intenta procesar.
    Devuelve DataFrame o None (no revienta la app).
    """
    analysis_file = os.path.join(_iv600_out_dir(), "iv_analysis.csv")
    if not os.path.exists(analysis_file):
        # Intentar procesar de forma segura
        st.info("Buscando resultados de IV600...")
        _try_process_iv600()

    if os.path.exists(analysis_file):
        try:
            df = pd.read_csv(analysis_file)
            # Homologar columnas si hiciera falta (por compatibilidad)
            expected = ["Filename","Date","Time","Module","Module_Category",
                        "Irradiance_W_m2","Temperature_C","Pmax_W","Vmp_V","Imp_A",
                        "Isc_A","Voc_V","FF","Efficiency_%"]
            for c in expected:
                if c not in df.columns:
                    df[c] = np.nan
            return df[expected]
        except Exception as e:
            st.warning(f"IV600: no se pudo leer el análisis existente ({e}).")
            return None
    return None

def load_iv_data():
    """Carga datos PVStand y, si existe el Excel, también IV600."""
    try:
        # PVStand (tu lógica original)
        data_dir = os.path.join(project_root, "pvstand", "datos")
        output_dir = os.path.join(project_root, "pvstand", "datos_procesados_analisis_integrado_py", "iv_curves")
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(os.path.join(output_dir, "iv_analysis.csv")):
            st.info("Procesando datos de curvas IV (PVStand)...")
            results = process_pvstand_iv_files(data_dir=data_dir, output_dir=output_dir)
            if not results:
                st.error("Error procesando datos PVStand")
                return None

        df_pv = pd.read_csv(os.path.join(output_dir, "iv_analysis.csv"))
        df_pv["Source"] = "PVStand"

        # IV600 desde el Excel (si está)
        report = _find_iv600_report()
        if report:
            df_600 = _load_iv600_analysis_from_report(report)
            if df_600 is not None and not df_600.empty:
                df_600["Source"] = "IV600"
                # Unir y ordenar columnas
                order_cols = ["Filename","Date","Time","Module","Module_Category",
                              "Irradiance_W_m2","Temperature_C","Pmax_W","Vmp_V","Imp_A",
                              "Isc_A","Voc_V","FF","Efficiency_%","Source"]
                combined = pd.concat([df_pv, df_600], ignore_index=True, sort=False)
                cols = [c for c in order_cols if c in combined.columns] + \
                       [c for c in combined.columns if c not in order_cols]
                return combined[cols]
            else:
                st.info("IV600: reporte encontrado pero sin datos legibles; se muestra solo PVStand.")
        else:
            st.info("No se encontró el reporte Excel de IV600; se muestra solo PVStand.")

        return df_pv

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None


def load_real_iv_data():
    """Carga los datos reales de las curvas IV (como tenías, sólo PVStand)."""
    try:
        data_dir = os.path.join(project_root, "pvstand", "datos")

        # Procesar archivos si no existen (sólo para asegurar IV real listo)
        output_dir = os.path.join(project_root, "pvstand", "datos_procesados_analisis_integrado_py", "iv_curves")
        if not os.path.exists(os.path.join(output_dir, "iv_analysis.csv")):
            results = process_pvstand_iv_files(data_dir=data_dir, output_dir=output_dir)
            if not results:
                return None

        # Cargar datos reales de curvas IV (PVStand .txt)
        real_curves = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='latin-1') as f:
                        lines = f.readlines()

                    iv_data = []
                    for line in lines[23:]:
                        if line.strip():
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                try:
                                    voltage = float(parts[0])
                                    current = float(parts[1])
                                    power = float(parts[2])
                                    iv_data.append([voltage, current, power])
                                except ValueError:
                                    continue

                    if iv_data:
                        time_str = lines[1].split('\t')[1] if len(lines) > 1 else ""
                        if "14:30:00" <= time_str <= "15:05:00":
                            module_category = "Minimódulo"; color = "red"
                        else:
                            module_category = "Módulo Risen"; color = "blue"

                        real_curves.append({
                            'filename': filename,
                            'time': time_str,
                            'module_category': module_category,
                            'color': color,
                            'iv_data': np.array(iv_data)
                        })
                except Exception:
                    continue

        return real_curves
    except Exception as e:
        st.error(f"Error cargando datos reales: {e}")
        return None

def create_interactive_plot(df_analysis):
    """Crea gráficos separados para Módulo Risen y Minimódulo (como tenías)."""
    real_curves = load_real_iv_data()
    if not real_curves:
        st.error("No se pudieron cargar los datos reales de las curvas IV")
        return

    grouped_curves = {"Módulo Risen": [], "Minimódulo": []}
    for curve in real_curves:
        grouped_curves[curve['module_category']].append(curve)

    for module_type, curves in grouped_curves.items():
        if not curves:
            st.warning(f"No se encontraron curvas para {module_type}")
            continue

        st.subheader(f"🔍 {module_type}")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Curvas I-V - {module_type}", f"Curvas P-V - {module_type}"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        for curve in curves:
            iv_data = curve['iv_data']
            voltage, current, power = iv_data[:,0], iv_data[:,1], iv_data[:,2]
            name = f"{curve['time']}"

            fig.add_trace(
                go.Scatter(
                    x=voltage, y=current, mode='lines', name=name,
                    line=dict(color=curve['color'], width=2),
                    hovertemplate=f'<b>{name}</b><br>V: %{{x:.2f}} V<br>I: %{{y:.2f}} A<extra></extra>'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=voltage, y=power, mode='lines', name=name, showlegend=False,
                    line=dict(color=curve['color'], width=2),
                    hovertemplate=f'<b>{name}</b><br>V: %{{x:.2f}} V<br>P: %{{y:.2f}} W<extra></extra>'
                ),
                row=1, col=2
            )

        fig.update_layout(
            height=500, showlegend=True, margin=dict(t=60),
            title_text=f"{module_type} - Curvas IV y PV", title_x=0.5,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        fig.update_xaxes(title_text="Voltaje [V]", row=1, col=1)
        fig.update_yaxes(title_text="Corriente [A]", row=1, col=1)
        fig.update_xaxes(title_text="Voltaje [V]", row=1, col=2)
        fig.update_yaxes(title_text="Potencia [W]", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

# --- Curvas IV600 desde el Excel (si existe) ---
iv600_report = _find_iv600_report()
if iv600_report:
    st.subheader("🟢 Curvas IV Interactivas – IV600 (.xlsx)")
    curves_iv600 = _load_iv600_curves_from_report(iv600_report)
    if curves_iv600:
        # Reusar tu mismo layout de 2 paneles (IV/PV)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Curvas I-V - IV600", "Curvas P-V - IV600"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        for c in curves_iv600:
            v, i, p = c["iv_data"][:,0], c["iv_data"][:,1], c["iv_data"][:,2]
            name = c["time"] or c["filename"]
            fig.add_trace(go.Scatter(x=v, y=i, mode="lines", name=name,
                                     line=dict(color=c["color"], width=2)),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=v, y=p, mode="lines", name=name,
                                     line=dict(color=c["color"], width=2), showlegend=False),
                          row=1, col=2)

        fig.update_layout(height=500, showlegend=True, margin=dict(t=60),
                          title_text="IV600 - Curvas IV y PV", title_x=0.5,
                          legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
        fig.update_xaxes(title_text="Voltaje [V]", row=1, col=1)
        fig.update_yaxes(title_text="Corriente [A]", row=1, col=1)
        fig.update_xaxes(title_text="Voltaje [V]", row=1, col=2)
        fig.update_yaxes(title_text="Potencia [W]", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("El Excel de IV600 no contiene hojas de curva legibles.")

# -----------------------------------------------------------------------------
# Tu main intacto (sin cambios en UI, solo ahora combina IV600 si existe)
# -----------------------------------------------------------------------------
def main():
    """Función principal de la aplicación"""

    st.set_page_config(
        page_title="Análisis PVStand - Curvas IV",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("☀️ Análisis de Curvas IV - PVStand")
    st.markdown("---")

    with st.sidebar:
        st.header("📊 Información del Análisis")
        st.markdown("""
        **Módulos analizados:**
        - 🔵 **Módulo Risen**: 13:30-14:05
        - 🔴 **Minimódulo**: 14:30-15:00

        **Parámetros calculados:**
        - Pmax, Vmp, Imp
        - Isc, Voc
        - Factor de llenado
        - Eficiencia
        """)

    # Cargar datos (ahora intenta combinar IV600 si existe)
    with st.spinner("Cargando datos de curvas IV..."):
        df_analysis = load_iv_data()

    if df_analysis is None:
        st.error("No se pudieron cargar los datos")
        return

    # Resumen
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Mediciones", len(df_analysis))
    with col2:
        risen_count = len(df_analysis[df_analysis['Module_Category'] == 'Módulo Risen']) if 'Module_Category' in df_analysis else 0
        st.metric("🔵 Módulo Risen", risen_count)
    with col3:
        minimodule_count = len(df_analysis[df_analysis['Module_Category'] == 'Minimódulo']) if 'Module_Category' in df_analysis else 0
        st.metric("🔴 Minimódulo", minimodule_count)
    with col4:
        avg_irradiance = df_analysis['Irradiance_W_m2'].mean() if 'Irradiance_W_m2' in df_analysis else np.nan
        st.metric("☀️ Irradiación Promedio", f"{avg_irradiance:.0f} W/m²" if np.isfinite(avg_irradiance) else "N/D")

    st.markdown("---")

    st.header("📊 Curvas IV Interactivas (Datos Reales)")
    real_curves = load_real_iv_data()
    if real_curves:
        st.info(f"✅ Cargadas {len(real_curves)} curvas reales de los archivos de datos")
        col1, col2 = st.columns(2)
        with col1:
            risen_count = len([c for c in real_curves if c['module_category'] == 'Módulo Risen'])
            st.metric("🔵 Curvas Risen", risen_count)
        with col2:
            minimodule_count = len([c for c in real_curves if c['module_category'] == 'Minimódulo'])
            st.metric("🔴 Curvas Minimódulo", minimodule_count)

    create_interactive_plot(df_analysis)

    st.header("⚙️ Curvas IV Corregidas a STC")
    corrected_curves = load_corrected_curves()
    if corrected_curves:
        fig_corr = go.Figure()
        for df in corrected_curves:
            name = df["Archivo"].iloc[0].removesuffix("_corregida.csv")
            fig_corr.add_trace(go.Scatter(x=df["V"], y=df["I"], mode='lines',
                                          name=f"{name} - Original", line=dict(dash="solid", width=2)))
            fig_corr.add_trace(go.Scatter(x=df["V_STC"], y=df["I_STC"], mode='lines',
                                          name=f"{name} - Corregida STC", line=dict(dash="dash", width=2)))
        fig_corr.update_layout(
            title="📊 Comparación de Curvas IV (Original vs Corregida a STC)",
            xaxis_title="Voltaje (V)", yaxis_title="Corriente (A)",
            height=600, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("⚠️ No se encontraron curvas corregidas a STC.")

    # Tabla y descarga
    st.header("📋 Datos del Análisis")
    col1, col2 = st.columns(2)
    with col1:
        module_filter = st.selectbox(
            "Filtrar por tipo de módulo:",
            ["Todos"] + list(df_analysis['Module_Category'].dropna().unique()) if 'Module_Category' in df_analysis else ["Todos"]
        )
    with col2:
        sort_by = st.selectbox(
            "Ordenar por:",
            [c for c in ["Time", "Pmax_W", "Efficiency_%", "Irradiance_W_m2"] if c in df_analysis.columns] or [df_analysis.columns[0]]
        )

    filtered_df = df_analysis.copy()
    if module_filter != "Todos" and 'Module_Category' in filtered_df:
        filtered_df = filtered_df[filtered_df['Module_Category'] == module_filter]
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_by)

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.header("💾 Descargar Datos")
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Descargar CSV",
        data=csv_data,
        file_name="iv_analysis.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Generado automáticamente por el sistema de análisis PVStand</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📐 Parámetros Calculados (Curvas Corregidas)")
    if corrected_curves:
        param_rows = []
        for df in corrected_curves:
            name = df["Archivo"].iloc[0].replace("_corregida.csv", "")
            pmax_idx = df["P_STC"].idxmax()
            pmax = df["P_STC"].iloc[pmax_idx]
            vmp = df["V_STC"].iloc[pmax_idx]
            imp = df["I_STC"].iloc[pmax_idx]
            voc_candidates = df[df["I_STC"] <= 0.05]
            voc = voc_candidates["V_STC"].max() if not voc_candidates.empty else np.nan
            isc_candidates = df[df["V_STC"] <= 0.05]
            isc = isc_candidates["I_STC"].max() if not isc_candidates.empty else np.nan
            ff = pmax / (voc * isc) if (pd.notna(voc) and pd.notna(isc) and voc > 0 and isc > 0) else np.nan
            param_rows.append({
                "Archivo": name,
                "Pmax (W)": round(pmax, 2),
                "Vmp (V)": round(vmp, 2),
                "Imp (A)": round(imp, 2),
                "Voc (V)": round(voc, 2) if pd.notna(voc) else "-",
                "Isc (A)": round(isc, 2) if pd.notna(isc) else "-",
                "FF": round(ff, 3) if pd.notna(ff) else "-"
            })
        df_params = pd.DataFrame(param_rows)
        st.dataframe(df_params, use_container_width=True)
    else:
        st.caption("Sin curvas corregidas para calcular parámetros.")

if __name__ == "__main__":
    main()
