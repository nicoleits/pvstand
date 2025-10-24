#!/usr/bin/env python3
"""
Aplicación Streamlit para visualización de curvas IV (PVStand + IV600)
- Respeta tus rutas desde config/paths.py
- Procesa si faltan outputs y luego carga iv_analysis.csv de ambos orígenes
- Gráficos I-V y P-V por categoría: Módulo Risen / Minimódulo / IV600
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# -------------------------------
# Paths del proyecto
# -------------------------------
# (Asegura que podamos importar config y pvstand.* desde donde corras streamlit)
this_dir = Path(__file__).resolve().parent
repo_root = this_dir  # ajusta si ejecutas la app en otra carpeta
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Config y processors
from config import paths
from pvstand.analysis.pvstand_iv_processor import (
    process_pvstand_iv_files,
    process_IV600_iv_files,
)

# Salidas (siguiendo tus constantes en paths.py)
PVSTAND_OUT_DIR = Path(paths.PVSTAND_OUTPUT_SUBDIR_CSV) / "iv_curves"
IV600_OUT_DIR   = Path(paths.IV600_OUTPUT_SUBDIR_CSV) / "iv_curves"

# Entradas
PVSTAND_IN_DIR = Path(paths.BASE_INPUT_DIR)  # .txt
# IV600 puede ser archivo o directorio, según tu paths.py
IV600_IN = Path(getattr(paths, "IV600_RAW_DATA_DIR", getattr(paths, "IV600_RAW_DATA_FILE", paths.BASE_INPUT_DIR)))


COLOR_BY_CAT = {
    "Módulo Risen": "blue",
    "Minimódulo": "red",
    "IV600": "green",
    "Desconocido": "gray",
}

# -------------------------------
# Utilidades
# -------------------------------
def _ensure_outputs():
    PVSTAND_OUT_DIR.mkdir(parents=True, exist_ok=True)
    IV600_OUT_DIR.mkdir(parents=True, exist_ok=True)

def _list_iv600_xlsx():
    """Devuelve lista de archivos .xlsx de IV600 desde lo definido en paths."""
    if IV600_IN.is_file() and IV600_IN.suffix.lower() == ".xlsx":
        return [str(IV600_IN)]
    if IV600_IN.is_dir():
        return [str(p) for p in IV600_IN.glob("*.xlsx")]
    # fallback: buscar en BASE_INPUT_DIR
    base = Path(paths.BASE_INPUT_DIR)
    return [str(p) for p in base.glob("*.xlsx")]

def _parse_iv600_samples(filepath: str):
    """
    Parser simple de la hoja 'samples' (case-insensitive) con tripletas VOpc/IOpc/POpc por filas.
    Devuelve lista de dicts con iv_data (Nx3), category=IV600, etc.
    """
    try:
        xls = pd.ExcelFile(filepath)
        # buscar hoja 'samples' sin importar mayúsculas
        samples_sheet = None
        for s in xls.sheet_names:
            if s.strip().lower() == "samples":
                samples_sheet = s
                break
        if not samples_sheet:
            return []

        df = xls.parse(samples_sheet, header=None)
        starts = df.index[df[0].astype(str).str.upper().eq("VOPC")].tolist()

        curves = []
        for i, start in enumerate(starts):
            v_row = pd.to_numeric(df.iloc[start,   1:], errors="coerce").to_numpy()
            i_row = pd.to_numeric(df.iloc[start+1, 1:], errors="coerce").to_numpy()
            p_row = pd.to_numeric(df.iloc[start+2, 1:], errors="coerce").to_numpy()

            n = min(len(v_row), len(i_row), len(p_row))
            v = v_row[:n]; i_ = i_row[:n]; p = p_row[:n]
            mask = ~(np.isnan(v) | np.isnan(i_) | np.isnan(p))
            v = v[mask]; i_ = i_[mask]; p = p[mask]
            if v.size == 0:
                continue

            arr = np.column_stack([v, i_, p])
            curves.append({
                "filename": f"{Path(filepath).name}_sample{i+1}",
                "time": f"{9+i:02d}:00:00",
                "module_category": "IV600",
                "color": COLOR_BY_CAT["IV600"],
                "iv_data": arr
            })
        return curves
    except Exception:
        return []

# -------------------------------
# Carga de outputs (procesa si falta)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_iv_analysis_combined():
    """
    Garantiza que existan los iv_analysis.csv de PVStand e IV600 y luego retorna el combinado.
    """
    _ensure_outputs()

    pv_csv = PVSTAND_OUT_DIR / "iv_analysis.csv"
    iv600_csv = IV600_OUT_DIR / "iv_analysis.csv"

    # Procesar PVStand si falta
    if not pv_csv.exists():
        process_pvstand_iv_files(data_dir=str(PVSTAND_IN_DIR), output_dir=str(PVSTAND_OUT_DIR))

    # Procesar IV600 si falta
    if not iv600_csv.exists():
        xlsx_candidates = _list_iv600_xlsx()
        if xlsx_candidates:
            # se pasa el directorio o el archivo; el processor soporta ambos
            data_dir = str(IV600_IN) if IV600_IN.exists() else xlsx_candidates[0]
            process_IV600_iv_files(data_dir=data_dir, output_dir=str(IV600_OUT_DIR))

    dfs = []
    if pv_csv.exists():
        df_pv = pd.read_csv(pv_csv)
        df_pv["Source"] = "PVStand"
        dfs.append(df_pv)

    if iv600_csv.exists():
        df_600 = pd.read_csv(iv600_csv)
        df_600["Source"] = "IV600"
        dfs.append(df_600)

    if not dfs:
        return pd.DataFrame()

    order_cols = [
        "Filename","Date","Time","Module","Module_Category",
        "Irradiance_W_m2","Temperature_C","Pmax_W","Vmp_V","Imp_A",
        "Isc_A","Voc_V","FF","Efficiency_%","Source"
    ]
    df_all = pd.concat(dfs, ignore_index=True)
    cols = [c for c in order_cols if c in df_all.columns] + [c for c in df_all.columns if c not in order_cols]
    return df_all[cols]

# -------------------------------
# Carga de curvas reales para gráficos
# -------------------------------
@st.cache_data(show_spinner=False)
def load_real_curves():
    """
    Carga curvas reales:
    - PVStand: .txt desde BASE_INPUT_DIR
    - IV600: .xlsx 'samples' (si existen)
    """
    curves = []

    # PVStand .txt
    try:
        for txt in Path(PVSTAND_IN_DIR).glob("*.txt"):
            try:
                lines = txt.read_text(encoding="latin-1").splitlines()
            except Exception:
                continue

            iv = []
            for line in lines[23:]:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        v = float(parts[0]); i = float(parts[1]); p = float(parts[2])
                        iv.append([v, i, p])
                    except ValueError:
                        pass

            if not iv:
                continue

            time_str = ""
            try:
                time_str = lines[1].split("\t")[1]
            except Exception:
                pass

            if "14:30:00" <= time_str <= "15:05:00":
                cat = "Minimódulo"
            else:
                cat = "Módulo Risen"
            curves.append({
                "filename": txt.name,
                "time": time_str,
                "module_category": cat,
                "color": COLOR_BY_CAT.get(cat, "gray"),
                "iv_data": np.array(iv)
            })
    except Exception as e:
        st.warning(f"Error leyendo .txt PVStand: {e}")

    # IV600 .xlsx
    try:
        for xlsx in _list_iv600_xlsx():
            curves.extend(_parse_iv600_samples(xlsx))
    except Exception as e:
        st.warning(f"Error leyendo .xlsx IV600: {e}")

    return curves

# -------------------------------
# Gráficos
# -------------------------------
def plot_category(curves, module_type: str):
    if not curves:
        st.warning(f"No se encontraron curvas para {module_type}")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Curvas I-V - {module_type}", f"Curvas P-V - {module_type}"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    for c in curves:
        arr = c["iv_data"]
        v, i, p = arr[:,0], arr[:,1], arr[:,2]
        name = c["time"] or c["filename"]

        fig.add_trace(
            go.Scatter(
                x=v, y=i, mode="lines", name=name,
                line=dict(color=c["color"], width=2),
                hovertemplate=f"<b>{name}</b><br>V: %{{x:.2f}} V<br>I: %{{y:.2f}} A<extra></extra>"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=v, y=p, mode="lines", name=name,
                line=dict(color=c["color"], width=2),
                hovertemplate=f"<b>{name}</b><br>V: %{{x:.2f}} V<br>P: %{{y:.2f}} W<extra></extra>",
                showlegend=False
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

def plot_corrected_curves():
    """Gráfico comparativo Original vs Corregida STC (tu bloque original)."""
    corrected_dir = Path(repo_root) / "pvstand" / "resultados_correccion"
    files = list(corrected_dir.glob("*_corregida.csv"))
    if not files:
        st.warning("⚠️ No se encontraron curvas corregidas a STC.")
        return

    fig = go.Figure()
    for f in files:
        df = pd.read_csv(f)
        name = f.name.replace("_corregida.csv", "")
        fig.add_trace(go.Scatter(x=df["V"], y=df["I"], mode="lines",
                                 name=f"{name} - Original", line=dict(dash="solid", width=2)))
        fig.add_trace(go.Scatter(x=df["V_STC"], y=df["I_STC"], mode="lines",
                                 name=f"{name} - Corregida STC", line=dict(dash="dash", width=2)))
    fig.update_layout(
        title="📊 Comparación de Curvas IV (Original vs Corregida a STC)",
        xaxis_title="Voltaje (V)", yaxis_title="Corriente (A)",
        height=600, legend=dict(orientation="v", y=1, x=1.02, yanchor="top", xanchor="left")
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="Análisis PVStand & IV600 - Curvas IV",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("☀️ Análisis de Curvas IV - PVStand & IV600")
    st.markdown("---")

    with st.sidebar:
        st.header("📊 Información del Análisis")
        st.markdown("""
        **Categorías:**
        - 🔵 Módulo Risen
        - 🔴 Minimódulo
        - 🟢 IV600 (.xlsx)

        **Parámetros:**
        - Pmax, Vmp, Imp
        - Isc, Voc
        - FF (Factor de llenado)
        - Eficiencia (si hay área + irradiancia)
        """)

    with st.spinner("Cargando/Procesando datos..."):
        df = load_iv_analysis_combined()

    if df is None or df.empty:
        st.error("No se pudieron cargar datos de análisis (PVStand/IV600). Revisa rutas en config/paths.py.")
        return

    # Métricas
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📈 Mediciones", len(df))
    with c2: st.metric("🔵 Módulo Risen", len(df[df.get("Module_Category","")== "Módulo Risen"]))
    with c3: st.metric("🔴 Minimódulo", len(df[df.get("Module_Category","")== "Minimódulo"]))
    with c4: st.metric("🟢 IV600", len(df[df.get("Module_Category","")== "IV600"]))
    if "Irradiance_W_m2" in df.columns:
        avg_irr = df["Irradiance_W_m2"].mean()
        st.caption(f"☀️ Irradiación promedio: **{avg_irr:.0f} W/m²**" if np.isfinite(avg_irr) else "☀️ Irradiación promedio no disponible")

    st.markdown("---")

    # Gráficos interactivos (datos reales)
    st.header("📊 Curvas IV Interactivas (Datos Reales)")
    real_curves = load_real_curves()
    if real_curves:
        st.info(f"✅ Cargadas {len(real_curves)} curvas reales (.txt / .xlsx)")
    cats = sorted({c["module_category"] for c in real_curves}) if real_curves else []
    for cat in cats:
        plot_category([c for c in real_curves if c["module_category"] == cat], cat)

    # Curvas corregidas STC
    st.header("⚙️ Curvas IV Corregidas a STC")
    plot_corrected_curves()

    # Tabla con filtros + descarga
    st.header("📋 Datos del Análisis")
    colf1, colf2 = st.columns(2)
    with colf1:
        module_filter = st.selectbox(
            "Filtrar por categoría:",
            ["Todos"] + sorted(df["Module_Category"].dropna().unique())
        )
    with colf2:
        sort_candidates = [c for c in ["Time", "Pmax_W", "Efficiency_%", "Irradiance_W_m2"] if c in df.columns]
        sort_by = st.selectbox("Ordenar por:", sort_candidates or [df.columns[0]])

    filtered = df.copy()
    if module_filter != "Todos":
        filtered = filtered[filtered["Module_Category"] == module_filter]
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by)

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.header("💾 Descargar Datos")
    csv_data = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Descargar CSV",
        data=csv_data,
        file_name="iv_analysis_combined.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666'>Generado automáticamente por el sistema de análisis PVStand & IV600</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
