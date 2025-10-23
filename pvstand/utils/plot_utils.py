import os
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def save_plot(fig, filename, subdir=None, dpi=300, bbox_inches='tight'):
    """
    Guarda un gráfico en el directorio de gráficos con una estructura consistente.
    
    Args:
        fig: Figura de matplotlib a guardar
        filename: Nombre del archivo (sin extensión)
        subdir: Subdirectorio dentro de graficos_analisis_integrado_py
        dpi: Resolución de la imagen (default: 300)
        bbox_inches: Ajuste de márgenes (default: 'tight')
    """
    # Directorio base para gráficos - usar ruta absoluta
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'graficos_analisis_integrado_py')
    
    # Crear ruta completa
    if subdir:
        output_dir = os.path.join(base_dir, subdir)
    else:
        output_dir = base_dir
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Asegurar que el nombre del archivo tenga extensión .png
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    # Construir ruta completa
    output_path = os.path.join(output_dir, filename)
    
    # Guardar figura
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Gráfico guardado en: {output_path}")
    
    return output_path

def get_plot_path(filename, subdir=None):
    """
    Obtiene la ruta donde se guardará un gráfico.
    
    Args:
        filename: Nombre del archivo (sin extensión)
        subdir: Subdirectorio dentro de graficos_analisis_integrado_py
    
    Returns:
        str: Ruta completa donde se guardará el gráfico
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'graficos_analisis_integrado_py')
    
    if subdir:
        output_dir = os.path.join(base_dir, subdir)
    else:
        output_dir = base_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    return os.path.join(output_dir, filename) 