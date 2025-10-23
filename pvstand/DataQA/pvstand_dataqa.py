#!/usr/bin/env python3
"""
Análisis de calidad de datos para archivo raw_pvstand_iv_data.csv
Detecta discontinuidades y problemas en las mediciones de módulos PV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para español
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

class PVStandDataQualityAnalyzer:
    def __init__(self, filepath):
        """
        Inicializa el analizador de calidad de datos
        
        Args:
            filepath (str): Ruta al archivo raw_pvstand_iv_data.csv
        """
        self.filepath = filepath
        self.data = None
        self.gaps_info = {}
        
    def load_data(self):
        """Carga y prepara los datos"""
        print("Cargando datos...")
        self.data = pd.read_csv(self.filepath)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp')
        print(f"Datos cargados: {len(self.data)} registros")
        print(f"Período: {self.data['timestamp'].min()} a {self.data['timestamp'].max()}")
        print(f"Módulos encontrados: {self.data['module'].unique()}")
        
    def detect_time_gaps(self, expected_interval_minutes=5):
        """
        Detecta gaps temporales en las mediciones
        
        Args:
            expected_interval_minutes (int): Intervalo esperado entre mediciones en minutos
        """
        print(f"\nDetectando gaps temporales (intervalo esperado: {expected_interval_minutes} minutos)...")
        
        gaps_by_module = {}
        
        for module in self.data['module'].unique():
            module_data = self.data[self.data['module'] == module].copy()
            module_data = module_data.sort_values('timestamp')
            
            # Calcular diferencias de tiempo
            time_diffs = module_data['timestamp'].diff()
            expected_delta = timedelta(minutes=expected_interval_minutes)
            
            # Identificar gaps (diferencias mayores al intervalo esperado + tolerancia)
            tolerance = timedelta(minutes=1)  # 1 minuto de tolerancia
            gaps = time_diffs > (expected_delta + tolerance)
            
            gap_info = []
            if gaps.any():
                # Reset index para trabajar con posiciones secuenciales
                module_data_reset = module_data.reset_index(drop=True)
                time_diffs_reset = module_data_reset['timestamp'].diff()
                gaps_reset = time_diffs_reset > (expected_delta + tolerance)
                
                gap_positions = module_data_reset[gaps_reset].index
                for pos in gap_positions:
                    gap_start = module_data_reset.loc[pos-1, 'timestamp'] if pos > 0 else None
                    gap_end = module_data_reset.loc[pos, 'timestamp']
                    gap_duration = time_diffs_reset.loc[pos]
                    
                    gap_info.append({
                        'inicio': gap_start,
                        'fin': gap_end,
                        'duracion': gap_duration,
                        'duracion_horas': gap_duration.total_seconds() / 3600
                    })
            
            gaps_by_module[module] = gap_info
            print(f"  {module}: {len(gap_info)} gaps encontrados")
        
        self.gaps_info = gaps_by_module
        return gaps_by_module
    
    def filter_gaps_by_duration(self, max_hours=2):
        """
        Filtra y muestra gaps menores a una duración específica
        
        Args:
            max_hours (float): Duración máxima en horas para filtrar gaps
        """
        if not self.gaps_info:
            print("Primero debe ejecutar detect_time_gaps()")
            return {}
        
        print(f"\nFiltrando gaps menores a {max_hours} horas...")
        
        filtered_gaps = {}
        total_short_gaps = 0
        
        for module, gaps in self.gaps_info.items():
            short_gaps = [gap for gap in gaps if gap['duracion_horas'] <= max_hours]
            filtered_gaps[module] = short_gaps
            total_short_gaps += len(short_gaps)
            
            if short_gaps:
                print(f"\n{module}: {len(short_gaps)} gaps ≤ {max_hours} horas")
                print("  Inicio                    Fin                      Duración")
                print("  " + "-"*65)
                
                for gap in short_gaps:
                    inicio_str = gap['inicio'].strftime('%Y-%m-%d %H:%M:%S') if gap['inicio'] else 'N/A'
                    fin_str = gap['fin'].strftime('%Y-%m-%d %H:%M:%S')
                    duracion_str = f"{gap['duracion_horas']:.2f} horas"
                    
                    print(f"  {inicio_str}  {fin_str}  {duracion_str}")
            else:
                print(f"\n{module}: Sin gaps ≤ {max_hours} horas")
        
        print(f"\nRESUMEN:")
        print(f"Total de gaps cortos (≤ {max_hours} horas): {total_short_gaps}")
        
        return filtered_gaps
    
    def plot_gaps_timeline(self, max_hours=None, save_plot=True):
        """
        Crea un gráfico de gaps con fecha en X y duración en Y
        
        Args:
            max_hours (float): Máxima duración de gaps a mostrar (None = todos)
            save_plot (bool): Si guardar el gráfico como archivo
        """
        if not self.gaps_info:
            print("Primero debe ejecutar detect_time_gaps()")
            return
        
        print(f"\nCreando gráfico de gaps temporales...")
        
        # Preparar datos para el gráfico
        gap_data = []
        
        for module, gaps in self.gaps_info.items():
            for gap in gaps:
                if max_hours is None or gap['duracion_horas'] <= max_hours:
                    gap_data.append({
                        'fecha': gap['fin'],  # Usar fecha de fin del gap
                        'duracion_horas': gap['duracion_horas'],
                        'modulo': module,
                        'inicio': gap['inicio'],
                        'fin': gap['fin']
                    })
        
        if not gap_data:
            print("No hay gaps para mostrar con los criterios especificados")
            return
        
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Definir categorías de duración y sus colores
        def get_duration_category(hours):
            """Categoriza la duración del gap"""
            minutes = hours * 60
            if minutes <= 10:
                return '≤ 10 min', '#2ecc71'  # Verde
            elif minutes <= 30:
                return '≤ 30 min', '#f39c12'  # Naranja
            elif hours <= 1:
                return '≤ 1 hora', '#e74c3c'  # Rojo
            elif hours <= 2:
                return '≤ 2 horas', '#9b59b6'  # Púrpura
            else:
                return '> 2 horas', '#34495e'  # Gris oscuro
        
        # Marcadores para cada módulo
        markers = {'perc1fixed': 'o', 'perc2fixed': 's'}
        
        # Agrupar datos por categoría de duración y módulo
        duration_categories = {}
        
        for gap in gap_data:
            category, color = get_duration_category(gap['duracion_horas'])
            module = gap['modulo']
            
            if category not in duration_categories:
                duration_categories[category] = {'color': color, 'data': {}}
            
            if module not in duration_categories[category]['data']:
                duration_categories[category]['data'][module] = {
                    'fechas': [],
                    'duraciones': [],
                    'marker': markers.get(module, 'o')
                }
            
            duration_categories[category]['data'][module]['fechas'].append(gap['fecha'])
            duration_categories[category]['data'][module]['duraciones'].append(gap['duracion_horas'])
        
        # Plotear por categoría de duración
        legend_elements = []
        
        # Orden de categorías para la leyenda
        category_order = ['≤ 10 min', '≤ 30 min', '≤ 1 hora', '≤ 2 horas', '> 2 horas']
        
        for category in category_order:
            if category in duration_categories:
                cat_info = duration_categories[category]
                color = cat_info['color']
                
                for module, module_data in cat_info['data'].items():
                    scatter = ax.scatter(
                        module_data['fechas'], 
                        module_data['duraciones'],
                        color=color,
                        marker=module_data['marker'],
                        s=80, 
                        alpha=0.8,
                        edgecolors='black', 
                        linewidth=0.8,
                        label=f'{category} ({module})'
                    )
                    
                    # Añadir a la leyenda solo la primera vez por categoría
                    if not any(element.get_label().startswith(category.split('(')[0]) for element in legend_elements):
                        from matplotlib.lines import Line2D
                        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=10, 
                                                    label=category, markeredgecolor='black'))
        
        # Personalizar el gráfico
        ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
        ax.set_ylabel('Duración del Gap (horas)', fontsize=12, fontweight='bold')
        
        title = f'Timeline de Gaps Temporales en Mediciones PVStand'
        if max_hours:
            title += f' (≤ {max_hours} horas)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Configurar eje X (fechas)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Configurar eje Y
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Añadir líneas de referencia horizontales
        ax.axhline(y=10/60, color='gray', linestyle=':', alpha=0.6, linewidth=1)  # 10 min
        ax.axhline(y=30/60, color='gray', linestyle=':', alpha=0.6, linewidth=1)  # 30 min
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.6, linewidth=1)      # 1 hora
        ax.axhline(y=2, color='gray', linestyle=':', alpha=0.6, linewidth=1)      # 2 horas
        
        # Añadir etiquetas a las líneas de referencia
        ax.text(ax.get_xlim()[1], 10/60, '10 min', ha='right', va='bottom', fontsize=9, alpha=0.7)
        ax.text(ax.get_xlim()[1], 30/60, '30 min', ha='right', va='bottom', fontsize=9, alpha=0.7)
        ax.text(ax.get_xlim()[1], 1, '1 hora', ha='right', va='bottom', fontsize=9, alpha=0.7)
        ax.text(ax.get_xlim()[1], 2, '2 horas', ha='right', va='bottom', fontsize=9, alpha=0.7)
        
        # Crear leyenda combinada (duración + forma por módulo)
        # Leyenda de duración por color
        duration_legend = ax.legend(handles=legend_elements, title='Duración del Gap', 
                                   loc='upper left', framealpha=0.9, fontsize=10)
        ax.add_artist(duration_legend)
        
        # Leyenda de módulos por forma
        from matplotlib.lines import Line2D
        module_elements = []
        for module in self.data['module'].unique():
            marker = markers.get(module, 'o')
            module_elements.append(Line2D([0], [0], marker=marker, color='gray', 
                                        markerfacecolor='gray', markersize=10, 
                                        label=module, markeredgecolor='black', linestyle='None'))
        
        module_legend = ax.legend(handles=module_elements, title='Módulo (Forma)', 
                                 loc='upper right', framealpha=0.9, fontsize=10)
        
        # Estadísticas en el gráfico con conteo por categorías
        total_gaps = len(gap_data)
        avg_duration = sum(g['duracion_horas'] for g in gap_data) / len(gap_data)
        max_duration = max(g['duracion_horas'] for g in gap_data)
        
        # Contar gaps por categoría
        category_counts = {}
        for gap in gap_data:
            category, _ = get_duration_category(gap['duracion_horas'])
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Crear texto de estadísticas
        stats_lines = [f'Total gaps: {total_gaps}']
        stats_lines.append(f'Duración promedio: {avg_duration:.2f}h')
        stats_lines.append(f'Duración máxima: {max_duration:.2f}h')
        stats_lines.append('')  # Línea en blanco
        stats_lines.append('Por categoría:')
        
        for category in category_order:
            if category in category_counts:
                count = category_counts[category]
                percentage = (count / total_gaps) * 100
                stats_lines.append(f'{category}: {count} ({percentage:.1f}%)')
        
        stats_text = '\\n'.join(stats_lines)
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.8), fontsize=9)
        
        plt.tight_layout()
        
        # Guardar el gráfico
        if save_plot:
            filename = f'gaps_timeline{"_filtered" if max_hours else ""}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado como: {filename}")
        
        plt.show()
        
        # Mostrar gaps por categoría
        print(f"\nRESUMEN POR CATEGORÍAS DE DURACIÓN:")
        print("-" * 50)
        
        for category in category_order:
            if category in category_counts:
                count = category_counts[category]
                percentage = (count / total_gaps) * 100
                print(f"{category}: {count} gaps ({percentage:.1f}%)")
        
        print(f"\nGaps más largos encontrados:")
        sorted_gaps = sorted(gap_data, key=lambda x: x['duracion_horas'], reverse=True)
        for i, gap in enumerate(sorted_gaps[:5]):
            category, _ = get_duration_category(gap['duracion_horas'])
            print(f"  {i+1}. {gap['modulo']}: {gap['duracion_horas']:.2f}h [{category}] ({gap['inicio']} - {gap['fin']})")
        
        return fig
    
    def plot_gap_frequency_analysis(self, max_hours=None, save_plot=True):
        """
        Crea gráficos de análisis de frecuencia de gaps por horario y día de la semana
        
        Args:
            max_hours (float): Máxima duración de gaps a incluir (None = todos)
            save_plot (bool): Si guardar el gráfico como archivo
        """
        if not self.gaps_info:
            print("Primero debe ejecutar detect_time_gaps()")
            return
        
        print(f"\nCreando análisis de frecuencia de gaps...")
        
        # Preparar datos
        gap_data = []
        for module, gaps in self.gaps_info.items():
            for gap in gaps:
                if max_hours is None or gap['duracion_horas'] <= max_hours:
                    gap_data.append({
                        'fecha_inicio': gap['inicio'],
                        'fecha_fin': gap['fin'],
                        'duracion_horas': gap['duracion_horas'],
                        'modulo': module,
                        'hora_inicio': gap['inicio'].hour if gap['inicio'] else None,
                        'hora_fin': gap['fin'].hour,
                        'dia_semana': gap['fin'].strftime('%A') if gap['fin'] else None,
                        'dia_semana_num': gap['fin'].weekday() if gap['fin'] else None
                    })
        
        if not gap_data:
            print("No hay gaps para mostrar con los criterios especificados")
            return
        
        # Crear subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Análisis de Frecuencia de Gaps Temporales', fontsize=16, fontweight='bold')
        
        # 1. Frecuencia por hora del día (histograma)
        ax1 = axes[0, 0]
        
        # Extraer horas de fin de gaps (cuando se detecta el gap)
        horas_fin = [gap['hora_fin'] for gap in gap_data if gap['hora_fin'] is not None]
        
        # Crear histograma
        bins = range(0, 25)  # 0-23 horas + 1 para el borde
        counts, _, patches = ax1.hist(horas_fin, bins=bins, alpha=0.7, color='skyblue', 
                                     edgecolor='black', linewidth=0.5)
        
        # Colorear barras según intensidad
        max_count = max(counts) if counts.size > 0 else 1
        for i, (count, patch) in enumerate(zip(counts, patches)):
            intensity = count / max_count
            patch.set_facecolor(plt.cm.Reds(0.3 + 0.7 * intensity))
        
        ax1.set_xlabel('Hora del Día', fontweight='bold')
        ax1.set_ylabel('Número de Gaps', fontweight='bold')
        ax1.set_title('Frecuencia de Gaps por Hora del Día')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores en las barras más altas
        for i, count in enumerate(counts):
            if count > max_count * 0.5:  # Solo mostrar en barras altas
                ax1.text(i + 0.5, count + max_count * 0.01, f'{int(count)}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Frecuencia por día de la semana
        ax2 = axes[0, 1]
        
        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        dias_num = [gap['dia_semana_num'] for gap in gap_data if gap['dia_semana_num'] is not None]
        
        # Contar gaps por día
        dia_counts = [dias_num.count(i) for i in range(7)]
        
        bars = ax2.bar(dias_semana, dia_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                                       '#9467bd', '#8c564b', '#e377c2'], 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Día de la Semana', fontweight='bold')
        ax2.set_ylabel('Número de Gaps', fontweight='bold')
        ax2.set_title('Frecuencia de Gaps por Día de la Semana')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for bar, count in zip(bars, dia_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dia_counts) * 0.01, 
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Heatmap: Hora vs Día de la semana
        ax3 = axes[1, 0]
        
        # Crear matriz de frecuencias
        frequency_matrix = np.zeros((7, 24))  # 7 días x 24 horas
        
        for gap in gap_data:
            if gap['dia_semana_num'] is not None and gap['hora_fin'] is not None:
                frequency_matrix[gap['dia_semana_num'], gap['hora_fin']] += 1
        
        # Crear heatmap
        im = ax3.imshow(frequency_matrix, cmap='Reds', aspect='auto', interpolation='nearest')
        
        # Configurar ejes
        ax3.set_xticks(range(0, 24, 2))
        ax3.set_xticklabels(range(0, 24, 2))
        ax3.set_yticks(range(7))
        ax3.set_yticklabels(dias_semana)
        ax3.set_xlabel('Hora del Día', fontweight='bold')
        ax3.set_ylabel('Día de la Semana', fontweight='bold')
        ax3.set_title('Heatmap: Gaps por Hora y Día')
        
        # Añadir colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Número de Gaps', rotation=270, labelpad=20)
        
        # Añadir valores en las celdas con más gaps
        max_freq = np.max(frequency_matrix)
        if max_freq > 0:
            for i in range(7):
                for j in range(24):
                    if frequency_matrix[i, j] > max_freq * 0.3:  # Solo mostrar valores altos
                        ax3.text(j, i, f'{int(frequency_matrix[i, j])}', 
                               ha='center', va='center', color='white', fontweight='bold', fontsize=8)
        
        # 4. Duración promedio de gaps por hora
        ax4 = axes[1, 1]
        
        # Calcular duración promedio por hora
        hour_durations = {}
        for gap in gap_data:
            if gap['hora_fin'] is not None:
                hora = gap['hora_fin']
                if hora not in hour_durations:
                    hour_durations[hora] = []
                hour_durations[hora].append(gap['duracion_horas'])
        
        horas_ordenadas = sorted(hour_durations.keys())
        duraciones_promedio = [np.mean(hour_durations[hora]) for hora in horas_ordenadas]
        duraciones_std = [np.std(hour_durations[hora]) for hora in horas_ordenadas]
        
        # Gráfico de líneas con barras de error
        ax4.errorbar(horas_ordenadas, duraciones_promedio, yerr=duraciones_std, 
                    marker='o', linewidth=2, markersize=6, capsize=5, 
                    color='darkred', alpha=0.8, label='Duración promedio ± std')
        
        ax4.set_xlabel('Hora del Día', fontweight='bold')
        ax4.set_ylabel('Duración Promedio (horas)', fontweight='bold')
        ax4.set_title('Duración Promedio de Gaps por Hora')
        ax4.set_xticks(range(0, 24, 2))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Resaltar horas con mayor duración promedio
        if duraciones_promedio:
            max_duration_idx = np.argmax(duraciones_promedio)
            max_hour = horas_ordenadas[max_duration_idx]
            max_duration = duraciones_promedio[max_duration_idx]
            
            ax4.annotate(f'Máx: {max_duration:.2f}h\\n({max_hour}:00)', 
                        xy=(max_hour, max_duration), 
                        xytext=(max_hour + 2, max_duration + max_duration * 0.1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        # Guardar el gráfico
        if save_plot:
            filename = f'gaps_frequency_analysis{"_filtered" if max_hours else ""}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico de frecuencias guardado como: {filename}")
        
        plt.show()
        
        # Análisis estadístico en consola
        self._print_frequency_analysis(gap_data, horas_fin, dias_num, hour_durations)
        
        return fig
    
    def _print_frequency_analysis(self, gap_data, horas_fin, dias_num, hour_durations):
        """Imprime análisis estadístico de frecuencias"""
        print(f"\n{'='*60}")
        print("ANÁLISIS DE FRECUENCIA DE GAPS")
        print(f"{'='*60}")
        
        total_gaps = len(gap_data)
        
        # Análisis por hora
        print(f"\n📊 ANÁLISIS POR HORA DEL DÍA:")
        if horas_fin:
            from collections import Counter
            hour_counter = Counter(horas_fin)
            
            print(f"  Horas con más gaps:")
            top_hours = hour_counter.most_common(5)
            for i, (hora, count) in enumerate(top_hours, 1):
                percentage = (count / total_gaps) * 100
                avg_duration = np.mean(hour_durations.get(hora, [0]))
                print(f"    {i}. {hora:02d}:00 - {count} gaps ({percentage:.1f}%) - Duración prom: {avg_duration:.2f}h")
        
        # Análisis por día de la semana
        print(f"\n📅 ANÁLISIS POR DÍA DE LA SEMANA:")
        if dias_num:
            from collections import Counter
            day_counter = Counter(dias_num)
            dias_nombres = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            
            print(f"  Días con más gaps:")
            for i, (dia_num, count) in enumerate(day_counter.most_common(), 1):
                percentage = (count / total_gaps) * 100
                dia_nombre = dias_nombres[dia_num]
                print(f"    {i}. {dia_nombre}: {count} gaps ({percentage:.1f}%)")
        
        # Patrones identificados
        print(f"\n🔍 PATRONES IDENTIFICADOS:")
        
        # Horario de mayor actividad
        if horas_fin:
            hour_counter = Counter(horas_fin)
            peak_hour = hour_counter.most_common(1)[0][0]
            peak_count = hour_counter.most_common(1)[0][1]
            print(f"  • Hora pico de gaps: {peak_hour:02d}:00 ({peak_count} gaps)")
        
        # Día de mayor actividad
        if dias_num:
            day_counter = Counter(dias_num)
            peak_day_num = day_counter.most_common(1)[0][0]
            peak_day_count = day_counter.most_common(1)[0][1]
            peak_day_name = dias_nombres[peak_day_num]
            print(f"  • Día pico de gaps: {peak_day_name} ({peak_day_count} gaps)")
        
        # Horarios de trabajo vs no trabajo
        work_hours = [h for h in horas_fin if 8 <= h <= 17]  # 8 AM - 5 PM
        non_work_hours = [h for h in horas_fin if h < 8 or h > 17]
        
        if work_hours or non_work_hours:
            work_percentage = (len(work_hours) / total_gaps) * 100
            non_work_percentage = (len(non_work_hours) / total_gaps) * 100
            print(f"  • Gaps en horario laboral (8-17h): {len(work_hours)} ({work_percentage:.1f}%)")
            print(f"  • Gaps fuera horario laboral: {len(non_work_hours)} ({non_work_percentage:.1f}%)")
        
        # Fin de semana vs días laborales
        weekend_gaps = [d for d in dias_num if d >= 5]  # Sábado (5) y Domingo (6)
        weekday_gaps = [d for d in dias_num if d < 5]
        
        if weekend_gaps or weekday_gaps:
            weekend_percentage = (len(weekend_gaps) / total_gaps) * 100
            weekday_percentage = (len(weekday_gaps) / total_gaps) * 100
            print(f"  • Gaps en días laborales: {len(weekday_gaps)} ({weekday_percentage:.1f}%)")
            print(f"  • Gaps en fin de semana: {len(weekend_gaps)} ({weekend_percentage:.1f}%)")
        
        print(f"\n{'='*60}")
    
    def analyze_data_completeness(self):
        """Analiza la completitud de los datos"""
        print("\nAnalizando completitud de datos...")
        
        completeness_stats = {}
        
        for module in self.data['module'].unique():
            module_data = self.data[self.data['module'] == module]
            
            total_records = len(module_data)
            
            # Contar valores faltantes por columna
            missing_stats = {}
            for col in ['pmax', 'imax', 'umax']:
                missing_count = module_data[col].isna().sum()
                zero_count = (module_data[col] == 0).sum()
                
                missing_stats[col] = {
                    'valores_faltantes': missing_count,
                    'valores_cero': zero_count,
                    'porcentaje_faltantes': (missing_count / total_records) * 100,
                    'porcentaje_cero': (zero_count / total_records) * 100
                }
            
            # Calcular período teórico vs real
            start_time = module_data['timestamp'].min()
            end_time = module_data['timestamp'].max()
            total_duration = end_time - start_time
            expected_records = int(total_duration.total_seconds() / (5 * 60)) + 1  # cada 5 minutos
            
            completeness_stats[module] = {
                'registros_reales': total_records,
                'registros_esperados': expected_records,
                'completitud_porcentaje': (total_records / expected_records) * 100 if expected_records > 0 else 0,
                'periodo_inicio': start_time,
                'periodo_fin': end_time,
                'duracion_total': total_duration,
                'missing_por_columna': missing_stats
            }
            
            print(f"  {module}:")
            print(f"    Registros: {total_records}/{expected_records} ({completeness_stats[module]['completitud_porcentaje']:.1f}%)")
            print(f"    Período: {start_time} - {end_time}")
        
        return completeness_stats
    
    def detect_anomalies(self):
        """Detecta anomalías en los datos"""
        print("\nDetectando anomalías...")
        
        anomalies = {}
        
        for module in self.data['module'].unique():
            module_data = self.data[self.data['module'] == module].copy()
            module_anomalies = {}
            
            for param in ['pmax', 'imax', 'umax']:
                param_data = module_data[param].dropna()
                
                if len(param_data) > 0:
                    # Estadísticas básicas
                    Q1 = param_data.quantile(0.25)
                    Q3 = param_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Límites para outliers
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Identificar outliers
                    outliers_mask = (param_data < lower_bound) | (param_data > upper_bound)
                    outliers = param_data[outliers_mask]
                    
                    module_anomalies[param] = {
                        'outliers_count': len(outliers),
                        'outliers_percentage': (len(outliers) / len(param_data)) * 100,
                        'min_value': param_data.min(),
                        'max_value': param_data.max(),
                        'mean_value': param_data.mean(),
                        'std_value': param_data.std(),
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
            
            anomalies[module] = module_anomalies
            
        return anomalies
    
    def create_visualizations(self):
        """Crea visualizaciones de la calidad de datos"""
        print("\nCreando visualizaciones...")
        
        # Configurar el estilo
        plt.style.use('default')
        
        # 1. Timeline de disponibilidad de datos
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Análisis de Calidad de Datos - PVStand', fontsize=16, fontweight='bold')
        
        # Timeline por módulo
        ax1 = axes[0, 0]
        modules = self.data['module'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, module in enumerate(modules):
            module_data = self.data[self.data['module'] == module]
            timestamps = module_data['timestamp']
            y_pos = [i] * len(timestamps)
            ax1.scatter(timestamps, y_pos, alpha=0.6, s=1, color=colors[i % len(colors)], label=module)
        
        ax1.set_yticks(range(len(modules)))
        ax1.set_yticklabels(modules)
        ax1.set_xlabel('Tiempo')
        ax1.set_ylabel('Módulo')
        ax1.set_title('Timeline de Mediciones por Módulo')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de valores por parámetro
        ax2 = axes[0, 1]
        param_data = []
        param_labels = []
        for param in ['pmax', 'imax', 'umax']:
            for module in modules:
                data_subset = self.data[self.data['module'] == module][param].dropna()
                if len(data_subset) > 0:
                    param_data.append(data_subset)
                    param_labels.append(f'{module}\n{param}')
        
        if param_data:
            ax2.boxplot(param_data, labels=param_labels)
            ax2.set_title('Distribución de Parámetros por Módulo')
            ax2.set_ylabel('Valor')
            plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # 3. Gaps temporales
        ax3 = axes[1, 0]
        gap_durations = []
        gap_modules = []
        
        for module, gaps in self.gaps_info.items():
            for gap in gaps:
                gap_durations.append(gap['duracion_horas'])
                gap_modules.append(module)
        
        if gap_durations:
            unique_modules = list(set(gap_modules))
            module_gaps = {mod: [] for mod in unique_modules}
            for i, module in enumerate(gap_modules):
                module_gaps[module].append(gap_durations[i])
            
            x_pos = []
            heights = []
            labels = []
            for i, (module, durations) in enumerate(module_gaps.items()):
                x_pos.extend([i] * len(durations))
                heights.extend(durations)
                if i == 0:
                    labels.extend([f'{module}'] * len(durations))
                else:
                    labels.extend([''] * len(durations))
            
            if heights:
                ax3.scatter(x_pos, heights, alpha=0.7)
                ax3.set_xticks(range(len(unique_modules)))
                ax3.set_xticklabels(unique_modules)
                ax3.set_ylabel('Duración del Gap (horas)')
                ax3.set_title('Duración de Gaps Temporales por Módulo')
                ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No se encontraron gaps temporales', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Gaps Temporales - Ninguno Detectado')
        
        # 4. Completitud de datos
        ax4 = axes[1, 1]
        completeness_stats = self.analyze_data_completeness()
        
        modules_comp = list(completeness_stats.keys())
        completeness_pct = [completeness_stats[mod]['completitud_porcentaje'] for mod in modules_comp]
        
        bars = ax4.bar(modules_comp, completeness_pct, color=['#2ca02c' if x >= 95 else '#ff7f0e' if x >= 80 else '#d62728' for x in completeness_pct])
        ax4.set_ylabel('Completitud (%)')
        ax4.set_title('Completitud de Datos por Módulo')
        ax4.set_ylim(0, 100)
        
        # Añadir valores en las barras
        for bar, pct in zip(bars, completeness_pct):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('pvstand_data_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report(self):
        """Genera un reporte resumen de la calidad de datos"""
        print("\n" + "="*80)
        print("REPORTE DE CALIDAD DE DATOS - PVSTAND")
        print("="*80)
        
        # Información general
        print(f"\n1. INFORMACIÓN GENERAL:")
        print(f"   Archivo: {self.filepath}")
        print(f"   Total de registros: {len(self.data)}")
        print(f"   Período de datos: {self.data['timestamp'].min()} a {self.data['timestamp'].max()}")
        print(f"   Módulos analizados: {', '.join(self.data['module'].unique())}")
        
        # Completitud
        print(f"\n2. COMPLETITUD DE DATOS:")
        completeness_stats = self.analyze_data_completeness()
        for module, stats in completeness_stats.items():
            print(f"   {module}:")
            print(f"     - Registros: {stats['registros_reales']}/{stats['registros_esperados']} ({stats['completitud_porcentaje']:.1f}%)")
            print(f"     - Duración: {stats['duracion_total']}")
        
        # Gaps temporales
        print(f"\n3. DISCONTINUIDADES TEMPORALES:")
        total_gaps = sum(len(gaps) for gaps in self.gaps_info.values())
        if total_gaps > 0:
            print(f"   Total de gaps encontrados: {total_gaps}")
            for module, gaps in self.gaps_info.items():
                if gaps:
                    print(f"   {module}: {len(gaps)} gaps")
                    for i, gap in enumerate(gaps[:5]):  # Mostrar solo los primeros 5
                        print(f"     - Gap {i+1}: {gap['duracion_horas']:.1f} horas ({gap['inicio']} - {gap['fin']})")
                    if len(gaps) > 5:
                        print(f"     - ... y {len(gaps)-5} gaps adicionales")
        else:
            print("   ✓ No se encontraron gaps temporales significativos")
        
        # Anomalías
        print(f"\n4. DETECCIÓN DE ANOMALÍAS:")
        anomalies = self.detect_anomalies()
        for module, module_anomalies in anomalies.items():
            print(f"   {module}:")
            for param, stats in module_anomalies.items():
                print(f"     {param}: {stats['outliers_count']} outliers ({stats['outliers_percentage']:.1f}%)")
                print(f"       Rango: {stats['min_value']:.3f} - {stats['max_value']:.3f} (media: {stats['mean_value']:.3f})")
        
        # Recomendaciones
        print(f"\n5. RECOMENDACIONES:")
        recommendations = []
        
        for module, stats in completeness_stats.items():
            if stats['completitud_porcentaje'] < 95:
                recommendations.append(f"   - {module}: Completitud baja ({stats['completitud_porcentaje']:.1f}%) - revisar sistema de adquisición")
        
        if total_gaps > 10:
            recommendations.append(f"   - Muchos gaps temporales ({total_gaps}) - verificar estabilidad del sistema de medición")
        
        for module, gaps in self.gaps_info.items():
            long_gaps = [g for g in gaps if g['duracion_horas'] > 24]
            if long_gaps:
                recommendations.append(f"   - {module}: {len(long_gaps)} gaps > 24 horas - revisar mantenimiento del equipo")
        
        if not recommendations:
            recommendations.append("   ✓ La calidad de los datos es buena en general")
        
        for rec in recommendations:
            print(rec)
        
        print("\n" + "="*80)
    
    def run_full_analysis(self, show_short_gaps=True, max_gap_hours=2, plot_gaps=True, plot_frequency=True):
        """Ejecuta el análisis completo"""
        self.load_data()
        self.detect_time_gaps()
        
        # Mostrar gaps cortos si se solicita
        if show_short_gaps:
            self.filter_gaps_by_duration(max_gap_hours)
        
        # Crear gráfico específico de gaps
        if plot_gaps:
            self.plot_gaps_timeline(max_gap_hours)
        
        # Crear análisis de frecuencias
        if plot_frequency:
            self.plot_gap_frequency_analysis(max_gap_hours)
        
        self.create_visualizations()
        self.generate_report()

# Función principal
def main():
    """Función principal para ejecutar el análisis"""
    # Ruta al archivo de datos
    filepath = "/home/nicole/SR/SOILING/datos/raw_pvstand_iv_data.csv"
    
    # Crear y ejecutar el analizador
    analyzer = PVStandDataQualityAnalyzer(filepath)
    analyzer.run_full_analysis()

def show_gaps_only(max_hours=2):
    """Función para mostrar solo los gaps filtrados"""
    filepath = "/home/nicole/SR/SOILING/datos/raw_pvstand_iv_data.csv"
    
    analyzer = PVStandDataQualityAnalyzer(filepath)
    analyzer.load_data()
    analyzer.detect_time_gaps()
    analyzer.filter_gaps_by_duration(max_hours)

def plot_gaps_only(max_hours=None):
    """Función para crear solo el gráfico de gaps"""
    filepath = "/home/nicole/SR/SOILING/datos/raw_pvstand_iv_data.csv"
    
    analyzer = PVStandDataQualityAnalyzer(filepath)
    analyzer.load_data()
    analyzer.detect_time_gaps()
    analyzer.plot_gaps_timeline(max_hours)

def plot_frequency_only(max_hours=None):
    """Función para crear solo el análisis de frecuencias"""
    filepath = "/home/nicole/SR/SOILING/datos/raw_pvstand_iv_data.csv"
    
    analyzer = PVStandDataQualityAnalyzer(filepath)
    analyzer.load_data()
    analyzer.detect_time_gaps()
    analyzer.plot_gap_frequency_analysis(max_hours)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "gaps":
            # Mostrar solo gaps filtrados
            max_hours = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
            show_gaps_only(max_hours)
            
        elif command == "plot":
            # Crear solo gráfico de gaps
            max_hours = float(sys.argv[2]) if len(sys.argv) > 2 else None
            plot_gaps_only(max_hours)
            
        elif command == "plot-filtered":
            # Crear gráfico de gaps filtrados
            max_hours = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
            plot_gaps_only(max_hours)
            
        elif command == "frequency":
            # Crear análisis de frecuencias
            max_hours = float(sys.argv[2]) if len(sys.argv) > 2 else None
            plot_frequency_only(max_hours)
            
        elif command == "frequency-filtered":
            # Crear análisis de frecuencias filtrado
            max_hours = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
            plot_frequency_only(max_hours)
            
        else:
            print("Comandos disponibles:")
            print("  python pvstand_dataqa.py                         # Análisis completo")
            print("  python pvstand_dataqa.py gaps [horas]            # Solo gaps ≤ X horas")
            print("  python pvstand_dataqa.py plot [horas]            # Solo gráfico de gaps")
            print("  python pvstand_dataqa.py plot-filtered [h]       # Gráfico gaps filtrados")
            print("  python pvstand_dataqa.py frequency [horas]       # Análisis de frecuencias")
            print("  python pvstand_dataqa.py frequency-filtered [h]  # Frecuencias filtradas")
    else:
        main()