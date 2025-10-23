# utils/solar_time.py

import pandas as pd
import pvlib
from pytz import timezone
from config.logging_config import logger
from config import settings # Importar settings para defaults

class UtilsMedioDiaSolar:
    def __init__(self, datei, datef, freq, inter, 
                 tz_local_str=None, lat=None, lon=None, alt=None): # Permitir None para usar defaults de settings
        """
        Inicializa UtilsMedioDiaSolar.
        Args:
            datei (str or datetime): Fecha de inicio del periodo.
            datef (str or datetime): Fecha de fin del periodo.
            freq (str): Frecuencia para el análisis (actualmente no usada directamente en msd).
            inter (int): Intervalo en minutos alrededor del mediodía solar para el filtro.
            tz_local_str (str, optional): String de la zona horaria local. Defaults a settings.DUSTIQ_LOCAL_TIMEZONE_STR.
            lat (float, optional): Latitud del sitio. Defaults a settings.SITE_LATITUDE.
            lon (float, optional): Longitud del sitio. Defaults a settings.SITE_LONGITUDE.
            alt (float, optional): Altitud del sitio en metros. Defaults a settings.SITE_ALTITUDE.
        """
        try:
            # Asegurar que son Timestamps antes de cualquier operación de zona horaria
            self.datei = pd.Timestamp(datei)
            self.datef = pd.Timestamp(datef)
        except Exception as e:
            logger.error(f"Error al convertir datei/datef a pd.Timestamp: {e}")
            raise
            
        self.freq = freq
        self.inter_minutes = inter
        
        # Usar valores de settings si los argumentos son None
        _tz_local_str = tz_local_str if tz_local_str is not None else settings.DUSTIQ_LOCAL_TIMEZONE_STR
        self.lat = lat if lat is not None else settings.SITE_LATITUDE
        self.lon = lon if lon is not None else settings.SITE_LONGITUDE
        self.alt = alt if alt is not None else settings.SITE_ALTITUDE

        try:
            self.tz_local = timezone(_tz_local_str)
        except Exception as e:
            logger.error(f"Zona horaria local inválida '{_tz_local_str}': {e}")
            logger.warning(f"Usando UTC como zona horaria local debido a error previo.")
            self.tz_local = timezone('UTC')

        logger.info(f"UtilsMedioDiaSolar inicializado: Lat={self.lat}, Lon={self.lon}, Alt={self.alt}, TZ={self.tz_local}")
        # Localizar datei y datef aquí para logging y para el rango
        try:
            # Asegurar que datei/datef son naive antes de localizar
            current_datei = self.datei.tz_localize(None) if self.datei.tzinfo is not None else self.datei
            current_datef = self.datef.tz_localize(None) if self.datef.tzinfo is not None else self.datef

            self.datei_local_for_range = current_datei.tz_localize(self.tz_local, nonexistent='shift_forward')
            self.datef_local_for_range = current_datef.tz_localize(self.tz_local, nonexistent='shift_forward')
            
            logger.info(f"Fechas de análisis (localizadas para rango): {self.datei_local_for_range} a {self.datef_local_for_range}. Intervalo de mediodía solar: +/- {self.inter_minutes/2} min.")
        except TypeError: # Si ya están localizadas (esto no debería ocurrir con el tz_localize(None) de arriba)
             logger.warning("TypeError al localizar fechas en __init__, intentando conversión directa.")
             self.datei_local_for_range = self.datei.tz_convert(self.tz_local)
             self.datef_local_for_range = self.datef.tz_convert(self.tz_local)
             logger.info(f"Fechas de análisis (convertidas para rango): {self.datei_local_for_range} a {self.datef_local_for_range}. Intervalo de mediodía solar: +/- {self.inter_minutes/2} min.")
        except Exception as e_localize_init:
            logger.error(f"Error localizando datei/datef en __init__: {e_localize_init}. El cálculo de msd puede fallar.")
            # Dejar datei/datef como naive, msd() intentará localizar de nuevo.
            self.datei_local_for_range = self.datei # Fallback a naive
            self.datef_local_for_range = self.datef # Fallback a naive


    def msd(self):
        """
        Calcula los intervalos de tiempo alrededor del mediodía solar para cada día en el rango.
        Returns:
            pd.DataFrame: DataFrame con columnas 0 y 1
                          conteniendo los datetimes naive en UTC del inicio y fin del intervalo.
        """
        logger.info("Iniciando cálculo de mediodías solares...")

        try:
            start_day_local = self.datei_local_for_range.normalize() 
            end_day_local = self.datef_local_for_range.normalize()   
        except Exception as e_norm:
            logger.error(f"Error al normalizar fechas pre-localizadas: {e_norm}. Intentando localizar de nuevo.")
            try:
                # Asegurar que son naive antes de localizar
                current_datei_msd = pd.Timestamp(self.datei).tz_localize(None) if pd.Timestamp(self.datei).tzinfo is not None else pd.Timestamp(self.datei)
                current_datef_msd = pd.Timestamp(self.datef).tz_localize(None) if pd.Timestamp(self.datef).tzinfo is not None else pd.Timestamp(self.datef)

                start_day_local = current_datei_msd.tz_localize(self.tz_local, nonexistent='shift_forward').normalize()
                end_day_local = current_datef_msd.tz_localize(self.tz_local, nonexistent='shift_forward').normalize()
            except Exception as e_fallback_loc:
                logger.error(f"Error crítico al localizar/normalizar fechas de inicio/fin en msd: {e_fallback_loc}")
                return pd.DataFrame(columns=[0, 1]) # Devolver con las nuevas columnas esperadas

        try:
            # Primary method: Generate naive daily timestamps, then localize the whole series.
            # This avoids pd.date_range having to deal with DST internally during generation.
            
            # Ensure start/end are converted to naive for pd.date_range
            # start_day_local and end_day_local are already localized to self.tz_local and normalized here
            s_naive = start_day_local.tz_localize(None) # Convert to naive 
            e_naive = end_day_local.tz_localize(None)   # Convert to naive
            
            # Generate naive date range
            naive_days_series = pd.date_range(start=s_naive, end=e_naive, freq='D')
            
            if naive_days_series.empty:
                logger.warning("El rango de fechas (naive) no generó días para procesar en UtilsMedioDiaSolar.msd().")
                # Localize even an empty series to maintain type if needed downstream, though an empty naive one is often fine.
                days_to_process_local = naive_days_series.tz_localize(self.tz_local, nonexistent='shift_forward', ambiguous='raise')
            else:
                # Localize the entire series
                days_to_process_local = naive_days_series.tz_localize(
                    self.tz_local, 
                    nonexistent='shift_forward', 
                    ambiguous='raise' # 'raise' is default for ambiguous, could also use 'infer' or 'NaT'
                )
            logger.info("UtilsMedioDiaSolar.msd(): generó rango de días naive y luego localizó la serie completa.")

        except Exception as e_drange_robust: # Catch any exception from this robust approach
            logger.error(f"Error crítico al generar y localizar el rango de días (método robusto): {e_drange_robust}")
            return pd.DataFrame(columns=[0, 1]) # Devolver con las nuevas columnas esperadas

        if days_to_process_local.empty:
            logger.warning("El rango de fechas no generó días para procesar en UtilsMedioDiaSolar.msd() (después de intento de localización)")
            return pd.DataFrame(columns=[0, 1]) # Devolver con las nuevas columnas esperadas

        logger.info(f"Procesando {len(days_to_process_local)} días desde {days_to_process_local.min()} hasta {days_to_process_local.max()}")

        solar_noon_utc_list = []

        for day_local_current in days_to_process_local:
            day_start_boundary_local = day_local_current
            day_end_boundary_local = day_local_current + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            times_for_solpos_local = pd.date_range(start=day_start_boundary_local, end=day_end_boundary_local, freq='1min', tz=self.tz_local)
            
            if times_for_solpos_local.empty:
                logger.warning(f"No se generaron timestamps para el día {day_local_current} en UtilsMedioDiaSolar.msd()")
                continue

            try:
                solpos = pvlib.solarposition.get_solarposition(
                    time=times_for_solpos_local, 
                    latitude=self.lat, 
                    longitude=self.lon, 
                    altitude=self.alt,
                    temperature=12, 
                    pressure=pvlib.atmosphere.alt2pres(self.alt)
                )
                
                solar_noon_local_dt = solpos['apparent_elevation'].idxmax()
                solar_noon_utc_dt = solar_noon_local_dt.tz_convert('UTC')
                solar_noon_utc_list.append(solar_noon_utc_dt)
            except Exception as e:
                logger.error(f"Error calculando mediodía solar para el día {day_local_current}: {e}")
                continue

        if not solar_noon_utc_list:
            logger.warning("No se pudieron calcular mediodías solares para ningún día.")
            return pd.DataFrame(columns=[0, 1]) # Devolver con las nuevas columnas esperadas

        df_solar_noons = pd.DataFrame({'SolarNoon_UTC': solar_noon_utc_list})

        interval_delta = pd.Timedelta(minutes=self.inter_minutes / 2)
        df_solar_noons['SolarNoon_Time_i_utc_aware'] = df_solar_noons['SolarNoon_UTC'] - interval_delta
        df_solar_noons['SolarNoon_Time_f_utc_aware'] = df_solar_noons['SolarNoon_UTC'] + interval_delta
        
        # Devolver columnas como 0 y 1, con datetimes naive UTC
        result_df = pd.DataFrame({
            0: df_solar_noons['SolarNoon_Time_i_utc_aware'].dt.tz_localize(None),
            1: df_solar_noons['SolarNoon_Time_f_utc_aware'].dt.tz_localize(None)
        })
        
        logger.info(f"Se calcularon {len(result_df)} intervalos de mediodía solar.")
        if not result_df.empty:
            log_message = f"Primeros 3 intervalos de mediodía solar (UTC naive, columnas 0 y 1):\n"
            logger.debug(log_message + str(result_df.head(3)))
        return result_df

# Ejemplo de uso (opcional, para prueba directa del módulo)
if __name__ == '__main__':
    # from config.settings import DUSTIQ_LOCAL_TIMEZONE_STR # Ya se importa settings arriba
    # Configurar logging básico si se ejecuta directamente
    import logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Probando UtilsMedioDiaSolar directamente...")

    # Parámetros de ejemplo
    # tz_example = settings.DUSTIQ_LOCAL_TIMEZONE_STR # Usa el default de settings ahora
    # lat_example = settings.SITE_LATITUDE
    # lon_example = settings.SITE_LONGITUDE
    # alt_example = settings.SITE_ALTITUDE

    start_date_example = "2024-07-15"
    end_date_example = "2024-07-20"
    interval_minutes_example = 120 # +/- 60 minutos

    # El constructor ahora toma los valores de settings por defecto si no se pasan
    mds_test = UtilsMedioDiaSolar(
        datei=start_date_example,
        datef=end_date_example,
        freq='1min', 
        inter=interval_minutes_example
        # tz_local_str, lat, lon, alt se tomarán de settings si no se especifican
    )
    
    df_intervals = mds_test.msd()
    
    if not df_intervals.empty:
        print("\nIntervalos de mediodía solar calculados (UTC naive, cols 0 y 1):")
        print(df_intervals)
        print(f"\nInformación del DataFrame:")
        df_intervals.info()
    else:
        print("\nNo se generaron intervalos de mediodía solar.")

    # Prueba con fechas que podrían cruzar DST o tener problemas
    start_date_example_dst = "2024-09-07" # Chile cambia hora el 2024-09-08 (medianoche del sábado al domingo)
    end_date_example_dst = "2024-09-10"
    logger.info(f"Probando con rango de fechas que incluye cambio de hora DST (adelanto): {start_date_example_dst} a {end_date_example_dst}")
    mds_test_dst_adelanto = UtilsMedioDiaSolar(
        datei=start_date_example_dst,
        datef=end_date_example_dst,
        freq='1min',
        inter=interval_minutes_example
    )
    df_intervals_dst_adelanto = mds_test_dst_adelanto.msd()
    if not df_intervals_dst_adelanto.empty:
        print("\nIntervalos de mediodía solar (cruzando DST - adelanto):")
        print(df_intervals_dst_adelanto)
    else:
        print("\nNo se generaron intervalos para prueba DST (adelanto).")

    start_date_example_dst_atraso = "2024-04-05" # Chile cambió hora el 2024-04-07 (atraso)
    end_date_example_dst_atraso = "2024-04-09"
    logger.info(f"Probando con rango de fechas que incluye cambio de hora DST (atraso): {start_date_example_dst_atraso} a {end_date_example_dst_atraso}")
    mds_test_dst_atraso = UtilsMedioDiaSolar(
        datei=start_date_example_dst_atraso,
        datef=end_date_example_dst_atraso,
        freq='1min',
        inter=interval_minutes_example
    )
    df_intervals_dst_atraso = mds_test_dst_atraso.msd()
    if not df_intervals_dst_atraso.empty:
        print("\nIntervalos de mediodía solar (cruzando DST - atraso):")
        print(df_intervals_dst_atraso)
    else:
        print("\nNo se generaron intervalos para prueba DST (atraso).") 