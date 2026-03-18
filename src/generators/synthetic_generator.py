"""
Synthetic weather data generator
Generates synthetic climate data by cycling through historical daily data,
adjusting to monthly predictions, and producing hourly output.

Flow:
1. Load historical daily data from DB
2. Cycle daily records to fill the generation period
3. Adjust daily records to match monthly predictions (if provided)
4. Generate hourly data from adjusted daily records
5. Verify hourly data match daily records
6. Save everything to DB
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from math import cos, sin, pi, exp
import numpy as np
import pandas as pd

from database.sqliteDB import (
    get_historical_daily_data, 
    insert_generation_jobs,
    insert_monthly_predictions,
    insert_generated_daily_data,
    insert_generated_hourly_data
)
#from generators.mbc_correction import MBCnCorrector
from generators.k_neighbors import KNeighborsCorrector
from generators.xgboost_model import XGBoostWeatherModel

class SyntheticWeatherGenerator:
    """
    Generador de datos meteorológicos sintéticos.
    
    Cicla datos históricos diarios para el período de generación,
    los ajusta a predicciones mensuales (si se proporcionan),
    y genera datos horarios interpolados.
    """
    
    def __init__(self, source: str, id_station, historical_start: str, 
                 historical_end: str, generation_start: str, generation_end: str):
        """
        Inicializa el generador.
        
        Args:
            source: Fuente de datos (ej: 'AEMET')
            id_station: ID de la estación
            historical_start: Fecha inicio históricos (YYYY-MM-DD)
            historical_end: Fecha fin históricos (YYYY-MM-DD)
            generation_start: Fecha inicio generación (YYYY-MM-DD)
            generation_end: Fecha fin generación (YYYY-MM-DD)
        """
        self.source = source
        self.id_station = id_station
        self.historical_start = historical_start
        self.historical_end = historical_end
        self.generation_start = generation_start
        self.generation_end = generation_end
        
        # Cargar datos históricos DIARIOS
        self.historical_data = get_historical_daily_data(
            source=source,
            id_station=id_station,
            start_date=historical_start,
            end_date=historical_end
        )
        
        if not self.historical_data:
            raise ValueError(
                f"No historical data found for station {id_station} "
                f"in period {historical_start} to {historical_end}"
            )
        
        # Ordenar por fecha
        self.historical_data.sort(key=lambda x: x['date'])
        
        print(f"📊 Datos históricos diarios cargados: {len(self.historical_data)} registros")
    
    # ========================================================================
    # STEP 1: Cycle daily records
    # ========================================================================
    
    def _generate_daily_synthetic(self) -> List[Dict]:
        """
        Genera datos diarios sintéticos ciclando los históricos por mes-día.
        
        Para cada fecha a generar, busca el mismo mes-día en los años históricos
        y cicla entre los años disponibles.
        
        Returns:
            Lista de diccionarios con datos diarios sintéticos
        """
        generated_daily = []
        
        gen_start = datetime.strptime(self.generation_start, '%Y-%m-%d')
        gen_end = datetime.strptime(self.generation_end, '%Y-%m-%d')
        
        # Indexar históricos por mes-día y año
        records_by_month_day = {}
        for record in self.historical_data:
            record_date = datetime.strptime(record['date'], '%Y-%m-%d')
            month_day_key = f"{record_date.month:02d}-{record_date.day:02d}"
            year = record_date.year
            
            if month_day_key not in records_by_month_day:
                records_by_month_day[month_day_key] = {}
            records_by_month_day[month_day_key][year] = record
        
        current_date = gen_start
        month_day_occurrence = {}
        
        while current_date <= gen_end:
            month_day_key = f"{current_date.month:02d}-{current_date.day:02d}"
            
            if month_day_key not in month_day_occurrence:
                month_day_occurrence[month_day_key] = 0
            month_day_occurrence[month_day_key] += 1
            occurrence_count = month_day_occurrence[month_day_key]
            
            source_record = None
            
            if month_day_key in records_by_month_day:
                available_years = sorted(records_by_month_day[month_day_key].keys())
                if available_years:
                    # TODO dejar aleatorio
                    #year_index = random.randint(0, len(available_years) - 1)
                    # CICLO: usa modulo para recorrer los años disponibles de forma determinista
                    # Sin esto (con random.randint()), los datos se distribuían aleatoriamente,
                    # ahora se cicla por año histórico para facilitar debugging
                    year_index = (occurrence_count - 1) % len(available_years)
                    selected_year = available_years[year_index]
                    source_record = records_by_month_day[month_day_key][selected_year]
            
            # Caso especial: 29 de febrero
            if source_record is None and current_date.month == 2 and current_date.day == 29:
                alt_key = "02-28"
                if alt_key in records_by_month_day:
                    available_years = sorted(records_by_month_day[alt_key].keys())
                    # TODO dejar aleatorio
                    #year_index = random.randint(0, len(available_years) - 1)
                    # CICLO: usa modulo para recorrer los años disponibles de forma determinista
                    # Sin esto (con random.randint()), los datos se distribuían aleatoriamente,
                    # ahora se cicla por año histórico para facilitar debugging
                    year_index = (occurrence_count - 1) % len(available_years)
                    selected_year = available_years[year_index]
                    source_record = records_by_month_day[alt_key][selected_year]
            
            # Fallback: usar ciclo sobre todos los datos históricos
            if source_record is None:
                # TODO dejar aleatorio
                #idx = random.randint(0, len(self.historical_data) - 1)
                # CICLO: en lugar de random.randint(), usa modulo para ciclar determinísticamente
                idx = (occurrence_count - 1) % len(self.historical_data)
                source_record = self.historical_data[idx]
            
            # Copiar registro con nueva fecha
            new_record = source_record.copy()
            new_record['date'] = current_date.strftime('%Y-%m-%d')
            generated_daily.append(new_record)
            
            current_date += timedelta(days=1)
        
        # 📊 Registros diarios sintéticos generados por ciclo histórico
        print(f"📅 Ciclo completado: {len(generated_daily)} registros diarios generados")
        return generated_daily
    
    # ========================================================================
    # STEP 2: Adjust daily records to monthly predictions
    # ========================================================================
    
    def _adjust_to_monthly_predictions(self, daily_data: List[Dict], 
                                       predictions_df: pd.DataFrame) -> List[Dict]:
        """
        Ajusta los datos diarios históricos para que coincidan con las predicciones mensuales.
        
        Para temperaturas:
        - Calcula estadísticas mensuales de los históricos
        - Aplica diferencias a cada día del mes
        - Verifica límites individuales (min/max) para cada temperatura
        - Compensa en extremo opuesto cuando se exceden límites
        - Fuerza orden: Tmax >= Tmean >= Tmin
        
        Para precipitación:
        - Aplica factor multiplicativo
        - Si predicción=0, pone todos los días a 0
        - Si histórico=0 pero predicción>0, genera 2 períodos de lluvia
        
        Args:
            daily_data: Lista de registros diarios
            predictions_df: DataFrame con predicciones mensuales (formato long)
            
        Returns:
            Lista de registros diarios ajustados
        """
        adjusted_data = [record.copy() for record in daily_data]
        
        # Crear índice de predicciones para búsqueda O(1): (year, month, variable) -> predicción
        predictions_index = {}
        for _, row in predictions_df.iterrows():
            year = int(row['Year'])
            month = int(row['Month'])
            variable = row['Variable']
            key = (year, month, variable)
            predictions_index[key] = {
                'mean': row.get('Mean'),
                'min': row.get('Minimum'),
                'max': row.get('Maximum')
            }
        
        # Agrupar registros diarios por año-mes
        daily_by_month = {}
        for i, record in enumerate(adjusted_data):
            date = datetime.strptime(record['date'], '%Y-%m-%d')
            month_key = (date.year, date.month)
            if month_key not in daily_by_month:
                daily_by_month[month_key] = []
            daily_by_month[month_key].append(i)
        
        # Procesar cada mes
        for (year, month), indices in daily_by_month.items():
            # ============================================================
            # TEMPERATURAS - Lógica mejorada con verificación individual
            # ============================================================
            self._adjust_temperatures_for_month(adjusted_data, indices, predictions_index, year, month)
            
            # ============================================================
            # PRECIPITACIÓN
            # ============================================================
            self._adjust_precipitation_for_month(adjusted_data, indices, predictions_index, year, month)
        
        return adjusted_data
    
    def _get_variable_stats(self, pred_df: pd.DataFrame, variable_name: str) -> Dict:
        """
        Extrae estadísticas (min, mean, max) de una variable del DataFrame en formato long.
        
        Args:
            pred_df: DataFrame filtrado por mes (Year, Month, Variable, Minimum, Mean, Maximum)
            variable_name: Nombre de la variable a buscar (ej: 'temperature_max', 'precipitation')
            
        Returns:
            Dict con 'min', 'mean', 'max' o None si no se encuentra
        """
        var_row = pred_df[pred_df['Variable'] == variable_name]
        if var_row.empty:
            return None
        
        var_row = var_row.iloc[0]
        return {
            'min': var_row.get('Minimum'),
            'mean': var_row.get('Mean'),
            'max': var_row.get('Maximum')
        }
    
    def _adjust_temperatures_for_month(self, daily_data: List[Dict], 
                                          indices: List[int], 
                                          predictions_index: Dict,
                                          year: int, month: int):
        """
        Ajuste de temperaturas con verificación individual de límites.
        
        Args:
            daily_data: Lista completa de registros diarios (se modifica in-place)
            indices: Índices de los días de este mes
            predictions_index: Diccionario (year, month, variable) -> {mean, min, max}
            year: Año del mes
            month: Mes (1-12)
        """
        # Obtener predicciones para este mes
        tmax_pred = predictions_index.get((year, month, 'temperature_max'))
        tmean_pred = predictions_index.get((year, month, 'temperature_mean'))
        tmin_pred = predictions_index.get((year, month, 'temperature_min'))
        
        if not all([tmax_pred, tmean_pred, tmin_pred]):
            return
        
        # Validar que tengan la media
        if any(pd.isna(pred.get('mean')) for pred in [tmax_pred, tmean_pred, tmin_pred]):
            return
        
        # Calcular estadísticas mensuales de los históricos actuales
        tmax_values = [daily_data[i].get('temperature_max') for i in indices if daily_data[i].get('temperature_max') is not None]
        tmean_values = [daily_data[i].get('temperature_mean') for i in indices if daily_data[i].get('temperature_mean') is not None]
        tmin_values = [daily_data[i].get('temperature_min') for i in indices if daily_data[i].get('temperature_min') is not None]
        
        if not all([tmax_values, tmean_values, tmin_values]):
            return
        
        hist_tmax_mean = np.mean(tmax_values)
        hist_tmean_mean = np.mean(tmean_values)
        hist_tmin_mean = np.mean(tmin_values)
        
        # Calcular diferencias (predicción - histórico)
        diff_tmax = tmax_pred['mean'] - hist_tmax_mean
        diff_tmean = tmean_pred['mean'] - hist_tmean_mean
        diff_tmin = tmin_pred['mean'] - hist_tmin_mean
        
        # Extraer límites de predicción
        tmax_min = float(tmax_pred['min']) if pd.notna(tmax_pred.get('min')) else None
        tmax_max = float(tmax_pred['max']) if pd.notna(tmax_pred.get('max')) else None
        tmean_min = float(tmean_pred['min']) if pd.notna(tmean_pred.get('min')) else None
        tmean_max = float(tmean_pred['max']) if pd.notna(tmean_pred.get('max')) else None
        tmin_min = float(tmin_pred['min']) if pd.notna(tmin_pred.get('min')) else None
        tmin_max = float(tmin_pred['max']) if pd.notna(tmin_pred.get('max')) else None
        
        # Ajustar cada día individualmente
        for idx in indices:
            record = daily_data[idx]
            
            tmax = record.get('temperature_max')
            tmean = record.get('temperature_mean')
            tmin = record.get('temperature_min')
            
            if tmax is None or tmean is None or tmin is None:
                continue
            
            # Aplicar diferencias
            new_tmax = tmax + diff_tmax
            new_tmean = tmean + diff_tmean
            new_tmin = tmin + diff_tmin
            
            # Ajustar Tmax a sus límites
            # TODO Eliminar comentado
            if tmax_max is not None and new_tmax > tmax_max:
                #excess = new_tmax - tmax_max
                new_tmax = tmax_max
                #new_tmin -= excess
                #if tmin_min is not None and new_tmin < tmin_min:
                #    new_tmin = tmin_min
            
            if tmax_min is not None and new_tmax < tmax_min:
                #deficit = tmax_min - new_tmax
                new_tmax = tmax_min
                #new_tmin += deficit
                #if tmin_max is not None and new_tmin > tmin_max:
                #    new_tmin = tmin_max
            
            # Ajustar Tmin a sus límites
            if tmin_max is not None and new_tmin > tmin_max:
                #excess = new_tmin - tmin_max
                new_tmin = tmin_max
                #new_tmax += excess
                #if tmax_max is not None and new_tmax > tmax_max:
                #    new_tmax = tmax_max
            
            if tmin_min is not None and new_tmin < tmin_min:
                #deficit = tmin_min - new_tmin
                new_tmin = tmin_min
                #new_tmax -= deficit
                #if tmax_min is not None and new_tmax < tmax_min:
                #    new_tmax = tmax_min
            
            # Ajustar Tmean a sus límites
            if tmean_max is not None and new_tmean > tmean_max:
                new_tmean = tmean_max
            if tmean_min is not None and new_tmean < tmean_min:
                new_tmean = tmean_min
            
            # Forzar orden: Tmax >= Tmean >= Tmin
            if new_tmean > new_tmax:
                avg = (new_tmean + new_tmax) / 2.0
                new_tmean = avg
                new_tmax = avg
            
            if new_tmean < new_tmin:
                avg = (new_tmean + new_tmin) / 2.0
                new_tmean = avg
                new_tmin = avg
            
            if new_tmax < new_tmin:
                avg = (new_tmax + new_tmin) / 2.0
                new_tmax = avg
                new_tmin = avg
            
            # Guardar valores ajustados con 1 decimal
            record['temperature_max'] = round(new_tmax, 1)
            record['temperature_mean'] = round(new_tmean, 1)
            record['temperature_min'] = round(new_tmin, 1)
    
    def _adjust_precipitation_for_month(self, daily_data: List[Dict],
                                           indices: List[int],
                                           predictions_index: Dict,
                                           year: int, month: int):
        """
        Dispatcher de ajuste de precipitación para un mes específico.

        Si hay predicción de 'number_days_rain', usa la ruta avanzada (Fases 0-4).
        En caso contrario, usa la ruta simple (factor multiplicativo).

        Args:
            daily_data: Lista completa de registros diarios (se modifica in-place)
            indices: Índices de los días de este mes
            predictions_index: Diccionario (year, month, variable) -> {mean, min, max}
            year: Año del mes
            month: Mes (1-12)
        """
        prec_pred = predictions_index.get((year, month, 'precipitation'))
        if not prec_pred:
            return

        pred_prec_mean = prec_pred.get('mean')
        if pred_prec_mean is None or pd.isna(pred_prec_mean):
            return
        pred_prec_mean = float(pred_prec_mean)

        days_rain_pred = predictions_index.get((year, month, 'number_days_rain'))
        if days_rain_pred is not None:
            pred_days_mean = days_rain_pred.get('mean')
            if pred_days_mean is not None and not pd.isna(pred_days_mean):
                self._adjust_precipitation_advanced(
                    daily_data, indices, pred_prec_mean, prec_pred, days_rain_pred, year, month
                )
                return

        self._adjust_precipitation_simple(daily_data, indices, pred_prec_mean, prec_pred)

    # ---- ruta simple (solo precipitación) ----------------------------------

    def _adjust_precipitation_simple(self, daily_data: List[Dict],
                                     indices: List[int],
                                     pred_prec_mean: float,
                                     prec_pred: Dict):
        """
        Ajuste simple de precipitación mediante factor multiplicativo.

        Casos:
        - pred == 0  → todos los días a 0
        - hist == 0  → generar 2 períodos de lluvia con valor mínimo de predicción
        - else       → factor multiplicativo
        """
        prec_values = [daily_data[i].get('precipitation', 0) or 0 for i in indices]
        hist_prec_mean = np.mean(prec_values)
        
        # CASO 1: Predicción es 0 -> todos los días a 0
        if pred_prec_mean == 0:
            for idx in indices:
                daily_data[idx]['precipitation'] = 0.0
            return
        
        # CASO 2: Histórico es 0 pero predicción > 0 -> generar períodos de lluvia
        # En este caso, usar el MÍNIMO de la predicción en lugar de la media
        if hist_prec_mean == 0:
            pred_prec_min = prec_pred.get('min')
            
            # Si no hay mínimo definido, usar la media como fallback
            if pred_prec_min is None or pd.isna(pred_prec_min):
                target_prec = pred_prec_mean
                target_label = "media"
            else:
                target_prec = float(pred_prec_min)
                target_label = "mínimo"

            first_date = datetime.strptime(daily_data[indices[0]]['date'], '%Y-%m-%d')
            month_names = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                           'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
            print(f"🌧️ Generando 2 períodos de lluvia para "
                  f"{month_names[first_date.month - 1]} {first_date.year} "
                  f"({target_label}: {target_prec:.1f} mm/día)")
            self._generate_rain_periods(daily_data, indices, target_prec)
            return
        
        # CASO 3: Aplicar factor multiplicativo
        factor = pred_prec_mean / hist_prec_mean
        for idx in indices:
            current_prec = daily_data[idx].get('precipitation', 0) or 0
            daily_data[idx]['precipitation'] = round(current_prec * factor, 1)

    # ---- ruta avanzada (precipitación + número de días) --------------------

    def _adjust_precipitation_advanced(self, daily_data: List[Dict],
                                       indices: List[int],
                                       pred_prec_mean: float,
                                       prec_pred: Dict,
                                       days_rain_pred: Dict,
                                       year: int, month: int):
        """
        Ajuste avanzado de precipitación en 4 fases:
          0. Caso trivial: pred==0 o días==0 → todo a 0
          1. Escanear estado actual del mes
          2. Validar vs predicciones (déficit / exceso de días)
          3. Ajustar número de días lluviosos (añadir o eliminar)
          4. Factor multiplicativo para alcanzar la precipitación objetivo exacta
        """
        first_date = datetime.strptime(daily_data[indices[0]]['date'], '%Y-%m-%d')
        month_names = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                       'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
        period_label = f"{month_names[first_date.month - 1]} {first_date.year}"
        
        # Fase 0: casos triviales
        pred_days_mean = float(days_rain_pred.get('mean'))
        #print(f"\n🌧️ === AJUSTE AVANZADO: {period_label} ===")
        #print(f"   Predicción: {pred_prec_mean:.1f} mm/día, {pred_days_mean:.1f} días lluvia")
        
        if pred_prec_mean == 0 or pred_days_mean == 0:
            #print(f"   ⚠️  FASE 0: Predicción nula → todos los días a 0")
            for idx in indices:
                daily_data[idx]['precipitation'] = 0.0
            return

        # Fase 1: escanear mes
        #print(f"\n   📊 FASE 1: Escanear precipitación actual...")
        analysis = self._scan_month_rainfall(daily_data, indices)

        # Si no hay lluvia histórica, generar períodos base antes de ajustar días
        if analysis['rainy_days'] == 0:
            #print(f"      ⚠️  Sin lluvia histórica → generando períodos base...")
            self._generate_rain_periods(daily_data, indices, pred_prec_mean)
            analysis = self._scan_month_rainfall(daily_data, indices)

        # Fase 2: validar vs predicción
        #print(f"\n   ✓ FASE 2: Validar número de días...")
        validation = self._validate_month_rainfall(analysis, days_rain_pred)
        days_status = validation['days_status']

        # Fase 3: ajustar número de días lluviosos
        #print(f"\n   ⚙️  FASE 3: Ajustar días lluviosos...")
        if days_status['deficit'] > 0:
            #print(f"      Faltan {days_status['deficit']} día(s) de lluvia → AÑADIR")
            self._add_rainy_days(daily_data, indices, analysis, days_status['deficit'])
        elif days_status['excess'] > 0:
            #print(f"      Sobran {days_status['excess']} día(s) de lluvia → ELIMINAR")
            self._remove_rainy_days(daily_data, indices, analysis, days_status['excess'])
        #else:
            #print(f"      OK - Número de días correcto")

        # Fase 4: factor multiplicativo con matriz combinada días × precipitación
        print(f"\n   📐 FASE 4: Factor multiplicativo (matriz combinada)...")
        days_in_month = len(indices)

        # Estado FINAL de días tras Fase 3
        current_days = sum(1 for idx in indices if (daily_data[idx].get('precipitation', 0) or 0) > 0)
        pred_days_min = float(days_status['min']) if days_status['min'] is not None else pred_days_mean
        pred_days_max = float(days_status['max']) if days_status['max'] is not None else pred_days_mean

        if current_days <= pred_days_min:
            days_state = "EN_MINIMO"
        elif current_days >= pred_days_max:
            days_state = "EN_MAXIMO"
        else:
            days_state = "EN_RANGO"

        # Totales de precipitación objetivo
        pred_prec_min = prec_pred.get('min')
        pred_prec_max = prec_pred.get('max')
        pred_prec_min = float(pred_prec_min) if pred_prec_min is not None and not pd.isna(pred_prec_min) else pred_prec_mean
        pred_prec_max = float(pred_prec_max) if pred_prec_max is not None and not pd.isna(pred_prec_max) else pred_prec_mean

        min_total  = pred_prec_min  * days_in_month
        max_total  = pred_prec_max  * days_in_month
        mean_total = pred_prec_mean * days_in_month

        actual_total = sum(daily_data[idx].get('precipitation', 0) or 0 for idx in indices)

        if actual_total < min_total:
            precip_state = "BAJO_MIN"
        elif actual_total > max_total:
            precip_state = "SOBRE_MAX"
        else:
            precip_state = "EN_RANGO"

        #print(f"      Estado días:   {days_state} ({current_days} días, rango [{pred_days_min:.0f}–{pred_days_max:.0f}])")
        #print(f"      Estado precip: {precip_state} ({actual_total:.1f} mm, rango [{min_total:.1f}–{max_total:.1f}], media {mean_total:.1f})")

        # Matriz combinada → target_total
        if days_state == "EN_MINIMO":
            if precip_state == "BAJO_MIN":
                target_total = min_total
            elif precip_state == "SOBRE_MAX":
                target_total = mean_total
            else:  # EN_RANGO
                target_total = actual_total
        elif days_state == "EN_MAXIMO":
            if precip_state == "BAJO_MIN":
                target_total = mean_total
            elif precip_state == "SOBRE_MAX":
                target_total = max_total
            else:  # EN_RANGO
                target_total = actual_total
        else:  # EN_RANGO
            if precip_state == "BAJO_MIN":
                target_total = min_total
            elif precip_state == "SOBRE_MAX":
                target_total = max_total
            else:  # EN_RANGO
                target_total = actual_total

        #print(f"      Objetivo total: {target_total:.1f} mm, Actual: {actual_total:.1f} mm")

        if actual_total == 0 and target_total > 0:
            #print(f"      ⚠️  Sin lluvia tras eliminar días → generando períodos de emergencia")
            self._generate_rain_periods(daily_data, indices, pred_prec_mean)
        elif actual_total > 0 and abs(target_total - actual_total) > 0.1:
            factor = target_total / actual_total
            #print(f"      Factor: {factor:.3f}")
            for idx in indices:
                current_prec = daily_data[idx].get('precipitation', 0) or 0
                if current_prec > 0:
                    daily_data[idx]['precipitation'] = round(current_prec * factor, 1)
        #else:
            #print(f"      Sin cambios necesarios")
        
        final_total = sum(daily_data[idx].get('precipitation', 0) or 0 for idx in indices)
        final_rainy_days = sum(1 for idx in indices if daily_data[idx].get('precipitation', 0) or 0 > 0)
        #print(f"\n   ✅ RESULTADO FINAL: {final_total:.1f} mm total, {final_rainy_days} días lluvia")
        #print(f"   {'='*50}\n")

    # ---- helpers de escaneo y validación -----------------------------------

    def _scan_month_rainfall(self, daily_data: List[Dict],
                             indices: List[int]) -> Dict:
        """
        Fase 1: escanea el estado actual de precipitación del mes.

        Returns:
            Dict con:
              total_precipitation, rainy_days, days_in_month, mean_precipitation,
              rain_series (lista de {start_pos, end_pos, duration, total_precip, daily_values}),
              precip_by_day (lista de valores por posición en indices)
        """
        precip_by_day = [daily_data[idx].get('precipitation', 0) or 0 for idx in indices]
        n = len(precip_by_day)
        rainy_days = sum(1 for p in precip_by_day if p > 0)
        total_precipitation = sum(precip_by_day)
        mean_precipitation = total_precipitation / n if n > 0 else 0.0

        rain_series = []
        i = 0
        while i < n:
            if precip_by_day[i] > 0:
                start_pos = i
                while i < n and precip_by_day[i] > 0:
                    i += 1
                end_pos = i - 1
                daily_values = precip_by_day[start_pos:end_pos + 1]
                rain_series.append({
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'duration': end_pos - start_pos + 1,
                    'total_precip': sum(daily_values),
                    'daily_values': list(daily_values),
                })
            else:
                i += 1

        #print(f"      Total: {total_precipitation:.1f} mm, Días: {rainy_days}, "
        #      f"Series: {len(rain_series)}")
        #for j, s in enumerate(rain_series, 1):
            #print(f"         Serie {j}: posiciones {s['start_pos']}-{s['end_pos']}, "
            #      f"duración {s['duration']}, total {s['total_precip']:.1f} mm")

        return {
            'total_precipitation': total_precipitation,
            'rainy_days': rainy_days,
            'days_in_month': n,
            'mean_precipitation': mean_precipitation,
            'rain_series': rain_series,
            'precip_by_day': precip_by_day,
        }

    def _validate_month_rainfall(self, analysis: Dict, days_rain_pred: Dict) -> Dict:
        """
        Fase 2: compara el estado actual con la predicción de días de lluvia.

        Returns:
            Dict con days_status: {current, min, max, mean, deficit, excess,
                                   margin_up, margin_down}
        """
        current_days = analysis['rainy_days']
        days_in_month = analysis['days_in_month']

        raw_mean = days_rain_pred.get('mean')
        raw_min  = days_rain_pred.get('min')
        raw_max  = days_rain_pred.get('max')

        pred_mean = int(round(float(raw_mean))) if raw_mean is not None and not pd.isna(raw_mean) else current_days
        pred_min  = int(round(float(raw_min)))  if raw_min  is not None and not pd.isna(raw_min)  else pred_mean
        pred_max  = int(round(float(raw_max)))  if raw_max  is not None and not pd.isna(raw_max)  else pred_mean

        pred_min  = max(0, min(pred_min,  days_in_month))
        pred_max  = max(0, min(pred_max,  days_in_month))
        pred_mean = max(0, min(pred_mean, days_in_month))

        deficit = max(0, pred_min - current_days)
        excess  = max(0, current_days - pred_max)

        #print(f"      Predicción: min={pred_min}, mean={pred_mean}, max={pred_max}")
        #print(f"      Actual: {current_days} días")
        #if deficit > 0:
        #    print(f"      → DÉFICIT: faltan {deficit} día(s)")
        #elif excess > 0:
        #    print(f"      → EXCESO: sobran {excess} día(s)")
        #else:
        #    print(f"      → Dentro del rango [min={pred_min}, max={pred_max}]")

        return {
            'days_status': {
                'current':     current_days,
                'min':         pred_min,
                'max':         pred_max,
                'mean':        pred_mean,
                'deficit':     deficit,
                'excess':      excess,
                'margin_up':   max(0, pred_max - current_days),
                'margin_down': max(0, current_days - pred_min),
            }
        }

    # ---- helpers de ajuste de días -----------------------------------------

    @staticmethod
    def _find_dry_gaps(precip_by_day: List[float]) -> List[Dict]:
        """Devuelve lista de {start_pos, end_pos, duration} para secuencias secas."""
        n = len(precip_by_day)
        gaps = []
        i = 0
        while i < n:
            if precip_by_day[i] == 0:
                start_pos = i
                while i < n and precip_by_day[i] == 0:
                    i += 1
                end_pos = i - 1
                gaps.append({
                    'start_pos': start_pos,
                    'end_pos':   end_pos,
                    'duration':  end_pos - start_pos + 1,
                })
            else:
                i += 1
        return gaps

    def _add_rainy_days(self, daily_data: List[Dict],
                        indices: List[int],
                        analysis: Dict,
                        deficit: int):
        """
        Fase 3A: añade días lluviosos hasta cubrir el déficit.

        Estrategia:
        - Si déficit == 1: intentar extender una serie existente por un extremo.
        - Déficit restante: crear series de hasta 5 días en huecos secos disponibles.
        """
        precip_by_day = list(analysis['precip_by_day'])   # copia mutable
        rainy_days    = analysis['rainy_days']
        total_precip  = analysis['total_precipitation']
        mean_precipitation = analysis['mean_precipitation']
        mean_per_rainy_day = (total_precip / rainy_days) if rainy_days > 0 else 1.0
        #print(f"         Media por día lluvioso: {mean_per_rainy_day:.2f} mm")

        deficit_remaining = deficit

        # Intento de extensión de serie existente cuando solo falta 1 día
        if deficit_remaining == 1 and analysis['rain_series']:
            for serie in analysis['rain_series']:
                right_pos = serie['end_pos'] + 1
                if right_pos < len(precip_by_day) and precip_by_day[right_pos] == 0:
                    new_val = max(0.1, round(mean_precipitation * random.uniform(0.5, 1.5), 1))
                    precip_by_day[right_pos] = new_val
                    daily_data[indices[right_pos]]['precipitation'] = new_val
                    #print(f"         ✓ Extendido serie pos {serie['end_pos']} → derecha pos {right_pos}: {new_val} mm")
                    deficit_remaining = 0
                    break
                left_pos = serie['start_pos'] - 1
                if left_pos >= 0 and precip_by_day[left_pos] == 0:
                    new_val = max(0.1, round(mean_precipitation * random.uniform(0.5, 1.5), 1))
                    precip_by_day[left_pos] = new_val
                    daily_data[indices[left_pos]]['precipitation'] = new_val
                    #print(f"         ✓ Extendido serie pos {serie['start_pos']} → izquierda pos {left_pos}: {new_val} mm")
                    deficit_remaining = 0
                    break

        # Crear series en huecos secos para el déficit restante
        series_count = 0
        while deficit_remaining > 0:
            series_size = min(5, deficit_remaining)
            series_total = mean_precipitation * series_size

            gaps = self._find_dry_gaps(precip_by_day)
            suitable = [g for g in gaps if g['duration'] >= series_size]

            if not suitable:
                # Usar el hueco más grande disponible
                if not gaps:
                    #print(f"         ⚠️  No quedan huecos secos para {deficit_remaining} día(s)")
                    break   # No quedan huecos
                gap = max(gaps, key=lambda g: g['duration'])
                series_size  = gap['duration']
                series_total = mean_precipitation * series_size
            else:
                gap = random.choice(suitable)

            max_start = gap['end_pos'] - series_size + 1
            start_pos = random.randint(gap['start_pos'], max_start)

            rainfall = self._generate_variable_rainfall(series_size, series_total)
            for i in range(series_size):
                pos = start_pos + i
                precip_by_day[pos] = rainfall[i]
                daily_data[indices[pos]]['precipitation'] = rainfall[i]

            series_count += 1
            #print(f"         ✓ Serie {series_count}: pos {start_pos}-{start_pos + series_size - 1}, "
            #      f"duración {series_size}, total {series_total:.1f} mm")
            deficit_remaining -= series_size

    def _remove_rainy_days(self, daily_data: List[Dict],
                           indices: List[int],
                           analysis: Dict,
                           excess: int):
        """
        Fase 3B: elimina días lluviosos hasta reducir el exceso.

        Estrategia (en orden):
        1. Eliminar serie completa con duración == exceso (coincidencia exacta).
           Si no existe, eliminar series pequeñas iterativamente (primero las más pequeñas)
           hasta cubrir el exceso. Esto incluye automáticamente series de 1 día.
        2. Quitar extremos de forma distribuida entre las series restantes
           (ordenadas por total_precip ascendente), manteniendo duración >= 2.
        """
        rain_series   = [dict(s) for s in analysis['rain_series']]
        precip_by_day = list(analysis['precip_by_day'])
        excess_remaining = excess

        def _zero_series(serie, reason):
            #print(f"         ✓ Eliminada serie pos {serie['start_pos']}-{serie['end_pos']} "
            #      f"(dur {serie['duration']}, {reason})")
            for pos in range(serie['start_pos'], serie['end_pos'] + 1):
                precip_by_day[pos] = 0.0
                daily_data[indices[pos]]['precipitation'] = 0.0

        # Paso 1: eliminación de series (exacta + pequeñas)
        #print(f"         Paso 1: Buscar coincidencia exacta (duración == {excess_remaining})...")
        exact = next((s for s in sorted(rain_series, key=lambda s: s['duration'])
                      if s['duration'] == excess_remaining), None)
        if exact:
            _zero_series(exact, "coincidencia exacta")
            rain_series.remove(exact)
            excess_remaining = 0
        else:
            # Eliminar series pequeñas iterativamente (primero las más pequeñas)
            series_eliminated = 0
            while excess_remaining > 0 and rain_series:
                candidates = [s for s in rain_series if s['duration'] <= excess_remaining]
                if not candidates:
                    #print(f"         No hay más series para eliminar (todas > {excess_remaining} días)")
                    break
                # Eliminar la más pequeña de las candidatas
                serie_to_remove = min(candidates, key=lambda s: s['duration'])
                reason = f"serie pequeña (dur {serie_to_remove['duration']})"
                _zero_series(serie_to_remove, reason)
                excess_remaining -= serie_to_remove['duration']
                rain_series.remove(serie_to_remove)
                series_eliminated += 1
            
            #if series_eliminated > 0:
            #    print(f"         Eliminadas {series_eliminated} serie(s) pequeña(s)")
            #    if excess_remaining > 0:
            #        print(f"         Todavía faltan eliminar {excess_remaining} día(s)")

        if excess_remaining <= 0:
            return

        # Paso 2: quitar extremos distribuidos entre series restantes
        #print(f"         Paso 2: Quitar {excess_remaining} extremo(s) de forma distribuida...")
        for serie in sorted(rain_series, key=lambda s: s['total_precip']):
            cur_start  = serie['start_pos']
            cur_end    = serie['end_pos']
            cur_values = list(serie['daily_values'])
            extremos_removidos = 0

            while excess_remaining > 0 and (cur_end - cur_start + 1) >= 2:
                if cur_values[0] <= cur_values[-1]:
                    precip_by_day[cur_start] = 0.0
                    daily_data[indices[cur_start]]['precipitation'] = 0.0
                    removed_val = cur_values[0]
                    cur_start += 1
                    cur_values = cur_values[1:]
                    lado = "izq"
                else:
                    precip_by_day[cur_end] = 0.0
                    daily_data[indices[cur_end]]['precipitation'] = 0.0
                    removed_val = cur_values[-1]
                    cur_end -= 1
                    cur_values = cur_values[:-1]
                    lado = "der"
                extremos_removidos += 1
                excess_remaining -= 1
            
            #if extremos_removidos > 0:
            #    print(f"         ✓ Serie pos {serie['start_pos']}-{serie['end_pos']}: "
            #          f"removidos {extremos_removidos} extremo(s)")
            
            if excess_remaining <= 0:
                break

    def _generate_rain_periods(self, daily_data: List[Dict], 
                               indices: List[int], target_mean: float):
        """
        Genera 2 períodos de 2-5 días seguidos con lluvia para alcanzar la media objetivo.
        La precipitación varía aleatoriamente entre días pero mantiene la media total.
        
        Args:
            daily_data: Lista completa de registros diarios (se modifica in-place)
            indices: Índices de los días de este mes
            target_mean: Media de precipitación objetivo (mm/día)
        """
        # Inicializar todos los días a 0
        for idx in indices:
            daily_data[idx]['precipitation'] = 0.0
        
        n_days = len(indices)
        if n_days == 0:
            return
        
        # Generar 2 períodos de lluvia
        total_prec_needed = target_mean * n_days
        
        # Duración de cada período: aleatorio entre 2 y 5 días
        period1_days = random.randint(2, min(5, n_days // 2))
        period2_days = random.randint(2, min(5, n_days // 2))
        
        total_rain_days = period1_days + period2_days
        if total_rain_days > n_days:
            period1_days = n_days // 2
            period2_days = n_days - period1_days
            total_rain_days = period1_days + period2_days
        
        # Posicionar el primer período (en la primera mitad del mes)
        max_start1 = max(0, n_days // 2 - period1_days)
        start1 = random.randint(0, max(0, max_start1))
        
        # Posicionar el segundo período (en la segunda mitad del mes)
        half_month = n_days // 2
        max_start2 = max(half_month, n_days - period2_days)
        start2 = random.randint(half_month, max_start2)
        
        # Distribuir la precipitación total entre los dos períodos (proporcional a duración)
        period1_total = total_prec_needed * (period1_days / total_rain_days)
        period2_total = total_prec_needed * (period2_days / total_rain_days)
        
        # Generar distribución aleatoria de lluvia para el primer período
        period1_values = self._generate_variable_rainfall(period1_days, period1_total)
        
        # Generar distribución aleatoria de lluvia para el segundo período
        period2_values = self._generate_variable_rainfall(period2_days, period2_total)
        
        # Aplicar lluvia al primer período
        for i in range(period1_days):
            idx = start1 + i
            if idx < len(indices):
                daily_data[indices[idx]]['precipitation'] = period1_values[i]
        
        # Aplicar lluvia al segundo período
        for i in range(period2_days):
            idx = start2 + i
            if idx < len(indices):
                daily_data[indices[idx]]['precipitation'] = period2_values[i]
    
    @staticmethod
    def _generate_variable_rainfall(n_days: int, total_rainfall: float) -> List[float]:
        """
        Genera una distribución variable de precipitación para n días que sume exactamente el total.
        
        Usa una distribución aleatoria para simular variabilidad natural de la lluvia,
        donde algunos días llueve más intensamente y otros menos.
        
        Args:
            n_days: Número de días con lluvia
            total_rainfall: Precipitación total a distribuir (mm)
            
        Returns:
            Lista de valores de precipitación para cada día (redondeados a 1 decimal)
        """
        if n_days <= 0:
            return []
        
        if total_rainfall <= 0:
            return [0.0] * n_days
        
        # Generar pesos aleatorios usando distribución exponencial
        # Esto simula que algunos días llueve más intensamente que otros
        weights = [random.expovariate(1.0) for _ in range(n_days)]
        
        # Normalizar los pesos para que sumen 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Distribuir la precipitación según los pesos
        rainfall_values = [total_rainfall * w for w in normalized_weights]
        
        # Redondear a 1 decimal
        rounded_values = [round(val, 1) for val in rainfall_values]
        
        # Ajustar para compensar errores de redondeo y mantener suma exacta
        current_sum = sum(rounded_values)
        diff = round(total_rainfall - current_sum, 1)
        
        if diff != 0:
            # Añadir la diferencia al día con mayor precipitación
            max_idx = rounded_values.index(max(rounded_values))
            rounded_values[max_idx] = round(rounded_values[max_idx] + diff, 1)
        
        return rounded_values
    
    # ========================================================================
    # STEP 3: Generate hourly data from daily records
    # ========================================================================
    
    def _generate_hourly_from_daily(self, daily_data: List[Dict]) -> List[Dict]:
        """
        Genera datos horarios (24 registros por día) a partir de registros diarios.
        
        Usa interpolación sinusoidal para temperatura, humedad y presión,
        distribución gaussiana para precipitación, y perfil gaussiano para viento.
        
        Args:
            daily_data: Lista de registros diarios
        
        Returns:
            Lista de diccionarios con datos horarios
        """
        all_hourly = []
        
        for daily_rec in daily_data:
            fecha = daily_rec['date']
            
            # Temperaturas
            tmin = daily_rec.get('temperature_min')
            tmax = daily_rec.get('temperature_max')
            hour_tmin = self._parse_hour(daily_rec.get('hour_tmin'))
            hour_tmax = self._parse_hour(daily_rec.get('hour_tmax'))
            
            tmean = daily_rec.get('temperature_mean')
            if tmin is not None and tmax is not None:
                hourly_temps = self._interpolate_hourly_temperature(tmin, tmax, tmean, hour_tmin, hour_tmax)
            else:
                hourly_temps = [tmean] * 24 if tmean is not None else [None] * 24
            
            # Precipitación
            prec = daily_rec.get('precipitation', 0.0) or 0.0
            hourly_precip = self._distribute_precipitation(prec)
            
            # Viento
            wind_mean = daily_rec.get('wind_speed_mean')
            wind_max = daily_rec.get('wind_speed_max')
            hour_wind = self._parse_hour(daily_rec.get('hour_wind_max'))
            hourly_wind = self._interpolate_wind_speed(wind_mean, wind_max, hour_wind)
            wind_dir = daily_rec.get('wind_direction')
            
            # Humedad
            hr_min = daily_rec.get('humidity_min')
            hr_max = daily_rec.get('humidity_max')
            hr_mean = daily_rec.get('humidity_mean')
            if hr_min is not None and hr_max is not None:
                hour_hrmin = self._parse_hour(daily_rec.get('hour_hrmin'))
                hour_hrmax = self._parse_hour(daily_rec.get('hour_hrmax'))
                hourly_humidity = self._interpolate_hourly_humidity(hr_min, hr_max, hr_mean, hour_hrmin, hour_hrmax)
            else:
                hourly_humidity = [None] * 24

            # Presión
            pres_min = daily_rec.get('pressure_min')
            pres_max = daily_rec.get('pressure_max')
            if pres_min is not None and pres_max is not None:
                hour_presmin = self._parse_hour(daily_rec.get('hour_presmin'))
                hour_presmax = self._parse_hour(daily_rec.get('hour_presmax'))
                hourly_pressure = self._interpolate_pressure(pres_min, pres_max, hour_presmin, hour_presmax)
            else:
                hourly_pressure = [None] * 24

            # Crear 24 registros horarios
            for hour in range(24):
                datetime_iso = f"{fecha}T{hour:02d}:00:00Z"
                all_hourly.append({
                    'datetime': datetime_iso,
                    'temperature': hourly_temps[hour],
                    'precipitation': hourly_precip[hour],
                    'wind_speed': hourly_wind[hour],
                    'wind_direction': wind_dir,
                    'humidity': hourly_humidity[hour],
                    'pressure': hourly_pressure[hour]
                })
        
        return all_hourly
    
    # ========================================================================
    # STEP 4: Verify hourly vs daily
    # ========================================================================
    
    def _verify_daily_vs_predictions(self, daily_data: List[Dict], 
                                     predictions_df: pd.DataFrame) -> bool:
        """
        Verifica que los datos diarios generados coinciden con las predicciones mensuales.
        Solo un método de depuración para validar coherencia matemática.
        
        Márgenes de error permitidos:
        - Temperatura: ±1.0°C en la media
        - Precipitación: ±0.5 mm en la media mensual
        
        Args:
            daily_data: Lista de registros diarios generados
            predictions_df: DataFrame con predicciones mensuales
            
        Returns:
            True si pasa la verificación, False si hay discrepancias
        """
        all_pass = True
        
        # Crear índice de predicciones para búsqueda O(1): (year, month, variable) -> predicción
        predictions_index = {}
        for _, row in predictions_df.iterrows():
            year = int(row['Year'])
            month = int(row['Month'])
            variable = str(row['Variable'])
            key = (year, month, variable)
            predictions_index[key] = {
                'mean': row.get('Mean'),
                'min': row.get('Minimum'),
                'max': row.get('Maximum')
            }
        
        # Agrupar datos diarios por año-mes
        daily_by_month = {}
        for record in daily_data:
            date = datetime.strptime(record['date'], '%Y-%m-%d')
            month_key = (date.year, date.month)
            if month_key not in daily_by_month:
                daily_by_month[month_key] = []
            daily_by_month[month_key].append(record)
        
        # Verificar cada mes que tenga datos diarios
        for (year, month), month_records in daily_by_month.items():
            # Verificar TEMPERATURAS
            for temp_var in ['temperature_max', 'temperature_mean', 'temperature_min']:
                pred_key = (year, month, temp_var)
                if pred_key not in predictions_index:
                    continue
                
                pred_data = predictions_index[pred_key]
                pred_mean = pred_data.get('mean')
                
                if pd.isna(pred_mean):
                    continue
                
                values = [r.get(temp_var) for r in month_records if r.get(temp_var) is not None]
                if not values:
                    continue
                
                actual_mean = np.mean(values)
                diff = abs(actual_mean - float(pred_mean))
                
                var_label = {'temperature_max': 'Tmax', 'temperature_mean': 'Tmean', 'temperature_min': 'Tmin'}[temp_var]
                
                if diff > 1.0:
                    print(f"  ⚠️ {var_label} {year}-{month:02d}: Media predicha={pred_mean:.1f}, Media real={actual_mean:.1f}, Diferencia={diff:.2f}°C")
                    all_pass = False
                
                # Verificar límites
                pred_min = pred_data.get('min')
                if pd.notna(pred_min):
                    actual_min = min(values)
                    if actual_min < float(pred_min) - 0.1:
                        print(f"  ❌  {var_label} {year}-{month:02d}: Mínimo real ({actual_min:.1f}°C) inferior a predicción ({pred_min:.1f}°C)")
                
                pred_max = pred_data.get('max')
                if pd.notna(pred_max):
                    actual_max = max(values)
                    if actual_max > float(pred_max) + 0.1:
                        print(f"  ❌  {var_label} {year}-{month:02d}: Máximo real ({actual_max:.1f}°C) superior a predicción ({pred_max:.1f}°C)")
            
            # Verificar PRECIPITACIÓN
            pred_key = (year, month, 'precipitation')
            if pred_key in predictions_index:
                pred_data = predictions_index[pred_key]
                pred_mean = pred_data.get('mean')
                
                if pd.notna(pred_mean):
                    values = [r.get('precipitation', 0) or 0 for r in month_records]
                    total_prec = sum(values)
                    n_days = len(values)
                    actual_mean = total_prec / n_days if n_days > 0 else 0
                    
                    pred_mean_val = float(pred_mean)
                    diff = abs(actual_mean - pred_mean_val)
                    if diff > 0.5:
                        print(f"  ⚠️ Precip {year}-{month:02d}: Media predicha={pred_mean_val:.1f} mm/día, Media real={actual_mean:.1f} mm/día, Diferencia={diff:.2f} mm")
                        all_pass = False
                    
                    # Verificar límites (comparar medias directamente)
                    pred_min = pred_data.get('min')
                    if pd.notna(pred_min):
                        pred_min_val = float(pred_min)
                        if actual_mean < pred_min_val - 0.1:
                            print(f"  ❌  Precip {year}-{month:02d}: Media real ({actual_mean:.1f} mm/día) inferior a mínimo predicho ({pred_min_val:.1f} mm/día)")
                    
                    pred_max = pred_data.get('max')
                    if pd.notna(pred_max):
                        pred_max_val = float(pred_max)
                        if actual_mean > pred_max_val + 0.1:
                            print(f"  ❌  Precip {year}-{month:02d}: Media real ({actual_mean:.1f} mm/día) superior a máximo predicho ({pred_max_val:.1f} mm/día)")
        
        return all_pass
    
    def _verify_hourly_vs_daily(self, hourly_data: List[Dict], 
                               daily_data: List[Dict]) -> bool:
        """
        Verifica que los datos horarios coinciden con los datos diarios.
        Solo un método de depuración para validar coherencia matemática.
        
        Márgenes de error permitidos:
        - Temperatura: ±1.5°C en la media diaria
        - Precipitación: ±0.5 mm
        - Humedad: ±5% en la media diaria
        - Presión: ±2 hPa en la media diaria
        - Viento: ±0.5 m/s
        
        Args:
            hourly_data: Lista de registros horarios generados
            daily_data: Lista de registros diarios base
            
        Returns:
            True si pasa la verificación, False si hay discrepancias
        """
        all_pass = True
        
        # Agrupar datos horarios por fecha
        hourly_by_date = {}
        for rec in hourly_data:
            date = rec['datetime'].split('T')[0]
            if date not in hourly_by_date:
                hourly_by_date[date] = []
            hourly_by_date[date].append(rec)
        
        # Verificar cada día
        for daily_rec in daily_data:
            date = daily_rec['date']
            
            if date not in hourly_by_date:
                print(f"  ❌ {date}: No hay datos horarios para este día")
                all_pass = False
                continue
            
            hourly_recs = hourly_by_date[date]
            
            # Verificar que hay 24 registros horarios
            if len(hourly_recs) != 24:
                print(f"  ❌ {date}: Se esperaban 24 registros horarios, se encontraron {len(hourly_recs)}")
                all_pass = False
                continue
            
            # Verificar TEMPERATURA
            hourly_temps = [r.get('temperature') for r in hourly_recs 
                           if r.get('temperature') is not None]
            if hourly_temps:
                mean_hourly_temp = np.mean(hourly_temps)
                daily_tmean = daily_rec.get('temperature_mean')
                
                if daily_tmean is not None:
                    diff_temp = abs(mean_hourly_temp - daily_tmean)
                    if diff_temp > 1.5:
                        print(f"  ⚠️  {date}: Temp media horaria ({mean_hourly_temp:.1f}°C) vs diaria ({daily_tmean:.1f}°C), Diff={diff_temp:.2f}°C")
                
                # Verificar rangos
                hourly_tmin = min(hourly_temps)
                hourly_tmax = max(hourly_temps)
                daily_tmin = daily_rec.get('temperature_min') or hourly_tmin
                daily_tmax = daily_rec.get('temperature_max') or hourly_tmax
                
                if hourly_tmin < daily_tmin - 0.1:
                    print(f"  ⚠️  {date}: Tmin horaria ({hourly_tmin:.1f}°C) inferior a diaria ({daily_tmin:.1f}°C)")
                
                if hourly_tmax > daily_tmax + 0.1:
                    print(f"  ⚠️  {date}: Tmax horaria ({hourly_tmax:.1f}°C) superior a diaria ({daily_tmax:.1f}°C)")
            
            # Verificar PRECIPITACIÓN
            hourly_precips = [r.get('precipitation') or 0 for r in hourly_recs]
            total_hourly_precip = sum(hourly_precips)
            daily_precip = daily_rec.get('precipitation') or 0
            
            diff_precip = abs(total_hourly_precip - daily_precip)
            if diff_precip > 0.5:
                print(f"  ⚠️  {date}: Precip total horaria ({total_hourly_precip:.1f} mm) vs diaria ({daily_precip:.1f} mm), Diff={diff_precip:.2f} mm")
            
            # Verificar HUMEDAD
            hourly_humidities = [r.get('humidity') for r in hourly_recs 
                                if r.get('humidity') is not None]
            if hourly_humidities:
                mean_hourly_humid = np.mean(hourly_humidities)
                daily_humid_mean = daily_rec.get('humidity_mean')
                
                if daily_humid_mean is not None:
                    diff_humid = abs(mean_hourly_humid - daily_humid_mean)
                    if diff_humid > 5:
                        print(f"  ⚠️  {date}: Humedad media horaria ({mean_hourly_humid:.1f}%) vs diaria ({daily_humid_mean:.1f}%), Diff={diff_humid:.1f}%")
            
            # Verificar PRESIÓN
            hourly_pressures = [r.get('pressure') for r in hourly_recs 
                               if r.get('pressure') is not None]
            if hourly_pressures:
                mean_hourly_pres = np.mean(hourly_pressures)
                daily_pres_min = daily_rec.get('pressure_min')
                daily_pres_max = daily_rec.get('pressure_max')
                if daily_pres_min is None or daily_pres_max is None:
                    continue
                daily_pres_mean = (daily_pres_min + daily_pres_max) / 2
                
                diff_pres = abs(mean_hourly_pres - daily_pres_mean)
                if diff_pres > 2.0:
                    print(f"  ⚠️  {date}: Presión media horaria ({mean_hourly_pres:.1f} hPa) vs diaria ({daily_pres_mean:.1f} hPa), Diff={diff_pres:.1f} hPa")
            
            # Verificar VIENTO
            hourly_winds = [r.get('wind_speed') for r in hourly_recs 
                           if r.get('wind_speed') is not None]
            if hourly_winds:
                mean_hourly_wind = np.mean(hourly_winds)
                daily_wind_mean = daily_rec.get('wind_speed_mean')
                max_hourly_wind = max(hourly_winds)
                daily_wind_max = daily_rec.get('wind_speed_max')
                
                if daily_wind_mean is not None:
                    diff_wind = abs(mean_hourly_wind - daily_wind_mean)
                    if diff_wind > 0.5:
                        print(f"  ⚠️  {date}: Viento medio horario ({mean_hourly_wind:.1f} m/s) vs diario ({daily_wind_mean:.1f} m/s), Diff={diff_wind:.2f} m/s")
                
                # Verificar que se alcanza la racha máxima de viento esperada
                if daily_wind_max is not None:
                    diff_wind_max = abs(max_hourly_wind - daily_wind_max)
                    if diff_wind_max > 0.1:
                        print(f"  ⚠️  {date}: Racha máxima de viento no alcanza valor esperado. Horaria={max_hourly_wind:.1f} m/s, Esperada={daily_wind_max:.1f} m/s, Diff={diff_wind_max:.2f} m/s")
        
        return all_pass
    
    # ========================================================================
    # Main generation flow
    # ========================================================================
    
    def generate(
        self,
        predictions_df: Optional[pd.DataFrame] = None,
        correction_method: str = 'knn',
    ) -> Tuple[List[Dict], List[Dict], int]:
        """
        Genera datos sintéticos completos (diarios + horarios).

        Flow:
        1. Ciclar datos históricos diarios al período de generación
        2. Ajustar a predicciones mensuales (si se proporcionan)
        3. Aplicar corrección de variables secundarias (KNN o XGBoost)
        4. Generar datos horarios
        5. Verificar que datos diarios cuadran con predicciones mensuales (si se proporcionan)
        6. Verificar que los datos horarios cuadran con los diarios

        Args:
            predictions_df:    DataFrame con predicciones mensuales (opcional).
            correction_method: Método de corrección para variables secundarias
                               (viento, humedad, presión). Sólo se aplica cuando
                               se proporcionan predicciones mensuales.
                               'knn'     - K-Nearest Neighbors.
                               'xgboost' - Modelo XGBoost de ventana deslizante.

        Returns:
            Tupla (daily_data, hourly_data, total_hourly_records)
        """
        # 1. Generar datos diarios sintéticos (ciclo del histórico)
        daily_data = self._generate_daily_synthetic()
        print(f"📅 Generados {len(daily_data)} registros diarios sintéticos")
        
        # 2. Ajustar a predicciones mensuales si se proporcionan
        if predictions_df is not None and not predictions_df.empty:
            daily_data = self._adjust_to_monthly_predictions(daily_data, predictions_df)

            # 2.5. Corrección multivariada MBCn para consistencia inter-variables
            #print("🔄 Aplicando corrección multivariada MBCn...")
            #mbc_corrector = MBCnCorrector(n_iter=30, extrapolation_quantile=0.95)
            #daily_data = mbc_corrector.correct(
            #    adjusted_data=daily_data,
            #    historical_data=self.historical_data
            #)
            #print("✅ Corrección MBCn aplicada")

            # 2.5. Corrección de variables numéricas secundarias (viento, humedad,
            #      presión). Sólo se aplica cuando hay predicciones mensuales.
            #      Las variables no numéricas (dirección viento, horas) se
            #      mantienen tal como vienen del ciclo histórico.
            if correction_method == 'xgboost':
                print("🔄 Aplicando modelo XGBoost (ventana deslizante) para variables secundarias...")
                xgb_model = XGBoostWeatherModel(window_size=5)
                daily_data = xgb_model.correct(
                    adjusted_data=daily_data,
                    historical_data=self.historical_data
                )
                print("✅ Corrección XGBoost aplicada")
            else:  # 'knn'
                # Corrección K-Vecinos: adaptar variables no modificadas
                # (viento, humedad, presión) usando los días históricos más
                # parecidos en temperatura y precipitación.
                print("🔄 Aplicando corrección K-Vecinos para variables secundarias...")
                knn_corrector = KNeighborsCorrector(k=3, month_weight=0.25)
                daily_data = knn_corrector.correct(
                    adjusted_data=daily_data,
                    historical_data=self.historical_data
                )
                print("✅ Corrección K-Vecinos aplicada")
        
        # 3. Generar datos horarios desde los diarios
        hourly_data = self._generate_hourly_from_daily(daily_data)
        print(f"🕐 Generados {len(hourly_data)} registros horarios")
        
        # 4. Verificar que los datos generados cuadran con los diarios (y/o predicciones)
        if predictions_df is not None and not predictions_df.empty:
            print("🔍 Verificando coherencia de datos diarios vs predicciones mensuales...")
            if self._verify_daily_vs_predictions(daily_data, predictions_df):
                print("✅ Datos diarios vs predicciones: coherencia verificada")
            else:
                print("⚠️ Se detectaron discrepancias en datos diarios vs predicciones (ver errores arriba)")
        
        print("🔍 Verificando coherencia de datos horarios vs diarios...")
        if self._verify_hourly_vs_daily(hourly_data, daily_data):
            print("✅ Datos horarios vs diarios: coherencia verificada")
        else:
            print("⚠️ Se detectaron discrepancias en datos horarios vs diarios (ver errores arriba)")
        
        return daily_data, hourly_data, len(hourly_data)

    def generate_and_save(
        self,
        latitude: float,
        longitude: float,
        predictions_df: Optional[pd.DataFrame] = None,
        correction_method: str = 'knn',
    ) -> Tuple[int, int, List[Dict]]:
        """
        Genera datos sintéticos, los guarda en BD y devuelve la info.

        Args:
            latitude:          Latitud de la ubicación.
            longitude:         Longitud de la ubicación.
            predictions_df:    DataFrame con predicciones mensuales (opcional).
            correction_method: Método de corrección para variables secundarias
                               ('knn' o 'xgboost').

        Returns:
            Tupla (job_id, cantidad_registros_horarios, datos_horarios)
        """
        # 1. Crear entrada en GenerationJob
        job_data = [(
            latitude,
            longitude,
            self.historical_start,
            self.historical_end,
            self.generation_start,
            self.generation_end
        )]
        
        job_ids = insert_generation_jobs(job_data)
        if not job_ids:
            raise ValueError("Error al crear el trabajo de generación en la base de datos")
        job_id = job_ids[0]
        
        # 2. Guardar predicciones mensuales si se proporcionan
        if predictions_df is not None and not predictions_df.empty:
            pred_tuples = []
            for _, row in predictions_df.iterrows():
                pred_tuples.append((
                    job_id,
                    int(row['Year']),
                    int(row['Month']),
                    str(row['Variable']),
                    float(row['Minimum']) if pd.notna(row.get('Minimum')) else None,
                    float(row['Mean']) if pd.notna(row.get('Mean')) else None,
                    float(row['Maximum']) if pd.notna(row.get('Maximum')) else None
                ))
            insert_monthly_predictions(pred_tuples)
            print(f"📊 Insertadas {len(pred_tuples)} predicciones mensuales")
        
        # 3. Generar datos
        daily_data, hourly_data, hourly_count = self.generate(predictions_df, correction_method)
        
        # 4. Insertar datos diarios generados
        daily_tuples = []
        for rec in daily_data:
            daily_tuples.append((
                job_id,
                rec['date'],
                rec.get('temperature_min'),
                rec.get('temperature_max'),
                rec.get('temperature_mean'),
                rec.get('precipitation'),
                rec.get('wind_speed_mean'),
                rec.get('wind_speed_max'),
                rec.get('wind_direction'),
                rec.get('humidity_min'),
                rec.get('humidity_max'),
                rec.get('humidity_mean'),
                rec.get('pressure_min'),
                rec.get('pressure_max')
            ))
        insert_generated_daily_data(daily_tuples)
        print(f"📅 Insertados {len(daily_tuples)} registros diarios generados")
        
        # 5. Insertar datos horarios generados
        hourly_tuples = []
        for rec in hourly_data:
            hourly_tuples.append((
                job_id,
                rec['datetime'],
                rec.get('temperature'),
                rec.get('precipitation'),
                rec.get('wind_speed'),
                rec.get('wind_direction'),
                rec.get('humidity'),
                rec.get('pressure')
            ))
        insert_generated_hourly_data(hourly_tuples)
        print(f"🕐 Insertados {len(hourly_tuples)} registros horarios generados")
        
        print(f"✅ Generación completa. Job ID: {job_id}")
        
        return job_id, hourly_count, hourly_data
    
    # ========================================================================
    # Interpolation helper methods
    # ========================================================================
    
    @staticmethod
    def _parse_hour(time_str) -> int:
        """Extrae la hora de un string HH:MM o similar. Default: 12"""
        try:
            if not time_str or str(time_str).lower() in ['varias', 'n/a', 'nd', '', 'none']:
                return 12
            time_str = str(time_str).strip()
            if time_str.isdigit():
                return int(time_str) % 24
            if ':' in time_str:
                return int(time_str.split(':')[0]) % 24
            return 12
        except (ValueError, IndexError):
            return 12

    @staticmethod
    def _interpolate_hourly_temperature(
        tmin: float, tmax: float, tmean: Optional[float],
        hour_min: int, hour_max: int
    ) -> List[float]:

        if tmean is None:
            tmean = (tmin + tmax) / 2

        if tmax == tmin:
            return [round(tmin, 1)] * 24

        hours_between = (hour_max - hour_min) % 24
        if hours_between == 0:
            hours_between = 12

        # Generar forma base normalizada f(h) ∈ [0,1]
        f_values = []

        for hour in range(24):

            hours_since_min = (hour - hour_min) % 24

            if hours_since_min <= hours_between:
                progress = hours_since_min / hours_between
                f = (1 - cos(pi * progress)) / 2
            else:
                hours_since_max = hours_since_min - hours_between
                hours_to_next_min = 24 - hours_between
                progress = hours_since_max / hours_to_next_min
                f = 1 - (1 - cos(pi * progress)) / 2

            f_values.append(f)

        # Buscar exponente α que conserve la media
        def compute_mean(alpha):
            temps = [
                tmin + (tmax - tmin) * (f ** alpha)
                for f in f_values
            ]
            return sum(temps) / 24

        # Bisección estable
        low, high = 0.1, 5.0

        for _ in range(50):
            mid = (low + high) / 2
            m = compute_mean(mid)

            if m > tmean:
                low = mid
            else:
                high = mid

        alpha = (low + high) / 2

        # Construir temperaturas finales
        hourly_temps = [
            tmin + (tmax - tmin) * (f ** alpha)
            for f in f_values
        ]

        # Forzar extremos exactos
        hourly_temps[hour_min] = tmin
        hourly_temps[hour_max] = tmax

        return [round(t, 1) for t in hourly_temps]
    
    @staticmethod
    def _interpolate_wind_speed(
        wind_avg: Optional[float], wind_max: Optional[float], hour_max: int
    ) -> List[Optional[float]]:
        """Interpola velocidad del viento horaria con pico gaussiano."""
        if wind_avg is None:
            return [None] * 24

        if wind_max is None or wind_max <= wind_avg:
            return [wind_avg] * 24

        # Caso especial: media 0
        if wind_avg == 0:
            hourly = [0.0] * 24
            hourly[hour_max] = round(wind_max, 1)
            return hourly

        # Generar forma gaussiana
        base = []
        for hour in range(24):
            dist = min(abs(hour - hour_max), 24 - abs(hour - hour_max))
            factor = exp(-(dist**2) / 2)
            base.append(factor)

        # Normalizar para que el máximo sea exactamente 1
        max_base = max(base)
        base = [b / max_base for b in base]

        mean_base = sum(base) / 24

        # Resolver sistema para cumplir media y máximo
        b = (wind_max - wind_avg) / (1 - mean_base)
        a = wind_max - b

        hourly = [max(0, round(a + b * f, 1)) for f in base]

        return hourly
    
    @staticmethod
    def _interpolate_hourly_humidity(
        hr_min: float, hr_max: float, hr_mean: Optional[float],
        hour_hrmin: int, hour_hrmax: int
    ) -> List[int]:
        """Interpola humedad relativa horaria con ajuste de media (similar a temperatura).
        
        Genera una curva coseno normalizada f(h) ∈ [0,1] entre hr_min y hr_max,
        y busca un exponente α tal que la media horaria coincida con hr_mean.
        
        Args:
            hr_min: Humedad relativa mínima del día (%)
            hr_max: Humedad relativa máxima del día (%)
            hr_mean: Humedad relativa media del día (%)
            hour_hrmin: Hora del mínimo de humedad
            hour_hrmax: Hora del máximo de humedad
            
        Returns:
            Lista de 24 valores enteros de humedad relativa (%)
        """
        if hr_mean is None:
            hr_mean = (hr_max + hr_min) / 2

        if hr_max == hr_min:
            return [int(round(hr_min))] * 24

        hours_between = (hour_hrmin - hour_hrmax) % 24
        if hours_between == 0:
            hours_between = 12

        # Generar forma base normalizada f(h) ∈ [0,1]
        # donde 0 = hr_min y 1 = hr_max
        f_values = []
        for hour in range(24):
            hours_since_max = (hour - hour_hrmax) % 24

            if hours_since_max <= hours_between:
                # Descenso de máximo a mínimo
                progress = hours_since_max / hours_between
                f = 1 - (1 - cos(pi * progress)) / 2
            else:
                # Ascenso de mínimo a máximo
                hours_since_min = hours_since_max - hours_between
                hours_to_next_max = 24 - hours_between
                progress = hours_since_min / hours_to_next_max
                f = (1 - cos(pi * progress)) / 2

            f_values.append(f)

        # Buscar exponente α que conserve la media
        def compute_mean(alpha):
            humidities = [
                hr_min + (hr_max - hr_min) * (f ** alpha)
                for f in f_values
            ]
            return sum(humidities) / 24

        # Bisección estable
        low, high = 0.1, 5.0
        for _ in range(50):
            mid = (low + high) / 2
            m = compute_mean(mid)
            if m > hr_mean:
                low = mid
            else:
                high = mid

        alpha = (low + high) / 2

        # Construir humedades finales
        hourly_humidity = [
            hr_min + (hr_max - hr_min) * (f ** alpha)
            for f in f_values
        ]

        # Forzar extremos exactos
        hourly_humidity[hour_hrmax] = hr_max
        hourly_humidity[hour_hrmin] = hr_min

        return [int(round(max(0, min(100, h)))) for h in hourly_humidity]
    
    @staticmethod
    def _distribute_precipitation(total_precip: float) -> List[float]:
        """Distribuye precipitación diaria en horas de forma realista."""
        hourly_precip = [0.0] * 24
        if total_precip <= 0:
            return hourly_precip
        
        random.seed(int(total_precip * 1000))
        
        if total_precip < 5:
            rain_hours = random.randint(2, 4)
        elif total_precip < 15:
            rain_hours = random.randint(4, 6)
        else:
            rain_hours = random.randint(6, 8)
        
        start_hour = random.randint(0, 24 - rain_hours)
        for i in range(rain_hours):
            hour = (start_hour + i) % 24
            factor = exp(-((i - rain_hours/2)**2) / (rain_hours/2))
            hourly_precip[hour] = factor
        
        total_distributed = sum(hourly_precip)
        if total_distributed > 0:
            hourly_precip = [p * total_precip / total_distributed for p in hourly_precip]
        
        return [round(p, 1) for p in hourly_precip]
    
    @staticmethod
    def _interpolate_pressure(
        pres_min: float, pres_max: float, hour_presmin: int, hour_presmax: int
    ) -> List[float]:
        """Interpola presión atmosférica horaria con curva coseno."""
        hourly_pressure = []
        for hour in range(24):
            hours_since_min = (hour - hour_presmin) % 24
            hours_between = (hour_presmax - hour_presmin) % 24
            if hours_between == 0:
                hours_between = 12
            
            if hours_since_min <= hours_between:
                progress = hours_since_min / hours_between
                smooth_progress = (1 - cos(pi * progress)) / 2
                pressure = pres_min + (pres_max - pres_min) * smooth_progress
            else:
                hours_since_max = hours_since_min - hours_between
                hours_to_next_min = 24 - hours_between
                progress = hours_since_max / hours_to_next_min
                smooth_progress = (1 - cos(pi * progress)) / 2
                pressure = pres_max - (pres_max - pres_min) * smooth_progress
            
            hourly_pressure.append(round(pressure, 1))
        return hourly_pressure
