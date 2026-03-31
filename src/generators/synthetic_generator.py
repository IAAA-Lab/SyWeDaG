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
from utils.system_utils import safe_print

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
    Synthetic weather data generator.

    Cycles historical daily data for the generation period,
    adjusts it to monthly predictions (if provided),
    and generates interpolated hourly data.
    """
    
    def __init__(self, source: str, id_station, historical_start: str, 
                 historical_end: str, generation_start: str, generation_end: str):
        """
        Initialize the generator.
        
        Args:
            source: Data source (e.g., 'AEMET')
            id_station: Station ID
            historical_start: Historical start date (YYYY-MM-DD)
            historical_end: Historical end date (YYYY-MM-DD)
            generation_start: Generation start date (YYYY-MM-DD)
            generation_end: Generation end date (YYYY-MM-DD)
        """
        self.source = source
        self.id_station = id_station
        self.historical_start = historical_start
        self.historical_end = historical_end
        self.generation_start = generation_start
        self.generation_end = generation_end
        
        # Load DAILY historical data
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
        
        # Sort by date
        self.historical_data.sort(key=lambda x: x['date'])
        
        safe_print(f"📊 Historical daily data loaded: {len(self.historical_data)} records")
    
    # ========================================================================
    # STEP 1: Cycle daily records
    # ========================================================================
    
    def _generate_daily_synthetic(self) -> List[Dict]:
        """
        Generate synthetic daily data by cycling historical records by month-day.

        For each target date, find the same month-day in historical years
        and cycle across available years.
        
        Returns:
            List of dictionaries with synthetic daily data
        """
        generated_daily = []
        
        gen_start = datetime.strptime(self.generation_start, '%Y-%m-%d')
        gen_end = datetime.strptime(self.generation_end, '%Y-%m-%d')
        
        # Index historical records by month-day and year
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
                    # CYCLE: use modulo to iterate available years deterministically
                    # Without this (with random.randint()), data was distributed randomly,
                    # now it cycles by historical year to simplify debugging
                    year_index = (occurrence_count - 1) % len(available_years)
                    selected_year = available_years[year_index]
                    source_record = records_by_month_day[month_day_key][selected_year]
            
            # Special case: February 29
            if source_record is None and current_date.month == 2 and current_date.day == 29:
                alt_key = "02-28"
                if alt_key in records_by_month_day:
                    available_years = sorted(records_by_month_day[alt_key].keys())
                    # TODO dejar aleatorio
                    #year_index = random.randint(0, len(available_years) - 1)
                    # CYCLE: use modulo to iterate available years deterministically
                    # Without this (with random.randint()), data was distributed randomly,
                    # now it cycles by historical year to simplify debugging
                    year_index = (occurrence_count - 1) % len(available_years)
                    selected_year = available_years[year_index]
                    source_record = records_by_month_day[alt_key][selected_year]
            
            # Fallback: cycle over all historical records
            if source_record is None:
                # TODO dejar aleatorio
                #idx = random.randint(0, len(self.historical_data) - 1)
                # CYCLE: instead of random.randint(), use modulo for deterministic cycling
                idx = (occurrence_count - 1) % len(self.historical_data)
                source_record = self.historical_data[idx]
            
            # Copy record with new date
            new_record = source_record.copy()
            new_record['date'] = current_date.strftime('%Y-%m-%d')
            generated_daily.append(new_record)
            
            current_date += timedelta(days=1)
        
        # 📊 Synthetic daily records generated from historical cycle
        safe_print(f"📅 Cycle completed: {len(generated_daily)} daily records generated")
        return generated_daily
    
    # ========================================================================
    # STEP 2: Adjust daily records to monthly predictions
    # ========================================================================
    
    def _adjust_to_monthly_predictions(self, daily_data: List[Dict], 
                                       predictions_df: pd.DataFrame) -> List[Dict]:
        """
        Adjust historical daily data to match monthly predictions.
        
        For temperatures:
        - Compute monthly historical statistics
        - Apply differences to each day of the month
        - Check individual limits (min/max) for each temperature
        - Compensate on the opposite extreme when limits are exceeded
        - Enforce order: Tmax >= Tmean >= Tmin

        For precipitation:
        - Apply multiplicative factor
        - If prediction=0, set all days to 0
        - If historical=0 but prediction>0, generate 2 rain periods
        
        Args:
            daily_data: List of daily records
            predictions_df: DataFrame with monthly predictions (long format)
            
        Returns:
            List of adjusted daily records
        """
        adjusted_data = [record.copy() for record in daily_data]
        
        # Build prediction index for O(1) lookup: (year, month, variable) -> prediction
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
        
        # Group daily records by year-month
        daily_by_month = {}
        for i, record in enumerate(adjusted_data):
            date = datetime.strptime(record['date'], '%Y-%m-%d')
            month_key = (date.year, date.month)
            if month_key not in daily_by_month:
                daily_by_month[month_key] = []
            daily_by_month[month_key].append(i)
        
        # Process each month
        for (year, month), indices in daily_by_month.items():
            # ============================================================
            # TEMPERATURES - Improved logic with per-record validation
            # ============================================================
            self._adjust_temperatures_for_month(adjusted_data, indices, predictions_index, year, month)
            
            # ============================================================
            # PRECIPITATION
            # ============================================================
            self._adjust_precipitation_for_month(adjusted_data, indices, predictions_index, year, month)
        
        return adjusted_data
    
    def _get_variable_stats(self, pred_df: pd.DataFrame, variable_name: str) -> Dict:
        """
        Extract variable statistics (min, mean, max) from long-format DataFrame.
        
        Args:
            pred_df: DataFrame filtrado por mes (Year, Month, Variable, Minimum, Mean, Maximum)
            variable_name: Variable name to search (e.g., 'temperature_max', 'precipitation')
            
        Returns:
            Dict with 'min', 'mean', 'max' or None if not found
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
        Temperature adjustment with per-record limit validation.
        
        Args:
            daily_data: Full list of daily records (modified in-place)
            indices: Indices of the days in this month
            predictions_index: Dictionary (year, month, variable) -> {mean, min, max}
            year: Year of the month
            month: Month (1-12)
        """
        # Get predictions for this month
        tmax_pred = predictions_index.get((year, month, 'temperature_max'))
        tmean_pred = predictions_index.get((year, month, 'temperature_mean'))
        tmin_pred = predictions_index.get((year, month, 'temperature_min'))
        
        if not all([tmax_pred, tmean_pred, tmin_pred]):
            return
        
        # Validate that mean values exist
        if any(pd.isna(pred.get('mean')) for pred in [tmax_pred, tmean_pred, tmin_pred]):
            return
        
        # Compute monthly statistics of current historical values
        tmax_values = [daily_data[i].get('temperature_max') for i in indices if daily_data[i].get('temperature_max') is not None]
        tmean_values = [daily_data[i].get('temperature_mean') for i in indices if daily_data[i].get('temperature_mean') is not None]
        tmin_values = [daily_data[i].get('temperature_min') for i in indices if daily_data[i].get('temperature_min') is not None]
        
        if not all([tmax_values, tmean_values, tmin_values]):
            return
        
        hist_tmax_mean = np.mean(tmax_values)
        hist_tmean_mean = np.mean(tmean_values)
        hist_tmin_mean = np.mean(tmin_values)
        
        # Compute differences (prediction - historical)
        diff_tmax = tmax_pred['mean'] - hist_tmax_mean
        diff_tmean = tmean_pred['mean'] - hist_tmean_mean
        diff_tmin = tmin_pred['mean'] - hist_tmin_mean
        
        # Extract prediction bounds
        tmax_min = float(tmax_pred['min']) if pd.notna(tmax_pred.get('min')) else None
        tmax_max = float(tmax_pred['max']) if pd.notna(tmax_pred.get('max')) else None
        tmean_min = float(tmean_pred['min']) if pd.notna(tmean_pred.get('min')) else None
        tmean_max = float(tmean_pred['max']) if pd.notna(tmean_pred.get('max')) else None
        tmin_min = float(tmin_pred['min']) if pd.notna(tmin_pred.get('min')) else None
        tmin_max = float(tmin_pred['max']) if pd.notna(tmin_pred.get('max')) else None
        
        # Adjust each day individually
        for idx in indices:
            record = daily_data[idx]
            
            tmax = record.get('temperature_max')
            tmean = record.get('temperature_mean')
            tmin = record.get('temperature_min')
            
            if tmax is None or tmean is None or tmin is None:
                continue
            
            # Apply differences
            new_tmax = tmax + diff_tmax
            new_tmean = tmean + diff_tmean
            new_tmin = tmin + diff_tmin
            
            # Adjust Tmax to its bounds
            if tmax_max is not None and new_tmax > tmax_max:
                new_tmax = tmax_max
            
            if tmax_min is not None and new_tmax < tmax_min:
                new_tmax = tmax_min
            
            # Adjust Tmin to its bounds
            if tmin_max is not None and new_tmin > tmin_max:
                new_tmin = tmin_max
            
            if tmin_min is not None and new_tmin < tmin_min:
                new_tmin = tmin_min
            
            # Adjust Tmean to its bounds
            if tmean_max is not None and new_tmean > tmean_max:
                new_tmean = tmean_max
            if tmean_min is not None and new_tmean < tmean_min:
                new_tmean = tmean_min
            
            # Enforce order: Tmax >= Tmean >= Tmin
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
            
            # Save adjusted values with 1 decimal
            record['temperature_max'] = round(new_tmax, 1)
            record['temperature_mean'] = round(new_tmean, 1)
            record['temperature_min'] = round(new_tmin, 1)
    
    def _adjust_precipitation_for_month(self, daily_data: List[Dict],
                                           indices: List[int],
                                           predictions_index: Dict,
                                           year: int, month: int):
        """
        Precipitation-adjustment dispatcher for a specific month.

        If there is a 'number_days_rain' prediction, use the advanced path (Phases 0-4).
        Otherwise, use the simple path (multiplicative factor).

        Args:
            daily_data: Full list of daily records (modified in-place)
            indices: Indices of this month's days
            predictions_index: Dictionary (year, month, variable) -> {mean, min, max}
            year: Year of the month
            month: Month (1-12)
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

    # ---- simple path (precipitation only) -----------------------------------

    def _adjust_precipitation_simple(self, daily_data: List[Dict],
                                     indices: List[int],
                                     pred_prec_mean: float,
                                     prec_pred: Dict):
        """
        Simple precipitation adjustment using a multiplicative factor.

        Cases:
        - pred == 0  -> all days set to 0
        - hist == 0  -> generate 2 rain periods with prediction minimum value
        - else       -> multiplicative factor
        """
        prec_values = [daily_data[i].get('precipitation', 0) or 0 for i in indices]
        hist_prec_mean = np.mean(prec_values)
        
        # CASE 1: Prediction is 0 -> set all days to 0
        if pred_prec_mean == 0:
            for idx in indices:
                daily_data[idx]['precipitation'] = 0.0
            return
        
        # CASE 2: Historical is 0 but prediction > 0 -> generate rain periods
        # In this case, use prediction MINIMUM instead of mean
        if hist_prec_mean == 0:
            pred_prec_min = prec_pred.get('min')
            
            # If no minimum is defined, use mean as fallback
            if pred_prec_min is None or pd.isna(pred_prec_min):
                target_prec = pred_prec_mean
                target_label = "mean"
            else:
                target_prec = float(pred_prec_min)
                target_label = "minimum"

            first_date = datetime.strptime(daily_data[indices[0]]['date'], '%Y-%m-%d')
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
            safe_print(f"🌧️ Generating 2 rain periods for "
                  f"{month_names[first_date.month - 1]} {first_date.year} "
                f"({target_label}: {target_prec:.1f} mm/day)")
            self._generate_rain_periods(daily_data, indices, target_prec)
            return
        
        # CASE 3: Apply multiplicative factor
        factor = pred_prec_mean / hist_prec_mean
        for idx in indices:
            current_prec = daily_data[idx].get('precipitation', 0) or 0
            daily_data[idx]['precipitation'] = round(current_prec * factor, 1)

    # ---- advanced path (precipitation + rainy-day count) -------------------

    def _adjust_precipitation_advanced(self, daily_data: List[Dict],
                                       indices: List[int],
                                       pred_prec_mean: float,
                                       prec_pred: Dict,
                                       days_rain_pred: Dict,
                                       year: int, month: int):
        """
        Advanced precipitation adjustment in 4 phases:
            0. Trivial case: pred==0 or days==0 -> set all to 0
            1. Scan current month state
            2. Validate vs predictions (day deficit / excess)
            3. Adjust rainy-day count (add or remove)
            4. Multiplicative factor to match exact target precipitation
        """
        # Phase 0: trivial cases
        pred_days_mean = float(days_rain_pred.get('mean'))

        if pred_prec_mean == 0 or pred_days_mean == 0:
            for idx in indices:
                daily_data[idx]['precipitation'] = 0.0
            return

        # Phase 1: scan month
        analysis = self._scan_month_rainfall(daily_data, indices)

        # If there is no historical rain, generate baseline periods before day adjustment
        if analysis['rainy_days'] == 0:
            self._generate_rain_periods(daily_data, indices, pred_prec_mean)
            analysis = self._scan_month_rainfall(daily_data, indices)

        # Phase 2: validate number of days against prediction
        validation = self._validate_month_rainfall(analysis, days_rain_pred)
        days_status = validation['days_status']

        # Phase 3: adjust rainy-day count
        if days_status['deficit'] > 0:
            self._add_rainy_days(daily_data, indices, analysis, days_status['deficit'])
        elif days_status['excess'] > 0:
            self._remove_rainy_days(daily_data, indices, analysis, days_status['excess'])

        # Phase 4: multiplicative factor with combined days x precipitation matrix
        days_in_month = len(indices)

        # FINAL day-state after Phase 3
        current_days = sum(1 for idx in indices if (daily_data[idx].get('precipitation', 0) or 0) > 0)
        pred_days_min = float(days_status['min']) if days_status['min'] is not None else pred_days_mean
        pred_days_max = float(days_status['max']) if days_status['max'] is not None else pred_days_mean

        if current_days <= pred_days_min:
            days_state = "AT_MINIMUM"
        elif current_days >= pred_days_max:
            days_state = "AT_MAXIMUM"
        else:
            days_state = "IN_RANGE"

        # Target precipitation totals
        pred_prec_min = prec_pred.get('min')
        pred_prec_max = prec_pred.get('max')
        pred_prec_min = float(pred_prec_min) if pred_prec_min is not None and not pd.isna(pred_prec_min) else pred_prec_mean
        pred_prec_max = float(pred_prec_max) if pred_prec_max is not None and not pd.isna(pred_prec_max) else pred_prec_mean

        min_total  = pred_prec_min  * days_in_month
        max_total  = pred_prec_max  * days_in_month
        mean_total = pred_prec_mean * days_in_month

        actual_total = sum(daily_data[idx].get('precipitation', 0) or 0 for idx in indices)

        if actual_total < min_total:
            precip_state = "BELOW_MIN"
        elif actual_total > max_total:
            precip_state = "ABOVE_MAX"
        else:
            precip_state = "IN_RANGE"

        # Matriz combinada → target_total
        if days_state == "AT_MINIMUM":
            if precip_state == "BELOW_MIN":
                target_total = min_total
            elif precip_state == "ABOVE_MAX":
                target_total = mean_total
            else:  # IN_RANGE
                target_total = actual_total
        elif days_state == "AT_MAXIMUM":
            if precip_state == "BELOW_MIN":
                target_total = mean_total
            elif precip_state == "ABOVE_MAX":
                target_total = max_total
            else:  # IN_RANGE
                target_total = actual_total
        else:  # IN_RANGE
            if precip_state == "BELOW_MIN":
                target_total = min_total
            elif precip_state == "ABOVE_MAX":
                target_total = max_total
            else:  # IN_RANGE
                target_total = actual_total

        # No rain after removing days -> generating emergency periods
        if actual_total == 0 and target_total > 0:
            self._generate_rain_periods(daily_data, indices, pred_prec_mean)
        
        # Apply multiplicative factor to match target total
        elif actual_total > 0 and abs(target_total - actual_total) > 0.1:
            factor = target_total / actual_total
            for idx in indices:
                current_prec = daily_data[idx].get('precipitation', 0) or 0
                if current_prec > 0:
                    daily_data[idx]['precipitation'] = round(current_prec * factor, 1)

    # ---- scan and validation helpers ----------------------------------------

    def _scan_month_rainfall(self, daily_data: List[Dict],
                             indices: List[int]) -> Dict:
        """
        Phase 1: scan current monthly precipitation state.

        Returns:
            Dict with:
              total_precipitation, rainy_days, days_in_month, mean_precipitation,
              rain_series (list of {start_pos, end_pos, duration, total_precip, daily_values}),
              precip_by_day (list of values by position in indices)
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
        Phase 2: compare current state with predicted rainy-day count.

        Returns:
            Dict with days_status: {current, min, max, mean, deficit, excess,
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

    # ---- day-adjustment helpers ---------------------------------------------

    @staticmethod
    def _find_dry_gaps(precip_by_day: List[float]) -> List[Dict]:
        """Return list of {start_pos, end_pos, duration} for dry sequences."""
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
        Phase 3A: add rainy days until deficit is covered.

        Strategy:
        - If deficit == 1: try extending an existing series at one edge.
        - Remaining deficit: create up to 5-day series in available dry gaps.
        """
        precip_by_day = list(analysis['precip_by_day'])   # copia mutable
        mean_precipitation = analysis['mean_precipitation']
        deficit_remaining = deficit

        # Try extending an existing series when only 1 day is missing
        if deficit_remaining == 1 and analysis['rain_series']:
            for serie in analysis['rain_series']:
                right_pos = serie['end_pos'] + 1
                if right_pos < len(precip_by_day) and precip_by_day[right_pos] == 0:
                    new_val = max(0.1, round(mean_precipitation * random.uniform(0.5, 1.5), 1))
                    precip_by_day[right_pos] = new_val
                    daily_data[indices[right_pos]]['precipitation'] = new_val
                    deficit_remaining = 0
                    break
                left_pos = serie['start_pos'] - 1
                if left_pos >= 0 and precip_by_day[left_pos] == 0:
                    new_val = max(0.1, round(mean_precipitation * random.uniform(0.5, 1.5), 1))
                    precip_by_day[left_pos] = new_val
                    daily_data[indices[left_pos]]['precipitation'] = new_val
                    deficit_remaining = 0
                    break

        # Create series in dry gaps for remaining deficit
        series_count = 0
        while deficit_remaining > 0:
            series_size = min(5, deficit_remaining)
            series_total = mean_precipitation * series_size

            gaps = self._find_dry_gaps(precip_by_day)
            suitable = [g for g in gaps if g['duration'] >= series_size]

            if not suitable:
                # Use the largest available gap
                if not gaps:
                    break   # No gaps left
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
            deficit_remaining -= series_size

    def _remove_rainy_days(self, daily_data: List[Dict],
                           indices: List[int],
                           analysis: Dict,
                           excess: int):
        """
        Phase 3B: remove rainy days until excess is reduced.

          Strategy (in order):
          1. Remove full series with duration == excess (exact match).
              If none exists, iteratively remove small series (smallest first)
              until excess is covered. This automatically includes 1-day series.
          2. Remove edges in a distributed way among remaining series
              (sorted by ascending total_precip), keeping duration >= 2.
        """
        rain_series   = [dict(s) for s in analysis['rain_series']]
        precip_by_day = list(analysis['precip_by_day'])
        excess_remaining = excess

        def _zero_series(serie):
            for pos in range(serie['start_pos'], serie['end_pos'] + 1):
                precip_by_day[pos] = 0.0
                daily_data[indices[pos]]['precipitation'] = 0.0

        # Step 1: remove series (exact + small)
        exact = next((s for s in sorted(rain_series, key=lambda s: s['duration'])
                      if s['duration'] == excess_remaining), None)
        if exact:
            _zero_series(exact)
            rain_series.remove(exact)
            excess_remaining = 0
        else:
            # Iteratively remove small series (smallest first)
            series_eliminated = 0
            while excess_remaining > 0 and rain_series:
                candidates = [s for s in rain_series if s['duration'] <= excess_remaining]
                if not candidates:
                    break
                # Remove the smallest candidate series
                serie_to_remove = min(candidates, key=lambda s: s['duration'])
                _zero_series(serie_to_remove)
                excess_remaining -= serie_to_remove['duration']
                rain_series.remove(serie_to_remove)
                series_eliminated += 1

        if excess_remaining <= 0:
            return

        # Step 2: remove distributed edges among remaining series
        for serie in sorted(rain_series, key=lambda s: s['total_precip']):
            cur_start  = serie['start_pos']
            cur_end    = serie['end_pos']
            cur_values = list(serie['daily_values'])
            extremos_removidos = 0

            while excess_remaining > 0 and (cur_end - cur_start + 1) >= 2:
                if cur_values[0] <= cur_values[-1]:
                    precip_by_day[cur_start] = 0.0
                    daily_data[indices[cur_start]]['precipitation'] = 0.0
                    cur_start += 1
                    cur_values = cur_values[1:]
                else:
                    precip_by_day[cur_end] = 0.0
                    daily_data[indices[cur_end]]['precipitation'] = 0.0
                    cur_end -= 1
                    cur_values = cur_values[:-1]
                extremos_removidos += 1
                excess_remaining -= 1
            
            if excess_remaining <= 0:
                break

    def _generate_rain_periods(self, daily_data: List[Dict], 
                               indices: List[int], target_mean: float):
        """
        Generate 2 periods of 2-5 consecutive rainy days to reach the target mean.
        Precipitation varies randomly across days while preserving total mean.
        
        Args:
            daily_data: Full list of daily records (modified in-place)
            indices: Indices of this month's days
            target_mean: Target precipitation mean (mm/day)
        """
        # Initialize all days to 0
        for idx in indices:
            daily_data[idx]['precipitation'] = 0.0
        
        n_days = len(indices)
        if n_days == 0:
            return
        
        # Generate 2 rain periods
        total_prec_needed = target_mean * n_days
        
        # Duration of each period: random between 2 and 5 days
        period1_days = random.randint(2, min(5, n_days // 2))
        period2_days = random.randint(2, min(5, n_days // 2))
        
        total_rain_days = period1_days + period2_days
        if total_rain_days > n_days:
            period1_days = n_days // 2
            period2_days = n_days - period1_days
            total_rain_days = period1_days + period2_days
        
        # Place first period (first half of month)
        max_start1 = max(0, n_days // 2 - period1_days)
        start1 = random.randint(0, max(0, max_start1))
        
        # Place second period (second half of month)
        half_month = n_days // 2
        max_start2 = max(half_month, n_days - period2_days)
        start2 = random.randint(half_month, max_start2)
        
        # Distribute total precipitation between periods (proportional to duration)
        period1_total = total_prec_needed * (period1_days / total_rain_days)
        period2_total = total_prec_needed * (period2_days / total_rain_days)
        
        # Generate random rainfall distribution for first period
        period1_values = self._generate_variable_rainfall(period1_days, period1_total)
        
        # Generate random rainfall distribution for second period
        period2_values = self._generate_variable_rainfall(period2_days, period2_total)
        
        # Apply rain to first period
        for i in range(period1_days):
            idx = start1 + i
            if idx < len(indices):
                daily_data[indices[idx]]['precipitation'] = period1_values[i]
        
        # Apply rain to second period
        for i in range(period2_days):
            idx = start2 + i
            if idx < len(indices):
                daily_data[indices[idx]]['precipitation'] = period2_values[i]
    
    @staticmethod
    def _generate_variable_rainfall(n_days: int, total_rainfall: float) -> List[float]:
        """
        Generate a variable precipitation distribution for n days that sums exactly to the total.

        Uses a random distribution to simulate natural rainfall variability,
        where some days are more intense and others less so.
        
        Args:
            n_days: Number of rainy days
            total_rainfall: Total precipitation to distribute (mm)
            
        Returns:
            List of precipitation values for each day (rounded to 1 decimal)
        """
        if n_days <= 0:
            return []
        
        if total_rainfall <= 0:
            return [0.0] * n_days
        
        # Generate random weights using exponential distribution
        # This simulates that some days rain more intensely than others
        weights = [random.expovariate(1.0) for _ in range(n_days)]
        
        # Normalize weights so they sum to 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Distribute precipitation according to weights
        rainfall_values = [total_rainfall * w for w in normalized_weights]
        
        # Round to 1 decimal
        rounded_values = [round(val, 1) for val in rainfall_values]
        
        # Adjust to compensate rounding errors and keep exact sum
        current_sum = sum(rounded_values)
        diff = round(total_rainfall - current_sum, 1)
        
        if diff != 0:
            # Add difference to the day with highest precipitation
            max_idx = rounded_values.index(max(rounded_values))
            rounded_values[max_idx] = round(rounded_values[max_idx] + diff, 1)
        
        return rounded_values
    
    # ========================================================================
    # STEP 3: Generate hourly data from daily records
    # ========================================================================
    
    def _generate_hourly_from_daily(self, daily_data: List[Dict]) -> List[Dict]:
        """
        Generate hourly data (24 records per day) from daily records.

        Uses sinusoidal interpolation for temperature, humidity, and pressure,
        Gaussian distribution for precipitation, and a Gaussian wind profile.
        
        Args:
            daily_data: List of daily records
        
        Returns:
            List of dictionaries with hourly data
        """
        all_hourly = []
        
        for daily_rec in daily_data:
            date_label = daily_rec['date']
            
            # Temperatures
            tmin = daily_rec.get('temperature_min')
            tmax = daily_rec.get('temperature_max')
            hour_tmin = self._parse_hour(daily_rec.get('hour_tmin'))
            hour_tmax = self._parse_hour(daily_rec.get('hour_tmax'))
            
            tmean = daily_rec.get('temperature_mean')
            if tmin is not None and tmax is not None:
                hourly_temps = self._interpolate_hourly_temperature(tmin, tmax, tmean, hour_tmin, hour_tmax)
            else:
                hourly_temps = [tmean] * 24 if tmean is not None else [None] * 24
            
            # Precipitation
            prec = daily_rec.get('precipitation', 0.0) or 0.0
            hourly_precip = self._distribute_precipitation(prec)
            
            # Wind
            wind_mean = daily_rec.get('wind_speed_mean')
            wind_max = daily_rec.get('wind_speed_max')
            hour_wind = self._parse_hour(daily_rec.get('hour_wind_max'))
            hourly_wind = self._interpolate_wind_speed(wind_mean, wind_max, hour_wind)
            wind_dir = daily_rec.get('wind_direction')
            
            # Humidity
            hr_min = daily_rec.get('humidity_min')
            hr_max = daily_rec.get('humidity_max')
            hr_mean = daily_rec.get('humidity_mean')
            if hr_min is not None and hr_max is not None:
                hour_hrmin = self._parse_hour(daily_rec.get('hour_hrmin'))
                hour_hrmax = self._parse_hour(daily_rec.get('hour_hrmax'))
                hourly_humidity = self._interpolate_hourly_humidity(hr_min, hr_max, hr_mean, hour_hrmin, hour_hrmax)
            else:
                hourly_humidity = [None] * 24

            # Pressure
            pres_min = daily_rec.get('pressure_min')
            pres_max = daily_rec.get('pressure_max')
            if pres_min is not None and pres_max is not None:
                hour_presmin = self._parse_hour(daily_rec.get('hour_presmin'))
                hour_presmax = self._parse_hour(daily_rec.get('hour_presmax'))
                hourly_pressure = self._interpolate_pressure(pres_min, pres_max, hour_presmin, hour_presmax)
            else:
                hourly_pressure = [None] * 24

            # Create 24 hourly records
            for hour in range(24):
                datetime_iso = f"{date_label}T{hour:02d}:00:00Z"
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
        Verify that generated daily data matches monthly predictions.
        Debug-only method to validate mathematical consistency.

        Allowed error margins:
        - Temperature: ±1.0°C in monthly mean
        - Precipitation: ±0.5 mm in monthly mean
        
        Args:
            daily_data: List of generated daily records
            predictions_df: DataFrame with monthly predictions
            
        Returns:
            True if validation passes, False if discrepancies exist
        """
        all_pass = True
        
        # Build prediction index for O(1) lookup: (year, month, variable) -> prediction
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
        
        # Group daily data by year-month
        daily_by_month = {}
        for record in daily_data:
            date = datetime.strptime(record['date'], '%Y-%m-%d')
            month_key = (date.year, date.month)
            if month_key not in daily_by_month:
                daily_by_month[month_key] = []
            daily_by_month[month_key].append(record)
        
        # Validate each month with available daily data
        for (year, month), month_records in daily_by_month.items():
            # Validate TEMPERATURES
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
                    safe_print(f"  ⚠️ {var_label} {year}-{month:02d}: Predicted mean={pred_mean:.1f}, Actual mean={actual_mean:.1f}, Difference={diff:.2f}°C")
                    all_pass = False
                
                # Validate bounds
                pred_min = pred_data.get('min')
                if pd.notna(pred_min):
                    actual_min = min(values)
                    if actual_min < float(pred_min) - 0.1:
                        safe_print(f"  ❌  {var_label} {year}-{month:02d}: Actual minimum ({actual_min:.1f}°C) below prediction ({pred_min:.1f}°C)")
                
                pred_max = pred_data.get('max')
                if pd.notna(pred_max):
                    actual_max = max(values)
                    if actual_max > float(pred_max) + 0.1:
                        safe_print(f"  ❌  {var_label} {year}-{month:02d}: Actual maximum ({actual_max:.1f}°C) above prediction ({pred_max:.1f}°C)")
            
            # Validate PRECIPITATION
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
                        safe_print(f"  ⚠️ Precip {year}-{month:02d}: Predicted mean={pred_mean_val:.1f} mm/day, Actual mean={actual_mean:.1f} mm/day, Difference={diff:.2f} mm")
                        all_pass = False
                    
                    # Validate bounds (compare means directly)
                    pred_min = pred_data.get('min')
                    if pd.notna(pred_min):
                        pred_min_val = float(pred_min)
                        if actual_mean < pred_min_val - 0.1:
                            safe_print(f"  ❌  Precip {year}-{month:02d}: Actual mean ({actual_mean:.1f} mm/day) below predicted minimum ({pred_min_val:.1f} mm/day)")
                    
                    pred_max = pred_data.get('max')
                    if pd.notna(pred_max):
                        pred_max_val = float(pred_max)
                        if actual_mean > pred_max_val + 0.1:
                            safe_print(f"  ❌  Precip {year}-{month:02d}: Actual mean ({actual_mean:.1f} mm/day) above predicted maximum ({pred_max_val:.1f} mm/day)")
        
        return all_pass
    
    def _verify_hourly_vs_daily(self, hourly_data: List[Dict], 
                               daily_data: List[Dict]) -> bool:
        """
        Verify that hourly data matches daily data.
        Debug-only method to validate mathematical consistency.

        Allowed error margins:
        - Temperature: ±1.5°C in daily mean
        - Precipitation: ±0.5 mm
        - Humidity: ±5% in daily mean
        - Pressure: ±2 hPa in daily mean
        - Wind: ±0.5 m/s
        
        Args:
            hourly_data: List of generated hourly records
            daily_data: List of base daily records
            
        Returns:
            True if validation passes, False if discrepancies exist
        """
        all_pass = True
        
        # Group hourly data by date
        hourly_by_date = {}
        for rec in hourly_data:
            date = rec['datetime'].split('T')[0]
            if date not in hourly_by_date:
                hourly_by_date[date] = []
            hourly_by_date[date].append(rec)
        
        # Validate each day
        for daily_rec in daily_data:
            date = daily_rec['date']
            
            if date not in hourly_by_date:
                safe_print(f"  ❌ {date}: No hourly data for this day")
                all_pass = False
                continue
            
            hourly_recs = hourly_by_date[date]
            
            # Check there are 24 hourly records
            if len(hourly_recs) != 24:
                safe_print(f"  ❌ {date}: Expected 24 hourly records, found {len(hourly_recs)}")
                all_pass = False
                continue
            
            # Validate TEMPERATURE
            hourly_temps = [r.get('temperature') for r in hourly_recs 
                           if r.get('temperature') is not None]
            if hourly_temps:
                mean_hourly_temp = np.mean(hourly_temps)
                daily_tmean = daily_rec.get('temperature_mean')
                
                if daily_tmean is not None:
                    diff_temp = abs(mean_hourly_temp - daily_tmean)
                    if diff_temp > 1.5:
                        safe_print(f"  ⚠️  {date}: Hourly mean temp ({mean_hourly_temp:.1f}°C) vs daily ({daily_tmean:.1f}°C), Diff={diff_temp:.2f}°C")
                
                # Validate ranges
                hourly_tmin = min(hourly_temps)
                hourly_tmax = max(hourly_temps)
                daily_tmin = daily_rec.get('temperature_min') or hourly_tmin
                daily_tmax = daily_rec.get('temperature_max') or hourly_tmax
                
                if hourly_tmin < daily_tmin - 0.1:
                    safe_print(f"  ⚠️  {date}: Hourly Tmin ({hourly_tmin:.1f}°C) below daily ({daily_tmin:.1f}°C)")
                
                if hourly_tmax > daily_tmax + 0.1:
                    safe_print(f"  ⚠️  {date}: Hourly Tmax ({hourly_tmax:.1f}°C) above daily ({daily_tmax:.1f}°C)")
            
            # Validate PRECIPITATION
            hourly_precips = [r.get('precipitation') or 0 for r in hourly_recs]
            total_hourly_precip = sum(hourly_precips)
            daily_precip = daily_rec.get('precipitation') or 0
            
            diff_precip = abs(total_hourly_precip - daily_precip)
            if diff_precip > 0.5:
                safe_print(f"  ⚠️  {date}: Hourly total precip ({total_hourly_precip:.1f} mm) vs daily ({daily_precip:.1f} mm), Diff={diff_precip:.2f} mm")
            
            # Validate HUMIDITY
            hourly_humidities = [r.get('humidity') for r in hourly_recs 
                                if r.get('humidity') is not None]
            if hourly_humidities:
                mean_hourly_humid = np.mean(hourly_humidities)
                daily_humid_mean = daily_rec.get('humidity_mean')
                
                if daily_humid_mean is not None:
                    diff_humid = abs(mean_hourly_humid - daily_humid_mean)
                    if diff_humid > 5:
                        safe_print(f"  ⚠️  {date}: Hourly humidity mean ({mean_hourly_humid:.1f}%) vs daily ({daily_humid_mean:.1f}%), Diff={diff_humid:.1f}%")
            
            # Validate PRESSURE
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
                    safe_print(f"  ⚠️  {date}: Hourly pressure mean ({mean_hourly_pres:.1f} hPa) vs daily ({daily_pres_mean:.1f} hPa), Diff={diff_pres:.1f} hPa")
            
            # Validate WIND
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
                        safe_print(f"  ⚠️  {date}: Hourly wind mean ({mean_hourly_wind:.1f} m/s) vs daily ({daily_wind_mean:.1f} m/s), Diff={diff_wind:.2f} m/s")
                
                # Validate expected max wind gust is reached
                if daily_wind_max is not None:
                    diff_wind_max = abs(max_hourly_wind - daily_wind_max)
                    if diff_wind_max > 0.1:
                        safe_print(f"  ⚠️  {date}: Max wind gust does not reach expected value. Hourly={max_hourly_wind:.1f} m/s, Expected={daily_wind_max:.1f} m/s, Diff={diff_wind_max:.2f} m/s")
        
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
        Generate complete synthetic data (daily + hourly).

        Flow:
        1. Cycle historical daily data to generation period
        2. Adjust to monthly predictions (if provided)
        3. Apply secondary-variable correction (KNN or XGBoost)
        4. Generate hourly data
        5. Verify daily data matches monthly predictions (if provided)
        6. Verify hourly data matches daily data

        Args:
            predictions_df:    DataFrame with monthly predictions (optional).
            correction_method: Correction method for secondary variables
                               (wind, humidity, pressure). Applied only when
                               monthly predictions are provided.
                               'knn'     - K-Nearest Neighbors.
                               'xgboost' - Modelo XGBoost de ventana deslizante.

        Returns:
            Tuple (daily_data, hourly_data, total_hourly_records)
        """
        # 1. Generate synthetic daily data (historical cycle)
        daily_data = self._generate_daily_synthetic()
        safe_print(f"📅 Generated {len(daily_data)} synthetic daily records")
        
        # 2. Adjust to monthly predictions if provided
        if predictions_df is not None and not predictions_df.empty:
            daily_data = self._adjust_to_monthly_predictions(daily_data, predictions_df)

            # 2.5. Correction of secondary numeric variables (wind, humidity,
            #      pressure). Applied only when monthly predictions are provided.
            #      Non-numeric variables (wind direction, hour fields)
            #      remain as in the historical cycle.
            if correction_method == 'xgboost':
                safe_print("🔄 Applying XGBoost model (sliding window) for secondary variables...")
                xgb_model = XGBoostWeatherModel(window_size=5)
                daily_data = xgb_model.correct(
                    adjusted_data=daily_data,
                    historical_data=self.historical_data
                )
                safe_print("✅ XGBoost correction applied")
            else:  # 'knn'
                # K-Neighbors correction: adapt non-modified variables
                # (wind, humidity, pressure) using the most similar
                # historical days in temperature and precipitation.
                safe_print("🔄 Applying K-Neighbors correction for secondary variables...")
                knn_corrector = KNeighborsCorrector(k=3, month_weight=0.25)
                daily_data = knn_corrector.correct(
                    adjusted_data=daily_data,
                    historical_data=self.historical_data
                )
                safe_print("✅ K-Neighbors correction applied")
        
        # 3. Generate hourly data from daily data
        hourly_data = self._generate_hourly_from_daily(daily_data)
        safe_print(f"🕐 Generated {len(hourly_data)} hourly records")
        
        # 4. Verify generated data matches daily data (and/or predictions)
        if predictions_df is not None and not predictions_df.empty:
            safe_print("🔍 Verifying consistency of daily data vs monthly predictions...")
            if self._verify_daily_vs_predictions(daily_data, predictions_df):
                safe_print("✅ Daily data vs predictions: consistency verified")
            else:
                safe_print("⚠️ Discrepancies detected in daily data vs predictions (see errors above)")
        
        safe_print("🔍 Verifying consistency of hourly data vs daily data...")
        if self._verify_hourly_vs_daily(hourly_data, daily_data):
            safe_print("✅ Hourly data vs daily data: consistency verified")
        else:
            safe_print("⚠️ Discrepancies detected in hourly data vs daily data (see errors above)")
        
        return daily_data, hourly_data, len(hourly_data)

    def generate_and_save(
        self,
        latitude: float,
        longitude: float,
        predictions_df: Optional[pd.DataFrame] = None,
        correction_method: str = 'knn',
    ) -> Tuple[int, int, List[Dict]]:
        """
        Generate synthetic data, save it to DB, and return metadata.

        Args:
            latitude:          Location latitude.
            longitude:         Location longitude.
            predictions_df:    DataFrame with monthly predictions (optional).
            correction_method: Correction method for secondary variables
                               ('knn' o 'xgboost').

        Returns:
            Tuple (job_id, hourly_record_count, hourly_data)
        """
        # 1. Create GenerationJob entry
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
            raise ValueError("Error creating generation job in database")
        job_id = job_ids[0]
        
        # 2. Save monthly predictions if provided
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
            safe_print(f"📊 Inserted {len(pred_tuples)} monthly predictions")
        
        # 3. Generate data
        daily_data, hourly_data, hourly_count = self.generate(predictions_df, correction_method)
        
        # 4. Insert generated daily data
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
        safe_print(f"📅 Inserted {len(daily_tuples)} generated daily records")
        
        # 5. Insert generated hourly data
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
        safe_print(f"🕐 Inserted {len(hourly_tuples)} generated hourly records")
        
        safe_print(f"✅ Generation completed. Job ID: {job_id}")
        
        return job_id, hourly_count, hourly_data
    
    # ========================================================================
    # Interpolation helper methods
    # ========================================================================
    
    @staticmethod
    def _parse_hour(time_str) -> int:
        """Extract hour from an HH:MM-like string. Default: 12."""
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

        # Generate normalized base shape f(h) in [0,1]
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

        # Find exponent alpha that preserves the mean
        def compute_mean(alpha):
            temps = [
                tmin + (tmax - tmin) * (f ** alpha)
                for f in f_values
            ]
            return sum(temps) / 24

        # Stable bisection
        low, high = 0.1, 5.0

        for _ in range(50):
            mid = (low + high) / 2
            m = compute_mean(mid)

            if m > tmean:
                low = mid
            else:
                high = mid

        alpha = (low + high) / 2

        # Build final temperatures
        hourly_temps = [
            tmin + (tmax - tmin) * (f ** alpha)
            for f in f_values
        ]

        # Force exact extrema
        hourly_temps[hour_min] = tmin
        hourly_temps[hour_max] = tmax

        return [round(t, 1) for t in hourly_temps]
    
    @staticmethod
    def _interpolate_wind_speed(
        wind_avg: Optional[float], wind_max: Optional[float], hour_max: int
    ) -> List[Optional[float]]:
        """Interpolate hourly wind speed with a Gaussian peak."""
        if wind_avg is None:
            return [None] * 24

        if wind_max is None or wind_max <= wind_avg:
            return [wind_avg] * 24

        # Special case: mean = 0
        if wind_avg == 0:
            hourly = [0.0] * 24
            hourly[hour_max] = round(wind_max, 1)
            return hourly

        # Generate Gaussian shape
        base = []
        for hour in range(24):
            dist = min(abs(hour - hour_max), 24 - abs(hour - hour_max))
            factor = exp(-(dist**2) / 2)
            base.append(factor)

        # Normalize so maximum is exactly 1
        max_base = max(base)
        base = [b / max_base for b in base]

        mean_base = sum(base) / 24

        # Solve system to satisfy mean and maximum
        b = (wind_max - wind_avg) / (1 - mean_base)
        a = wind_max - b

        hourly = [max(0, round(a + b * f, 1)) for f in base]

        return hourly
    
    @staticmethod
    def _interpolate_hourly_humidity(
        hr_min: float, hr_max: float, hr_mean: Optional[float],
        hour_hrmin: int, hour_hrmax: int
    ) -> List[int]:
        """Interpolate hourly relative humidity with mean adjustment (similar to temperature).
        
        Generate a normalized cosine curve f(h) in [0,1] between hr_min and hr_max,
        and find exponent alpha so hourly mean matches hr_mean.
        
        Args:
            hr_min: Daily minimum relative humidity (%)
            hr_max: Daily maximum relative humidity (%)
            hr_mean: Daily mean relative humidity (%)
            hour_hrmin: Hour of minimum humidity
            hour_hrmax: Hour of maximum humidity
            
        Returns:
            List of 24 integer relative-humidity values (%)
        """
        if hr_mean is None:
            hr_mean = (hr_max + hr_min) / 2

        if hr_max == hr_min:
            return [int(round(hr_min))] * 24

        hours_between = (hour_hrmin - hour_hrmax) % 24
        if hours_between == 0:
            hours_between = 12

        # Generate normalized base shape f(h) in [0,1]
        # where 0 = hr_min and 1 = hr_max
        f_values = []
        for hour in range(24):
            hours_since_max = (hour - hour_hrmax) % 24

            if hours_since_max <= hours_between:
                # Decrease from maximum to minimum
                progress = hours_since_max / hours_between
                f = 1 - (1 - cos(pi * progress)) / 2
            else:
                # Increase from minimum to maximum
                hours_since_min = hours_since_max - hours_between
                hours_to_next_max = 24 - hours_between
                progress = hours_since_min / hours_to_next_max
                f = (1 - cos(pi * progress)) / 2

            f_values.append(f)

        # Find exponent alpha that preserves the mean
        def compute_mean(alpha):
            humidities = [
                hr_min + (hr_max - hr_min) * (f ** alpha)
                for f in f_values
            ]
            return sum(humidities) / 24

        # Stable bisection
        low, high = 0.1, 5.0
        for _ in range(50):
            mid = (low + high) / 2
            m = compute_mean(mid)
            if m > hr_mean:
                low = mid
            else:
                high = mid

        alpha = (low + high) / 2

        # Build final humidity values
        hourly_humidity = [
            hr_min + (hr_max - hr_min) * (f ** alpha)
            for f in f_values
        ]

        # Force exact extrema
        hourly_humidity[hour_hrmax] = hr_max
        hourly_humidity[hour_hrmin] = hr_min

        return [int(round(max(0, min(100, h)))) for h in hourly_humidity]
    
    @staticmethod
    def _distribute_precipitation(total_precip: float) -> List[float]:
        """Distribute daily precipitation across hours in a realistic way."""
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
        """Interpolate hourly atmospheric pressure with a cosine curve."""
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
