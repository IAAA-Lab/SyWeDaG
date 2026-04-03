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

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
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
from generators.daily_correctors.k_neighbors import KNeighborsCorrector
from generators.daily_correctors.xgboost_model import XGBoostWeatherModel
from generators.monthly_adjustments.temperature_adjuster import adjust_temperatures_for_month
from generators.monthly_adjustments.precipitation_adjuster import adjust_precipitation_for_month
from generators.hourly_generation.hourly_interpolator import (
    interpolate_hourly_temperature,
    distribute_precipitation,
    interpolate_wind_speed,
    interpolate_hourly_humidity,
    interpolate_pressure,
)

class SyntheticWeatherGenerator:
    """
    Synthetic weather data generator.

    Cycles historical daily data for the generation period,
    adjusts it to monthly predictions (if provided),
    and generates interpolated hourly data.
    """
    
    def __init__(self, source: str, id_station: str, historical_start: str, 
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
            adjust_temperatures_for_month(adjusted_data, indices, predictions_index, year, month)
            
            # ============================================================
            # PRECIPITATION
            # ============================================================
            adjust_precipitation_for_month(adjusted_data, indices, predictions_index, year, month)
        
        return adjusted_data
    
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
                hourly_temps = interpolate_hourly_temperature(tmin, tmax, tmean, hour_tmin, hour_tmax)
            else:
                hourly_temps = [tmean] * 24 if tmean is not None else [None] * 24
            
            # Precipitation
            prec = daily_rec.get('precipitation', 0.0) or 0.0
            hourly_precip = distribute_precipitation(prec)
            
            # Wind
            wind_mean = daily_rec.get('wind_speed_mean')
            wind_max = daily_rec.get('wind_speed_max')
            hour_wind = self._parse_hour(daily_rec.get('hour_wind_max'))
            hourly_wind = interpolate_wind_speed(wind_mean, wind_max, hour_wind)
            wind_dir = daily_rec.get('wind_direction')
            
            # Humidity
            hr_min = daily_rec.get('humidity_min')
            hr_max = daily_rec.get('humidity_max')
            hr_mean = daily_rec.get('humidity_mean')
            if hr_min is not None and hr_max is not None:
                hour_hrmin = self._parse_hour(daily_rec.get('hour_hrmin'))
                hour_hrmax = self._parse_hour(daily_rec.get('hour_hrmax'))
                hourly_humidity = interpolate_hourly_humidity(hr_min, hr_max, hr_mean, hour_hrmin, hour_hrmax)
            else:
                hourly_humidity = [None] * 24

            # Pressure
            pres_min = daily_rec.get('pressure_min')
            pres_max = daily_rec.get('pressure_max')
            if pres_min is not None and pres_max is not None:
                hour_presmin = self._parse_hour(daily_rec.get('hour_presmin'))
                hour_presmax = self._parse_hour(daily_rec.get('hour_presmax'))
                hourly_pressure = interpolate_pressure(pres_min, pres_max, hour_presmin, hour_presmax)
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
    # Helper methods
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

