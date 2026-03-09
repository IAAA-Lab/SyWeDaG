import requests
import json
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt, pi
import numpy as np

from data_sources.base_source import BaseWeatherSource, WeatherStation, WeatherData, DailyWeatherRecord
from database.sqliteDB import insert_weather_stations, insert_historical_daily_data


class AemetWeatherSource(BaseWeatherSource):
    """
    AEMET weather source implementation (Spanish State Meteorological Agency)
    
    Accesses public meteorological data from AEMET via its REST API.
    """

    def __init__(self, config: dict):
        """
        Initialize the AEMET data source.
        
        Args:
            config: Dictionary with configuration (must contain api_url and api_key)
        """
        super().__init__(config)
        self.api_url = config.get('api_url', 'https://opendata.aemet.es/opendata/api')
        self.api_key = config.get('api_key')

    def _parse_coordinates(self, coord_str: str) -> Optional[float]:
        """
        Parse AEMET format coordinates (e.g. '413938N' or '010015W') to decimal.
        
        Format: DDMMSSd where:
        - DD = degrees
        - MM = minutes
        - SS = seconds
        - d = direction (N/S for latitude, E/W/O for longitude)
        
        Args:
            coord_str: String with the coordinate (e.g. '413938N', '010015W')
            
        Returns:
            Coordinate in decimal format, or None if parsing fails
        """
        try:
            if not coord_str:
                return None
            
            # Remove whitespace
            coord_str = str(coord_str).strip()
            
            # If already a decimal number, return directly
            if coord_str.replace('.', '').replace('-', '').isdigit():
                return float(coord_str)
            
            # Extract direction from end (N, S, E, W, O)
            if coord_str and coord_str[-1] in 'NSEWOnsawo':
                direction = coord_str[-1].upper()
                coord_value = coord_str[:-1]
            else:
                # No explicit direction, assume positive
                direction = 'N'
                coord_value = coord_str
            
            # Parse DDMMSSd format
            if len(coord_value) >= 6:
                try:
                    degrees = float(coord_value[:2])
                    minutes = float(coord_value[2:4])
                    seconds = float(coord_value[4:6])
                    
                    # Convert to decimal
                    decimal = degrees + minutes / 60 + seconds / 3600
                    
                    # Apply direction (S, W, O are negative)
                    if direction in ['S', 'W', 'O']:
                        decimal = -decimal
                    
                    return decimal
                except (ValueError, IndexError):
                    # If parsing fails, try as direct float
                    return float(coord_value)
            else:
                return float(coord_value)
                
        except (ValueError, TypeError):
            return None

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Haversine distance in km between two points.
        
        Args:
            lat1, lon1: Latitude and longitude of point 1
            lat2, lon2: Latitude and longitude of point 2
            
        Returns:
            Distance in kilometers
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return 6371 * c

    def _fetch_from_aemet_api(self, url: str) -> Optional[dict]:
        """
        Make a request to the AEMET API with automatic retries.
        
        AEMET returns a response with data that contains a URL
        to obtain the actual data.
        
        Automatically retries on:
        - Error 429 (Too Many Requests - rate limiting)
        - Error 5xx (server errors)
        
        Args:
            url: URL of the endpoint
            
        Returns:
            Dictionary with the data or None if there's an error
        """
        max_retries = 3
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                # Wait 1 second between requests to respect AEMET rate limiting
                time.sleep(1)

                querystring = {"api_key": self.api_key}
                headers = {'cache-control': 'no-cache'}
                
                response = requests.request("GET", url, headers=headers, params=querystring)
                response.raise_for_status()
                
                data = response.json()
                
                # AEMET returns data and a URL to obtain the actual data
                if 'datos' in data:
                    # Get data from the provided URL
                    data_url = data['datos']
                    data_response = requests.get(data_url)
                    data_response.raise_for_status()
                    return data_response.json()
                
                return data
                
            except requests.HTTPError as e:
                # Get status code if available
                status_code = e.response.status_code if hasattr(e, 'response') and e.response is not None else None
                
                # Retry on error 429 or 5xx
                if status_code in [429] or (status_code and 500 <= status_code < 600):
                    if attempt < max_retries - 1:
                        print(f"⚠️ AEMET Error {status_code}. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        # Increase delay for next retries
                        retry_delay = min(retry_delay * 2, 30)  # Maximum 30 seconds
                        continue
                    else:
                        print(f"❌ AEMET Error {status_code} after {max_retries} attempts: {e}")
                        return None
                else:
                    # Other HTTP errors - don't retry
                    print(f"❌ AEMET HTTP Error {status_code}: {e}")
                    return None
                    
            except requests.ConnectionError as e:
                # Connection error - retry
                if attempt < max_retries - 1:
                    print(f"⚠️ Connection error. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                    continue
                else:
                    print(f"❌ Connection error after {max_retries} attempts: {e}")
                    return None
                    
            except requests.Timeout as e:
                # Timeout - retry
                if attempt < max_retries - 1:
                    print(f"⚠️ Connection timeout. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                    continue
                else:
                    print(f"❌ Timeout after {max_retries} attempts: {e}")
                    return None
                    
            except requests.RequestException as e:
                # Other requests errors - don't retry
                print(f"❌ Error fetching AEMET API: {e}")
                return None
                
            except Exception as e:
                # Other unexpected errors
                print(f"❌ Unexpected error in AEMET API: {e}")
                return None
        
        return None

    def get_nearest_station(self, latitude: float, longitude: float) -> Optional[WeatherStation]:
        """
        Get the nearest weather station to given coordinates.
        
        Makes request to AEMET, calculates the nearest one, inserts it in DB and returns it.
        
        Args:
            latitude: Latitude of the point
            longitude: Longitude of the point
            
        Returns:
            WeatherStation with the nearest station data, or None if there's an error
        """
        # Get list of all stations
        url = f"{self.api_url}/valores/climatologicos/inventarioestaciones/todasestaciones/"
        
        stations_data = self._fetch_from_aemet_api(url)
        
        if not stations_data:
            return None

        # Convert to list if necessary
        if isinstance(stations_data, dict):
            stations_data = [stations_data]

        if not isinstance(stations_data, list):
            print("❌ Unexpected format from AEMET data")
            return None

        # Calculate distances and find the nearest
        closest_station = None
        min_distance = float('inf')
        stations_to_insert = []

        for station in stations_data:
            try:
                # Parse AEMET format coordinates (e.g. 413938N, 010015W)
                # Direction is already included in the string
                station_lat = self._parse_coordinates(station.get('latitud', '0'))
                station_lon = self._parse_coordinates(station.get('longitud', '0'))
                
                if station_lat is None or station_lon is None:
                    continue
                
                station_id = station.get('indicativo', '')
                station_name = station.get('nombre', '')
                station_height = int(station.get('altitud', 0))
                station_region = station.get('provincia', None)

                # Calculate distance
                distance = self._calculate_distance(latitude, longitude, station_lat, station_lon)

                # Save for DB insertion
                stations_to_insert.append((
                    self.source_name,           # source
                    station_id,                 # id_station (can be string)
                    station_name,               # name
                    station_region,             # region
                    station_lat,                # latitude
                    station_lon,                # longitude
                    station_height              # height
                ))

                # Find the nearest
                if distance < min_distance:
                    min_distance = distance
                    closest_station = WeatherStation(
                        source=self.source_name,
                        id_station=station_id,
                        name=station_name,
                        region=station_region,
                        latitude=station_lat,
                        longitude=station_lon,
                        height=station_height
                    )

            except (ValueError, TypeError) as e:
                print(f"⚠️ Error processing station: {e}")
                continue

        # Insert all stations in DB
        if stations_to_insert:
            try:
                insert_weather_stations(stations_to_insert)
                print(f"✅ Inserted {len(stations_to_insert)} stations in DB")
            except Exception as e:
                print(f"⚠️ Error inserting stations in DB: {e}")

        if closest_station:
            print(f"✅ Nearest station: {closest_station.name} at {min_distance:.2f} km")

        return closest_station

    def get_weather_data(
        self,
        start_year: int,
        end_year: int,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        station_id: Optional[int] = None
    ) -> WeatherData:
        """
        Get historical weather data for a period and location.
        
        Args:
            start_year: Start year
            end_year: End year
            latitude: Latitude (optional)
            longitude: Longitude (optional)
            station_id: Station ID (optional)
            
        Returns:
            WeatherData with the meteorological data
            
        Raises:
            ValueError: If neither latitude/longitude nor station_id is provided
        """
        # Validate that station identification is provided
        if station_id is None:
            raise ValueError(
                "Must provide station_id"
            )

        # Convert station_id to string if necessary (AEMET uses strings)
        station_id = str(station_id)

        all_daily_records = []

        # Get data in 6-month periods (AEMET API does not support more than 6 months)
        from datetime import date, timedelta
        
        current_date = date(start_year, 1, 1)
        
        # If final year is current, limit to today at 00h
        # If previous year, reach until 31/12
        today = date.today()
        if end_year == today.year:
            end_date_obj = today
        else:
            end_date_obj = date(end_year, 12, 31)

        while current_date <= end_date_obj:
            # Calculate period end date (maximum 6 months)
            if current_date.month + 6 <= 12:
                next_period_start = current_date.replace(month=current_date.month + 6)
            else:
                next_period_start = current_date.replace(year=current_date.year + 1, month=current_date.month + 6 - 12)
            
            # End date is the day before next period, to avoid overlaps
            from datetime import timedelta
            period_end = (next_period_start - timedelta(days=1)) if next_period_start <= end_date_obj else end_date_obj
            
            # Make sure period_end is not before current_date
            if period_end < current_date:
                period_end = end_date_obj

            # Format dates for API
            start_date_str = f"{current_date.strftime('%Y-%m-%d')}T00%3A00%3A00UTC"
            end_date_str = f"{period_end.strftime('%Y-%m-%d')}T00%3A00%3A00UTC"

            url = (
                f"{self.api_url}/valores/climatologicos/diarios/datos/"
                f"fechaini/{start_date_str}/fechafin/{end_date_str}/estacion/{station_id}"
            )

            print(f"📥 Getting AEMET data for {station_id} ({current_date.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')})...")
            
            period_data = self._fetch_from_aemet_api(url)

            if not period_data:
                print(f"⚠️ No data obtained for period {current_date.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
                # Continue to next period
                if current_date.month + 6 <= 12:
                    current_date = current_date.replace(month=current_date.month + 6)
                else:
                    current_date = current_date.replace(year=current_date.year + 1, month=current_date.month + 6 - 12)
                continue

            # Convert to list if necessary
            if isinstance(period_data, dict):
                period_data = [period_data]

            if not isinstance(period_data, list):
                print(f"⚠️ Unexpected format for period {current_date.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
                # Continue to next period
                if current_date.month + 6 <= 12:
                    current_date = current_date.replace(month=current_date.month + 6)
                else:
                    current_date = current_date.replace(year=current_date.year + 1, month=current_date.month + 6 - 12)
                continue

            # 1. Fill missing days by interpolating
            complete_period_data = self._fill_missing_days(period_data)
            
            # 2. Interpolate missing values (NULLs) within existing days
            self._interpolate_missing_values_in_period(complete_period_data)
            
            # 3. Store daily records directly (without hourly interpolation)
            for record_index, daily_record in enumerate(complete_period_data):
                try:
                    fecha = daily_record.get('fecha', '')
                    tmin = self._parse_float(daily_record.get('tmin'))
                    tmax = self._parse_float(daily_record.get('tmax'))
                    tmed = self._parse_float(daily_record.get('tmed'))
                    hour_tmin = self._process_hour_field(daily_record.get('horatmin'), 'horatmin', complete_period_data, record_index)
                    hour_tmax = self._process_hour_field(daily_record.get('horatmax'), 'horatmax', complete_period_data, record_index)
                    
                    prec = self._parse_float(daily_record.get('prec'))
                    
                    wind_speed_mean = self._parse_float(daily_record.get('velmedia'))
                    wind_speed_max = self._parse_float(daily_record.get('racha'))
                    wind_dir = self._convert_wind_direction(daily_record.get('dir'))
                    hour_wind_max = self._process_hour_field(daily_record.get('horaracha'), 'horaracha', complete_period_data, record_index)
                    
                    hr_min = self._parse_int(daily_record.get('hrMin'))
                    hr_max = self._parse_int(daily_record.get('hrMax'))
                    hr_med = self._parse_int(daily_record.get('hrMedia'))
                    hour_hrmin = self._process_hour_field(daily_record.get('horaHrMin'), 'horaHrMin', complete_period_data, record_index)
                    hour_hrmax = self._process_hour_field(daily_record.get('horaHrMax'), 'horaHrMax', complete_period_data, record_index)
                    
                    pres_min = self._parse_float(daily_record.get('presMin'))
                    pres_max = self._parse_float(daily_record.get('presMax'))
                    hour_presmin = self._process_hour_field(daily_record.get('horaPresMin'), 'horaPresMin', complete_period_data, record_index)
                    hour_presmax = self._process_hour_field(daily_record.get('horaPresMax'), 'horaPresMax', complete_period_data, record_index)
                    
                    record = DailyWeatherRecord(
                        date=fecha,
                        temperature_min=tmin,
                        temperature_max=tmax,
                        temperature_mean=tmed,
                        hour_tmin=hour_tmin,
                        hour_tmax=hour_tmax,
                        precipitation=prec,
                        wind_speed_mean=wind_speed_mean,
                        wind_speed_max=wind_speed_max,
                        wind_direction=wind_dir,
                        hour_wind_max=hour_wind_max,
                        humidity_min=hr_min,
                        humidity_max=hr_max,
                        humidity_mean=hr_med,
                        hour_hrmin=hour_hrmin,
                        hour_hrmax=hour_hrmax,
                        pressure_min=pres_min,
                        pressure_max=pres_max,
                        hour_presmin=hour_presmin,
                        hour_presmax=hour_presmax
                    )
                    all_daily_records.append(record)

                except (ValueError, TypeError, KeyError) as e:
                    print(f"⚠️ Error processing daily record: {e}")
                    continue

            # Move to next 6-month period
            if current_date.month + 6 <= 12:
                current_date = current_date.replace(month=current_date.month + 6)
            else:
                current_date = current_date.replace(year=current_date.year + 1, month=current_date.month + 6 - 12)

        if not all_daily_records:
            raise ValueError(f"No data obtained for station {station_id} ({start_year}-{end_year})")

        print(f"✅ Obtained {len(all_daily_records)} daily records from AEMET")

        # Insert daily historical data in the database
        historical_tuples = []
        for rec in all_daily_records:
            historical_tuples.append((
                rec.date,                # date
                self.source_name,        # source
                station_id,              # id_station
                rec.temperature_min,     # temperature_min
                rec.temperature_max,     # temperature_max
                rec.temperature_mean,    # temperature_mean
                rec.hour_tmin,           # hour_tmin
                rec.hour_tmax,           # hour_tmax
                rec.precipitation,       # precipitation
                rec.wind_speed_mean,     # wind_speed_mean
                rec.wind_speed_max,      # wind_speed_max
                rec.wind_direction,      # wind_direction
                rec.hour_wind_max,       # hour_wind_max
                rec.humidity_min,        # humidity_min
                rec.humidity_max,        # humidity_max
                rec.humidity_mean,       # humidity_mean
                rec.hour_hrmin,          # hour_hrmin
                rec.hour_hrmax,          # hour_hrmax
                rec.pressure_min,        # pressure_min
                rec.pressure_max,        # pressure_max
                rec.hour_presmin,        # hour_presmin
                rec.hour_presmax         # hour_presmax
            ))
        
        if historical_tuples:
            try:
                rows_inserted = insert_historical_daily_data(historical_tuples)
                print(f"✅ Inserted {rows_inserted} daily historical records in DB")
            except Exception as e:
                print(f"⚠️ Error inserting historical data: {e}")

        return WeatherData(daily_records=all_daily_records)

    def get_mandatory_data(
        self,
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int
    ) -> tuple[Optional[WeatherStation], Optional[WeatherData]]:
        """
        Get the nearest station and required meteorological data.
        
        Combines get_nearest_station() and get_weather_data() in a single call.
        
        Args:
            latitude: Latitude of the point
            longitude: Longitude of the point
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            
        Returns:
            Tuple (WeatherStation, WeatherData) with the nearest station and its data
            
        Raises:
            ValueError: If required data cannot be obtained
        """
        # Get the nearest station
        nearest_station = self.get_nearest_station(latitude, longitude)
        
        if nearest_station is None:
            raise ValueError(f"No weather station found near ({latitude}, {longitude})")
        
        # Get meteorological data for the station
        try:
            weather_data = self.get_weather_data(
                start_year=start_year,
                end_year=end_year,
                station_id=nearest_station.id_station
            )
        except Exception as e:
            print(f"❌ Error getting meteorological data: {e}")
            raise ValueError(f"Could not obtain meteorological data for station {nearest_station.name}")
        
        print(f"✅ Data successfully obtained for {nearest_station.name} ({start_year}-{end_year})")
        
        return (nearest_station, weather_data)

    @staticmethod
    def _convert_wind_direction(direction_value) -> Optional[str]:
        """
        Convert wind direction from AEMET format (degrees in tens) to cardinal.
        
        AEMET Format: values 0-35 represent 0° to 350° (each unit = 10°)
        - 99 = variable direction
        - 88 = no data
        
        Args:
            direction_value: Numeric or string direction value
            
        Returns:
            Cardinal direction (N, NE, E, SE, S, SW, W, NW) or 'N' by default
        """
        try:
            if direction_value is None or direction_value == '':
                return None
            # Convert to int
            if isinstance(direction_value, str):
                value = int(direction_value.replace(',', '.').split('.')[0])
            else:
                value = int(float(direction_value) if direction_value else 0)
            
            # Special cases
            if value == 99 or value == 88:  # Variable or no data
                return 'N'
            
            # Map value (0-35, each = 10°) to cardinal direction
            # Divide circle into 8 directions
            # N: 348.75° to 11.25° -> values 35, 0, 1
            # NE: 11.25° to 56.25° -> values 2, 3, 4, 5
            # E: 56.25° to 101.25° -> values 6, 7, 8, 9, 10
            # SE: 101.25° to 146.25° -> values 11, 12, 13, 14
            # S: 146.25° to 191.25° -> values 15, 16, 17, 18, 19
            # SW: 191.25° to 236.25° -> values 20, 21, 22, 23
            # W: 236.25° to 281.25° -> values 24, 25, 26, 27
            # NW: 281.25° to 326.25° -> values 28, 29, 30, 31, 32, 33, 34
            
            if value in [0, 1, 2, 34, 35]:
                return 'N'
            elif value in [3, 4, 5, 6]:
                return 'NE'
            elif value in [7, 8, 9, 10]:
                return 'E'
            elif value in [11, 12, 13, 14, 15]:
                return 'SE'
            elif value in [16, 17, 18, 19]:
                return 'S'
            elif value in [20, 21, 22, 23, 24]:
                return 'SO'
            elif value in [25, 26, 27, 28]:
                return 'O'
            elif value in [29, 30, 31, 32, 33]:
                return 'NO'
            else:
                return 'N'
                
        except (ValueError, TypeError):
            return 'N'

    @staticmethod
    def _parse_float(value) -> Optional[float]:
        """Convert value to float, returning None if it fails or is an AEMET special value"""
        try:
            if value is None or value == '':
                return None
            
            # Convert to string for processing
            value_str = str(value).strip()
            
            # Handle AEMET special values
            if value_str.upper() in ['IP', 'IND', 'VV', 'N/A', 'ND']:
                # IP = negligible (< 0.1mm), treat as 0.0 for precipitation
                if value_str.upper() == 'IP':
                    return 0.0
                return None
            
            # Replace Spanish decimal comma with period
            value_str = value_str.replace(',', '.')
            
            return float(value_str)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_int(value) -> Optional[int]:
        """Convert value to int, returning None if it fails"""
        try:
            if value is None or value == '':
                return None
            
            value_str = str(value).strip()
            
            # Handle special values
            if value_str.upper() in ['IP', 'IND', 'VV', 'N/A', 'ND']:
                return None
            
            # Replace decimal comma with period and convert to int
            value_str = value_str.replace(',', '.')
            return int(float(value_str))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_time(time_str: str, time_type: str) -> str:
        """
        Apply a sensible time value for AEMET data when it is "Multiple" (Varias).
        
        Returns a sensible hour based on data type. 
        - 'horatmin': Hour when minimum temperature occurred (early morning, typically ~4 AM),
        - 'horatmax': Hour when maximum temperature occurred (afternoon, typically ~14:00 / 2 PM),
        - 'horaracha': Hour when maximum wind gust (racha) occurred (neutral hour, typically ~12 PM),
        - 'horahrmin': Hour when minimum humidity occurred (afternoon, typically ~15:00 / 3 PM, coincides with max temperature),
        - 'horahrmax': Hour when maximum humidity occurred (early morning, typically ~5 AM, coincides with min temperature),
        - 'horaPresMin': Hour when minimum atmospheric pressure occurred (afternoon/evening, typically ~18:00 / 6 PM),
        - 'horaPresMax': Hour when maximum atmospheric pressure occurred (early morning, typically ~6 AM)
        
        Default hours used:
        - 'horatmin': 4 (early morning - when minimum temperatures typically occur)
        - 'horatmax': 14 (afternoon - when maximum temperatures typically occur)
        - 'horaracha': 12 (noon - neutral hour for wind gusts)
        - 'horahrmin': 15 (afternoon - when minimum humidity occurs, with maximum temperatures)
        - 'horahrmax': 5 (early morning - when maximum humidity occurs, with minimum temperatures)
        - 'horaPresMin': 18 (afternoon/evening - when minimum pressure occurs)
        - 'horaPresMax': 6 (early morning - when maximum pressure occurs)
        
        Args:
            time_str: String with the hour (e.g "03:36", "Multiple" or "Varias")
            time_type: Meteorological data type ('horatmin', 'horatmax', 'horaracha', 'horahrmin', 'horahrmax', 'horaPresMin', 'horaPresMax')
            
        Returns:
            Hour as string 'HH:00'.
        """
        # Default hours based on data type
        default_hours = {
            'horatmin': 4,      # Early morning
            'horatmax': 14,     # Afternoon
            'horaracha': 12,    # Noon
            'horahrmin': 15,    # Afternoon (minimum humidity)
            'horahrmax': 5,     # Early morning (maximum humidity)
            'horaPresMin': 18,  # Afternoon/evening (minimum pressure)
            'horaPresMax': 6    # Early morning (maximum pressure)
        }
        
        default_hour = default_hours.get(time_type, 12)
        
        try:
            if not time_str or time_str.lower() == 'varias':
                return f"{default_hour:02d}:00"
            else:
                return time_str
            
        except (ValueError, IndexError):
            return time_str

    def _get_hour_from_nearest_day(self, daily_records: List[dict], current_index: int, hour_field: str) -> Optional[str]:
        """
        Get hour from the nearest day (previous or following) for a specific hour field.
        
        First searches backwards from the current index to find a valid hour value
        (not None and not 'Varias'/'Multiple'). If not found, searches forwards.
        
        Args:
            daily_records: Complete list of daily records
            current_index: Index of current record
            hour_field: Name of the hour field (e.g., 'horatmin', 'horatmax')
            
        Returns:
            Hour string in format 'HH:MM' or None if no valid hour found
        """
        # Search backwards from current position
        for prev_index in range(current_index - 1, -1, -1):
            prev_record = daily_records[prev_index]
            hour_value = prev_record.get(hour_field)
            
            # Check if value exists and is not 'Varias'/'Multiple'
            if hour_value and hour_value.lower() not in ['varias', 'multiple', '', 'n/a', 'nd']:
                return hour_value
        
        # If not found backwards, search forwards from current position
        for next_index in range(current_index + 1, len(daily_records)):
            next_record = daily_records[next_index]
            hour_value = next_record.get(hour_field)
            
            # Check if value exists and is not 'Varias'/'Multiple'
            if hour_value and hour_value.lower() not in ['varias', 'multiple', '', 'n/a', 'nd']:
                return hour_value
        
        return None

    def _process_hour_field(self, hour_raw: str, hour_type: str, daily_records: List[dict] = None, current_index: int = None) -> Optional[str]:
        """
        Process hour field with intelligent logic:
        - If None/empty → search for value in nearest neighbors (backward first, then forward)
        - If valid format (e.g., '03:36') → return as is
        - If 'Varias'/'Multiple' → search for value in nearest neighbors
        - If no valid value found anywhere → return None
        
        Args:
            hour_raw: Raw hour value from AEMET data
            hour_type: Type of hour field for context
            daily_records: Complete list of daily records (for interpolation)
            current_index: Current index in daily_records (for interpolation)
            
        Returns:
            Hour string 'HH:MM' or None
        """
        # If it's a valid time format, return as is
        if hour_raw and hour_raw != '':
            hour_str_lower = str(hour_raw).lower().strip()
            if hour_str_lower not in ['varias', 'multiple', 'n/a', 'nd', 'ind', 'vv']:
                return hour_raw
        
        # If None/empty or 'Varias'/'Multiple', search in neighbors
        if daily_records is not None and current_index is not None:
            nearest_hour = self._get_hour_from_nearest_day(daily_records, current_index, hour_type)
            if nearest_hour:
                return nearest_hour
        
        return None

    def _fill_missing_days(self, daily_records: List[dict]) -> List[dict]:
        """
        Fill missing days by interpolating between available days or extrapolating at extremes.
        
        If there are previous and next days, interpolates between them.
        If only previous day exists (e.g. final extreme), extrapolates using that day.
        If only next day exists (e.g. initial extreme), extrapolates using that day.
        
        Args:
            daily_records: List of daily records (may have missing days)
            
        Returns:
            Complete list of daily records without gaps
        """
        if not daily_records:
            return []
        
        # Sort by date
        daily_records = sorted(daily_records, key=lambda x: x.get('fecha', ''))
        
        # Create dictionary of records by date
        records_by_date = {r['fecha']: r for r in daily_records}
        
        # Find date range
        start_date = datetime.strptime(daily_records[0]['fecha'], '%Y-%m-%d').date()
        end_date = datetime.strptime(daily_records[-1]['fecha'], '%Y-%m-%d').date()
        
        # Generate all dates in range
        complete_records = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            
            if date_str in records_by_date:
                # Existing day
                complete_records.append(records_by_date[date_str])
            else:
                # Missing day - search for nearest previous and next days
                prev_date = current_date - timedelta(days=1)
                next_date = current_date + timedelta(days=1)
                
                # Search backwards
                while prev_date >= start_date and prev_date.strftime('%Y-%m-%d') not in records_by_date:
                    prev_date -= timedelta(days=1)
                
                # Search forwards
                while next_date <= end_date and next_date.strftime('%Y-%m-%d') not in records_by_date:
                    next_date += timedelta(days=1)
                
                # Determine if records exist before and after
                has_prev = prev_date >= start_date
                has_next = next_date <= end_date
                
                if has_prev and has_next:
                    # Interpolate using both records
                    prev_record = records_by_date[prev_date.strftime('%Y-%m-%d')]
                    next_record = records_by_date[next_date.strftime('%Y-%m-%d')]
                    
                    filled_record = self._interpolate_daily_record(
                        prev_record, 
                        next_record, 
                        date_str
                    )
                    complete_records.append(filled_record)
                elif has_prev:
                    # Extrapolate forwards using previous day
                    prev_record = records_by_date[prev_date.strftime('%Y-%m-%d')]
                    filled_record = self._extrapolate_daily_record(prev_record, date_str)
                    complete_records.append(filled_record)
                elif has_next:
                    # Extrapolate backwards using next day
                    next_record = records_by_date[next_date.strftime('%Y-%m-%d')]
                    filled_record = self._extrapolate_daily_record(next_record, date_str)
                    complete_records.append(filled_record)
            
            current_date += timedelta(days=1)
        
        return complete_records

    def _extrapolate_daily_record(self, reference_record: dict, target_date: str) -> dict:
        """
        Extrapolate a daily record using a reference one when days are missing at extremes.
        
        When there are no previous or next days available (e.g. 1-2 January, 31 December),
        uses the values from the nearest available day as reference for the missing day.
        
        Args:
            reference_record: Record from the nearest known day
            target_date: Date of the day to extrapolate (format YYYY-MM-DD)
            
        Returns:
            Extrapolated daily record with values similar to the reference
        """
        # Copy the reference record maintaining all its meteorological values
        extrapolated = reference_record.copy()
        extrapolated['fecha'] = target_date
        
        # Meteorological values are maintained from the reference day
        # This is a simple/conservative extrapolation that assumes similar conditions
        
        return extrapolated

    def _interpolate_daily_record(
        self, 
        prev_record: dict, 
        next_record: dict, 
        target_date: str
    ) -> dict:
        """
        Linearly interpolate a daily record between two known records.
        
        For each variable:
        - If both records have values: returns the average
        - If only one has a value: returns that value
        - If neither has a value: returns None
        
        Args:
            prev_record: Previous day record
            next_record: Next day record
            target_date: Date of the day to interpolate (format YYYY-MM-DD)
            
        Returns:
            Interpolated daily record
        """
        def interpolate_value(key, parse_as='float'):
            """
            Interpolate a value between two records.
            If both have values: average them.
            If one has value: use it.
            If neither has value: return None.
            """
            prev_val_raw = prev_record.get(key)
            next_val_raw = next_record.get(key)
            
            if parse_as == 'float':
                prev_val = self._parse_float(prev_val_raw)
                next_val = self._parse_float(next_val_raw)
                
                if prev_val is not None and next_val is not None:
                    return str(round((prev_val + next_val) / 2, 1))
                elif prev_val is not None:
                    return str(round(prev_val, 1))
                elif next_val is not None:
                    return str(round(next_val, 1))
                else:
                    return None
            
            elif parse_as == 'int':
                prev_val = self._parse_int(prev_val_raw)
                next_val = self._parse_int(next_val_raw)
                
                if prev_val is not None and next_val is not None:
                    return str(int((prev_val + next_val) / 2))
                elif prev_val is not None:
                    return str(prev_val)
                elif next_val is not None:
                    return str(next_val)
                else:
                    return None
            
            else:  # 'string' - take from prev if exists, else from next
                if prev_val_raw is not None and prev_val_raw != '':
                    return prev_val_raw
                elif next_val_raw is not None and next_val_raw != '':
                    return next_val_raw
                else:
                    return None
        
        interpolated_record = {
            'fecha': target_date,
            'indicativo': prev_record.get('indicativo') or next_record.get('indicativo'),
            'nombre': prev_record.get('nombre') or next_record.get('nombre'),
            'provincia': prev_record.get('provincia') or next_record.get('provincia'),
            'altitud': prev_record.get('altitud') or next_record.get('altitud'),
        }
        
        # Temperature
        interpolated_record['tmed'] = interpolate_value('tmed', 'float')
        interpolated_record['tmin'] = interpolate_value('tmin', 'float')
        interpolated_record['tmax'] = interpolate_value('tmax', 'float')
        
        # Hour of temperature extremes
        interpolated_record['horatmin'] = interpolate_value('horatmin', 'string')
        interpolated_record['horatmax'] = interpolate_value('horatmax', 'string')
        
        # Precipitation
        interpolated_record['prec'] = interpolate_value('prec', 'float')
        
        # Wind
        interpolated_record['dir'] = interpolate_value('dir', 'string')
        interpolated_record['velmedia'] = interpolate_value('velmedia', 'float')
        interpolated_record['racha'] = interpolate_value('racha', 'float')
        interpolated_record['horaracha'] = interpolate_value('horaracha', 'string')
        
        # Humidity
        interpolated_record['hrmedia'] = interpolate_value('hrmedia', 'int')
        interpolated_record['hrmin'] = interpolate_value('hrmin', 'int')
        interpolated_record['hrmax'] = interpolate_value('hrmax', 'int')
        interpolated_record['horahrmin'] = interpolate_value('horahrmin', 'string')
        interpolated_record['horahrmax'] = interpolate_value('horahrmax', 'string')
        
        # Pressure
        interpolated_record['presmin'] = interpolate_value('presmin', 'float')
        interpolated_record['presmax'] = interpolate_value('presmax', 'float')
        interpolated_record['horapresmin'] = interpolate_value('horapresmin', 'string')
        interpolated_record['horapresmax'] = interpolate_value('horapresmax', 'string')
        
        return interpolated_record

    def _interpolate_missing_values_in_period(self, daily_records: List[dict]) -> None:
        """
        Interpola valores faltantes (NULLs) dentro de días existentes.
        
        Detecta cuales variables nunca se devuelven (para no interpolarlas)
        y para las variables que sí se devuelven pero tienen NULLs,
        interpola esos valores usando días cercanos.
        
        Si hay días anteriores y posteriores, interpola entre ellos.
        Si solo hay día anterior, extrapola hacia adelante usando ese día.
        Si solo hay día posterior, extrapola hacia atrás usando ese día.
        Si no hay días con valores cercanos, el NULL se mantiene.
        
        Args:
            daily_records: Lista de registros diarios (modificada in-place)
        """
        if not daily_records or len(daily_records) < 2:
            return
        
        # Map de variables numéricas que pueden ser interpoladas (varios nombres posibles)
        # Cada variable se mapea a sus posibles nombres en las claves (para manejar mayúsculas/minúsculas)
        numeric_vars_keys = {
            'tmin': ['tmin'],
            'tmax': ['tmax'],
            'tmed': ['tmed'],
            'prec': ['prec'],
            'velmedia': ['velmedia', 'velMedia'],
            'racha': ['racha'],
            'hrmin': ['hrmin', 'hrMin'],
            'hrmax': ['hrmax', 'hrMax'],
            'hrmedia': ['hrmedia', 'hrMedia'],
            'presmin': ['presmin', 'presMin'],
            'presmax': ['presmax', 'presMax']
        }
        
        # Detectar qué claves reales existen en los registros
        actual_keys_used = {}
        for var, possible_keys in numeric_vars_keys.items():
            actual_key = None
            for record in daily_records:
                for key in possible_keys:
                    if key in record:
                        actual_key = key
                        break
                if actual_key:
                    break
            if actual_key:
                actual_keys_used[var] = actual_key
        
        # Detectar variables que NUNCA se devuelven (siempre NULL)
        never_present = set()
        for var, actual_key in actual_keys_used.items():
            has_any_value = False
            for record in daily_records:
                val = record.get(actual_key)
                if val is not None and val != '' and self._parse_float(val) is not None:
                    has_any_value = True
                    break
            if not has_any_value:
                never_present.add(var)
        
        if never_present:
            print(f"⚠️ Variables without data in period (not interpolating): {', '.join(sorted(never_present))}")
        
        # Para cada variable que SÍ se devuelve, interpolar sus NULLs
        for var, actual_key in actual_keys_used.items():
            if var in never_present:
                continue  # No interpolar variables que nunca se devuelven
            
            # Buscar NULLs en esta variable
            for i in range(len(daily_records)):
                val = daily_records[i].get(actual_key)
                parsed_val = self._parse_float(val)
                
                if parsed_val is None:  # Hay un NULL o valor inválido
                    # Buscar valores anteriores y posteriores
                    prev_val = None
                    next_val = None
                    
                    # Buscar hacia atrás para encontrar el valor anterior más cercano
                    for j in range(i - 1, -1, -1):
                        prev_parsed = self._parse_float(daily_records[j].get(actual_key))
                        if prev_parsed is not None:
                            prev_val = prev_parsed
                            break
                    
                    # Buscar hacia adelante para encontrar el valor posterior más cercano
                    for j in range(i + 1, len(daily_records)):
                        next_parsed = self._parse_float(daily_records[j].get(actual_key))
                        if next_parsed is not None:
                            next_val = next_parsed
                            break
                    
                    # Interpolar o extrapolar según lo que se encuentre
                    if prev_val is not None and next_val is not None:
                        # Interpolación lineal simple (promedio de los dos valores)
                        interpolated = (prev_val + next_val) / 2
                        daily_records[i][actual_key] = str(round(interpolated, 1))
                    elif prev_val is not None:
                        # Extrapolar hacia adelante usando el valor anterior
                        daily_records[i][actual_key] = str(prev_val)
                    elif next_val is not None:
                        # Extrapolar hacia atrás usando el valor posterior
                        daily_records[i][actual_key] = str(next_val)
                    # Si no hay prev ni next, el NULL se mantiene
