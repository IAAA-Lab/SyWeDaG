import requests
import json
import time
import os
from typing import Optional, List, Dict
from datetime import date, timedelta

from data_sources.base_source import BaseWeatherSource, WeatherStation, WeatherData, DailyWeatherRecord
from utils.data_parsing import parse_coordinates, convert_wind_direction, parse_float, parse_int
from utils.geospatial import calculate_distance_km
from utils.system_utils import safe_print

class AemetWeatherSource(BaseWeatherSource):
    """
    AEMET weather source implementation (Spanish State Meteorological Agency)
    
    Accesses public meteorological data from AEMET via its REST API.
    """

    def __init__(self, config: dict):
        """
        Initialize the AEMET data source.
        
        Args:
            config: Dictionary with configuration (must contain api_url and api_key_env_var)
                   api_key is read from environment variable specified in config
        """
        super().__init__(config)
        self.api_url = config.get('api_url', 'https://opendata.aemet.es/opendata/api')
        # Get API key from environment variable specified in config
        env_var_name = config.get('api_key_env_var', 'AEMET_API_KEY')
        self.api_key = os.getenv(env_var_name, '')

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
                        safe_print(f"⚠️ AEMET Error {status_code}. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        # Increase delay for next retries
                        retry_delay = min(retry_delay * 2, 30)  # Maximum 30 seconds
                        continue
                    else:
                        safe_print(f"❌ AEMET Error {status_code} after {max_retries} attempts: {e}")
                        return None
                else:
                    # Other HTTP errors - don't retry
                    safe_print(f"❌ AEMET HTTP Error {status_code}: {e}")
                    return None
                    
            except requests.ConnectionError as e:
                # Connection error - retry
                if attempt < max_retries - 1:
                    safe_print(f"⚠️ Connection error. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                    continue
                else:
                    safe_print(f"❌ Connection error after {max_retries} attempts: {e}")
                    return None
                    
            except requests.Timeout as e:
                # Timeout - retry
                if attempt < max_retries - 1:
                    safe_print(f"⚠️ Connection timeout. Retrying in {retry_delay} seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                    continue
                else:
                    safe_print(f"❌ Timeout after {max_retries} attempts: {e}")
                    return None
                    
            except requests.RequestException as e:
                # Other requests errors - don't retry
                safe_print(f"❌ Error fetching AEMET API: {e}")
                return None
                
            except Exception as e:
                # Other unexpected errors
                safe_print(f"❌ Unexpected error in AEMET API: {e}")
                return None
        
        return None

    def get_nearest_station(self, latitude: float, longitude: float) -> Optional[WeatherStation]:
        """
        Get the nearest weather station to given coordinates.
        
        Makes request to AEMET, calculates the nearest one and returns it.
        
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
            safe_print("❌ Unexpected format from AEMET data")
            return None

        # Calculate distances and find the nearest
        closest_station = None
        min_distance = float('inf')

        for station in stations_data:
            try:
                # Parse AEMET format coordinates (e.g. 413938N, 010015W)
                # Direction is already included in the string
                station_lat = parse_coordinates(station.get('latitud', '0'))
                station_lon = parse_coordinates(station.get('longitud', '0'))
                
                if station_lat is None or station_lon is None:
                    continue
                
                station_id = station.get('indicativo', '')
                station_name = station.get('nombre', '')
                station_height = int(station.get('altitud', 0))
                station_region = station.get('provincia', None)

                # Calculate distance
                distance = calculate_distance_km(latitude, longitude, station_lat, station_lon)

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
                safe_print(f"⚠️ Error processing station: {e}")
                continue

        if closest_station:
            safe_print(f"✅ Nearest station: {closest_station.name} at {min_distance:.2f} km")

        return closest_station

    def get_weather_data(
        self,
        start_year: int,
        end_year: int,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        station_id: Optional[str] = None
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

        all_daily_records = []

        # Get data in 6-month periods (AEMET API does not support more than 6 months)
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

            safe_print(f"📥 Getting AEMET data for {station_id} ({current_date.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')})...")
            
            period_data = self._fetch_from_aemet_api(url)

            if not period_data:
                safe_print(f"⚠️ No data obtained for period {current_date.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
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
                safe_print(f"⚠️ Unexpected format for period {current_date.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
                # Continue to next period
                if current_date.month + 6 <= 12:
                    current_date = current_date.replace(month=current_date.month + 6)
                else:
                    current_date = current_date.replace(year=current_date.year + 1, month=current_date.month + 6 - 12)
                continue

            # Store available records as returned by source
            for record_index, daily_record in enumerate(period_data):
                try:
                    fecha = daily_record.get('fecha', '')
                    tmin = parse_float(daily_record.get('tmin'))
                    tmax = parse_float(daily_record.get('tmax'))
                    tmed = parse_float(daily_record.get('tmed'))

                    hour_tmin = daily_record.get('horatmin')
                    hour_tmax = daily_record.get('horatmax')
                    
                    prec = parse_float(daily_record.get('prec'))
                    
                    wind_speed_mean = parse_float(daily_record.get('velmedia'))
                    wind_speed_max = parse_float(daily_record.get('racha'))

                    wind_dir = convert_wind_direction(daily_record.get('dir'))
                    hour_wind_max = daily_record.get('horaracha')
                    
                    hr_min = parse_int(daily_record.get('hrMin'))
                    hr_max = parse_int(daily_record.get('hrMax'))
                    hr_med = parse_int(daily_record.get('hrMedia'))

                    hour_hrmin = daily_record.get('horaHrMin')
                    hour_hrmax = daily_record.get('horaHrMax')
                    
                    pres_min = parse_float(daily_record.get('presMin'))
                    pres_max = parse_float(daily_record.get('presMax'))

                    hour_presmin = daily_record.get('horaPresMin')
                    hour_presmax = daily_record.get('horaPresMax')
                    
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
                    safe_print(f"⚠️ Error processing daily record: {e}")
                    continue

            # Move to next 6-month period
            if current_date.month + 6 <= 12:
                current_date = current_date.replace(month=current_date.month + 6)
            else:
                current_date = current_date.replace(year=current_date.year + 1, month=current_date.month + 6 - 12)

        if not all_daily_records:
            raise ValueError(f"No data obtained for station {station_id} ({start_year}-{end_year})")

        safe_print(f"✅ Obtained {len(all_daily_records)} daily records from AEMET")

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
            safe_print(f"❌ Error getting meteorological data: {e}")
            raise ValueError(f"Could not obtain meteorological data for station {nearest_station.name}")
        
        safe_print(f"✅ Data successfully obtained for {nearest_station.name} ({start_year}-{end_year})")
        
        return (nearest_station, weather_data)
