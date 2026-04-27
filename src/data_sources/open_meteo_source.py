from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

from data_sources.base_source import (
    BaseWeatherSource,
    DailyWeatherRecord,
    WeatherData,
    WeatherStation,
)
from utils.data_parsing import (
    degrees_to_cardinal,
    series_hour_of_max,
    series_hour_of_min,
    series_max,
    series_max_int,
    series_mean,
    series_mean_int,
    series_min,
    series_min_int,
    to_float_or_none,
)
from utils.system_utils import safe_print

class OpenMeteoWeatherSource(BaseWeatherSource):
    """
    Open-Meteo archive implementation.

    Uses daily + hourly variables and transforms them to the internal
    DailyWeatherRecord structure used by MeteoSynthetic.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_url = config.get("api_url", "https://archive-api.open-meteo.com/v1/archive")
        self.request_timeout = int(config.get("request_timeout_seconds", 60))

    def get_nearest_station(self, latitude: float, longitude: float) -> Optional[WeatherStation]:
        """
        Open-Meteo is grid-based (no physical station IDs in this workflow),
        so we expose a deterministic virtual station.
        """
        station_id = f"OPENMETEO_{latitude:.4f}_{longitude:.4f}"
        return WeatherStation(
            source=self.source_name,
            id_station=station_id,
            name="Open-Meteo Grid Point",
            region=None,
            latitude=float(latitude),
            longitude=float(longitude),
            height=0,
        )

    def get_weather_data(
        self,
        start_year: int,
        end_year: int,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        station_id: Optional[str] = None,
    ) -> WeatherData:
        if latitude is None or longitude is None:
            raise ValueError("Open-Meteo requires latitude and longitude")

        today = date.today()
        current = date(start_year, 1, 1)
        end_date = today if end_year == today.year else date(end_year, 12, 31)

        # Start with short chunks and reduce adaptively if needed.
        chunk_days = int(self.config.get("chunk_days", 730))
        all_records: list[DailyWeatherRecord] = []

        while current <= end_date:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
            try:
                chunk_records = self._fetch_chunk(latitude, longitude, current, chunk_end)
                all_records.extend(chunk_records)
                safe_print(
                    f"✅ Open-Meteo chunk loaded: {current.isoformat()} to {chunk_end.isoformat()}"
                )
                current = chunk_end + timedelta(days=1)
            except Exception as error:
                safe_print(
                    f"⚠️ Open-Meteo chunk failed ({current.isoformat()} to {chunk_end.isoformat()}): {error}"
                )
                # Retry same range with smaller chunks.
                if chunk_days > 7:
                    chunk_days = max(chunk_days // 2, 7)
                    safe_print(f"↘️ Reducing Open-Meteo chunk size to {chunk_days} days and retrying")
                else:
                    # Skip one week if even minimum chunk keeps failing.
                    safe_print("⚠️ Minimum chunk size reached; skipping problematic week")
                    current = min(current + timedelta(days=7), end_date + timedelta(days=1))

        if not all_records:
            raise ValueError(
                f"No Open-Meteo data obtained for ({latitude}, {longitude}) in {start_year}-{end_year}"
            )

        all_records.sort(key=lambda record: record.date)
        safe_print(f"✅ Open-Meteo total records: {len(all_records)}")
        return WeatherData(daily_records=all_records)

    def _fetch_chunk(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> list[DailyWeatherRecord]:
        """
        Fetch data from Open-Meteo Archive API using the official openmeteo-requests client.
        Uses requests_cache for persistent caching and retry_requests for resilience.
        """
        # Setup client with cache and retry logic
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": ["precipitation_sum", "wind_direction_10m_dominant"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m", "surface_pressure"],
            "timezone": "auto",
        }

        # Call the official API with flatbuffers encoding
        responses = openmeteo.weather_api(self.api_url, params=params)
        response = responses[0]

        safe_print(
            f"📍 Open-Meteo location: {response.Latitude()}°N {response.Longitude()}°E, "
            f"Elevation: {response.Elevation()}m, Timezone: {response.Timezone()}"
        )

        # Extract hourly data using Variables(index).ValuesAsNumpy()
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
        hourly_wind_gusts_10m = hourly.Variables(3).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(4).ValuesAsNumpy()

        # Create timezone-aware datetime index using UTC offset
        hourly_times = pd.date_range(
            start=pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        hourly_df = pd.DataFrame({
            "datetime": hourly_times,
            "temperature_2m": hourly_temperature_2m,
            "relative_humidity_2m": hourly_relative_humidity_2m,
            "wind_speed_10m": hourly_wind_speed_10m,
            "wind_gusts_10m": hourly_wind_gusts_10m,
            "surface_pressure": hourly_surface_pressure,
        })
        hourly_df["date"] = hourly_df["datetime"].dt.strftime("%Y-%m-%d")

        # Extract daily data using Variables(index).ValuesAsNumpy()
        daily = response.Daily()
        daily_precipitation_sum = daily.Variables(0).ValuesAsNumpy()
        daily_wind_direction_10m_dominant = daily.Variables(1).ValuesAsNumpy()

        # Create timezone-aware date index
        daily_times = pd.date_range(
            start=pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )

        daily_df = pd.DataFrame({
            "date": daily_times.strftime("%Y-%m-%d"),
            "precipitation_sum": daily_precipitation_sum,
            "wind_direction_10m_dominant": daily_wind_direction_10m_dominant,
        })

        records: list[DailyWeatherRecord] = []

        for _, daily_row in daily_df.iterrows():
            date_str = daily_row["date"]
            day_hourly = hourly_df[hourly_df["date"] == date_str]

            temperature_min = series_min(day_hourly["temperature_2m"])
            temperature_max = series_max(day_hourly["temperature_2m"])
            temperature_mean = series_mean(day_hourly["temperature_2m"])
            hour_tmin = series_hour_of_min(day_hourly, "temperature_2m")
            hour_tmax = series_hour_of_max(day_hourly, "temperature_2m")

            humidity_min = series_min_int(day_hourly["relative_humidity_2m"])
            humidity_max = series_max_int(day_hourly["relative_humidity_2m"])
            humidity_mean = series_mean_int(day_hourly["relative_humidity_2m"])
            hour_hrmin = series_hour_of_min(day_hourly, "relative_humidity_2m")
            hour_hrmax = series_hour_of_max(day_hourly, "relative_humidity_2m")

            pressure_min = series_min(day_hourly["surface_pressure"])
            pressure_max = series_max(day_hourly["surface_pressure"])
            hour_presmin = series_hour_of_min(day_hourly, "surface_pressure")
            hour_presmax = series_hour_of_max(day_hourly, "surface_pressure")

            wind_speed_mean = series_mean(day_hourly["wind_speed_10m"])
            wind_speed_max = series_max(day_hourly["wind_gusts_10m"])
            hour_wind_max = series_hour_of_max(day_hourly, "wind_gusts_10m")

            wind_direction = degrees_to_cardinal(daily_row["wind_direction_10m_dominant"])

            records.append(
                DailyWeatherRecord(
                    date=date_str,
                    temperature_min=temperature_min,
                    temperature_max=temperature_max,
                    temperature_mean=temperature_mean,
                    hour_tmin=hour_tmin,
                    hour_tmax=hour_tmax,
                    precipitation=to_float_or_none(daily_row["precipitation_sum"]),
                    wind_speed_mean=wind_speed_mean,
                    wind_speed_max=wind_speed_max,
                    wind_direction=wind_direction,
                    hour_wind_max=hour_wind_max,
                    humidity_min=humidity_min,
                    humidity_max=humidity_max,
                    humidity_mean=humidity_mean,
                    hour_hrmin=hour_hrmin,
                    hour_hrmax=hour_hrmax,
                    pressure_min=pressure_min,
                    pressure_max=pressure_max,
                    hour_presmin=hour_presmin,
                    hour_presmax=hour_presmax,
                )
            )

        return records

    def get_mandatory_data(
        self,
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int,
    ) -> tuple[Optional[WeatherStation], Optional[WeatherData]]:
        nearest_station = self.get_nearest_station(latitude, longitude)

        try:
            weather_data = self.get_weather_data(
                start_year=start_year,
                end_year=end_year,
                latitude=latitude,
                longitude=longitude,
            )
        except Exception as error:
            safe_print(f"❌ Error getting Open-Meteo historical data: {error}")
            raise ValueError("Could not obtain meteorological data from Open-Meteo")

        safe_print(f"✅ Open-Meteo data successfully obtained ({start_year}-{end_year})")
        return nearest_station, weather_data
