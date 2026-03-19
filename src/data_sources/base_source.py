from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

@dataclass
class WeatherStation:
    """Represents a weather station"""
    source: str             # Source name (e.g., "AEMET")
    id_station: int         # Unique station ID (source-specific)
    name: str               # Station name (e.g., "Madrid-Barajas")
    region: Optional[str]   # Region or province (e.g., "Madrid")
    latitude: float         # Latitude of the station (e.g., 39.735)
    longitude: float        # Longitude of the station (e.g., -2.785)
    height: int             # Height above sea level in meters (e.g., 610)


@dataclass
class DailyWeatherRecord:
    """Daily meteorological record with all fields"""
    date: str                                   # Date (YYYY-MM-DD)
    temperature_min: Optional[float] = None     # Minimum temperature in °C (e.g., 5.2)
    temperature_max: Optional[float] = None     # Maximum temperature in °C (e.g., 15.8)
    temperature_mean: Optional[float] = None    # Mean temperature in °C (e.g., 12.3)
    hour_tmin: Optional[str] = None             # Hour of minimum temperature (HH:MM, e.g., "04:00")
    hour_tmax: Optional[str] = None             # Hour of maximum temperature (HH:MM, e.g., "14:00")
    precipitation: Optional[float] = None       # Daily precipitation in mm (e.g., 0.0)
    wind_speed_mean: Optional[float] = None     # Wind speed mean in km/h (e.g., 12.5)
    wind_speed_max: Optional[float] = None      # Wind speed max in km/h (e.g., 25.0)
    wind_direction: Optional[str] = None        # Wind direction (N, NE, E, SE, S, SW, W, NW)
    hour_wind_max: Optional[str] = None         # Hour of maximum wind speed (HH:MM, e.g., "16:00")
    humidity_min: Optional[int] = None          # Humidity minimum in % (e.g., 30)
    humidity_max: Optional[int] = None          # Humidity maximum in % (e.g., 80)
    humidity_mean: Optional[int] = None         # Humidity mean in % (e.g., 55)
    hour_hrmin: Optional[str] = None            # Hour of minimum humidity (HH:MM, e.g., "06:00")
    hour_hrmax: Optional[str] = None            # Hour of maximum humidity (HH:MM, e.g., "18:00")
    pressure_min: Optional[float] = None        # Pressure minimum in hPa (e.g., 1010.5)
    pressure_max: Optional[float] = None        # Pressure maximum in hPa (e.g., 1025.0)
    hour_presmin: Optional[str] = None          # Hour of minimum pressure (HH:MM, e.g., "08:00")
    hour_presmax: Optional[str] = None          # Hour of maximum pressure (HH:MM, e.g., "20:00")


@dataclass
class WeatherData:
    """Daily meteorological data for a specific period"""
    daily_records: List[DailyWeatherRecord]  # List of daily records


class BaseWeatherSource(ABC):
    """
    Abstract base class for all meteorological data sources.
    Defines the operations that each data source must implement.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data source.
        
        Args:
            config: Dictionary with source-specific configuration
                   (API keys, URLs, etc.)
        """
        self.config = config
        self.source_name = config.get('name', self.__class__.__name__)

    @abstractmethod
    def get_nearest_station(self, latitude: float, longitude: float) -> Optional[WeatherStation]:
        """
        Get the nearest weather station to given coordinates.
        
        Args:
            latitude: Latitude of the point
            longitude: Longitude of the point
            
        Returns:
            WeatherStation with the nearest station data,
            or None if no station is found
        """
        pass

    @abstractmethod
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
        
        At least one of the following options must be specified:
        - (latitude, longitude) for coordinate-based sources
        - station_id for station-based sources
        
        Args:
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            latitude: Latitude (optional)
            longitude: Longitude (optional)
            station_id: Weather station ID (optional)
            
        Returns:
            WeatherData with historical meteorological data:
            - datetime: List of ISO 8601 dates
            - temperature: Temperature in °C
            - precipitation: Precipitation in mm
            - wind_speed: Wind speed in km/h
            - wind_direction: Wind direction (N, NE, E, SE, S, SW, W, NW)
            - humidity: Humidity in %
            - pressure: Pressure in hPa
            
        Raises:
            ValueError: If neither latitude/longitude nor station_id is provided,
                       or if the data source cannot process the given parameters
        """
        pass

    @abstractmethod
    def get_mandatory_data(
        self,
        latitude: float,
        longitude: float,
        start_year: int,
        end_year: int
    ) -> tuple[Optional[WeatherStation], Optional[WeatherData]]:
        """
        Get the nearest station and required meteorological data.
        
        Convenience method that combines get_nearest_station() and get_weather_data()
        to obtain all required information in a single call.
        
        Args:
            latitude: Latitude of the point
            longitude: Longitude of the point
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            
        Returns:
            Tuple (WeatherStation, WeatherData) with:
            - WeatherStation: The nearest weather station (None if not found)
            - WeatherData: The historical meteorological data (None if not obtained)
            
        Raises:
            ValueError: If required data cannot be obtained
        """
        pass
