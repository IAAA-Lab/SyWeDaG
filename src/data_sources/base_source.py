from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dataclasses import dataclass


@dataclass
class WeatherStation:
    """Represents a weather station"""
    source: str
    id_station: int
    name: str
    region: Optional[str]
    latitude: float
    longitude: float
    height: int


@dataclass
class DailyWeatherRecord:
    """Daily meteorological record with all fields"""
    date: str                        # Date YYYY-MM-DD
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    temperature_mean: Optional[float] = None
    hour_tmin: Optional[str] = None
    hour_tmax: Optional[str] = None
    precipitation: Optional[float] = None
    wind_speed_mean: Optional[float] = None
    wind_speed_max: Optional[float] = None
    wind_direction: Optional[str] = None
    hour_wind_max: Optional[str] = None
    humidity_min: Optional[int] = None
    humidity_max: Optional[int] = None
    humidity_mean: Optional[int] = None
    hour_hrmin: Optional[str] = None
    hour_hrmax: Optional[str] = None
    pressure_min: Optional[float] = None
    pressure_max: Optional[float] = None
    hour_presmin: Optional[str] = None
    hour_presmax: Optional[str] = None


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
