"""
Application services for map-related business logic.
"""

from functools import lru_cache
import json
from typing import Optional

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from shapely.geometry import Point, shape

from utils.system_utils import get_resource_path


@lru_cache(maxsize=1)
def load_geojson_files() -> dict:
    """
    Load and cache GeoJSON files declared in config.

    Returns:
        dict: Dictionary with source names as keys and GeoJSON data as values.
    """
    geojson_cache = {}

    try:
        config_path = get_resource_path("config/config.json")
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        for source in config.get("data_sources", []):
            geojson_path = source.get("geojson_path")
            source_name = source.get("name")

            if geojson_path and source_name:
                full_path = get_resource_path(geojson_path)
                if full_path.exists():
                    try:
                        with open(full_path, "r", encoding="utf-8") as file:
                            geojson_cache[source_name] = json.load(file)
                    except Exception as error:
                        print(f"⚠️ Error loading {geojson_path}: {error}")

    except Exception as error:
        print(f"⚠️ Error in load_geojson_files: {error}")

    return geojson_cache


def geocode_location(search_query: str) -> Optional[tuple[float, float]]:
    """
    Geocode a search query to coordinates.

    Args:
        search_query: Location to search for.

    Returns:
        tuple: (latitude, longitude) or None if not found.
    """
    try:
        geolocator = Nominatim(user_agent="MeteoSynthetic")
        location = geolocator.geocode(search_query, timeout=10)
        if location:
            return (location.latitude, location.longitude)
    except (GeocoderTimedOut, GeocoderServiceError) as error:
        print(f"⚠️ Geocoding error: {error}")
    return None


def get_data_source_for_point(lat: float, lon: float, config: dict) -> Optional[dict]:
    """
    Determine which data source covers a given point using GeoJSON boundaries.

    Args:
        lat: Latitude.
        lon: Longitude.
        config: Configuration dictionary.

    Returns:
        dict: Data source configuration or None.
    """
    point = Point(lon, lat)
    geojson_cache = load_geojson_files()

    for source in config.get("data_sources", []):
        source_name = source.get("name")

        if source_name and source_name in geojson_cache:
            geojson_data = geojson_cache[source_name]

            try:
                for feature in geojson_data.get("features", []):
                    geometry = feature.get("geometry")
                    if geometry:
                        shapely_geom = shape(geometry)
                        if shapely_geom.contains(point):
                            return source

            except Exception as error:
                print(f"⚠️ Error checking GeoJSON {source_name}: {error}")

    return None
