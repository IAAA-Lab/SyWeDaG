"""
Application services for map-related business logic.
"""

from functools import lru_cache
import json
from typing import Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from shapely.geometry import Point, shape

from utils.system_utils import get_resource_path, safe_print

@lru_cache(maxsize=1)
def load_geojson_files() -> dict:
    """
    Load and cache GeoJSON files declared in config.

    Returns:
        dict: Dictionary with source names as keys and GeoJSON data as values.
    """
    geojson_cache = {}
    geojson_by_path = {}

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
                        path_key = str(full_path)
                        if path_key not in geojson_by_path:
                            with open(full_path, "r", encoding="utf-8") as file:
                                geojson_by_path[path_key] = json.load(file)
                        geojson_cache[source_name] = geojson_by_path[path_key]
                    except Exception as error:
                        safe_print(f"⚠️ Error loading {geojson_path}: {error}")

    except Exception as error:
        safe_print(f"⚠️ Error in load_geojson_files: {error}")

    return geojson_cache


def get_data_sources_for_point(lat: float, lon: float, config: dict) -> list[dict]:
    """
    Determine all data sources that cover a given point using GeoJSON boundaries.

    Args:
        lat: Latitude.
        lon: Longitude.
        config: Configuration dictionary.

    Returns:
        list[dict]: Matching data source configurations.
    """
    point = Point(lon, lat)
    geojson_cache = load_geojson_files()
    matching_sources = []

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
                            matching_sources.append(source)
                            break

            except Exception as error:
                safe_print(f"⚠️ Error checking GeoJSON {source_name}: {error}")

    return matching_sources


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
        safe_print(f"⚠️ Geocoding error: {error}")
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
    matching_sources = get_data_sources_for_point(lat, lon, config)
    if matching_sources:
        return matching_sources[0]
    return None
