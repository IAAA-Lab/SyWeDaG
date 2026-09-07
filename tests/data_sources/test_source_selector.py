from data_sources.aemet_source import AemetWeatherSource
from data_sources.open_meteo_source import OpenMeteoWeatherSource
from data_sources.source_selector import get_data_source_instance

CONFIG = {
    "data_sources": [
        {"name": "AEMET", "api_url": "https://opendata.aemet.es/opendata/api", "api_key_env_var": "AEMET_API_KEY"},
        {"name": "Open-Meteo", "api_url": "https://archive-api.open-meteo.com/v1/archive"},
    ]
}


def test_returns_aemet_instance():
    instance = get_data_source_instance("AEMET", CONFIG)
    assert isinstance(instance, AemetWeatherSource)


def test_returns_open_meteo_instance():
    instance = get_data_source_instance("Open-Meteo", CONFIG)
    assert isinstance(instance, OpenMeteoWeatherSource)


def test_unknown_source_returns_none():
    assert get_data_source_instance("NotConfigured", CONFIG) is None


def test_unregistered_but_configured_source_returns_none():
    config = {"data_sources": [{"name": "SomeOtherSource", "api_url": "https://example.com"}]}
    assert get_data_source_instance("SomeOtherSource", config) is None
