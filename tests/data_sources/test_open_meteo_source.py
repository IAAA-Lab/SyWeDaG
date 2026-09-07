import pytest

from data_sources.base_source import DailyWeatherRecord
from data_sources.open_meteo_source import OpenMeteoWeatherSource


def _make_source():
    return OpenMeteoWeatherSource({"api_url": "https://fake.example/archive"})


def test_get_nearest_station_is_deterministic_virtual_station():
    source = _make_source()
    station = source.get_nearest_station(40.4168, -3.7038)

    assert station.source == source.source_name
    assert station.latitude == 40.4168
    assert station.longitude == -3.7038
    assert "OPENMETEO" in station.id_station


def test_get_weather_data_requires_coordinates():
    source = _make_source()
    with pytest.raises(ValueError):
        source.get_weather_data(2021, 2021)


def test_get_weather_data_sorts_and_aggregates_chunks(monkeypatch):
    source = _make_source()

    def fake_fetch_chunk(latitude, longitude, start_date, end_date):
        return [
            DailyWeatherRecord(date=end_date.isoformat(), temperature_mean=10.0),
            DailyWeatherRecord(date=start_date.isoformat(), temperature_mean=5.0),
        ]

    monkeypatch.setattr(source, "_fetch_chunk", fake_fetch_chunk)

    weather = source.get_weather_data(2021, 2021, latitude=40.0, longitude=-3.0)

    dates = [record.date for record in weather.daily_records]
    assert dates == sorted(dates)


def test_get_weather_data_raises_when_all_chunks_fail(monkeypatch):
    source = _make_source()

    def failing_fetch_chunk(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(source, "_fetch_chunk", failing_fetch_chunk)

    with pytest.raises(ValueError):
        source.get_weather_data(2021, 2021, latitude=40.0, longitude=-3.0)
