import pytest

from data_sources.aemet_source import AemetWeatherSource


def _make_source():
    return AemetWeatherSource({"api_url": "https://fake.example/api", "api_key_env_var": "AEMET_API_KEY"})


def test_get_nearest_station_picks_closest(monkeypatch):
    source = _make_source()
    stations_payload = [
        {"indicativo": "A", "nombre": "Near Station", "provincia": "Madrid",
         "altitud": "600", "latitud": "403000N", "longitud": "003000W"},
        {"indicativo": "B", "nombre": "Far Station", "provincia": "Barcelona",
         "altitud": "10", "latitud": "413000N", "longitud": "021000E"},
    ]
    monkeypatch.setattr(source, "_fetch_from_aemet_api", lambda url: stations_payload)

    station = source.get_nearest_station(40.5, -0.5)

    assert station is not None
    assert station.id_station == "A"
    assert station.name == "Near Station"


def test_get_nearest_station_returns_none_on_empty_response(monkeypatch):
    source = _make_source()
    monkeypatch.setattr(source, "_fetch_from_aemet_api", lambda url: None)

    assert source.get_nearest_station(40.5, -0.5) is None


def test_get_weather_data_requires_station_id():
    source = _make_source()
    with pytest.raises(ValueError):
        source.get_weather_data(2021, 2021)


def test_get_weather_data_parses_records(monkeypatch):
    source = _make_source()
    payload = [{
        "fecha": "2021-01-01",
        "tmin": "5,0", "tmax": "15,0", "tmed": "10,0",
        "horatmin": "06:00", "horatmax": "16:00",
        "prec": "0,0",
        "velmedia": "10,0", "racha": "20,0", "dir": "8", "horaracha": "14:00",
        "hrMin": "40", "hrMax": "70", "hrMedia": "55",
        "horaHrMin": "15:00", "horaHrMax": "05:00",
        "presMin": "1010,0", "presMax": "1020,0",
        "horaPresMin": "13:00", "horaPresMax": "03:00",
    }]
    monkeypatch.setattr(source, "_fetch_from_aemet_api", lambda url: payload)

    weather = source.get_weather_data(2021, 2021, station_id="3195")

    assert len(weather.daily_records) >= 1
    record = weather.daily_records[0]
    assert record.date == "2021-01-01"
    assert record.temperature_min == 5.0
    assert record.temperature_max == 15.0
    assert record.wind_direction == "E"
    assert record.humidity_min == 40
    assert record.pressure_min == 1010.0


def test_get_weather_data_raises_when_no_records(monkeypatch):
    source = _make_source()
    monkeypatch.setattr(source, "_fetch_from_aemet_api", lambda url: None)

    with pytest.raises(ValueError):
        source.get_weather_data(2021, 2021, station_id="3195")


def test_get_mandatory_data_raises_when_no_station(monkeypatch):
    source = _make_source()
    monkeypatch.setattr(source, "get_nearest_station", lambda lat, lon: None)

    with pytest.raises(ValueError):
        source.get_mandatory_data(40.0, -3.0, 2021, 2021)
