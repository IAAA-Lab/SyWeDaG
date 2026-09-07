import pytest

from generators.daily_correctors.xgboost_model import XGBoostWeatherModel


def test_rejects_invalid_window_size():
    with pytest.raises(ValueError):
        XGBoostWeatherModel(window_size=0)


def test_output_length_matches_input(historical_records_factory):
    historical = historical_records_factory(2018, 2018)
    adjusted = historical_records_factory(2021, 2021)

    model = XGBoostWeatherModel(window_size=3)
    corrected = model.correct(adjusted, historical)

    assert len(corrected) == len(adjusted)


def test_temperature_and_precipitation_are_untouched(historical_records_factory):
    historical = historical_records_factory(2018, 2018)
    adjusted = historical_records_factory(2021, 2021)

    model = XGBoostWeatherModel(window_size=3)
    corrected = model.correct(adjusted, historical)

    for before, after in zip(adjusted, corrected):
        assert after["temperature_min"] == before["temperature_min"]
        assert after["temperature_max"] == before["temperature_max"]
        assert after["precipitation"] == before["precipitation"]


def test_humidity_and_wind_stay_within_physical_bounds(historical_records_factory):
    historical = historical_records_factory(2018, 2018)
    adjusted = historical_records_factory(2021, 2021)

    model = XGBoostWeatherModel(window_size=3)
    corrected = model.correct(adjusted, historical)

    for record in corrected:
        assert 0 <= record["humidity_min"] <= 100
        assert 0 <= record["humidity_max"] <= 100
        assert record["humidity_min"] <= record["humidity_max"]
        assert record["wind_speed_max"] >= record["wind_speed_mean"]


def test_empty_adjusted_data_returns_empty():
    model = XGBoostWeatherModel(window_size=3)
    assert model.correct([], []) == []


def test_falls_back_when_not_enough_historical_records(historical_records_factory):
    historical = historical_records_factory(2020, 2020)[:2]
    adjusted = historical_records_factory(2021, 2021)[:5]

    model = XGBoostWeatherModel(window_size=7)
    corrected = model.correct(adjusted, historical)

    assert len(corrected) == len(adjusted)
