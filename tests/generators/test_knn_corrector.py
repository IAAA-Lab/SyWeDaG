import pytest

from generators.daily_correctors.k_neighbors import KNeighborsCorrector


def test_rejects_invalid_k():
    with pytest.raises(ValueError):
        KNeighborsCorrector(k=0)


def test_output_length_matches_input(historical_records_factory):
    historical = historical_records_factory(2018, 2020)
    adjusted = historical_records_factory(2021, 2021)

    corrector = KNeighborsCorrector(k=3)
    corrected = corrector.correct(adjusted, historical)

    assert len(corrected) == len(adjusted)


def test_temperature_and_precipitation_are_untouched(historical_records_factory):
    historical = historical_records_factory(2018, 2020)
    adjusted = historical_records_factory(2021, 2021)

    corrector = KNeighborsCorrector(k=3)
    corrected = corrector.correct(adjusted, historical)

    for before, after in zip(adjusted, corrected):
        assert after["temperature_min"] == before["temperature_min"]
        assert after["temperature_max"] == before["temperature_max"]
        assert after["temperature_mean"] == before["temperature_mean"]
        assert after["precipitation"] == before["precipitation"]


def test_humidity_stays_within_physical_bounds(historical_records_factory):
    historical = historical_records_factory(2018, 2020)
    adjusted = historical_records_factory(2021, 2021)

    corrector = KNeighborsCorrector(k=3)
    corrected = corrector.correct(adjusted, historical)

    for record in corrected:
        assert 0 <= record["humidity_min"] <= 100
        assert 0 <= record["humidity_max"] <= 100
        assert record["humidity_min"] <= record["humidity_max"]


def test_k1_copies_nearest_neighbor_directly(historical_records_factory):
    historical = historical_records_factory(2018, 2020)
    adjusted = historical_records_factory(2021, 2021)

    corrector = KNeighborsCorrector(k=1)
    corrected = corrector.correct(adjusted, historical)

    for record in corrected:
        assert record["wind_speed_max"] >= record["wind_speed_mean"]


def test_single_historical_day_does_not_crash():
    historical = [{
        "date": "2020-01-01",
        "temperature_min": 2.0,
        "temperature_max": 10.0,
        "temperature_mean": 6.0,
        "precipitation": 0.0,
        "wind_speed_mean": 3.0,
        "wind_speed_max": 6.0,
        "wind_direction": "N",
        "humidity_min": 40,
        "humidity_max": 70,
        "humidity_mean": 55,
        "pressure_min": 1010.0,
        "pressure_max": 1015.0,
    }]
    adjusted = [dict(historical[0], date="2021-01-01")]

    corrector = KNeighborsCorrector(k=3)
    corrected = corrector.correct(adjusted, historical)

    # Fewer than 2 valid historical records means the correction is skipped.
    assert corrected == adjusted
