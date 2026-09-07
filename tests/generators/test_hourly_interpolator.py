from generators.hourly_generation.hourly_interpolator import (
    distribute_precipitation,
    generate_continuous_humidity,
    generate_continuous_pressure,
    generate_continuous_temperature,
    generate_continuous_wind,
)
from generators.synthetic_generator import SyntheticWeatherGenerator

parse_hour = SyntheticWeatherGenerator._parse_hour


def test_temperature_series_has_one_value_per_hour(historical_records_factory):
    daily_data = historical_records_factory(2020, 2020)
    series = generate_continuous_temperature(daily_data, parse_hour)

    assert len(series) == len(daily_data) * 24
    assert all(value is not None for value in series)


def test_daily_mean_temperature_is_reasonably_close(historical_records_factory):
    daily_data = historical_records_factory(2020, 2020)
    series = generate_continuous_temperature(daily_data, parse_hour)

    for day_idx, record in enumerate(daily_data):
        day_hours = series[day_idx * 24: day_idx * 24 + 24]
        hourly_mean = sum(day_hours) / 24
        assert abs(hourly_mean - record["temperature_mean"]) <= 1.5


def test_no_large_jump_at_day_boundary(historical_records_factory):
    daily_data = historical_records_factory(2020, 2020)
    series = generate_continuous_temperature(daily_data, parse_hour)

    max_daily_range = max(r["temperature_max"] - r["temperature_min"] for r in daily_data)
    for day_idx in range(len(daily_data) - 1):
        boundary_hour = day_idx * 24 + 23
        jump = abs(series[boundary_hour + 1] - series[boundary_hour])
        assert jump <= max_daily_range


def test_humidity_stays_within_physical_bounds(historical_records_factory):
    daily_data = historical_records_factory(2020, 2020)
    series = generate_continuous_humidity(daily_data, parse_hour)

    assert all(0 <= value <= 100 for value in series)


def test_pressure_series_length(historical_records_factory):
    daily_data = historical_records_factory(2020, 2020)
    series = generate_continuous_pressure(daily_data, parse_hour)

    assert len(series) == len(daily_data) * 24


def test_wind_series_never_exceeds_daily_max(historical_records_factory):
    daily_data = historical_records_factory(2020, 2020)
    series = generate_continuous_wind(daily_data, parse_hour)

    for day_idx, record in enumerate(daily_data):
        day_hours = series[day_idx * 24: day_idx * 24 + 24]
        assert max(day_hours) <= record["wind_speed_max"] + 0.1


def test_no_data_returns_none_series():
    daily_data = [{"date": "2020-01-01"}, {"date": "2020-01-02"}]
    series = generate_continuous_temperature(daily_data, parse_hour)
    assert series == [None] * 48


class TestDistributePrecipitation:
    def test_zero_precipitation_returns_zeros(self):
        assert distribute_precipitation(0.0) == [0.0] * 24

    def test_total_is_conserved(self):
        for total in [1.0, 5.5, 12.3, 25.0]:
            hourly = distribute_precipitation(total)
            assert len(hourly) == 24
            assert abs(sum(hourly) - total) <= 0.2

    def test_negative_precipitation_returns_zeros(self):
        assert distribute_precipitation(-1.0) == [0.0] * 24
