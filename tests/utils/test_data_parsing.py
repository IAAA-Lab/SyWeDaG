import pandas as pd
import pytest

from utils.data_parsing import (
    convert_wind_direction,
    degrees_to_cardinal,
    parse_coordinates,
    parse_float,
    parse_int,
    series_hour_of_max,
    series_hour_of_min,
    series_max,
    series_mean,
    series_min,
    to_float_or_none,
)


class TestParseCoordinates:
    def test_dms_north_east(self):
        # 41 deg 39' 38" = 41 + 39/60 + 38/3600
        assert parse_coordinates("413938N") == pytest.approx(41.660556, abs=1e-4)

    def test_dms_west_is_negative(self):
        value = parse_coordinates("010015W")
        assert value < 0

    def test_dms_south_is_negative(self):
        value = parse_coordinates("413938S")
        assert value < 0

    def test_already_decimal_passthrough(self):
        assert parse_coordinates("41.66") == 41.66

    def test_empty_returns_none(self):
        assert parse_coordinates("") is None
        assert parse_coordinates(None) is None

    def test_malformed_returns_none(self):
        assert parse_coordinates("not-a-coord") is None


class TestConvertWindDirection:
    def test_north_bucket(self):
        assert convert_wind_direction(0) == "N"
        assert convert_wind_direction(35) == "N"

    def test_east_bucket(self):
        assert convert_wind_direction(8) == "E"

    def test_variable_and_missing_sentinels(self):
        assert convert_wind_direction(99) == "N"
        assert convert_wind_direction(88) == "N"

    def test_none_returns_none(self):
        assert convert_wind_direction(None) is None

    def test_string_input(self):
        assert convert_wind_direction("8") == "E"


class TestParseFloat:
    def test_spanish_decimal_comma(self):
        assert parse_float("12,5") == 12.5

    def test_ip_sentinel_is_zero(self):
        assert parse_float("Ip") == 0.0

    def test_missing_sentinels_return_none(self):
        for token in ["IND", "VV", "N/A", "ND"]:
            assert parse_float(token) is None

    def test_none_and_empty(self):
        assert parse_float(None) is None
        assert parse_float("") is None

    def test_malformed_returns_none(self):
        assert parse_float("abc") is None


class TestParseInt:
    def test_basic(self):
        assert parse_int("42") == 42

    def test_decimal_comma_truncates(self):
        assert parse_int("42,9") == 42

    def test_sentinels_return_none(self):
        assert parse_int("ND") is None


class TestToFloatOrNone:
    def test_nan_returns_none(self):
        assert to_float_or_none(float("nan")) is None

    def test_none_returns_none(self):
        assert to_float_or_none(None) is None

    def test_valid_value(self):
        assert to_float_or_none("3.5") == 3.5


class TestSeriesAggregates:
    def _series(self):
        return pd.Series([5.0, 10.0, None, 15.0])

    def test_series_min_max_mean(self):
        series = self._series()
        assert series_min(series) == 5.0
        assert series_max(series) == 15.0
        assert series_mean(series) == 10.0

    def test_all_nan_series_returns_none(self):
        series = pd.Series([None, None])
        assert series_min(series) is None
        assert series_max(series) is None
        assert series_mean(series) is None

    def test_series_hour_of_min_and_max(self):
        day_hourly = pd.DataFrame({
            "datetime": pd.to_datetime([
                "2021-01-01T00:00:00Z", "2021-01-01T06:00:00Z", "2021-01-01T14:00:00Z",
            ]),
            "temperature_2m": [5.0, 1.0, 9.0],
        })

        assert series_hour_of_min(day_hourly, "temperature_2m") == "06:00"
        assert series_hour_of_max(day_hourly, "temperature_2m") == "14:00"

    def test_empty_dataframe_returns_none(self):
        empty = pd.DataFrame({"datetime": [], "temperature_2m": []})
        assert series_hour_of_min(empty, "temperature_2m") is None
        assert series_hour_of_max(empty, "temperature_2m") is None


class TestDegreesToCardinal:
    def test_zero_is_north(self):
        assert degrees_to_cardinal(0) == "N"

    def test_wraparound_near_360(self):
        assert degrees_to_cardinal(359) == "N"

    def test_east(self):
        assert degrees_to_cardinal(90) == "E"

    def test_invalid_returns_none(self):
        assert degrees_to_cardinal("not-a-number") is None
