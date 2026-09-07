from data_sources.base_source import DailyWeatherRecord, WeatherData
from utils.historical_data_treatment import (
    apply_historical_treatment_if_needed,
    fill_missing_days,
    interpolate_missing_values_in_period,
)


def _record(date, temperature_mean=None, precipitation=None, humidity_mean=None):
    return {
        "date": date,
        "temperature_mean": temperature_mean,
        "precipitation": precipitation,
        "humidity_mean": humidity_mean,
    }


class TestFillMissingDays:
    def test_empty_input_returns_empty(self):
        assert fill_missing_days([]) == []

    def test_gap_is_interpolated_between_neighbours(self):
        records = [
            _record("2021-01-01", temperature_mean="10"),
            _record("2021-01-03", temperature_mean="20"),
        ]

        filled = fill_missing_days(records)

        assert [r["date"] for r in filled] == ["2021-01-01", "2021-01-02", "2021-01-03"]
        middle = filled[1]
        assert middle["temperature_mean"] == "15.0"

    def test_multi_day_gap_is_filled_for_every_missing_day(self):
        records = [
            _record("2021-01-01", temperature_mean="10"),
            _record("2021-01-04", temperature_mean="10"),
        ]

        filled = fill_missing_days(records)

        assert len(filled) == 4
        for record in filled:
            assert record["temperature_mean"] in ("10", "10.0")

    def test_no_gaps_returns_same_records(self):
        records = [_record("2021-01-01", temperature_mean="10"), _record("2021-01-02", temperature_mean="11")]
        filled = fill_missing_days(records)
        assert [r["date"] for r in filled] == ["2021-01-01", "2021-01-02"]


class TestInterpolateMissingValuesInPeriod:
    def test_interior_none_is_averaged(self):
        records = [
            _record("2021-01-01", temperature_mean="10"),
            _record("2021-01-02", temperature_mean=None),
            _record("2021-01-03", temperature_mean="20"),
        ]

        interpolate_missing_values_in_period(records)

        assert records[1]["temperature_mean"] == "15.0"

    def test_edge_none_uses_nearest_available_value(self):
        records = [
            _record("2021-01-01", temperature_mean=None),
            _record("2021-01-02", temperature_mean="10"),
        ]

        interpolate_missing_values_in_period(records)

        assert records[0]["temperature_mean"] == "10.0"

    def test_variable_never_present_is_left_untouched(self):
        records = [
            _record("2021-01-01", humidity_mean=None),
            _record("2021-01-02", humidity_mean=None),
        ]

        interpolate_missing_values_in_period(records)

        assert all(r["humidity_mean"] is None for r in records)

    def test_single_record_is_a_no_op(self):
        records = [_record("2021-01-01", temperature_mean=None)]
        interpolate_missing_values_in_period(records)
        assert records[0]["temperature_mean"] is None


class TestApplyHistoricalTreatmentIfNeeded:
    def _daily_record(self, date, temperature_mean=10.0, **overrides):
        base_mean = temperature_mean if temperature_mean is not None else 10.0
        fields = dict(
            date=date,
            temperature_min=base_mean - 5,
            temperature_max=base_mean + 5,
            temperature_mean=temperature_mean,
            precipitation=0.0,
            wind_speed_mean=5.0,
            wind_speed_max=10.0,
            wind_direction="N",
            humidity_min=40,
            humidity_max=70,
            humidity_mean=55,
            pressure_min=1010.0,
            pressure_max=1020.0,
        )
        fields.update(overrides)
        return DailyWeatherRecord(**fields)

    def test_none_input_passes_through(self):
        assert apply_historical_treatment_if_needed(None) is None

    def test_complete_data_is_returned_unchanged(self):
        weather_data = WeatherData(daily_records=[
            self._daily_record("2021-01-01"),
            self._daily_record("2021-01-02"),
        ])

        result = apply_historical_treatment_if_needed(weather_data)

        assert result is weather_data

    def test_missing_day_gets_filled_in(self):
        weather_data = WeatherData(daily_records=[
            self._daily_record("2021-01-01", temperature_mean=10.0),
            self._daily_record("2021-01-03", temperature_mean=20.0),
        ])

        result = apply_historical_treatment_if_needed(weather_data)

        assert [r.date for r in result.daily_records] == ["2021-01-01", "2021-01-02", "2021-01-03"]
        assert result.daily_records[1].temperature_mean == 15.0

    def test_missing_field_gets_interpolated(self):
        weather_data = WeatherData(daily_records=[
            self._daily_record("2021-01-01", temperature_mean=10.0),
            self._daily_record("2021-01-02", temperature_mean=None),
            self._daily_record("2021-01-03", temperature_mean=20.0),
        ])

        result = apply_historical_treatment_if_needed(weather_data)

        assert result.daily_records[1].temperature_mean == 15.0
