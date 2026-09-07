import copy

import pandas as pd
import pytest

from generators import synthetic_generator as sg
from generators.synthetic_generator import SyntheticWeatherGenerator


def _make_generator(monkeypatch, historical_data, gen_start, gen_end,
                     hist_start="2018-01-01", hist_end="2020-12-31"):
    monkeypatch.setattr(sg, "get_historical_daily_data", lambda **kwargs: historical_data)
    return SyntheticWeatherGenerator(
        source="TEST",
        id_station="0000X",
        historical_start=hist_start,
        historical_end=hist_end,
        generation_start=gen_start,
        generation_end=gen_end,
    )


def test_missing_historical_data_raises(monkeypatch):
    monkeypatch.setattr(sg, "get_historical_daily_data", lambda **kwargs: [])
    with pytest.raises(ValueError):
        SyntheticWeatherGenerator(
            source="TEST",
            id_station="0000X",
            historical_start="2018-01-01",
            historical_end="2020-12-31",
            generation_start="2021-01-01",
            generation_end="2021-12-31",
        )


def test_one_record_per_calendar_day(monkeypatch, historical_records_factory):
    historical = historical_records_factory(2018, 2020)
    generator = _make_generator(monkeypatch, historical, "2021-01-01", "2021-01-31")

    daily = generator._generate_daily_synthetic("annual")

    assert len(daily) == 31
    assert [r["date"] for r in daily] == [f"2021-01-{d:02d}" for d in range(1, 32)]


def test_annual_mode_uses_single_historical_year_per_generated_year(monkeypatch, historical_records_factory):
    historical = historical_records_factory(2018, 2020)
    generator = _make_generator(monkeypatch, historical, "2021-01-01", "2021-12-31")

    daily = generator._generate_daily_synthetic("annual")

    years_used = {r["_source_year"] for r in daily}
    assert len(years_used) == 1
    assert years_used.issubset({2018, 2019, 2020})


def test_monthly_mode_produces_valid_full_year(monkeypatch, historical_records_factory):
    historical = historical_records_factory(2018, 2020)
    generator = _make_generator(monkeypatch, historical, "2021-01-01", "2021-12-31")

    daily = generator._generate_daily_synthetic("monthly")

    assert len(daily) == 365
    assert all(r["_source_year"] in {2018, 2019, 2020} for r in daily)


def test_feb29_falls_back_to_feb28_same_historical_year(monkeypatch, historical_records_factory):
    # Neither historical year has a Feb 29, so the generator must fall back.
    historical = historical_records_factory(2019, 2019) + historical_records_factory(2021, 2021)
    generator = _make_generator(monkeypatch, historical, "2024-01-01", "2024-12-31")

    daily = generator._generate_daily_synthetic("annual")
    feb29 = next(r for r in daily if r["date"] == "2024-02-29")

    source_year = feb29["_source_year"]
    feb28_source = next(
        r for r in historical if r["date"] == f"{source_year}-02-28"
    )
    assert feb29["precipitation"] == feb28_source["precipitation"]
    assert feb29["wind_speed_mean"] == feb28_source["wind_speed_mean"]


def test_generate_end_to_end_without_predictions(monkeypatch, historical_records_factory):
    historical = historical_records_factory(2018, 2020)
    generator = _make_generator(monkeypatch, historical, "2021-01-01", "2021-01-31")

    daily_data, hourly_data, hourly_count = generator.generate()

    assert len(daily_data) == 31
    assert hourly_count == 31 * 24
    assert len(hourly_data) == 31 * 24


class TestVerifyDailyVsPredictions:
    def _predictions_df(self):
        return pd.DataFrame([
            {"Year": 2021, "Month": 1, "Variable": "temperature_max", "Minimum": None, "Mean": 15.0, "Maximum": None},
            {"Year": 2021, "Month": 1, "Variable": "temperature_mean", "Minimum": None, "Mean": 10.0, "Maximum": None},
            {"Year": 2021, "Month": 1, "Variable": "temperature_min", "Minimum": None, "Mean": 5.0, "Maximum": None},
            {"Year": 2021, "Month": 1, "Variable": "precipitation", "Minimum": None, "Mean": 2.0, "Maximum": None},
        ])

    def _matching_daily_data(self):
        return [
            {
                "date": f"2021-01-{day:02d}",
                "temperature_max": 15.0,
                "temperature_mean": 10.0,
                "temperature_min": 5.0,
                "precipitation": 2.0,
            }
            for day in range(1, 32)
        ]

    def test_passes_on_matching_data(self, monkeypatch, historical_records_factory):
        generator = _make_generator(
            monkeypatch, historical_records_factory(2018, 2020), "2021-01-01", "2021-01-31"
        )
        assert generator._verify_daily_vs_predictions(
            self._matching_daily_data(), self._predictions_df()
        ) is True

    def test_fails_when_mean_is_off(self, monkeypatch, historical_records_factory):
        generator = _make_generator(
            monkeypatch, historical_records_factory(2018, 2020), "2021-01-01", "2021-01-31"
        )
        corrupted = self._matching_daily_data()
        for record in corrupted:
            record["temperature_mean"] = 20.0  # 10 degrees off target of 10.0

        assert generator._verify_daily_vs_predictions(corrupted, self._predictions_df()) is False


class TestVerifyHourlyVsDaily:
    def _daily(self):
        return [{
            "date": "2021-01-01",
            "temperature_mean": 10.0,
            "temperature_min": 5.0,
            "temperature_max": 15.0,
            "precipitation": 2.4,
            "humidity_mean": 60,
            "pressure_min": 1010.0,
            "pressure_max": 1018.0,
            "wind_speed_mean": 5.0,
            "wind_speed_max": 9.0,
        }]

    def _matching_hourly(self):
        return [
            {
                "datetime": f"2021-01-01T{hour:02d}:00:00Z",
                "temperature": 10.0,
                "precipitation": 0.1,
                "humidity": 60,
                "pressure": 1014.0,
                "wind_speed": 5.0,
            }
            for hour in range(24)
        ]

    def test_passes_on_consistent_data(self, monkeypatch, historical_records_factory):
        generator = _make_generator(
            monkeypatch, historical_records_factory(2018, 2020), "2021-01-01", "2021-01-31"
        )
        assert generator._verify_hourly_vs_daily(self._matching_hourly(), self._daily()) is True

    def test_fails_when_hourly_mean_diverges_from_daily(self, monkeypatch, historical_records_factory):
        generator = _make_generator(
            monkeypatch, historical_records_factory(2018, 2020), "2021-01-01", "2021-01-31"
        )
        corrupted_hourly = copy.deepcopy(self._matching_hourly())
        for record in corrupted_hourly:
            record["temperature"] = 30.0  # far from the daily mean of 10.0

        assert generator._verify_hourly_vs_daily(corrupted_hourly, self._daily()) is False

    def test_fails_when_a_day_is_missing(self, monkeypatch, historical_records_factory):
        generator = _make_generator(
            monkeypatch, historical_records_factory(2018, 2020), "2021-01-01", "2021-01-31"
        )
        assert generator._verify_hourly_vs_daily([], self._daily()) is False


class TestParseHour:
    def test_default_for_missing_or_placeholder(self):
        for value in [None, "", "varias", "n/a", "nd"]:
            assert SyntheticWeatherGenerator._parse_hour(value) == 12

    def test_plain_digit_string(self):
        assert SyntheticWeatherGenerator._parse_hour("14") == 14

    def test_hh_mm_format(self):
        assert SyntheticWeatherGenerator._parse_hour("09:30") == 9

    def test_hour_24_maps_to_23(self):
        assert SyntheticWeatherGenerator._parse_hour("24:00") == 23
        assert SyntheticWeatherGenerator._parse_hour("2400") == 23
