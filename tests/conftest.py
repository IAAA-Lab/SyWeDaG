import math
import random
from datetime import date, timedelta
from pathlib import Path

import pytest


def _build_daily_records(start_year: int, end_year: int) -> list[dict]:
    """Deterministic synthetic daily records shaped like the app's daily record dicts."""
    records = []
    current = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    while current <= end:
        day_of_year = current.timetuple().tm_yday
        seasonal = 12.0 - 10.0 * math.cos(2 * math.pi * (day_of_year - 15) / 365.0)
        tmean = round(seasonal, 1)
        tmin = round(tmean - 5.0, 1)
        tmax = round(tmean + 5.0, 1)

        is_rain_day = (current.toordinal() % 5) == 0
        precipitation = 3.5 if is_rain_day else 0.0

        wind_mean = round(3.0 + (current.toordinal() % 7) * 0.5, 1)
        wind_max = round(wind_mean + 4.0, 1)

        hmean = 55 + (current.toordinal() % 20)
        hmin = max(0, hmean - 15)
        hmax = min(100, hmean + 15)

        pmin = 1005.0 + (current.toordinal() % 10)
        pmax = pmin + 8.0

        records.append({
            "date": current.strftime("%Y-%m-%d"),
            "_source_year": current.year,
            "temperature_min": tmin,
            "temperature_max": tmax,
            "temperature_mean": tmean,
            "hour_tmin": "06:00",
            "hour_tmax": "16:00",
            "precipitation": precipitation,
            "wind_speed_mean": wind_mean,
            "wind_speed_max": wind_max,
            "wind_direction": "N",
            "hour_wind_max": "14:00",
            "humidity_min": hmin,
            "humidity_max": hmax,
            "humidity_mean": hmean,
            "hour_hrmin": "15:00",
            "hour_hrmax": "05:00",
            "pressure_min": round(pmin, 1),
            "pressure_max": round(pmax, 1),
            "hour_presmin": "13:00",
            "hour_presmax": "03:00",
        })
        current += timedelta(days=1)
    return records


@pytest.fixture
def historical_records_factory():
    return _build_daily_records


@pytest.fixture
def synthetic_historical_daily():
    return _build_daily_records(2019, 2021)


@pytest.fixture(autouse=True)
def _seed_random():
    random.seed(42)


@pytest.fixture
def tmp_weather_db(monkeypatch, tmp_path):
    """Point database.sqliteDB at an isolated SQLite file and create its schema."""
    from database import sqliteDB

    db_dir = tmp_path / "data"

    def _fake_resource_path(relative_path):
        if str(relative_path) == "data/weather.db":
            return db_dir / "weather.db"
        return Path(relative_path)

    monkeypatch.setattr(sqliteDB, "get_resource_path", _fake_resource_path)
    sqliteDB.createDB()
    sqliteDB.createTables()
    return sqliteDB
