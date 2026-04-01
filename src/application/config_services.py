"""
Application services for configuration and generation workflows.
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from data_sources.source_selector import get_data_source_instance
from generators.synthetic_generator import SyntheticWeatherGenerator
from database.sqliteDB import insert_weather_stations, insert_historical_daily_data
from utils.system_utils import safe_print
from utils.historical_data_treatment import apply_historical_treatment_if_needed

METHOD_OPTIONS_MAP = {
    "K-Nearest Neighbors": "knn",
    "Machine Learning (XGBoost)": "xgboost",
}

VALID_PREDICTION_VARIABLES = {
    "precipitation",
    "temperature_max",
    "temperature_mean",
    "temperature_min",
    "number_days_rain",
}


def validate_predictions(df: pd.DataFrame) -> None:
    """
    Validate that the predictions DataFrame has the correct format.

    Raises:
        ValueError: If the format is invalid.
    """
    # Normalize CSV headers and value types before validation.
    df.columns = [str(column).strip() for column in df.columns]

    required_columns = {"Year", "Month", "Variable", "Minimum", "Mean", "Maximum"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Variable"] = df["Variable"].astype(str).str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    for column in ["Minimum", "Mean", "Maximum"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid = set(df["Variable"].unique()) - VALID_PREDICTION_VARIABLES
    if invalid:
        raise ValueError(
            f"Invalid variables: {invalid}. Valid: {VALID_PREDICTION_VARIABLES}"
        )

    if df["Month"].isna().any():
        raise ValueError("Month column contains invalid or empty values")

    if not all(df["Month"].between(1, 12)):
        raise ValueError("Month values must be between 1 and 12")

    if df["Year"].isna().any():
        raise ValueError("Year column contains invalid or empty values")

    if not all(df["Month"] % 1 == 0):
        raise ValueError("Month values must be whole numbers between 1 and 12")

    value_columns = ["Minimum", "Mean", "Maximum"]
    for column in value_columns:
        if df[column].isna().any():
            missing_rows = df[df[column].isna()]
            raise ValueError(
                f"Missing values in '{column}' column. "
                f"Found empty cells in rows with Year={missing_rows['Year'].iloc[0]}, "
                f"Month={missing_rows['Month'].iloc[0]}, Variable={missing_rows['Variable'].iloc[0]}"
            )

    invalid_rows = df[(df["Minimum"] > df["Mean"]) | (df["Mean"] > df["Maximum"])]
    if not invalid_rows.empty:
        first_invalid = invalid_rows.iloc[0]
        raise ValueError(
            f"Invalid value ranges for Year={first_invalid['Year']}, Month={first_invalid['Month']}, "
            f"Variable={first_invalid['Variable']}: "
            f"Minimum ({first_invalid['Minimum']}) must be <= Mean ({first_invalid['Mean']}) "
            f"must be <= Maximum ({first_invalid['Maximum']})"
        )


def get_mandatory_weather_data(
    latitude: float,
    longitude: float,
    start_year: int,
    end_year: int,
    config: dict,
    selected_data_source: Optional[str],
):
    """
    Get nearest station and required weather data, then persist historical data.

    Returns:
        Tuple (nearest_station, weather_data)
    """
    if selected_data_source is None:
        raise ValueError("No data source available for this location")

    data_source = get_data_source_instance(selected_data_source, config)
    if data_source is None:
        raise ValueError(f"Data source '{selected_data_source}' not configured")

    nearest_station, weather_data = data_source.get_mandatory_data(
        latitude=latitude,
        longitude=longitude,
        start_year=start_year,
        end_year=end_year,
    )

    weather_data = apply_historical_treatment_if_needed(weather_data)

    if nearest_station is not None:
        stations_to_insert = [
            (
                nearest_station.source,
                nearest_station.id_station,
                nearest_station.name,
                nearest_station.region,
                nearest_station.latitude,
                nearest_station.longitude,
                nearest_station.height,
            )
        ]

        try:
            rows_inserted = insert_weather_stations(stations_to_insert)
            safe_print(f"✅ Inserted {rows_inserted} weather station(s) in DB")
        except Exception as error:
            safe_print(f"⚠️ Error inserting stations in DB: {error}")

    if nearest_station is not None and weather_data is not None and weather_data.daily_records:
        historical_tuples = [
            (
                record.date,
                nearest_station.source,
                nearest_station.id_station,
                record.temperature_min,
                record.temperature_max,
                record.temperature_mean,
                record.hour_tmin,
                record.hour_tmax,
                record.precipitation,
                record.wind_speed_mean,
                record.wind_speed_max,
                record.wind_direction,
                record.hour_wind_max,
                record.humidity_min,
                record.humidity_max,
                record.humidity_mean,
                record.hour_hrmin,
                record.hour_hrmax,
                record.pressure_min,
                record.pressure_max,
                record.hour_presmin,
                record.hour_presmax,
            )
            for record in weather_data.daily_records
        ]

        try:
            rows_inserted = insert_historical_daily_data(historical_tuples)
            safe_print(f"✅ Inserted {rows_inserted} daily historical records in DB")
        except Exception as error:
            safe_print(f"⚠️ Error inserting historical data in DB: {error}")

    return nearest_station, weather_data


def compute_generation_dates(
    weather_data,
    end_year: int,
    gen_start_year: int,
):
    """
    Compute effective historical end and generation start dates.

    Returns:
        Tuple (actual_historical_end, actual_generation_start)
    """
    if weather_data and weather_data.daily_records:
        last_record = weather_data.daily_records[-1]
        last_historical_date = datetime.strptime(last_record.date, "%Y-%m-%d").date()
        actual_historical_end = last_historical_date.isoformat()
    else:
        actual_historical_end = f"{end_year}-12-31"
        last_historical_date = datetime(end_year, 12, 31).date()

    actual_generation_start = f"{gen_start_year}-01-01"
    if gen_start_year == datetime.now().year and gen_start_year == end_year:
        next_day = last_historical_date + timedelta(days=1)
        actual_generation_start = next_day.isoformat()
        safe_print(f"📅 Adjusting generation start to: {actual_generation_start}")

    return actual_historical_end, actual_generation_start


def generate_synthetic_data(
    source: str,
    station_id: str,
    start_year: int,
    gen_end_year: int,
    actual_historical_end: str,
    actual_generation_start: str,
    latitude: float,
    longitude: float,
    predictions_df: Optional[pd.DataFrame],
    correction_method_label: str,
):
    """
    Run synthetic data generation and persistence.

    Returns:
        Tuple (job_id, rows_inserted, generated_data)
    """
    generator = SyntheticWeatherGenerator(
        source=source,
        id_station=station_id,
        historical_start=f"{start_year}-01-01",
        historical_end=actual_historical_end,
        generation_start=actual_generation_start,
        generation_end=f"{gen_end_year}-12-31",
    )

    correction_method = METHOD_OPTIONS_MAP.get(correction_method_label, "knn")

    return generator.generate_and_save(
        latitude,
        longitude,
        predictions_df,
        correction_method,
    )
