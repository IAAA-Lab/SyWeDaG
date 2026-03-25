"""
Application services for results page workflows.
"""

from datetime import datetime
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile
import pandas as pd

from database.sqliteDB import (
    get_generated_hourly_data,
    insert_generation_jobs,
    insert_generated_hourly_data,
)

# Import results logic
VALID_WIND_DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
REQUIRED_CSV_COLUMNS = [
    "datetime",
    "temperature",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "humidity",
    "pressure",
]


def _find_zip_member(zip_file: ZipFile, target_name: str) -> str | None:
    """Find a ZIP member by basename, allowing optional folder prefixes."""
    for member_name in zip_file.namelist():
        if member_name.rstrip("/").split("/")[-1] == target_name:
            return member_name
    return None


def validate_hourly_data_import_zip(zip_bytes: bytes) -> tuple[dict, pd.DataFrame]:
    """
    Validate an imported results ZIP package and return parsed content.

    Expected files:
    - metadata.json
    - hourly_data.csv

    Returns:
        Tuple (metadata_dict, validated_hourly_dataframe)

    Raises:
        ValueError: If ZIP structure or content is invalid.
    """
    if not zip_bytes:
        raise ValueError("The ZIP file is empty")

    try:
        zip_buffer = BytesIO(zip_bytes)
        with ZipFile(zip_buffer, mode="r") as zip_file:
            metadata_member = _find_zip_member(zip_file, "metadata.json")
            csv_member = _find_zip_member(zip_file, "hourly_data.csv")

            if metadata_member is None:
                raise ValueError("Missing required file: metadata.json")
            if csv_member is None:
                raise ValueError("Missing required file: hourly_data.csv")

            try:
                metadata = json.loads(zip_file.read(metadata_member).decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ValueError("metadata.json is not valid UTF-8") from error
            except json.JSONDecodeError as error:
                raise ValueError(f"metadata.json is not valid JSON: {error}") from error

            if not isinstance(metadata, dict):
                raise ValueError("metadata.json must contain a JSON object")

            for key in ("location", "periods", "nearest_station","data_source"):
                if key not in metadata:
                    raise ValueError(f"metadata.json is missing required key: '{key}'")

            location = metadata.get("location", {})
            if not isinstance(location, dict):
                raise ValueError("metadata.location must be an object")

            try:
                float(location.get("latitude"))
                float(location.get("longitude"))
            except (TypeError, ValueError) as error:
                raise ValueError("metadata.location latitude/longitude must be numeric") from error

            periods = metadata.get("periods", {})
            if not isinstance(periods, dict):
                raise ValueError("metadata.periods must be an object")

            nearest_station = metadata.get("nearest_station", {})
            if not isinstance(nearest_station, dict):
                raise ValueError("metadata.nearest_station must be an object")

            required_station_keys = {"id_station"}
            missing_station_keys = required_station_keys - set(nearest_station.keys())
            if missing_station_keys:
                raise ValueError(
                    f"metadata.nearest_station is missing keys: {sorted(missing_station_keys)}"
                )

            station_id = nearest_station.get("id_station")
            if station_id is None or str(station_id).strip() == "":
                raise ValueError("metadata.nearest_station.id_station must be non-empty")

            required_period_keys = {
                "historical_start",
                "historical_end",
                "generated_start",
                "generated_end",
            }
            missing_periods = required_period_keys - set(periods.keys())
            if missing_periods:
                raise ValueError(
                    f"metadata.periods is missing keys: {sorted(missing_periods)}"
                )

            for key in required_period_keys:
                value = periods.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"metadata.periods.{key} must be a non-empty date string")

            if metadata.get("records_count") is not None:
                try:
                    int(metadata["records_count"])
                except (TypeError, ValueError) as error:
                    raise ValueError("metadata.records_count must be an integer") from error

            try:
                dataframe = pd.read_csv(BytesIO(zip_file.read(csv_member)))
            except Exception as error:
                raise ValueError(f"hourly_data.csv could not be read: {error}") from error

    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"Invalid ZIP file: {error}") from error

    if dataframe.empty:
        raise ValueError("hourly_data.csv does not contain any rows")

    missing_columns = set(REQUIRED_CSV_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"hourly_data.csv is missing columns: {sorted(missing_columns)}")

    dataframe = dataframe[list(REQUIRED_CSV_COLUMNS)].copy()

    parsed_datetime = pd.to_datetime(dataframe["datetime"], errors="coerce")
    if parsed_datetime.isna().any():
        invalid_count = int(parsed_datetime.isna().sum())
        raise ValueError(f"hourly_data.csv has {invalid_count} invalid datetime values")

    numeric_columns = ["temperature", "precipitation", "wind_speed", "humidity", "pressure"]
    for column in numeric_columns:
        converted = pd.to_numeric(dataframe[column], errors="coerce")
        invalid_mask = dataframe[column].notna() & converted.isna()
        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())
            raise ValueError(f"hourly_data.csv has {invalid_count} invalid numeric values in '{column}'")
        dataframe[column] = converted.astype(object).where(converted.notna(), None)

    wind_raw = dataframe["wind_direction"]
    wind_values = wind_raw.astype(str).str.strip().str.upper()
    wind_values = wind_values.where(wind_raw.notna(), None)
    wind_values = wind_values.mask(wind_values == "", None)

    non_null_wind = wind_values[wind_values.notna()]
    invalid_wind = ~non_null_wind.isin(VALID_WIND_DIRECTIONS)
    if invalid_wind.any():
        invalid_sample = sorted(set(non_null_wind[invalid_wind].head(5).tolist()))
        raise ValueError(
            "hourly_data.csv has invalid wind_direction values: "
            f"{invalid_sample}. Allowed: {sorted(VALID_WIND_DIRECTIONS)}"
        )

    dataframe["wind_direction"] = wind_values.astype(object).where(wind_values.notna(), None)
    dataframe["datetime"] = parsed_datetime.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    expected_count = metadata.get("records_count")
    if expected_count is not None and int(expected_count) != len(dataframe):
        raise ValueError(
            "records_count in metadata does not match hourly_data.csv row count "
            f"({expected_count} != {len(dataframe)})"
        )

    dataframe = dataframe.sort_values("datetime").reset_index(drop=True)
    return metadata, dataframe


def persist_imported_hourly_package(metadata: dict, dataframe: pd.DataFrame) -> tuple[int, int]:
    """
    Persist validated imported package content into local database.

    Args:
        metadata: Parsed metadata.json dictionary.
        dataframe: Validated hourly data DataFrame.

    Returns:
        Tuple (job_id, records_count).
    """
    location = metadata.get("location", {})
    periods = metadata.get("periods", {})

    job_data = [(
        float(location.get("latitude")),
        float(location.get("longitude")),
        str(periods.get("historical_start")),
        str(periods.get("historical_end")),
        str(periods.get("generated_start")),
        str(periods.get("generated_end")),
    )]

    job_ids = insert_generation_jobs(job_data)
    if not job_ids:
        raise ValueError("Error creating imported generation job in database")

    job_id = job_ids[0]

    hourly_tuples = []
    for _, row in dataframe.iterrows():
        hourly_tuples.append((
            job_id,
            row["datetime"],
            row.get("temperature"),
            row.get("precipitation"),
            row.get("wind_speed"),
            row.get("wind_direction"),
            row.get("humidity"),
            row.get("pressure"),
        ))

    insert_generated_hourly_data(hourly_tuples)
    return job_id, len(hourly_tuples)

# Exportation results logic
def build_hourly_data_export_zip(
    job_id: int,
    job_info: dict,
    nearest_station,
    records_count: int | None,
) -> tuple[bytes, str]:
    """
    Build a ZIP file (in memory) containing hourly_data.csv and metadata.json.

    Args:
        job_id: Generation job identifier.
        job_info: Generation job metadata from database.
        nearest_station: Station object associated with the generation job.
        records_count: Number of generated rows shown in UI.

    Returns:
        tuple[bytes, str]: (zip_bytes, filename).
        Returns (b"", filename) when there is no data.
    """
    hourly_data = get_generated_hourly_data(job_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_hourly_data_job_{job_id}_{timestamp}.zip"

    if not hourly_data:
        return b"", filename

    dataframe = pd.DataFrame(hourly_data)
    if "datetime" in dataframe.columns:
        dataframe = dataframe.sort_values("datetime").reset_index(drop=True)

    station_name = getattr(nearest_station, "name", None)
    station_region = getattr(nearest_station, "region", None)

    metadata = {
        "job_id": job_id,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "location": {
            "latitude": job_info.get("latitude"),
            "longitude": job_info.get("longitude"),
        },
        "data_source": getattr(nearest_station, "source", None),
        "nearest_station": {
            "id_station": getattr(nearest_station, "id_station", None),
            "name": station_name,
            "region": station_region,
            "display": f"{station_name}, {station_region}" if station_name and station_region else station_name,
        },
        "periods": {
            "historical_start": job_info.get("historicalStartDate"),
            "historical_end": job_info.get("historicalEndDate"),
            "generated_start": job_info.get("generatedStartDate"),
            "generated_end": job_info.get("generatedEndDate"),
        },
        "records_count": records_count if records_count is not None else len(dataframe),
    }

    csv_text = dataframe.to_csv(index=False)
    metadata_text = json.dumps(metadata, indent=2, ensure_ascii=False)

    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr("hourly_data.csv", csv_text)
        zip_file.writestr("metadata.json", metadata_text)

    return zip_buffer.getvalue(), filename
