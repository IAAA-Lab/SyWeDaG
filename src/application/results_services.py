"""
Application services for results page workflows.
"""

from datetime import datetime
from io import BytesIO
import json
from zipfile import ZIP_DEFLATED, ZipFile
import pandas as pd

from database.sqliteDB import get_generated_hourly_data


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
