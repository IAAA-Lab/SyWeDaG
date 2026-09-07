import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from application import results_services as rs
from data_sources.base_source import WeatherStation


def _build_zip(files: dict) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as zip_file:
        for name, content in files.items():
            zip_file.writestr(name, content)
    return buffer.getvalue()


def _valid_metadata():
    return {
        "location": {"latitude": 40.0, "longitude": -3.0},
        "periods": {
            "historical_start": "2018-01-01",
            "historical_end": "2020-12-31",
            "generated_start": "2021-01-01",
            "generated_end": "2021-12-31",
        },
        "nearest_station": {"id_station": "ST1", "name": "Test Station"},
        "data_source": "TEST",
        "records_count": 2,
    }


def _valid_csv():
    return (
        "datetime,temperature,precipitation,wind_speed,wind_direction,humidity,pressure\n"
        "2021-01-01T00:00:00Z,10.0,0.0,5.0,N,55,1015.0\n"
        "2021-01-01T01:00:00Z,9.5,0.0,5.2,N,56,1015.0\n"
    )


class TestValidateHourlyDataImportZip:
    def test_valid_package_round_trips(self):
        zip_bytes = _build_zip({
            "metadata.json": json.dumps(_valid_metadata()),
            "hourly_data.csv": _valid_csv(),
        })

        metadata, dataframe = rs.validate_hourly_data_import_zip(zip_bytes)

        assert metadata["nearest_station"]["id_station"] == "ST1"
        assert len(dataframe) == 2
        assert list(dataframe["wind_direction"]) == ["N", "N"]

    def test_empty_zip_bytes_raises(self):
        with pytest.raises(ValueError):
            rs.validate_hourly_data_import_zip(b"")

    def test_missing_metadata_json_raises(self):
        zip_bytes = _build_zip({"hourly_data.csv": _valid_csv()})
        with pytest.raises(ValueError, match="metadata.json"):
            rs.validate_hourly_data_import_zip(zip_bytes)

    def test_missing_hourly_csv_raises(self):
        zip_bytes = _build_zip({"metadata.json": json.dumps(_valid_metadata())})
        with pytest.raises(ValueError, match="hourly_data.csv"):
            rs.validate_hourly_data_import_zip(zip_bytes)

    def test_malformed_json_raises(self):
        zip_bytes = _build_zip({
            "metadata.json": "{not valid json",
            "hourly_data.csv": _valid_csv(),
        })
        with pytest.raises(ValueError, match="not valid JSON"):
            rs.validate_hourly_data_import_zip(zip_bytes)

    def test_missing_required_metadata_key_raises(self):
        metadata = _valid_metadata()
        del metadata["periods"]
        zip_bytes = _build_zip({
            "metadata.json": json.dumps(metadata),
            "hourly_data.csv": _valid_csv(),
        })
        with pytest.raises(ValueError, match="periods"):
            rs.validate_hourly_data_import_zip(zip_bytes)

    def test_invalid_wind_direction_raises(self):
        csv_text = (
            "datetime,temperature,precipitation,wind_speed,wind_direction,humidity,pressure\n"
            "2021-01-01T00:00:00Z,10.0,0.0,5.0,NOTREAL,55,1015.0\n"
        )
        zip_bytes = _build_zip({
            "metadata.json": json.dumps(_valid_metadata()),
            "hourly_data.csv": csv_text,
        })
        with pytest.raises(ValueError, match="wind_direction"):
            rs.validate_hourly_data_import_zip(zip_bytes)

    def test_records_count_mismatch_raises(self):
        metadata = _valid_metadata()
        metadata["records_count"] = 99
        zip_bytes = _build_zip({
            "metadata.json": json.dumps(metadata),
            "hourly_data.csv": _valid_csv(),
        })
        with pytest.raises(ValueError, match="records_count"):
            rs.validate_hourly_data_import_zip(zip_bytes)


class TestExportImportRoundTrip:
    def test_export_then_import_preserves_row_count(self, tmp_weather_db):
        job_id = tmp_weather_db.insert_generation_jobs([
            (40.0, -3.0, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ])[0]
        tmp_weather_db.insert_generated_hourly_data([
            (job_id, "2021-01-01T00:00:00Z", 8.0, 0.0, 10.0, "N", 55, 1015.0),
            (job_id, "2021-01-01T01:00:00Z", 7.5, 0.0, 9.0, "N", 56, 1015.0),
        ])
        job_info = tmp_weather_db.get_generation_job_info(job_id)
        station = WeatherStation(
            source="TEST", id_station="ST1", name="Test Station",
            region="Test Region", latitude=40.0, longitude=-3.0, height=600,
        )

        zip_bytes, filename = rs.build_hourly_data_export_zip(job_id, job_info, station, records_count=2)
        assert filename.endswith(".zip")

        metadata, dataframe = rs.validate_hourly_data_import_zip(zip_bytes)
        assert len(dataframe) == 2
        assert metadata["nearest_station"]["id_station"] == "ST1"

        new_job_id, imported_count = rs.persist_imported_hourly_package(metadata, dataframe)
        assert imported_count == 2

        imported_hourly = tmp_weather_db.get_generated_hourly_data(new_job_id)
        assert len(imported_hourly) == 2

    def test_export_with_no_data_returns_empty_bytes(self, tmp_weather_db):
        job_id = tmp_weather_db.insert_generation_jobs([
            (40.0, -3.0, "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ])[0]
        job_info = tmp_weather_db.get_generation_job_info(job_id)
        station = WeatherStation(
            source="TEST", id_station="ST1", name="Test Station",
            region=None, latitude=40.0, longitude=-3.0, height=600,
        )

        zip_bytes, filename = rs.build_hourly_data_export_zip(job_id, job_info, station, records_count=0)

        assert zip_bytes == b""
        assert filename.endswith(".zip")
