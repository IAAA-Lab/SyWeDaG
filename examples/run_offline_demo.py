"""
Offline demo / tutorial for SyWeDaG's generation pipeline.

Runs the full cycle -> adjust -> interpolate flow on a bundled sample
historical dataset, with no AEMET API key and no network access required.

Usage:
    python examples/run_offline_demo.py
"""

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from generators.synthetic_generator import SyntheticWeatherGenerator  # noqa: E402

EXAMPLES_DIR = Path(__file__).resolve().parent
HISTORICAL_CSV = EXAMPLES_DIR / "sample_historical_data.csv"
PREDICTIONS_CSV = REPO_ROOT / "sample_pred_excels" / "combined_data.csv"
OUTPUT_DIR = EXAMPLES_DIR / "output"

NUMERIC_FIELDS = {
    "temperature_min", "temperature_max", "temperature_mean", "precipitation",
    "wind_speed_mean", "wind_speed_max", "humidity_min", "humidity_max",
    "humidity_mean", "pressure_min", "pressure_max",
}


def load_historical_records(csv_path: Path) -> list[dict]:
    """Load the bundled sample dataset into the dict shape the generator expects."""
    records = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            record = dict(row)
            for field in NUMERIC_FIELDS:
                record[field] = float(record[field]) if record[field] != "" else None
            records.append(record)
    return records


def load_monthly_predictions(csv_path: Path, generation_year: int) -> pd.DataFrame:
    """Load one year of monthly predictions in the format the generator expects."""
    df = pd.read_csv(csv_path)
    return df[df["Year"] == generation_year].reset_index(drop=True)


def main():
    historical_records = load_historical_records(HISTORICAL_CSV)
    generation_year = 2021
    predictions_df = load_monthly_predictions(PREDICTIONS_CSV, generation_year)

    # The generator normally loads historical data from the local SQLite
    # database; here we hand it the bundled sample dataset directly so the
    # demo needs no database, no API key, and no network access.
    with patch(
        "generators.synthetic_generator.get_historical_daily_data",
        return_value=historical_records,
    ):
        generator = SyntheticWeatherGenerator(
            source="DEMO",
            id_station="DEMO001",
            historical_start=historical_records[0]["date"],
            historical_end=historical_records[-1]["date"],
            generation_start=f"{generation_year}-01-01",
            generation_end=f"{generation_year}-12-31",
        )
        daily_data, hourly_data, hourly_count = generator.generate(
            predictions_df=predictions_df,
            correction_method="knn",
            generation_mode="annual",
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"generated_hourly_{generation_year}.csv"

    fieldnames = ["datetime", "temperature", "precipitation", "wind_speed", "wind_direction", "humidity", "pressure"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(hourly_data)

    print(f"Generated {len(daily_data)} daily records and {hourly_count} hourly records.")
    print(f"Hourly output written to: {output_path}")


if __name__ == "__main__":
    main()
