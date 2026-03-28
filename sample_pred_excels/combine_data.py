"""
XLS Files to CSV Combiner Script
Data Source:
    https://escenarios.adaptecca.es/
Description:
    This script processes Excel files containing meteorological data (precipitation and temperature)
    organized in separate folders by variable type (Prec, TMax, TMed, TMin).
    It reads all .xls files from each folder, extracts monthly data for each year, and combines
    them into a single CSV file with a standardized structure.

Data Structure:
    Input: Individual Excel files with sheets named 'Datos' containing columns:
        - Año (Year)
        - Mínimo (Minimum value)
        - Media (Mean value)
        - Máximo (Maximum value)

    Output: combined_data.csv with columns:
        - Year
        - Month
        - Variable (precipitation, temperature_max, temperature_mean, temperature_min,
                    number_days_rain)
        - Minimum
        - Mean
        - Maximum

Usage:
    1. Organize your Excel files in subdirectories named: Prec/, TMax/, TMed/, TMin/, NDaysRain/
    2. Place them in the same directory as this script
    3. Run: python combine_data.py
    4. The combined_data.csv file will be generated in the same directory

Requirements:
    - pandas
    - xlrd (for reading .xls files)

Note:
    - Files must be named with month abbreviations (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep,
      Oct, Nov, Dec)
    - Excel files should be in .xls format
    - The 'Datos' sheet should have a header row and data starting from row 2
"""

from pathlib import Path

import pandas as pd

# Folder and variable configuration
BASE_FOLDER = Path(__file__).parent
VARIABLE_MAPPING = {
    "Prec": {
        "name": "precipitation",
        "folder": BASE_FOLDER / "Prec",
    },
    "TMax": {
        "name": "temperature_max",
        "folder": BASE_FOLDER / "TMax",
    },
    "TMean": {
        "name": "temperature_mean",
        "folder": BASE_FOLDER / "TMean",
    },
    "TMin": {
        "name": "temperature_min",
        "folder": BASE_FOLDER / "TMin",
    },
    "NDaysRain": {
        "name": "number_days_rain",
        "folder": BASE_FOLDER / "NDaysRain",
    },
}

# Month to number mapping
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def extract_month_from_filename(filename):
    """Extract month from a filename using English abbreviations."""
    for month_str, month_num in MONTHS.items():
        if month_str in filename:
            return month_num
    return None


def read_xls_data(file_path):
    """Read data from an .xls file and return a cleaned DataFrame."""
    try:
        # Automatic engine detection is enough for .xls when dependencies are installed.
        dataframe = pd.read_excel(file_path, sheet_name="Datos", skiprows=1)
        dataframe.columns = dataframe.columns.str.strip()

        expected_columns = ["Año", "Mínimo", "Media", "Máximo"]
        if not all(column in dataframe.columns for column in expected_columns):
            print(f"Warning: Unexpected columns in {file_path}")
            print(f"Found columns: {dataframe.columns.tolist()}")
            return None

        return dataframe.dropna(how="all")
    except Exception as error:
        print(f"Error reading {file_path}: {error}")
        return None


def process_all_excels():
    """Process all .xls files and generate a single combined CSV file."""
    combined_data = []

    for _, variable_info in VARIABLE_MAPPING.items():
        variable_name = variable_info["name"]
        folder = variable_info["folder"]

        if not folder.exists():
            print(f"Warning: folder {folder} does not exist")
            continue

        print(f"\nProcessing variable: {variable_name}")

        for file_path in sorted(folder.glob("*.xls")):
            month_num = extract_month_from_filename(file_path.name)
            if month_num is None:
                print(f"  Could not extract month from: {file_path.name}")
                continue

            print(f"  Reading: {file_path.name} (month {month_num})")
            dataframe = read_xls_data(file_path)
            if dataframe is None:
                continue

            for _, row in dataframe.iterrows():
                try:
                    combined_data.append(
                        {
                            "Year": int(row["Año"]),
                            "Month": month_num,
                            "Variable": variable_name,
                            "Minimum": float(row["Mínimo"]),
                            "Mean": float(row["Media"]),
                            "Maximum": float(row["Máximo"]),
                        }
                    )
                except (ValueError, TypeError) as error:
                    print(f"    Error in row: {row.to_dict()} - {error}")

    result_dataframe = pd.DataFrame(combined_data)
    if result_dataframe.empty:
        print("\nError: No data was successfully processed")
        return False

    result_dataframe = (
        result_dataframe.sort_values(["Year", "Month", "Variable"]).reset_index(drop=True)
    )

    output_file = BASE_FOLDER / "combined_data.csv"
    try:
        result_dataframe.to_csv(output_file, index=False, encoding="utf-8")
        print(f"\n✓ File successfully generated: {output_file}")
        print(f"  Total rows: {len(result_dataframe)}")
        print("\nFirst rows:")
        print(result_dataframe.head(10))
        return True
    except Exception as error:
        print(f"Error saving file: {error}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("XLS TO CSV COMBINER")
    print("=" * 60)
    process_all_excels()
