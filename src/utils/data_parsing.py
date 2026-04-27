from typing import Optional

import numpy as np
import pandas as pd

################################################
## Functions to parse and convert various data formats, especially for AEMET weather data.
#################################################
def parse_coordinates(coord_str: str) -> Optional[float]:
    """
    Parse Packed DMS format coordinates (e.g. '413938N' or '010015W') to decimal.

    Format: DDMMSSd where:
    - DD = degrees
    - MM = minutes
    - SS = seconds
    - d = direction (N/S for latitude, E/W/O for longitude)

    Args:
        coord_str: String with the coordinate (e.g. '413938N', '010015W')

    Returns:
        Coordinate in decimal format, or None if parsing fails
    """
    try:
        if not coord_str:
            return None

        # Remove whitespace
        coord_str = str(coord_str).strip()

        # If already a decimal number, return directly
        if coord_str.replace('.', '').replace('-', '').isdigit():
            return float(coord_str)

        # Extract direction from end (N, S, E, W, O)
        if coord_str and coord_str[-1] in 'NSEWOnsawo':
            direction = coord_str[-1].upper()
            coord_value = coord_str[:-1]
        else:
            # No explicit direction, assume positive
            direction = 'N'
            coord_value = coord_str

        # Parse DDMMSSd format
        if len(coord_value) >= 6:
            try:
                degrees = float(coord_value[:2])
                minutes = float(coord_value[2:4])
                seconds = float(coord_value[4:6])

                # Convert to decimal
                decimal = degrees + minutes / 60 + seconds / 3600

                # Apply direction (S, W, O are negative)
                if direction in ['S', 'W', 'O']:
                    decimal = -decimal

                return decimal
            except (ValueError, IndexError):
                # If parsing fails, try as direct float
                return float(coord_value)
        else:
            return float(coord_value)

    except (ValueError, TypeError):
        return None


def convert_wind_direction(direction_value) -> Optional[str]:
    """
    Convert wind direction from degrees format to cardinal.

    Degrees Format: values 0-35 represent 0° to 350° (each unit = 10°)
    - 99 = variable direction
    - 88 = no data

    Args:
        direction_value: Numeric or string direction value

    Returns:
        Cardinal direction (N, NE, E, SE, S, SW, W, NW) or 'N' by default
    """
    try:
        if direction_value is None or direction_value == '':
            return None
        # Convert to int
        if isinstance(direction_value, str):
            value = int(direction_value.replace(',', '.').split('.')[0])
        else:
            value = int(float(direction_value) if direction_value else 0)

        # Special cases
        if value == 99 or value == 88:  # Variable or no data
            return 'N'

        # Map value (0-35, each = 10°) to cardinal direction
        # Divide circle into 8 directions
        # N: 348.75° to 11.25° -> values 35, 0, 1
        # NE: 11.25° to 56.25° -> values 2, 3, 4, 5
        # E: 56.25° to 101.25° -> values 6, 7, 8, 9, 10
        # SE: 101.25° to 146.25° -> values 11, 12, 13, 14
        # S: 146.25° to 191.25° -> values 15, 16, 17, 18, 19
        # SW: 191.25° to 236.25° -> values 20, 21, 22, 23
        # W: 236.25° to 281.25° -> values 24, 25, 26, 27
        # NW: 281.25° to 326.25° -> values 28, 29, 30, 31, 32, 33, 34

        if value in [0, 1, 2, 34, 35]:
            return 'N'
        elif value in [3, 4, 5, 6]:
            return 'NE'
        elif value in [7, 8, 9, 10]:
            return 'E'
        elif value in [11, 12, 13, 14, 15]:
            return 'SE'
        elif value in [16, 17, 18, 19]:
            return 'S'
        elif value in [20, 21, 22, 23, 24]:
            return 'SW'
        elif value in [25, 26, 27, 28]:
            return 'W'
        elif value in [29, 30, 31, 32, 33]:
            return 'NW'
        else:
            return 'N'

    except (ValueError, TypeError):
        return 'N'


def parse_float(value) -> Optional[float]:
    """Convert value to float, returning None if it fails or is a special value"""
    try:
        if value is None or value == '':
            return None

        # Convert to string for processing
        value_str = str(value).strip()

        # Handle special values
        if value_str.upper() in ['IP', 'IND', 'VV', 'N/A', 'ND']:
            # IP = negligible (< 0.1mm), treat as 0.0 for precipitation
            if value_str.upper() == 'IP':
                return 0.0
            return None

        # Replace Spanish decimal comma with period
        value_str = value_str.replace(',', '.')

        return float(value_str)
    except (ValueError, TypeError):
        return None


def parse_int(value) -> Optional[int]:
    """Convert value to int, returning None if it fails"""
    try:
        if value is None or value == '':
            return None

        value_str = str(value).strip()

        # Handle special values
        if value_str.upper() in ['IP', 'IND', 'VV', 'N/A', 'ND']:
            return None

        # Replace decimal comma with period and convert to int
        value_str = value_str.replace(',', '.')
        return int(float(value_str))
    except (ValueError, TypeError):
        return None

################################################
## Functions to compute statistics from pandas Series, especially from Open Meteo data.
#################################################
def to_float_or_none(value) -> Optional[float]:
    """Convert value to float, returning None when value is empty/invalid/NaN."""
    if value is None:
        return None
    try:
        value = float(value)
        if np.isnan(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def series_min(series: pd.Series) -> Optional[float]:
    value = to_float_or_none(series.min(skipna=True))
    return None if value is None else float(round(value, 2))


def series_max(series: pd.Series) -> Optional[float]:
    value = to_float_or_none(series.max(skipna=True))
    return None if value is None else float(round(value, 2))


def series_mean(series: pd.Series) -> Optional[float]:
    value = to_float_or_none(series.mean(skipna=True))
    return None if value is None else float(round(value, 2))


def series_min_int(series: pd.Series) -> Optional[int]:
    value = to_float_or_none(series.min(skipna=True))
    return None if value is None else int(round(value))


def series_max_int(series: pd.Series) -> Optional[int]:
    value = to_float_or_none(series.max(skipna=True))
    return None if value is None else int(round(value))


def series_mean_int(series: pd.Series) -> Optional[int]:
    value = to_float_or_none(series.mean(skipna=True))
    return None if value is None else int(round(value))


def series_hour_of_min(day_hourly: pd.DataFrame, column: str) -> Optional[str]:
    if day_hourly.empty:
        return None
    series = pd.to_numeric(day_hourly[column], errors="coerce")
    if series.notna().sum() == 0:
        return None
    idx = series.idxmin()
    timestamp = day_hourly.loc[idx, "datetime"]
    return pd.to_datetime(timestamp).strftime("%H:%M")


def series_hour_of_max(day_hourly: pd.DataFrame, column: str) -> Optional[str]:
    if day_hourly.empty:
        return None
    series = pd.to_numeric(day_hourly[column], errors="coerce")
    if series.notna().sum() == 0:
        return None
    idx = series.idxmax()
    timestamp = day_hourly.loc[idx, "datetime"]
    return pd.to_datetime(timestamp).strftime("%H:%M")


def degrees_to_cardinal(direction_degrees) -> Optional[str]:
    """Convert direction in degrees (0-360) to cardinal direction."""
    try:
        value = float(direction_degrees)
        if np.isnan(value):
            return None
    except (TypeError, ValueError):
        return None

    value = value % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((value + 22.5) // 45) % 8
    return directions[index]
