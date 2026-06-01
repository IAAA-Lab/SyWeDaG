"""Hourly interpolation functions for synthetic weather generation.
Provides continuous-series generation for temperature, humidity and pressure
based on chronologically-ordered extremes (min/max) across the entire time
series, eliminating day-boundary discontinuities.  Wind and precipitation
helpers remain unchanged.
"""
import random
from math import cos, exp, pi
from typing import Callable, List, Optional, Tuple
# Type alias for a single extreme point: (global_hour_index, value, "min"|"max")
Extreme = Tuple[int, float, str]
# ============================================================================
# Internal helpers
# ============================================================================
def _cosine_interpolate(
    start_hour: int,
    start_value: float,
    end_hour: int,
    end_value: float,
) -> List[float]:
    """Interpolate smoothly between two extreme points using a cosine curve.
    Generates values for every integer hour in ``[start_hour, end_hour)``.
    The cosine formula guarantees C¹ continuity at junction points.
    Args:
        start_hour:  Global hour index of the first extreme.
        start_value: Value at *start_hour*.
        end_hour:    Global hour index of the second extreme.
        end_value:   Value at *end_hour*.
    Returns:
        List of interpolated values (length = ``end_hour - start_hour``).
        Returns an empty list when ``start_hour == end_hour``.
    """
    duration = end_hour - start_hour
    if duration <= 0:
        return []
    values: List[float] = []
    for i in range(duration):
        progress = i / duration
        smooth = (1 - cos(pi * progress)) / 2
        value = start_value + (end_value - start_value) * smooth
        values.append(value)
    return values
# ---------------------------------------------------------------------------
# Pseudo-extreme insertion
# ---------------------------------------------------------------------------

def _fix_boundary_flatness(
    extremes: List[Extreme], 
    daily_data: List[dict], 
    min_key: str, 
    max_key: str, 
    margin_func: Callable[[], float]
) -> List[Extreme]:
    """Prevent flat segments at the start and end of the series."""
    if not extremes:
        return extremes
    
    new_extremes = list(extremes)
    
    # Fix start
    first_hour, first_val, first_type = new_extremes[0]
    if first_hour > 0:
        day0_min = float(daily_data[0].get(min_key, first_val))
        day0_max = float(daily_data[0].get(max_key, first_val))
        if first_type == "max":
            pseudo_val = day0_min + margin_func()
            pseudo_val = min(pseudo_val, first_val - 0.1)
            new_extremes.insert(0, (0, pseudo_val, "min"))
        else:
            pseudo_val = day0_max - margin_func()
            pseudo_val = max(pseudo_val, first_val + 0.1)
            new_extremes.insert(0, (0, pseudo_val, "max"))
            
    # Fix end
    last_hour, last_val, last_type = new_extremes[-1]
    total_hours = len(daily_data) * 24
    if last_hour < total_hours - 1:
        day_last = daily_data[-1]
        day_last_min = float(day_last.get(min_key, last_val))
        day_last_max = float(day_last.get(max_key, last_val))
        target_hour = total_hours - 1
        if last_type == "max":
            pseudo_val = day_last_min + margin_func()
            pseudo_val = min(pseudo_val, last_val - 0.1)
            new_extremes.append((target_hour, pseudo_val, "min"))
        else:
            pseudo_val = day_last_max - margin_func()
            pseudo_val = max(pseudo_val, last_val + 0.1)
            new_extremes.append((target_hour, pseudo_val, "max"))
            
    return new_extremes


def _insert_pseudo_extremes_temperature(extremes: List[Extreme]) -> List[Extreme]:
    """Insert pseudo-extremes for temperature series using surrounding minimums."""
    changed = True
    while changed:
        changed = False
        new_extremes: List[Extreme] = [extremes[0]]
        for i in range(1, len(extremes)):
            prev = new_extremes[-1]
            curr = extremes[i]
            if prev[2] == curr[2]:
                mid_hour = (prev[0] + curr[0]) // 2

                if prev[2] == "max":
                    # max-max -> insert pseudo-min based on surrounding minimums
                    prev_min_val = prev[1] - 5.0
                    for j in range(len(new_extremes)-1, -1, -1):
                        if new_extremes[j][2] == "min":
                            prev_min_val = new_extremes[j][1]
                            break
                    
                    next_min_val = curr[1] - 5.0
                    for j in range(i, len(extremes)):
                        if extremes[j][2] == "min":
                            next_min_val = extremes[j][1]
                            break
                            
                    base_val = max(prev_min_val, next_min_val)
                    pseudo_val = base_val + random.uniform(0.5, 2.0)
                    pseudo_val = min(pseudo_val, prev[1] - 0.1, curr[1] - 0.1)
                    new_extremes.append((mid_hour, pseudo_val, "min"))
                else:
                    # min-min -> insert pseudo-max based on these minimums
                    base_val = max(prev[1], curr[1])
                    pseudo_val = base_val# + random.uniform(0.5, 2.0)
                    new_extremes.append((mid_hour, pseudo_val, "max"))
                changed = True
            new_extremes.append(curr)
        extremes = new_extremes
    return extremes

def _insert_pseudo_extremes_humidity(extremes: List[Extreme]) -> List[Extreme]:
    """Insert pseudo-extremes for humidity series."""
    changed = True
    while changed:
        changed = False
        new_extremes: List[Extreme] = [extremes[0]]
        for i in range(1, len(extremes)):
            prev = new_extremes[-1]
            curr = extremes[i]
            if prev[2] == curr[2]:
                mid_hour = (prev[0] + curr[0]) // 2
                if prev[2] == "max":
                    offset = max(1.0, abs(prev[1] - curr[1]) * 0.1)
                    pseudo_val = max(0.0, min(prev[1], curr[1]) - offset)
                    new_extremes.append((mid_hour, pseudo_val, "min"))
                else:
                    offset = max(1.0, abs(prev[1] - curr[1]) * 0.1)
                    pseudo_val = min(100.0, max(prev[1], curr[1]) + offset)
                    new_extremes.append((mid_hour, pseudo_val, "max"))
                changed = True
            new_extremes.append(curr)
        extremes = new_extremes
    return extremes
def _insert_pseudo_extremes_pressure(extremes: List[Extreme]) -> List[Extreme]:
    """Insert pseudo-extremes for pressure series.
    * **max → max**: pseudo-minimum at the midpoint, slightly below the lower
      of the two maxima.
    * **min → min**: pseudo-maximum at the midpoint, slightly above the higher
      of the two minima.
    """
    changed = True
    while changed:
        changed = False
        new_extremes: List[Extreme] = [extremes[0]]
        for i in range(1, len(extremes)):
            prev = new_extremes[-1]
            curr = extremes[i]
            if prev[2] == curr[2]:
                mid_hour = (prev[0] + curr[0]) // 2
                if prev[2] == "max":
                    offset = max(0.3, abs(prev[1] - curr[1]) * 0.1)
                    pseudo_val = min(prev[1], curr[1]) - offset
                    new_extremes.append((mid_hour, pseudo_val, "min"))
                else:
                    offset = max(0.3, abs(prev[1] - curr[1]) * 0.1)
                    pseudo_val = max(prev[1], curr[1]) + offset
                    new_extremes.append((mid_hour, pseudo_val, "max"))
                changed = True
            new_extremes.append(curr)
        extremes = new_extremes
    return extremes
# ---------------------------------------------------------------------------
# Extreme extraction helpers
# ---------------------------------------------------------------------------
def _fill_missing_extremes(
    daily_data: List[dict],
    min_key: str,
    max_key: str,
    hour_min_key: str,
    hour_max_key: str,
    parse_hour_fn: Callable,
) -> None:
    """Fill missing min/max values in *daily_data* in-place using nearest neighbour.
    Scans forward and backward so that every day has a usable pair of
    min/max values for the requested variable.
    """
    n = len(daily_data)
    # Forward fill
    for i in range(1, n):
        if daily_data[i].get(min_key) is None and daily_data[i - 1].get(min_key) is not None:
            daily_data[i][min_key] = daily_data[i - 1][min_key]
        if daily_data[i].get(max_key) is None and daily_data[i - 1].get(max_key) is not None:
            daily_data[i][max_key] = daily_data[i - 1][max_key]
        if daily_data[i].get(hour_min_key) is None and daily_data[i - 1].get(hour_min_key) is not None:
            daily_data[i][hour_min_key] = daily_data[i - 1][hour_min_key]
        if daily_data[i].get(hour_max_key) is None and daily_data[i - 1].get(hour_max_key) is not None:
            daily_data[i][hour_max_key] = daily_data[i - 1][hour_max_key]
    # Backward fill
    for i in range(n - 2, -1, -1):
        if daily_data[i].get(min_key) is None and daily_data[i + 1].get(min_key) is not None:
            daily_data[i][min_key] = daily_data[i + 1][min_key]
        if daily_data[i].get(max_key) is None and daily_data[i + 1].get(max_key) is not None:
            daily_data[i][max_key] = daily_data[i + 1][max_key]
        if daily_data[i].get(hour_min_key) is None and daily_data[i + 1].get(hour_min_key) is not None:
            daily_data[i][hour_min_key] = daily_data[i + 1][hour_min_key]
        if daily_data[i].get(hour_max_key) is None and daily_data[i + 1].get(hour_max_key) is not None:
            daily_data[i][hour_max_key] = daily_data[i + 1][hour_max_key]
def _extract_extremes(
    daily_data: List[dict],
    min_key: str,
    max_key: str,
    hour_min_key: str,
    hour_max_key: str,
    parse_hour_fn: Callable,
) -> List[Extreme]:
    """Build a chronologically-sorted list of extreme points from daily records.
    For each day the function emits two extremes (min and max) placed at
    the global hour index derived from the day index and the recorded hour.
    The two extremes are always emitted in chronological order.
    Args:
        daily_data:    Full list of daily records.
        min_key:       Dict key for the daily minimum value.
        max_key:       Dict key for the daily maximum value.
        hour_min_key:  Dict key for the hour of daily minimum.
        hour_max_key:  Dict key for the hour of daily maximum.
        parse_hour_fn: Callable that converts a raw hour string to ``int``.
    Returns:
        Sorted list of ``(global_hour, value, type)`` tuples.
    """
    extremes: List[Extreme] = []
    for day_idx, rec in enumerate(daily_data):
        val_min = rec.get(min_key)
        val_max = rec.get(max_key)
        if val_min is None or val_max is None:
            continue
        h_min = parse_hour_fn(rec.get(hour_min_key))
        h_max = parse_hour_fn(rec.get(hour_max_key))
        global_min = day_idx * 24 + h_min
        global_max = day_idx * 24 + h_max
        # Emit in chronological order
        if global_min <= global_max:
            extremes.append((global_min, float(val_min), "min"))
            extremes.append((global_max, float(val_max), "max"))
        else:
            extremes.append((global_max, float(val_max), "max"))
            extremes.append((global_min, float(val_min), "min"))
    return extremes
# ---------------------------------------------------------------------------
# Core: build a full continuous series from a list of extremes
# ---------------------------------------------------------------------------
def _build_continuous_series(
    extremes: List[Extreme],
    total_hours: int,
) -> List[float]:
    """Interpolate a continuous hourly series from sorted extremes.
    Handles the leading segment (before the first extreme) and the trailing
    segment (after the last extreme) by holding the nearest extreme's value
    constant.
    Args:
        extremes:    Sorted list of ``(global_hour, value, type)`` tuples.
        total_hours: Total length of the output series (``N_days * 24``).
    Returns:
        List of *total_hours* float values.
    """
    series: List[float] = [0.0] * total_hours
    if not extremes:
        return series
    # Leading flat segment
    first_hour, first_val, _ = extremes[0]
    for h in range(first_hour):
        series[h] = first_val
    # Interpolate between consecutive extremes
    for k in range(len(extremes) - 1):
        h0, v0, _ = extremes[k]
        h1, v1, _ = extremes[k + 1]
        segment = _cosine_interpolate(h0, v0, h1, v1)
        for j, val in enumerate(segment):
            series[h0 + j] = val
    # Trailing flat segment (+ set the last extreme value)
    last_hour, last_val, _ = extremes[-1]
    for h in range(last_hour, total_hours):
        series[h] = last_val
    return series
# ============================================================================
# Public API – continuous series generators
# ============================================================================
def generate_continuous_temperature(
    daily_data: List[dict],
    parse_hour_fn: Callable,
) -> List[Optional[float]]:
    """Generate a continuous hourly temperature series for the entire period.
    Args:
        daily_data:    List of daily record dicts (must contain
                       ``temperature_min``, ``temperature_max``,
                       ``hour_tmin``, ``hour_tmax``).
        parse_hour_fn: Callable that converts a raw hour field to ``int``.
    Returns:
        List of length ``len(daily_data) * 24`` with hourly temperatures
        rounded to 1 decimal place, or ``[None] * N`` if no data is available.
    """
    total_hours = len(daily_data) * 24
    # Check if any day has valid temperature data
    has_data = any(
        rec.get('temperature_min') is not None and rec.get('temperature_max') is not None
        for rec in daily_data
    )
    if not has_data:
        return [None] * total_hours
    # Work on a shallow copy to avoid mutating the caller's data
    working_data = [rec.copy() for rec in daily_data]
    _fill_missing_extremes(
        working_data,
        'temperature_min', 'temperature_max',
        'hour_tmin', 'hour_tmax',
        parse_hour_fn,
    )
    extremes = _extract_extremes(
        working_data,
        'temperature_min', 'temperature_max',
        'hour_tmin', 'hour_tmax',
        parse_hour_fn,
    )
    if not extremes:
        return [None] * total_hours
    extremes = _fix_boundary_flatness(
        extremes, working_data, 'temperature_min', 'temperature_max', 
        lambda: random.uniform(0.5, 2.0)
    )
    extremes = _insert_pseudo_extremes_temperature(extremes)
    series = _build_continuous_series(extremes, total_hours)
    return [round(v, 1) for v in series]
def generate_continuous_humidity(
    daily_data: List[dict],
    parse_hour_fn: Callable,
) -> List[Optional[int]]:
    """Generate a continuous hourly humidity series for the entire period.
    Args:
        daily_data:    List of daily record dicts (must contain
                       ``humidity_min``, ``humidity_max``,
                       ``hour_hrmin``, ``hour_hrmax``).
        parse_hour_fn: Callable that converts a raw hour field to ``int``.
    Returns:
        List of length ``len(daily_data) * 24`` with hourly integer
        humidity values clamped to [0, 100], or ``[None] * N``.
    """
    total_hours = len(daily_data) * 24
    has_data = any(
        rec.get('humidity_min') is not None and rec.get('humidity_max') is not None
        for rec in daily_data
    )
    if not has_data:
        return [None] * total_hours
    working_data = [rec.copy() for rec in daily_data]
    _fill_missing_extremes(
        working_data,
        'humidity_min', 'humidity_max',
        'hour_hrmin', 'hour_hrmax',
        parse_hour_fn,
    )
    extremes = _extract_extremes(
        working_data,
        'humidity_min', 'humidity_max',
        'hour_hrmin', 'hour_hrmax',
        parse_hour_fn,
    )
    if not extremes:
        return [None] * total_hours
    extremes = _fix_boundary_flatness(
        extremes, working_data, 'humidity_min', 'humidity_max', 
        lambda: random.uniform(2.0, 5.0)
    )
    extremes = _insert_pseudo_extremes_humidity(extremes)
    series = _build_continuous_series(extremes, total_hours)
    return [int(round(max(0, min(100, v)))) for v in series]
def generate_continuous_pressure(
    daily_data: List[dict],
    parse_hour_fn: Callable,
) -> List[Optional[float]]:
    """Generate a continuous hourly pressure series for the entire period.
    Args:
        daily_data:    List of daily record dicts (must contain
                       ``pressure_min``, ``pressure_max``,
                       ``hour_presmin``, ``hour_presmax``).
        parse_hour_fn: Callable that converts a raw hour field to ``int``.
    Returns:
        List of length ``len(daily_data) * 24`` with hourly pressure
        values rounded to 1 decimal, or ``[None] * N``.
    """
    total_hours = len(daily_data) * 24
    has_data = any(
        rec.get('pressure_min') is not None and rec.get('pressure_max') is not None
        for rec in daily_data
    )
    if not has_data:
        return [None] * total_hours
    working_data = [rec.copy() for rec in daily_data]
    _fill_missing_extremes(
        working_data,
        'pressure_min', 'pressure_max',
        'hour_presmin', 'hour_presmax',
        parse_hour_fn,
    )
    extremes = _extract_extremes(
        working_data,
        'pressure_min', 'pressure_max',
        'hour_presmin', 'hour_presmax',
        parse_hour_fn,
    )
    if not extremes:
        return [None] * total_hours
    extremes = _fix_boundary_flatness(
        extremes, working_data, 'pressure_min', 'pressure_max', 
        lambda: random.uniform(1.0, 3.0)
    )
    extremes = _insert_pseudo_extremes_pressure(extremes)
    series = _build_continuous_series(extremes, total_hours)
    return [round(v, 1) for v in series]
# ============================================================================
# Unchanged helpers – precipitation & wind
# ============================================================================
def distribute_precipitation(total_precip: float) -> List[float]:
    """Distribute daily precipitation across hours in a realistic way."""
    hourly_precip = [0.0] * 24
    if total_precip <= 0:
        return hourly_precip
    random.seed(int(total_precip * 1000))
    if total_precip < 5:
        rain_hours = random.randint(2, 4)
    elif total_precip < 15:
        rain_hours = random.randint(4, 6)
    else:
        rain_hours = random.randint(6, 8)
    start_hour = random.randint(0, 24 - rain_hours)
    for i in range(rain_hours):
        hour = (start_hour + i) % 24
        factor = exp(-((i - rain_hours / 2) ** 2) / (rain_hours / 2))
        hourly_precip[hour] = factor
    total_distributed = sum(hourly_precip)
    if total_distributed > 0:
        hourly_precip = [p * total_precip / total_distributed for p in hourly_precip]
    return [round(p, 1) for p in hourly_precip]
def interpolate_wind_speed(
    wind_avg: Optional[float],
    wind_max: Optional[float],
    hour_max: int,
) -> List[Optional[float]]:
    """Interpolate hourly wind speed with a Gaussian peak."""
    if wind_avg is None:
        return [None] * 24
    if wind_max is None or wind_max <= wind_avg:
        return [wind_avg] * 24
    # Special case: mean = 0
    if wind_avg == 0:
        hourly = [0.0] * 24
        hourly[hour_max] = round(wind_max, 1)
        return hourly
    # Generate Gaussian shape
    base = []
    for hour in range(24):
        dist = min(abs(hour - hour_max), 24 - abs(hour - hour_max))
        factor = exp(-(dist ** 2) / 2)
        base.append(factor)
    # Normalize so maximum is exactly 1
    max_base = max(base)
    base = [b / max_base for b in base]
    mean_base = sum(base) / 24
    # Solve system to satisfy mean and maximum
    b = (wind_max - wind_avg) / (1 - mean_base)
    a = wind_max - b
    hourly = [max(0, round(a + b * f, 1)) for f in base]
    return hourly
