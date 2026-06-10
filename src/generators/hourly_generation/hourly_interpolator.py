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
def _smoothstep(x: float) -> float:
    """Cosine easing for smooth interpolation (0 → 1)."""
    x = max(0.0, min(1.0, x))
    return (1 - cos(pi * x)) / 2


def _fix_boundary_flatness(
    extremes: List[Extreme], 
    daily_data: List[dict], 
    min_key: str, 
    max_key: str
) -> List[Extreme]:
    """Prevent flat segments at the start and end using temporal interpolation."""

    if not extremes:
        return extremes

    new_extremes = list(extremes)
    total_hours = len(daily_data) * 24

    # Fix start
    first_hour, first_val, first_type = new_extremes[0]
    if first_hour > 0:
        d0 = daily_data[0]
        day_min = float(d0.get(min_key, first_val))
        day_max = float(d0.get(max_key, first_val))
        range_day = max(day_max - day_min, 0.1)
        d_max = 15
        d = min(first_hour, d_max)
        r = _smoothstep(d / d_max)

        if first_type == "max":
            pseudo_val = first_val - range_day * r
        else:
            pseudo_val = first_val + range_day * r

        new_extremes.insert(0, (0, pseudo_val, "min" if first_type == "max" else "max"))

    # Fix end
    last_hour, last_val, last_type = new_extremes[-1]
    if last_hour < total_hours - 1:
        d_last = daily_data[-1]
        day_min = float(d_last.get(min_key, last_val))
        day_max = float(d_last.get(max_key, last_val))

        range_day = max(day_max - day_min, 0.1)
        d_max = 9
        d = min(total_hours - last_hour, d_max)
        r = _smoothstep(d / d_max)

        if last_type == "max":
            pseudo_val = last_val - range_day * r
        else:
            pseudo_val = last_val + range_day * r
        
        new_extremes.append(
            (total_hours - 1,
             pseudo_val,
             "min" if last_type == "max" else "max")
        )

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

# ---------------------------------------------------------------------------
# Mean correction
# ---------------------------------------------------------------------------
def _apply_mean_correction(
    series: List[float],
    daily_data: List[dict],
    parse_hour_fn: Callable,
    mean_key: str,
    min_key: str,
    max_key: str,
    hour_min_key: str,
    hour_max_key: str,
    tolerance: float,
    max_iterations: int = 15
) -> List[float]:
    """Correct continuous series to approximate daily means while respecting extremes."""
    total_hours = len(series)
    if total_hours == 0:
        return series
        
    corrected_series = list(series)

    # Phase 3: Weights
    weights = [1.0] * total_hours
    for day_idx, rec in enumerate(daily_data):
        h_min_str = rec.get(hour_min_key)
        h_max_str = rec.get(hour_max_key)
        if h_min_str is None or h_max_str is None:
            continue
            
        h_min = parse_hour_fn(h_min_str)
        h_max = parse_hour_fn(h_max_str)
        
        for h in range(24):
            global_h = day_idx * 24 + h
            if global_h < total_hours:

                # Peso respecto a Tmin/Tmax
                dist_min = abs(h - h_min)
                dist_max = abs(h - h_max)
                dist_extreme = min(dist_min, dist_max)
                extreme_weight = _smoothstep(min(1.0, dist_extreme / 6.0))

                # Peso respecto al cambio de día (0h y 23h)
                dist_boundary = min(h, 23 - h)
                boundary_weight = _smoothstep(min(1.0, dist_boundary / 4.0))

                weights[global_h] = extreme_weight * boundary_weight

    for _ in range(max_iterations):
        # Phase 1: Calculate daily error
        errors = []
        max_abs_error = 0.0
        has_mean_data = False
        
        for day_idx, rec in enumerate(daily_data):
            target_mean = rec.get(mean_key)
            if target_mean is None:
                errors.append(0.0)
                continue
                
            has_mean_data = True
            start_h = day_idx * 24
            end_h = start_h + 24
            day_slice = corrected_series[start_h:end_h]
            if not day_slice:
                errors.append(0.0)
                continue
            
            generated_mean = sum(day_slice) / len(day_slice)
            error_day = float(target_mean) - generated_mean
            errors.append(error_day)
            max_abs_error = max(max_abs_error, abs(error_day))
            
        if not has_mean_data or max_abs_error <= tolerance:
            break
            
        # Phase 2: Continuous correction signal
        correction_points = []
        for day_idx, error in enumerate(errors):
            correction_points.append((day_idx * 24 + 12, error))
            
        if correction_points:
            correction_points.insert(0, (0, correction_points[0][1]))
            correction_points.append((total_hours, correction_points[-1][1]))
            
        correction_series = [0.0] * total_hours
        if len(correction_points) > 1:
            for i in range(len(correction_points) - 1):
                h1, v1 = correction_points[i]
                h2, v2 = correction_points[i+1]
                segment = _cosine_interpolate(h1, v1, h2, v2)
                for j, val in enumerate(segment):
                    if h1 + j < total_hours:
                        correction_series[h1 + j] = val
        
        # Phase 4 & 5: Apply correction & Limit
        for h in range(total_hours):
            corrected = corrected_series[h] + correction_series[h] * weights[h]
            
            day_idx = h // 24
            if day_idx < len(daily_data):
                rec = daily_data[day_idx]
                day_min = rec.get(min_key)
                day_max = rec.get(max_key)
                if day_min is not None:
                    corrected = max(corrected, float(day_min))
                if day_max is not None:
                    corrected = min(corrected, float(day_max))
            
            corrected_series[h] = corrected

    return corrected_series

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
        extremes, working_data, 'temperature_min', 'temperature_max'
    )
    extremes = _insert_pseudo_extremes_temperature(extremes)
    series = _build_continuous_series(extremes, total_hours)
    
    series = _apply_mean_correction(
        series=series,
        daily_data=working_data,
        parse_hour_fn=parse_hour_fn,
        mean_key='temperature_mean',
        min_key='temperature_min',
        max_key='temperature_max',
        hour_min_key='hour_tmin',
        hour_max_key='hour_tmax',
        tolerance=0.05
    )
    
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
        extremes, working_data, 'humidity_min', 'humidity_max'
    )
    extremes = _insert_pseudo_extremes_humidity(extremes)
    series = _build_continuous_series(extremes, total_hours)
    
    series = _apply_mean_correction(
        series=series,
        daily_data=working_data,
        parse_hour_fn=parse_hour_fn,
        mean_key='humidity_mean',
        min_key='humidity_min',
        max_key='humidity_max',
        hour_min_key='hour_hrmin',
        hour_max_key='hour_hrmax',
        tolerance=0.5
    )
    
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
        extremes, working_data, 'pressure_min', 'pressure_max'
    )
    extremes = _insert_pseudo_extremes_pressure(extremes)
    series = _build_continuous_series(extremes, total_hours)
    
    series = _apply_mean_correction(
        series=series,
        daily_data=working_data,
        parse_hour_fn=parse_hour_fn,
        mean_key='pressure_mean',
        min_key='pressure_min',
        max_key='pressure_max',
        hour_min_key='hour_presmin',
        hour_max_key='hour_presmax',
        tolerance=0.05
    )
    
    return [round(v, 1) for v in series]

# ============================================================================
# Unchanged helpers – precipitation & wind
# ============================================================================
def generate_continuous_wind(
    daily_data: List[dict],
    parse_hour_fn: Callable,
) -> List[Optional[float]]:
    """Generate a continuous hourly wind speed series for the entire period.
    
    This replaces the older per-day approach, treating `wind_max` as a soft 
    constraint (target max between 80% and 95%) and using a multiplicative
    scaling factor interpolated over time to gently approximate the daily
    mean (`wind_avg`) without introducing day-boundary discontinuities.
    
    The series is composed of:
      V(t) = (B(t) + N(t)) * Factor(t) + G(t) * Factor(t)
    where B is the trend, N is smoothed natural noise, G is the gust episode.
    
    Args:
        daily_data:    List of daily record dicts.
        parse_hour_fn: Callable that converts a raw hour field to `int`.
        
    Returns:
        List of length `len(daily_data) * 24` with hourly wind speeds
        rounded to 1 decimal, or `[None] * N`.
    """
    total_hours = len(daily_data) * 24
    has_data = any(rec.get('wind_speed_mean') is not None for rec in daily_data)
    if not has_data:
        return [None] * total_hours

    # 1. Base component B(t)
    B = [0.0] * total_hours
    mean_points = []
    for day_idx, rec in enumerate(daily_data):
        val = rec.get('wind_speed_mean')
        if val is None:
            val = 0.0
        mean_points.append((day_idx * 24 + 12, val))
        
    if not mean_points:
        return [None] * total_hours

    mean_points.insert(0, (0, mean_points[0][1]))
    mean_points.append((total_hours, mean_points[-1][1]))
    
    for i in range(len(mean_points) - 1):
        h1, v1 = mean_points[i]
        h2, v2 = mean_points[i+1]
        segment = _cosine_interpolate(h1, v1, h2, v2)
        for j, val in enumerate(segment):
            if h1 + j < total_hours:
                B[h1 + j] = val

    # 2. Turbulence / Noise component N(t)
    # Combine a slow wave and a fast wave for continuous natural turbulence
    slow_noise_points = []
    for h in range(0, total_hours + 6, 6):
        slow_noise_points.append((h, random.uniform(-1.0, 1.0)))
        
    fast_noise_points = []
    for h in range(0, total_hours + 2, 2):  # Higher frequency: every 2 hours
        fast_noise_points.append((h, random.uniform(-1.0, 1.0)))
        
    slow_N = [0.0] * total_hours
    for i in range(len(slow_noise_points) - 1):
        h1, v1 = slow_noise_points[i]
        h2, v2 = slow_noise_points[i+1]
        segment = _cosine_interpolate(h1, v1, h2, v2)
        for j, val in enumerate(segment):
            if h1 + j < total_hours:
                slow_N[h1 + j] = val
                
    fast_N = [0.0] * total_hours
    for i in range(len(fast_noise_points) - 1):
        h1, v1 = fast_noise_points[i]
        h2, v2 = fast_noise_points[i+1]
        segment = _cosine_interpolate(h1, v1, h2, v2)
        for j, val in enumerate(segment):
            if h1 + j < total_hours:
                fast_N[h1 + j] = val

    Combined_N = [0.0] * total_hours
    for h in range(total_hours):
        # Blend slow and fast noise for a spiky but connected profile
        Combined_N[h] = 0.4 * slow_N[h] + 0.6 * fast_N[h]

    # 3. Gust component G(t)
    G = [0.0] * total_hours
    for day_idx, rec in enumerate(daily_data):
        wind_max = rec.get('wind_speed_max')
        wind_mean = rec.get('wind_speed_mean')
        
        if wind_max is None or wind_mean is None or wind_max <= wind_mean:
            continue
            
        hour_max_str = rec.get('hour_wind_max')
        hour_max = parse_hour_fn(hour_max_str)
        global_hour_max = day_idx * 24 + hour_max
        
        # Less aggressive max target (60% to 90%)
        target_max = wind_max * random.uniform(0.60, 0.90)
        
        # Reduced amplitude so G(t) doesn't dominate the series
        diff = target_max - wind_mean
        amplitude = max(0.0, diff * random.uniform(0.50, 0.60))
        
        # Much wider episode (8 to 12 hours half-width)
        width = random.randint(8, 12)
        
        for h in range(global_hour_max - width, global_hour_max + width + 1):
            if 0 <= h < total_hours:
                dist = abs(h - global_hour_max)
                factor = (1 + cos(pi * dist / width)) / 2
                G[h] += amplitude * factor

    # 4. Combine into Raw(t)
    raw_series = [0.0] * total_hours
    for h in range(total_hours):
        # Base level is B(t) + G(t). We apply turbulence proportionally.
        # Combined_N is in [-1, 1]. A 50% turbulence means * (1 + 0.5 * N).
        base_level = B[h] + G[h]
        val = base_level * (1 + 0.5 * Combined_N[h])
        raw_series[h] = max(0.1, val)

    # 5. Smooth Multiplicative Scaling
    factor_points = []
    for day_idx, rec in enumerate(daily_data):
        target_mean = rec.get('wind_speed_mean')
        if target_mean is None:
            target_mean = 0.0
            
        start_h = day_idx * 24
        end_h = start_h + 24
        day_raw = raw_series[start_h:end_h]
        current_mean = sum(day_raw) / 24
        
        if current_mean > 0:
            daily_factor = target_mean / current_mean
        else:
            daily_factor = 1.0
            
        factor_points.append((day_idx * 24 + 12, daily_factor))

    factor_points.insert(0, (0, factor_points[0][1]))
    factor_points.append((total_hours, factor_points[-1][1]))
    
    factor_series = [1.0] * total_hours
    for i in range(len(factor_points) - 1):
        h1, v1 = factor_points[i]
        h2, v2 = factor_points[i+1]
        segment = _cosine_interpolate(h1, v1, h2, v2)
        for j, val in enumerate(segment):
            if h1 + j < total_hours:
                factor_series[h1 + j] = val

    # 6. Final calculation
    final_series = []
    for h in range(total_hours):
        final_val = raw_series[h] * factor_series[h]
        
        day_idx = h // 24
        wind_max = daily_data[day_idx].get('wind_speed_max')
        if wind_max is not None:
            final_val = min(final_val, wind_max)
            
        final_series.append(round(max(0.0, final_val), 1))
        
    return final_series

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
    """Interpolate hourly wind speed with a Gaussian peak.
    
    DEPRECATED: Use `generate_continuous_wind` instead. This function 
    is kept for backward compatibility and generates wind for a single
    day independently, which causes discontinuities at day boundaries.
    """
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
