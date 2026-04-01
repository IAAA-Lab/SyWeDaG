"""Hourly interpolation functions for synthetic weather generation."""

import random
from math import cos, exp, pi
from typing import List, Optional


def interpolate_hourly_temperature(
    tmin: float,
    tmax: float,
    tmean: Optional[float],
    hour_min: int,
    hour_max: int,
) -> List[float]:
    """Interpolate 24 hourly temperatures from daily min/max/mean values."""
    if tmean is None:
        tmean = (tmin + tmax) / 2

    if tmax == tmin:
        return [round(tmin, 1)] * 24

    hours_between = (hour_max - hour_min) % 24
    if hours_between == 0:
        hours_between = 12

    # Generate normalized base shape f(h) in [0,1]
    f_values = []

    for hour in range(24):
        hours_since_min = (hour - hour_min) % 24

        if hours_since_min <= hours_between:
            progress = hours_since_min / hours_between
            f = (1 - cos(pi * progress)) / 2
        else:
            hours_since_max = hours_since_min - hours_between
            hours_to_next_min = 24 - hours_between
            progress = hours_since_max / hours_to_next_min
            f = 1 - (1 - cos(pi * progress)) / 2

        f_values.append(f)

    # Find exponent alpha that preserves the mean
    def compute_mean(alpha: float) -> float:
        temps = [tmin + (tmax - tmin) * (f ** alpha) for f in f_values]
        return sum(temps) / 24

    # Stable bisection
    low, high = 0.1, 5.0

    for _ in range(50):
        mid = (low + high) / 2
        m = compute_mean(mid)

        if m > tmean:
            low = mid
        else:
            high = mid

    alpha = (low + high) / 2

    # Build final temperatures
    hourly_temps = [tmin + (tmax - tmin) * (f ** alpha) for f in f_values]

    # Force exact extrema
    hourly_temps[hour_min] = tmin
    hourly_temps[hour_max] = tmax

    return [round(t, 1) for t in hourly_temps]


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


def interpolate_hourly_humidity(
    hr_min: float,
    hr_max: float,
    hr_mean: Optional[float],
    hour_hrmin: int,
    hour_hrmax: int,
) -> List[int]:
    """Interpolate hourly relative humidity with mean adjustment (similar to temperature).

    Generate a normalized cosine curve f(h) in [0,1] between hr_min and hr_max,
    and find exponent alpha so hourly mean matches hr_mean.

    Args:
        hr_min: Daily minimum relative humidity (%)
        hr_max: Daily maximum relative humidity (%)
        hr_mean: Daily mean relative humidity (%)
        hour_hrmin: Hour of minimum humidity
        hour_hrmax: Hour of maximum humidity

    Returns:
        List of 24 integer relative-humidity values (%)
    """
    if hr_mean is None:
        hr_mean = (hr_max + hr_min) / 2

    if hr_max == hr_min:
        return [int(round(hr_min))] * 24

    hours_between = (hour_hrmin - hour_hrmax) % 24
    if hours_between == 0:
        hours_between = 12

    # Generate normalized base shape f(h) in [0,1]
    # where 0 = hr_min and 1 = hr_max
    f_values = []
    for hour in range(24):
        hours_since_max = (hour - hour_hrmax) % 24

        if hours_since_max <= hours_between:
            # Decrease from maximum to minimum
            progress = hours_since_max / hours_between
            f = 1 - (1 - cos(pi * progress)) / 2
        else:
            # Increase from minimum to maximum
            hours_since_min = hours_since_max - hours_between
            hours_to_next_max = 24 - hours_between
            progress = hours_since_min / hours_to_next_max
            f = (1 - cos(pi * progress)) / 2

        f_values.append(f)

    # Find exponent alpha that preserves the mean
    def compute_mean(alpha: float) -> float:
        humidities = [hr_min + (hr_max - hr_min) * (f ** alpha) for f in f_values]
        return sum(humidities) / 24

    # Stable bisection
    low, high = 0.1, 5.0
    for _ in range(50):
        mid = (low + high) / 2
        m = compute_mean(mid)
        if m > hr_mean:
            low = mid
        else:
            high = mid

    alpha = (low + high) / 2

    # Build final humidity values
    hourly_humidity = [hr_min + (hr_max - hr_min) * (f ** alpha) for f in f_values]

    # Force exact extrema
    hourly_humidity[hour_hrmax] = hr_max
    hourly_humidity[hour_hrmin] = hr_min

    return [int(round(max(0, min(100, h)))) for h in hourly_humidity]


def interpolate_pressure(
    pres_min: float,
    pres_max: float,
    hour_presmin: int,
    hour_presmax: int,
) -> List[float]:
    """Interpolate hourly atmospheric pressure with a cosine curve."""
    hourly_pressure = []
    for hour in range(24):
        hours_since_min = (hour - hour_presmin) % 24
        hours_between = (hour_presmax - hour_presmin) % 24
        if hours_between == 0:
            hours_between = 12

        if hours_since_min <= hours_between:
            progress = hours_since_min / hours_between
            smooth_progress = (1 - cos(pi * progress)) / 2
            pressure = pres_min + (pres_max - pres_min) * smooth_progress
        else:
            hours_since_max = hours_since_min - hours_between
            hours_to_next_min = 24 - hours_between
            progress = hours_since_max / hours_to_next_min
            smooth_progress = (1 - cos(pi * progress)) / 2
            pressure = pres_max - (pres_max - pres_min) * smooth_progress

        hourly_pressure.append(round(pressure, 1))
    return hourly_pressure
