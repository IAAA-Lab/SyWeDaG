"""Monthly precipitation adjustment helpers."""

import random
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

from utils.system_utils import safe_print

def adjust_precipitation_for_month(
    daily_data: List[Dict],
    indices: List[int],
    predictions_index: Dict,
    year: int,
    month: int,
) -> None:
    """
    Precipitation-adjustment dispatcher for a specific month.

    If there is a 'number_days_rain' prediction, use the advanced path (Phases 0-4).
    Otherwise, use the simple path (multiplicative factor).

    Args:
        daily_data: Full list of daily records (modified in-place)
        indices: Indices of this month's days
        predictions_index: Dictionary (year, month, variable) -> {mean, min, max}
        year: Year of the month
        month: Month (1-12)
    """
    prec_pred = predictions_index.get((year, month, "precipitation"))
    if not prec_pred:
        return

    pred_prec_mean = prec_pred.get("mean")
    if pred_prec_mean is None or pd.isna(pred_prec_mean):
        return
    pred_prec_mean = float(pred_prec_mean)

    days_rain_pred = predictions_index.get((year, month, "number_days_rain"))
    if days_rain_pred is not None:
        pred_days_mean = days_rain_pred.get("mean")
        if pred_days_mean is not None and not pd.isna(pred_days_mean):
            _adjust_precipitation_advanced(
                daily_data, indices, pred_prec_mean, prec_pred, days_rain_pred, year, month
            )
            return

    _adjust_precipitation_simple(daily_data, indices, pred_prec_mean, prec_pred)

# Simple adjustment
def _adjust_precipitation_simple(
    daily_data: List[Dict],
    indices: List[int],
    pred_prec_mean: float,
    prec_pred: Dict,
) -> None:
    """
    Simple precipitation adjustment using a multiplicative factor.

    Cases:
    - pred == 0  -> all days set to 0
    - hist == 0  -> generate 2 rain periods with prediction minimum value
    - else       -> multiplicative factor
    """
    
    prec_values = [daily_data[i].get("precipitation", 0) or 0 for i in indices]
    hist_prec_mean = np.mean(prec_values)

    # CASE 1: Prediction is 0 -> set all days to 0
    if pred_prec_mean == 0:
        for idx in indices:
            daily_data[idx]["precipitation"] = 0.0
        return

    # CASE 2: Historical is 0 but prediction > 0 -> generate rain periods
    # In this case, use prediction MINIMUM instead of mean
    if hist_prec_mean == 0:
        pred_prec_min = prec_pred.get("min")

        # If no minimum is defined, use mean as fallback
        if pred_prec_min is None or pd.isna(pred_prec_min):
            target_prec = pred_prec_mean
            target_label = "mean"
        else:
            target_prec = float(pred_prec_min)
            target_label = "minimum"

        first_date = datetime.strptime(daily_data[indices[0]]["date"], "%Y-%m-%d")
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        safe_print(
            f"🌧️ Generating 2 rain periods for "
            f"{month_names[first_date.month - 1]} {first_date.year} "
            f"({target_label}: {target_prec:.1f} mm/day)"
        )
        _generate_rain_periods(daily_data, indices, target_prec)
        return

    # CASE 3: Apply multiplicative factor
    factor = pred_prec_mean / hist_prec_mean
    for idx in indices:
        current_prec = daily_data[idx].get("precipitation", 0) or 0
        daily_data[idx]["precipitation"] = round(current_prec * factor, 1)

# Advanced adjustment
def _adjust_precipitation_advanced(
    daily_data: List[Dict],
    indices: List[int],
    pred_prec_mean: float,
    prec_pred: Dict,
    days_rain_pred: Dict,
    year: int,
    month: int,
) -> None:
    """
    Advanced precipitation adjustment in 4 phases:
        0. Trivial case: pred==0 or days==0 -> set all to 0
        1. Scan current month state
        2. Validate vs predictions (day deficit / excess)
        3. Adjust rainy-day count (add or remove)
        4. Multiplicative factor to match exact target precipitation
    """
    # Phase 0: trivial cases
    pred_days_mean = float(days_rain_pred.get("mean"))

    if pred_prec_mean == 0 or pred_days_mean == 0:
        for idx in indices:
            daily_data[idx]["precipitation"] = 0.0
        return

    # Phase 1: scan month
    analysis = _scan_month_rainfall(daily_data, indices)

    # If there is no historical rain, generate baseline periods before day adjustment
    if analysis["rainy_days"] == 0:
        _generate_rain_periods(daily_data, indices, pred_prec_mean)
        analysis = _scan_month_rainfall(daily_data, indices)

    # Phase 2: validate number of days against prediction
    validation = _validate_month_rainfall(analysis, days_rain_pred)
    days_status = validation["days_status"]

    # Phase 3: adjust rainy-day count
    if days_status["deficit"] > 0:
        _add_rainy_days(daily_data, indices, analysis, days_status["deficit"])
    elif days_status["excess"] > 0:
        _remove_rainy_days(daily_data, indices, analysis, days_status["excess"])

    # Phase 4: multiplicative factor with combined days x precipitation matrix
    days_in_month = len(indices)

    # FINAL day-state after Phase 3
    current_days = sum(
        1 for idx in indices if (daily_data[idx].get("precipitation", 0) or 0) > 0
    )
    pred_days_min = float(days_status["min"]) if days_status["min"] is not None else pred_days_mean
    pred_days_max = float(days_status["max"]) if days_status["max"] is not None else pred_days_mean

    if current_days <= pred_days_min:
        days_state = "AT_MINIMUM"
    elif current_days >= pred_days_max:
        days_state = "AT_MAXIMUM"
    else:
        days_state = "IN_RANGE"

    # Target precipitation totals
    pred_prec_min = prec_pred.get("min")
    pred_prec_max = prec_pred.get("max")
    pred_prec_min = (
        float(pred_prec_min)
        if pred_prec_min is not None and not pd.isna(pred_prec_min)
        else pred_prec_mean
    )
    pred_prec_max = (
        float(pred_prec_max)
        if pred_prec_max is not None and not pd.isna(pred_prec_max)
        else pred_prec_mean
    )

    min_total = pred_prec_min * days_in_month
    max_total = pred_prec_max * days_in_month
    mean_total = pred_prec_mean * days_in_month

    actual_total = sum(daily_data[idx].get("precipitation", 0) or 0 for idx in indices)

    # Combined matrix → target_total
    if actual_total < min_total:
        precip_state = "BELOW_MIN"
    elif actual_total > max_total:
        precip_state = "ABOVE_MAX"
    else:
        precip_state = "IN_RANGE"

    if days_state == "AT_MINIMUM":
        if precip_state == "BELOW_MIN":
            target_total = min_total
        elif precip_state == "ABOVE_MAX":
            target_total = mean_total
        else:  # IN_RANGE
            target_total = actual_total
    elif days_state == "AT_MAXIMUM":
        if precip_state == "BELOW_MIN":
            target_total = mean_total
        elif precip_state == "ABOVE_MAX":
            target_total = max_total
        else:  # IN_RANGE
            target_total = actual_total
    else:  # IN_RANGE
        if precip_state == "BELOW_MIN":
            target_total = min_total
        elif precip_state == "ABOVE_MAX":
            target_total = max_total
        else:  # IN_RANGE
            target_total = actual_total

    # No rain after removing days -> generating emergency periods
    if actual_total == 0 and target_total > 0:
        _generate_rain_periods(daily_data, indices, pred_prec_mean)
    
    # Apply multiplicative factor to match target total
    elif actual_total > 0 and abs(target_total - actual_total) > 0.1:
        factor = target_total / actual_total
        for idx in indices:
            current_prec = daily_data[idx].get("precipitation", 0) or 0
            if current_prec > 0:
                daily_data[idx]["precipitation"] = round(current_prec * factor, 1)

# Other helper functions for advanced adjustment phases
def _scan_month_rainfall(daily_data: List[Dict], indices: List[int]) -> Dict:
    """
    Phase 1: scan current monthly precipitation state.

    Returns:
        Dict with:
            total_precipitation, rainy_days, days_in_month, mean_precipitation,
            rain_series (list of {start_pos, end_pos, duration, total_precip, daily_values}),
            precip_by_day (list of values by position in indices)
    """
    precip_by_day = [daily_data[idx].get("precipitation", 0) or 0 for idx in indices]
    n = len(precip_by_day)
    rainy_days = sum(1 for p in precip_by_day if p > 0)
    total_precipitation = sum(precip_by_day)
    mean_precipitation = total_precipitation / n if n > 0 else 0.0

    rain_series = []
    i = 0
    while i < n:
        if precip_by_day[i] > 0:
            start_pos = i
            while i < n and precip_by_day[i] > 0:
                i += 1
            end_pos = i - 1
            daily_values = precip_by_day[start_pos : end_pos + 1]
            rain_series.append(
                {
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "duration": end_pos - start_pos + 1,
                    "total_precip": sum(daily_values),
                    "daily_values": list(daily_values),
                }
            )
        else:
            i += 1

    return {
        "total_precipitation": total_precipitation,
        "rainy_days": rainy_days,
        "days_in_month": n,
        "mean_precipitation": mean_precipitation,
        "rain_series": rain_series,
        "precip_by_day": precip_by_day,
    }


def _validate_month_rainfall(analysis: Dict, days_rain_pred: Dict) -> Dict:
    """
    Phase 2: compare current state with predicted rainy-day count.

    Returns:
        Dict with days_status: {current, min, max, mean, deficit, excess,
                                margin_up, margin_down}
    """
    current_days = analysis["rainy_days"]
    days_in_month = analysis["days_in_month"]

    raw_mean = days_rain_pred.get("mean")
    raw_min = days_rain_pred.get("min")
    raw_max = days_rain_pred.get("max")

    pred_mean = (
        int(round(float(raw_mean)))
        if raw_mean is not None and not pd.isna(raw_mean)
        else current_days
    )
    pred_min = (
        int(round(float(raw_min))) if raw_min is not None and not pd.isna(raw_min) else pred_mean
    )
    pred_max = (
        int(round(float(raw_max))) if raw_max is not None and not pd.isna(raw_max) else pred_mean
    )

    pred_min = max(0, min(pred_min, days_in_month))
    pred_max = max(0, min(pred_max, days_in_month))
    pred_mean = max(0, min(pred_mean, days_in_month))

    deficit = max(0, pred_min - current_days)
    excess = max(0, current_days - pred_max)

    return {
        "days_status": {
            "current": current_days,
            "min": pred_min,
            "max": pred_max,
            "mean": pred_mean,
            "deficit": deficit,
            "excess": excess,
            "margin_up": max(0, pred_max - current_days),
            "margin_down": max(0, current_days - pred_min),
        }
    }


def _find_dry_gaps(precip_by_day: List[float]) -> List[Dict]:
    """Return list of {start_pos, end_pos, duration} for dry sequences."""
    n = len(precip_by_day)
    gaps = []
    i = 0
    while i < n:
        if precip_by_day[i] == 0:
            start_pos = i
            while i < n and precip_by_day[i] == 0:
                i += 1
            end_pos = i - 1
            gaps.append(
                {
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "duration": end_pos - start_pos + 1,
                }
            )
        else:
            i += 1
    return gaps


def _add_rainy_days(
    daily_data: List[Dict], indices: List[int], analysis: Dict, deficit: int
) -> None:
    """
    Phase 3A: add rainy days until deficit is covered.

    Strategy:
    - If deficit == 1: try extending an existing series at one edge.
    - Remaining deficit: create up to 5-day series in available dry gaps.
    """
    precip_by_day = list(analysis["precip_by_day"])
    mean_precipitation = analysis["mean_precipitation"]
    deficit_remaining = deficit

    # Try extending an existing series when only 1 day is missing
    if deficit_remaining == 1 and analysis["rain_series"]:
        for serie in analysis["rain_series"]:
            right_pos = serie["end_pos"] + 1
            if right_pos < len(precip_by_day) and precip_by_day[right_pos] == 0:
                new_val = max(0.1, round(mean_precipitation * random.uniform(0.5, 1.5), 1))
                precip_by_day[right_pos] = new_val
                daily_data[indices[right_pos]]["precipitation"] = new_val
                deficit_remaining = 0
                break
            left_pos = serie["start_pos"] - 1
            if left_pos >= 0 and precip_by_day[left_pos] == 0:
                new_val = max(0.1, round(mean_precipitation * random.uniform(0.5, 1.5), 1))
                precip_by_day[left_pos] = new_val
                daily_data[indices[left_pos]]["precipitation"] = new_val
                deficit_remaining = 0
                break

    # Create series in dry gaps for remaining deficit
    while deficit_remaining > 0:
        series_size = min(5, deficit_remaining)
        series_total = mean_precipitation * series_size

        gaps = _find_dry_gaps(precip_by_day)
        suitable = [g for g in gaps if g["duration"] >= series_size]

        if not suitable:
            # Use the largest available gap
            if not gaps:
                break # No gaps left
            gap = max(gaps, key=lambda g: g["duration"])
            series_size = gap["duration"]
            series_total = mean_precipitation * series_size
        else:
            gap = random.choice(suitable)

        max_start = gap["end_pos"] - series_size + 1
        start_pos = random.randint(gap["start_pos"], max_start)

        rainfall = _generate_variable_rainfall(series_size, series_total)
        for i in range(series_size):
            pos = start_pos + i
            precip_by_day[pos] = rainfall[i]
            daily_data[indices[pos]]["precipitation"] = rainfall[i]

        deficit_remaining -= series_size


def _remove_rainy_days(
    daily_data: List[Dict], indices: List[int], analysis: Dict, excess: int
) -> None:
    """
    Phase 3B: remove rainy days until excess is reduced.

        Strategy (in order):
        1. Remove full series with duration == excess (exact match).
            If none exists, iteratively remove small series (smallest first)
            until excess is covered. This automatically includes 1-day series.
        2. Remove edges in a distributed way among remaining series
            (sorted by ascending total_precip), keeping duration >= 2.
    """
    rain_series = [dict(s) for s in analysis["rain_series"]]
    precip_by_day = list(analysis["precip_by_day"])
    excess_remaining = excess

    def _zero_series(serie: Dict) -> None:
        for pos in range(serie["start_pos"], serie["end_pos"] + 1):
            precip_by_day[pos] = 0.0
            daily_data[indices[pos]]["precipitation"] = 0.0

    # Step 1: remove series (exact + small)
    exact = next(
        (s for s in sorted(rain_series, key=lambda s: s["duration"]) if s["duration"] == excess_remaining),
        None,
    )
    if exact:
        _zero_series(exact)
        rain_series.remove(exact)
        excess_remaining = 0
    else:
        # Iteratively remove small series (smallest first)
        while excess_remaining > 0 and rain_series:
            candidates = [s for s in rain_series if s["duration"] <= excess_remaining]
            if not candidates:
                break
            # Remove the smallest candidate series
            serie_to_remove = min(candidates, key=lambda s: s["duration"])
            _zero_series(serie_to_remove)
            excess_remaining -= serie_to_remove["duration"]
            rain_series.remove(serie_to_remove)

    if excess_remaining <= 0:
        return

    # Step 2: remove distributed edges among remaining series
    for serie in sorted(rain_series, key=lambda s: s["total_precip"]):
        cur_start = serie["start_pos"]
        cur_end = serie["end_pos"]
        cur_values = list(serie["daily_values"])

        while excess_remaining > 0 and (cur_end - cur_start + 1) >= 2:
            if cur_values[0] <= cur_values[-1]:
                precip_by_day[cur_start] = 0.0
                daily_data[indices[cur_start]]["precipitation"] = 0.0
                cur_start += 1
                cur_values = cur_values[1:]
            else:
                precip_by_day[cur_end] = 0.0
                daily_data[indices[cur_end]]["precipitation"] = 0.0
                cur_end -= 1
                cur_values = cur_values[:-1]
            excess_remaining -= 1

        if excess_remaining <= 0:
            break


def _generate_rain_periods(
    daily_data: List[Dict], indices: List[int], target_mean: float
) -> None:
    """
    Generate 2 periods of 2-5 consecutive rainy days to reach the target mean.
    Precipitation varies randomly across days while preserving total mean.
    
    Args:
        daily_data: Full list of daily records (modified in-place)
        indices: Indices of this month's days
        target_mean: Target precipitation mean (mm/day)
    """
    # Initialize all days to 0
    for idx in indices:
        daily_data[idx]["precipitation"] = 0.0

    n_days = len(indices)
    if n_days == 0:
        return

    # Generate 2 rain periods
    total_prec_needed = target_mean * n_days

    # Duration of each period: random between 2 and 5 days
    period1_days = random.randint(2, min(5, n_days // 2))
    period2_days = random.randint(2, min(5, n_days // 2))

    total_rain_days = period1_days + period2_days
    if total_rain_days > n_days:
        period1_days = n_days // 2
        period2_days = n_days - period1_days
        total_rain_days = period1_days + period2_days

    # Place first period (first half of month)
    max_start1 = max(0, n_days // 2 - period1_days)
    start1 = random.randint(0, max(0, max_start1))

    # Place second period (second half of month)
    half_month = n_days // 2
    max_start2 = max(half_month, n_days - period2_days)
    start2 = random.randint(half_month, max_start2)

    # Distribute total precipitation between periods (proportional to duration)
    period1_total = total_prec_needed * (period1_days / total_rain_days)
    period2_total = total_prec_needed * (period2_days / total_rain_days)

    # Generate random rainfall distribution for first period
    period1_values = _generate_variable_rainfall(period1_days, period1_total)
    
    # Generate random rainfall distribution for second period
    period2_values = _generate_variable_rainfall(period2_days, period2_total)

    # Apply rain to first period
    for i in range(period1_days):
        idx = start1 + i
        if idx < len(indices):
            daily_data[indices[idx]]["precipitation"] = period1_values[i]

    # Apply rain to second period
    for i in range(period2_days):
        idx = start2 + i
        if idx < len(indices):
            daily_data[indices[idx]]["precipitation"] = period2_values[i]


def _generate_variable_rainfall(n_days: int, total_rainfall: float) -> List[float]:
    """
    Generate a variable precipitation distribution for n days that sums exactly to the total.

    Uses a random distribution to simulate natural rainfall variability,
    where some days are more intense and others less so.
    
    Args:
        n_days: Number of rainy days
        total_rainfall: Total precipitation to distribute (mm)
        
    Returns:
        List of precipitation values for each day (rounded to 1 decimal)
    """
    if n_days <= 0:
        return []

    if total_rainfall <= 0:
        return [0.0] * n_days

    # Generate random weights using exponential distribution
    # This simulates that some days rain more intensely than others
    weights = [random.expovariate(1.0) for _ in range(n_days)]
    
    # Normalize weights so they sum to 1
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Distribute precipitation according to weights
    rainfall_values = [total_rainfall * w for w in normalized_weights]
    
    # Round to 1 decimal
    rounded_values = [round(val, 1) for val in rainfall_values]

    # Adjust to compensate rounding errors and keep exact sum
    current_sum = sum(rounded_values)
    diff = round(total_rainfall - current_sum, 1)

    if diff != 0:
        # Add difference to the day with highest precipitation
        max_idx = rounded_values.index(max(rounded_values))
        rounded_values[max_idx] = round(rounded_values[max_idx] + diff, 1)

    return rounded_values
