"""Monthly temperature adjustment helpers."""

from typing import Dict, List
import numpy as np
import pandas as pd

def adjust_temperatures_for_month(
    daily_data: List[Dict],
    indices: List[int],
    predictions_index: Dict,
    year: int,
    month: int,
) -> None:
    """
    Temperature adjustment with per-record limit validation.
    
    Args:
        daily_data: Full list of daily records (modified in-place)
        indices: Indices of the days in this month
        predictions_index: Dictionary (year, month, variable) -> {mean, min, max}
        year: Year of the month
        month: Month (1-12)
    """
    # Get predictions for this month
    tmax_pred = predictions_index.get((year, month, "temperature_max"))
    tmean_pred = predictions_index.get((year, month, "temperature_mean"))
    tmin_pred = predictions_index.get((year, month, "temperature_min"))

    if not all([tmax_pred, tmean_pred, tmin_pred]):
        return

    # Validate that mean values exist
    if any(pd.isna(pred.get("mean")) for pred in [tmax_pred, tmean_pred, tmin_pred]):
        return

    # Compute monthly statistics of current historical values
    tmax_values = [
        daily_data[i].get("temperature_max")
        for i in indices
        if daily_data[i].get("temperature_max") is not None
    ]
    tmean_values = [
        daily_data[i].get("temperature_mean")
        for i in indices
        if daily_data[i].get("temperature_mean") is not None
    ]
    tmin_values = [
        daily_data[i].get("temperature_min")
        for i in indices
        if daily_data[i].get("temperature_min") is not None
    ]

    if not all([tmax_values, tmean_values, tmin_values]):
        return

    hist_tmax_mean = np.mean(tmax_values)
    hist_tmean_mean = np.mean(tmean_values)
    hist_tmin_mean = np.mean(tmin_values)

    # Compute differences (prediction - historical)
    diff_tmax = tmax_pred["mean"] - hist_tmax_mean
    diff_tmean = tmean_pred["mean"] - hist_tmean_mean
    diff_tmin = tmin_pred["mean"] - hist_tmin_mean

    # Extract prediction bounds
    tmax_min = float(tmax_pred["min"]) if pd.notna(tmax_pred.get("min")) else None
    tmax_max = float(tmax_pred["max"]) if pd.notna(tmax_pred.get("max")) else None
    tmean_min = float(tmean_pred["min"]) if pd.notna(tmean_pred.get("min")) else None
    tmean_max = float(tmean_pred["max"]) if pd.notna(tmean_pred.get("max")) else None
    tmin_min = float(tmin_pred["min"]) if pd.notna(tmin_pred.get("min")) else None
    tmin_max = float(tmin_pred["max"]) if pd.notna(tmin_pred.get("max")) else None

    # Adjust each day individually
    for idx in indices:
        record = daily_data[idx]

        tmax = record.get("temperature_max")
        tmean = record.get("temperature_mean")
        tmin = record.get("temperature_min")

        if tmax is None or tmean is None or tmin is None:
            continue

        # Apply differences
        new_tmax = tmax + diff_tmax
        new_tmean = tmean + diff_tmean
        new_tmin = tmin + diff_tmin

        # Adjust Tmax to its bounds
        if tmax_max is not None and new_tmax > tmax_max:
            new_tmax = tmax_max
        if tmax_min is not None and new_tmax < tmax_min:
            new_tmax = tmax_min

        # Adjust Tmin to its bounds
        if tmin_max is not None and new_tmin > tmin_max:
            new_tmin = tmin_max
        if tmin_min is not None and new_tmin < tmin_min:
            new_tmin = tmin_min

        # Adjust Tmean to its bounds
        if tmean_max is not None and new_tmean > tmean_max:
            new_tmean = tmean_max
        if tmean_min is not None and new_tmean < tmean_min:
            new_tmean = tmean_min

        if new_tmean > new_tmax:
            avg = (new_tmean + new_tmax) / 2.0
            new_tmean = avg
            new_tmax = avg

        if new_tmean < new_tmin:
            avg = (new_tmean + new_tmin) / 2.0
            new_tmean = avg
            new_tmin = avg

        if new_tmax < new_tmin:
            avg = (new_tmax + new_tmin) / 2.0
            new_tmax = avg
            new_tmin = avg

        # Save adjusted values with 1 decimal
        record["temperature_max"] = round(new_tmax, 1)
        record["temperature_mean"] = round(new_tmean, 1)
        record["temperature_min"] = round(new_tmin, 1)
