import copy

import numpy as np
import pandas as pd

from generators.monthly_adjustments.temperature_adjuster import adjust_temperatures_for_month


def _make_january_days(n_days=31, tmin=2.0, tmean=6.0, tmax=10.0):
    daily_data = []
    for day in range(1, n_days + 1):
        daily_data.append({
            "date": f"2020-01-{day:02d}",
            "temperature_min": tmin,
            "temperature_max": tmax,
            "temperature_mean": tmean,
        })
    return daily_data


def _predictions_index(tmax, tmean, tmin, tmax_bounds=None, tmean_bounds=None, tmin_bounds=None):
    tmax_bounds = tmax_bounds or (None, None)
    tmean_bounds = tmean_bounds or (None, None)
    tmin_bounds = tmin_bounds or (None, None)
    return {
        (2020, 1, "temperature_max"): {"mean": tmax, "min": tmax_bounds[0], "max": tmax_bounds[1]},
        (2020, 1, "temperature_mean"): {"mean": tmean, "min": tmean_bounds[0], "max": tmean_bounds[1]},
        (2020, 1, "temperature_min"): {"mean": tmin, "min": tmin_bounds[0], "max": tmin_bounds[1]},
    }


def test_adjusted_monthly_mean_matches_prediction():
    daily_data = _make_january_days()
    indices = list(range(len(daily_data)))
    predictions = _predictions_index(tmax=15.0, tmean=10.0, tmin=5.0)

    adjust_temperatures_for_month(daily_data, indices, predictions, 2020, 1)

    actual_tmax_mean = np.mean([r["temperature_max"] for r in daily_data])
    actual_tmean_mean = np.mean([r["temperature_mean"] for r in daily_data])
    actual_tmin_mean = np.mean([r["temperature_min"] for r in daily_data])

    assert actual_tmax_mean == 15.0
    assert actual_tmean_mean == 10.0
    assert actual_tmin_mean == 5.0


def test_order_invariant_holds_after_adjustment():
    # Historical days already have Tmin < Tmean < Tmax; push predictions
    # that would otherwise cross the order (very hot mean, cold min/max targets).
    daily_data = _make_january_days(tmin=2.0, tmean=6.0, tmax=10.0)
    indices = list(range(len(daily_data)))
    predictions = _predictions_index(tmax=6.0, tmean=20.0, tmin=1.0)

    adjust_temperatures_for_month(daily_data, indices, predictions, 2020, 1)

    for record in daily_data:
        assert record["temperature_min"] <= record["temperature_mean"] <= record["temperature_max"]


def test_bounds_are_respected_when_tighter_than_history():
    daily_data = _make_january_days(tmin=2.0, tmean=6.0, tmax=10.0)
    indices = list(range(len(daily_data)))
    predictions = _predictions_index(
        tmax=15.0, tmean=10.0, tmin=5.0,
        tmax_bounds=(12.0, 14.0),
        tmean_bounds=(8.0, 9.0),
        tmin_bounds=(4.0, 4.5),
    )

    adjust_temperatures_for_month(daily_data, indices, predictions, 2020, 1)

    for record in daily_data:
        assert record["temperature_max"] <= 14.0
        assert record["temperature_mean"] <= 9.0
        assert record["temperature_min"] <= 4.5


def test_missing_predictions_leaves_data_unchanged():
    daily_data = _make_january_days()
    original = copy.deepcopy(daily_data)
    indices = list(range(len(daily_data)))

    adjust_temperatures_for_month(daily_data, indices, {}, 2020, 1)

    assert daily_data == original


def test_nan_mean_prediction_leaves_data_unchanged():
    daily_data = _make_january_days()
    original = copy.deepcopy(daily_data)
    indices = list(range(len(daily_data)))
    predictions = _predictions_index(tmax=float("nan"), tmean=10.0, tmin=5.0)
    predictions[(2020, 1, "temperature_max")]["mean"] = pd.NA

    adjust_temperatures_for_month(daily_data, indices, predictions, 2020, 1)

    assert daily_data == original
