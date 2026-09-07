import random

from generators.monthly_adjustments.precipitation_adjuster import adjust_precipitation_for_month


def _make_days(n_days, precipitation_values):
    return [
        {"date": f"2020-06-{day:02d}", "precipitation": precipitation_values[day - 1]}
        for day in range(1, n_days + 1)
    ]


def test_zero_prediction_zeroes_all_days():
    daily_data = _make_days(30, [5.0] * 30)
    indices = list(range(30))
    predictions = {(2020, 6, "precipitation"): {"mean": 0.0, "min": None, "max": None}}

    adjust_precipitation_for_month(daily_data, indices, predictions, 2020, 6)

    assert all(r["precipitation"] == 0.0 for r in daily_data)


def test_dry_history_with_wet_prediction_injects_rain():
    random.seed(1)
    daily_data = _make_days(30, [0.0] * 30)
    indices = list(range(30))
    predictions = {(2020, 6, "precipitation"): {"mean": 3.0, "min": 2.0, "max": 4.0}}

    adjust_precipitation_for_month(daily_data, indices, predictions, 2020, 6)

    total = sum(r["precipitation"] for r in daily_data)
    assert total > 0
    assert any(r["precipitation"] > 0 for r in daily_data)


def test_multiplicative_scaling_moves_mean_toward_prediction():
    daily_data = _make_days(30, [2.0] * 30)
    indices = list(range(30))
    predictions = {(2020, 6, "precipitation"): {"mean": 4.0, "min": None, "max": None}}

    adjust_precipitation_for_month(daily_data, indices, predictions, 2020, 6)

    actual_mean = sum(r["precipitation"] for r in daily_data) / 30
    assert actual_mean == 4.0


def test_no_prediction_leaves_data_unchanged():
    daily_data = _make_days(30, [1.0] * 30)
    indices = list(range(30))

    adjust_precipitation_for_month(daily_data, indices, {}, 2020, 6)

    assert all(r["precipitation"] == 1.0 for r in daily_data)


def test_advanced_zero_days_rain_zeroes_all_days():
    random.seed(2)
    daily_data = _make_days(30, [5.0] * 10 + [0.0] * 20)
    indices = list(range(30))
    predictions = {
        (2020, 6, "precipitation"): {"mean": 3.0, "min": None, "max": None},
        (2020, 6, "number_days_rain"): {"mean": 0, "min": None, "max": None},
    }

    adjust_precipitation_for_month(daily_data, indices, predictions, 2020, 6)

    assert all(r["precipitation"] == 0.0 for r in daily_data)


def test_advanced_adds_rainy_days_to_meet_deficit():
    random.seed(3)
    daily_data = _make_days(30, [0.0] * 30)
    indices = list(range(30))
    predictions = {
        (2020, 6, "precipitation"): {"mean": 2.0, "min": 1.0, "max": 3.0},
        (2020, 6, "number_days_rain"): {"mean": 10, "min": 8, "max": 12},
    }

    adjust_precipitation_for_month(daily_data, indices, predictions, 2020, 6)

    rainy_days = sum(1 for r in daily_data if r["precipitation"] > 0)
    assert 8 <= rainy_days <= 12


def test_advanced_removes_rainy_days_to_meet_excess():
    random.seed(4)
    # Every day rains, well above the predicted maximum day count.
    daily_data = _make_days(30, [2.0] * 30)
    indices = list(range(30))
    predictions = {
        (2020, 6, "precipitation"): {"mean": 2.0, "min": 1.0, "max": 3.0},
        (2020, 6, "number_days_rain"): {"mean": 5, "min": 3, "max": 5},
    }

    adjust_precipitation_for_month(daily_data, indices, predictions, 2020, 6)

    rainy_days = sum(1 for r in daily_data if r["precipitation"] > 0)
    assert rainy_days <= 5
