import pandas as pd
import pytest

from application.config_services import validate_predictions


def _valid_df():
    return pd.DataFrame([
        {"Year": 2021, "Month": 1, "Variable": "temperature_mean", "Minimum": 5.0, "Mean": 10.0, "Maximum": 15.0},
        {"Year": 2021, "Month": 2, "Variable": "precipitation", "Minimum": 0.0, "Mean": 2.0, "Maximum": 5.0},
    ])


def test_valid_predictions_pass():
    validate_predictions(_valid_df())  # should not raise


def test_missing_columns_raise():
    df = _valid_df().drop(columns=["Mean"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_predictions(df)


def test_invalid_variable_name_raises():
    df = _valid_df()
    df.loc[0, "Variable"] = "not_a_real_variable"
    with pytest.raises(ValueError, match="Invalid variables"):
        validate_predictions(df)


def test_month_out_of_range_raises():
    df = _valid_df()
    df.loc[0, "Month"] = 13
    with pytest.raises(ValueError, match="Month values must be between"):
        validate_predictions(df)


def test_missing_month_raises():
    df = _valid_df()
    df.loc[0, "Month"] = None
    with pytest.raises(ValueError, match="Month column contains invalid"):
        validate_predictions(df)


def test_missing_value_cell_raises():
    df = _valid_df()
    df.loc[0, "Mean"] = None
    with pytest.raises(ValueError, match="Missing values"):
        validate_predictions(df)


def test_min_greater_than_mean_raises():
    df = _valid_df()
    df.loc[0, "Minimum"] = 20.0  # greater than Mean (10.0)
    with pytest.raises(ValueError, match="must be <="):
        validate_predictions(df)


def test_mean_greater_than_maximum_raises():
    df = _valid_df()
    df.loc[0, "Mean"] = 20.0  # greater than Maximum (15.0)
    with pytest.raises(ValueError, match="must be <="):
        validate_predictions(df)


def test_whitespace_in_headers_is_normalized():
    df = _valid_df()
    df.columns = [f" {column} " for column in df.columns]
    validate_predictions(df)  # should not raise
