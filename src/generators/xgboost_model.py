"""
XGBoost sliding-window model for secondary weather variable generation.

Uses a rolling window of `window_size` consecutive days (temperature_min,
temperature_max, temperature_mean, precipitation) as input features to
predict secondary meteorological variables (wind_speed_mean, wind_speed_max,
humidity_min, humidity_max, humidity_mean, pressure_min, pressure_max)
for the last day in the window.

Captures inter-day dependencies: e.g. a rainy day raises soil moisture and
humidity the following days.

Strategy:
    1. Detect target columns that have data in the historical dataset.
       All-None columns are excluded from training and output as None.
    2. Build rolling windows of `window_size` consecutive historical days.
       Input features: [tmin, tmax, tmean, precip] x window_size (flattened).
       Target: secondary numeric vars of the LAST day in each window.
    3. Train a MultiOutputRegressor wrapping XGBRegressor.
    4. For the first window_size-1 generated days (not enough window history),
       use KNeighborsCorrector as a seed.
    5. For day >= window_size-1: apply XGBoost with the sliding window of
       already-processed records. Temperature/precipitation are taken directly
       from the generated (adjusted) data and never modified.
    6. Categorical variables (wind_direction) and hour fields are handled
       exclusively by the KNN corrector — XGBoost only predicts numeric vars.
    7. Physical constraints are enforced on every predicted record.

Input features per window step:
    temperature_min, temperature_max, temperature_mean, precipitation

Numeric target variables (predicted by XGBoost for the LAST day):
    wind_speed_mean, wind_speed_max,
    humidity_min, humidity_max, humidity_mean,
    pressure_min, pressure_max

Non-numeric targets (preserved as-is from adjusted_data):
    wind_direction, hour_wind_max, hour_hrmin, hour_hrmax,
    hour_presmin, hour_presmax
"""

import numpy as np
from typing import List, Dict, Optional

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer

from generators.k_neighbors import KNeighborsCorrector


# ---------------------------------------------------------------------------
# Feature / target variable definitions
# ---------------------------------------------------------------------------

INPUT_FEATURES = [
    'temperature_min',
    'temperature_max',
    'temperature_mean',
    'precipitation',
]

ALL_TARGET_VARS_NUMERIC = [
    'wind_speed_mean',
    'wind_speed_max',
    'humidity_min',
    'humidity_max',
    'humidity_mean',
    'pressure_min',
    'pressure_max',
]


class XGBoostWeatherModel:
    """
    Sliding-window XGBoost model for secondary weather variable generation.

    Trains a multi-output XGBoost regressor on historical records using
    rolling windows of `window_size` days. For each new generated day the
    model receives the window of temperatures and precipitation (including the
    current day) and predicts wind speed, humidity and pressure.

    For the first `window_size - 1` days (insufficient window history) the
    existing K-Nearest Neighbors corrector is used as a seed.

    Args:
        window_size: Number of consecutive days in each input window.
                     Must be >= 1. Default: 7.
    """

    def __init__(self, window_size: int = 7):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self.model: Optional[MultiOutputRegressor] = None
        self.valid_target_vars: List[str] = []
        self._x_imputer: Optional[SimpleImputer] = None
        self._y_imputer: Optional[SimpleImputer] = None
        self._is_fitted: bool = False

    # ================================================================
    # Column validation
    # ================================================================

    def _detect_valid_columns(self, historical_data: List[Dict]) -> List[str]:
        """
        Detect numeric target columns that have at least one non-None value
        in the historical dataset. Stores result in ``self.valid_target_vars``.

        Args:
            historical_data: List of historical weather record dicts.

        Returns:
            List of valid target column names.
        """
        valid = [
            col for col in ALL_TARGET_VARS_NUMERIC
            if any(r.get(col) is not None for r in historical_data)
        ]
        self.valid_target_vars = valid

        excluded = set(ALL_TARGET_VARS_NUMERIC) - set(valid)
        if excluded:
            print(
                f"  ℹ️  XGBoost: Excluded all-None target columns: "
                f"{', '.join(sorted(excluded))}"
            )

        return valid

    # ================================================================
    # Window-building helpers
    # ================================================================

    def _extract_input_row(self, window: List[Dict]) -> List[float]:
        """
        Flatten a window of records into a 1-D feature vector.

        Layout: [day0_tmin, day0_tmax, day0_tmean, day0_precip, day1_tmin, ...]

        None values are replaced with np.nan (handled later by imputer).
        """
        row: List[float] = []
        for record in window:
            for feat in INPUT_FEATURES:
                val = record.get(feat)
                row.append(float(val) if val is not None else np.nan)
        return row

    def _build_training_windows(self, data: List[Dict]):
        """
        Build (X, y) training arrays from a list of records using
        rolling windows of `window_size` days.

        A window ending at day ``i`` forms one sample:
          - X: flattened input features of days [i-window_size+1 … i]
          - y: valid numeric target values of day ``i``

        Samples where ALL targets of day ``i`` are None are skipped.

        Returns:
            (X, y) as np.ndarray or (None, None) if not enough data.
        """
        X_rows: List[List[float]] = []
        y_rows: List[List[float]] = []

        for i in range(self.window_size - 1, len(data)):
            last_day = data[i]

            # Skip if target day has no useful target data at all
            if all(last_day.get(col) is None for col in self.valid_target_vars):
                continue

            window = data[i - self.window_size + 1 : i + 1]
            x_row = self._extract_input_row(window)
            y_row = [
                float(last_day.get(col)) if last_day.get(col) is not None else np.nan
                for col in self.valid_target_vars
            ]

            X_rows.append(x_row)
            y_rows.append(y_row)

        if not X_rows:
            return None, None

        return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64)

    # ================================================================
    # Training
    # ================================================================

    def fit(self, historical_data: List[Dict]) -> None:
        """
        Train the XGBoost multi-output model on historical data.

        Detects valid target columns first if not already done.
        Uses median imputation for missing input and target values.

        Args:
            historical_data: List of historical weather record dicts.
        """
        if not self.valid_target_vars:
            self._detect_valid_columns(historical_data)

        if not self.valid_target_vars:
            print(
                "  ⚠️  XGBoost: No valid target columns found in historical data. "
                "Model cannot be trained."
            )
            return

        if len(historical_data) < self.window_size + 1:
            print(
                f"  ⚠️  XGBoost: Not enough historical records "
                f"({len(historical_data)}) for window_size={self.window_size}."
            )
            return

        X, y = self._build_training_windows(historical_data)

        if X is None or len(X) == 0:
            print("  ⚠️  XGBoost: No valid training windows could be built.")
            return

        # Impute NaN with column median
        self._x_imputer = SimpleImputer(strategy='median')
        self._y_imputer = SimpleImputer(strategy='median')

        X = self._x_imputer.fit_transform(X)
        y = self._y_imputer.fit_transform(y)

        xgb_base = XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self.model = MultiOutputRegressor(xgb_base, n_jobs=-1)
        self.model.fit(X, y)
        self._is_fitted = True

        print(
            f"  ✅ XGBoost trained: {len(X)} windows | "
            f"window_size={self.window_size} | "
            f"targets={self.valid_target_vars}"
        )

    # ================================================================
    # Prediction
    # ================================================================

    def predict_day(self, window: List[Dict]) -> Dict[str, Optional[float]]:
        """
        Predict secondary numeric weather variables for the last day in the window.

        If the window length is shorter than ``window_size``, it is left-padded
        by repeating the first record (fallback for very early days).

        Args:
            window: List of dicts with at least temperature/precipitation fields.
                    The last element is the day to predict; all others provide
                    temporal context.

        Returns:
            Dict mapping each numeric target variable to its predicted value.
            Variables not in ``valid_target_vars`` are mapped to None.
        """
        if not self._is_fitted:
            return {col: None for col in ALL_TARGET_VARS_NUMERIC}

        # Pad window to exact window_size if needed
        padded = list(window)
        while len(padded) < self.window_size:
            padded = [padded[0]] + padded
        padded = padded[-self.window_size:]   # keep last window_size days

        x_row = np.array(
            self._extract_input_row(padded), dtype=np.float64
        ).reshape(1, -1)
        x_row = self._x_imputer.transform(x_row)

        raw_preds = self.model.predict(x_row)[0]  # shape (n_valid_targets,)

        result: Dict[str, Optional[float]] = {col: None for col in ALL_TARGET_VARS_NUMERIC}
        for i, col in enumerate(self.valid_target_vars):
            result[col] = float(raw_preds[i])

        self._apply_physical_constraints(result)
        return result

    # ================================================================
    # Physical constraints
    # ================================================================

    def _apply_physical_constraints(self, record: Dict[str, Optional[float]]) -> None:
        """
        Clamp predicted values to physically valid ranges and ensure
        min <= max consistency for humidity and pressure.
        """
        if record.get('humidity_min') is not None:
            record['humidity_min'] = int(round(max(0.0, min(100.0, record['humidity_min']))))
        if record.get('humidity_max') is not None:
            record['humidity_max'] = int(round(max(0.0, min(100.0, record['humidity_max']))))
        if record.get('humidity_mean') is not None:
            record['humidity_mean'] = int(round(max(0.0, min(100.0, record['humidity_mean']))))

        if record.get('pressure_min') is not None:
            record['pressure_min'] = round(max(850.0, min(1100.0, record['pressure_min'])), 1)
        if record.get('pressure_max') is not None:
            record['pressure_max'] = round(max(850.0, min(1100.0, record['pressure_max'])), 1)

        if record.get('wind_speed_mean') is not None:
            record['wind_speed_mean'] = round(max(0.0, record['wind_speed_mean']), 1)
        if record.get('wind_speed_max') is not None:
            record['wind_speed_max'] = round(max(0.0, record['wind_speed_max']), 1)

        # Ensure wind_speed_max >= wind_speed_mean
        if (record.get('wind_speed_mean') is not None
                and record.get('wind_speed_max') is not None):
            if record['wind_speed_max'] < record['wind_speed_mean']:
                record['wind_speed_max'] = record['wind_speed_mean']

        # Ensure min <= max for humidity and pressure
        for prefix in ('humidity', 'pressure'):
            min_key = f'{prefix}_min'
            max_key = f'{prefix}_max'
            if (record.get(min_key) is not None
                    and record.get(max_key) is not None
                    and record[min_key] > record[max_key]):
                record[min_key], record[max_key] = record[max_key], record[min_key]

    # ================================================================
    # Main correction pipeline
    # ================================================================

    def correct(
        self,
        adjusted_data: List[Dict],
        historical_data: List[Dict],
    ) -> List[Dict]:
        """
        Full pipeline: detect valid columns, train model, and fill numeric
        secondary weather variables for all days in ``adjusted_data``.

        Non-numeric fields (wind_direction, hour_wind_max, hour_hrmin,
        hour_hrmax, hour_presmin, hour_presmax) are NEVER modified: they
        retain whatever values are already present in adjusted_data.

        Steps:
            1. Detect valid numeric target columns from historical data.
            2. Train XGBoost on historical records.
            3. Work-copy of adjusted_data preserving all original fields.
            4. For the first ``window_size - 1`` seed days (not enough window
               history): use KNeighborsCorrector and copy ONLY the valid
               numeric target vars into the work-copy.
            5. For days ``>= window_size - 1``: apply XGBoost sliding-window
               prediction and write ONLY valid numeric target vars.
            6. Return the corrected work-copy.

        If training fails (no valid columns or too few records), falls back
        to KNeighborsCorrector for numeric vars only.

        Args:
            adjusted_data:   Generated data with temperature/precipitation set.
            historical_data: Historical records used for training and KNN fallback.

        Returns:
            Copy of ``adjusted_data`` with numeric secondary variables filled in.
            Non-numeric fields are unchanged.
        """
        n = len(adjusted_data)
        if n == 0:
            return adjusted_data

        # -- 1. Detect valid columns ----------------------------------
        self._detect_valid_columns(historical_data)

        # Helper: copy only numeric target vars from src into dst
        def _copy_numeric(dst: dict, src: dict) -> None:
            for col in self.valid_target_vars:
                dst[col] = src.get(col)

        # Fallback: KNN numeric-only
        def _knn_numeric_fallback(data: List[Dict]) -> List[Dict]:
            knn = KNeighborsCorrector(k=3, month_weight=0.25)
            knn_result = knn.correct(data, historical_data)
            out = [r.copy() for r in data]
            for i, knn_rec in enumerate(knn_result):
                _copy_numeric(out[i], knn_rec)
            return out

        if not self.valid_target_vars:
            print("  ⚠️  XGBoost: No target columns available — falling back to KNN (numeric vars only)")
            return _knn_numeric_fallback(adjusted_data)

        # -- 2. Train XGBoost -----------------------------------------
        print(
            f"🔄 Training XGBoost model "
            f"(window_size={self.window_size}, "
            f"targets={len(self.valid_target_vars)})..."
        )
        self.fit(historical_data)

        if not self._is_fitted:
            print("  ⚠️  XGBoost: Training failed — falling back to KNN (numeric vars only)")
            return _knn_numeric_fallback(adjusted_data)

        print("✅ XGBoost model trained")

        # -- 3. Work-copy (preserves wind_direction, hour fields, etc.) --
        corrected = [r.copy() for r in adjusted_data]

        # -- 4. KNN seed for first window_size-1 days -----------------
        seed_days = min(self.window_size - 1, n)

        if seed_days > 0:
            print(f"🔄 Applying KNN to first {seed_days} seed days (numeric vars only)...")
            knn = KNeighborsCorrector(k=3, month_weight=0.25)
            knn_seed = knn.correct(adjusted_data[:seed_days], historical_data)
            for i, knn_rec in enumerate(knn_seed):
                _copy_numeric(corrected[i], knn_rec)
            print(f"✅ KNN seed applied ({seed_days} days)")

        # -- 5. XGBoost sliding window --------------------------------
        remaining = n - seed_days
        if remaining <= 0:
            print(
                f"  ℹ️  XGBoost: All {n} days are seed days "
                f"(window_size={self.window_size}). KNN result kept."
            )
            return corrected

        print(
            f"🔄 Applying XGBoost sliding window to {remaining} days "
            f"(days {seed_days}–{n - 1})..."
        )

        for i in range(seed_days, n):
            window = corrected[i - self.window_size + 1 : i + 1]
            day_preds = self.predict_day(window)

            # Write only valid numeric target vars; non-numeric fields stay
            # untouched (wind_direction, hour_* preserve adjusted_data values)
            for col in self.valid_target_vars:
                corrected[i][col] = day_preds[col]

            if i > seed_days and (i - seed_days) % 500 == 0:
                print(f"  📍 XGBoost progress: {i}/{n - 1} days")

        print(f"✅ XGBoost correction complete ({n} days total)")
        return corrected
