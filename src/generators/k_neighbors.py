"""
K-Nearest Neighbors corrector for non-modified weather variables.

After temperature and precipitation have been adjusted to monthly predictions,
this corrector adapts the remaining variables (wind, humidity, pressure) by
finding the most similar historical days based on the modified variables
and copying/interpolating the non-modified variables from those neighbors.

Uses sklearn.neighbors.NearestNeighbors for efficient neighbor search
(KD-Tree / Ball-Tree) and sklearn.preprocessing.StandardScaler for
feature standardisation.

Strategy:
    1. Build a feature matrix from historical data using the modified variables
       (temperature_min, temperature_max, temperature_mean, precipitation)
       plus cyclic month encoding (sin/cos) with adjustable weight.
    2. StandardScaler to normalise, then multiply month columns by month_weight.
    3. Fit a NearestNeighbors model on the historical feature matrix.
    4. For each generated day, query the K nearest historical neighbours.
    5. K=1  →  copy all target variables from the single nearest neighbour.
       K>1  →  inverse-distance weighted average for numeric variables,
               weighted majority vote for categorical (wind_direction),
               hour fields always from nearest neighbour.
    6. Apply physical constraints (humidity [0,100], pressure >0, etc.).

Reference variables (distance features):
    temperature_min, temperature_max, temperature_mean, precipitation

Target variables (corrected by KNN):
    wind_speed_mean, wind_speed_max, wind_direction,
    humidity_min, humidity_max, humidity_mean,
    pressure_min, pressure_max
"""

import numpy as np
from math import pi, sin, cos
from datetime import datetime
from typing import List, Dict, Optional

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Reference variables (used to compute distance)
# ---------------------------------------------------------------------------
REFERENCE_VARS = [
    'temperature_min',
    'temperature_max',
    'temperature_mean',
    'precipitation',
]

# ---------------------------------------------------------------------------
# Numeric target variables (corrected by KNN)
# ---------------------------------------------------------------------------
TARGET_VARS_NUMERIC = [
    'wind_speed_mean',
    'wind_speed_max',
    'humidity_min',
    'humidity_max',
    'humidity_mean',
    'pressure_min',
    'pressure_max',
]

# ---------------------------------------------------------------------------
# Categorical target variables (copied from nearest neighbor)
# ---------------------------------------------------------------------------
TARGET_VARS_CATEGORICAL = [
    'wind_direction',
]

# ---------------------------------------------------------------------------
# Associated hour fields (always copied from nearest neighbor)
# ---------------------------------------------------------------------------
HOUR_FIELDS = [
    'hour_wind_max',
    'hour_hrmin',
    'hour_hrmax',
    'hour_presmin',
    'hour_presmax',
]

HOUR_REFERENCE_FIELDS = [
    'hour_tmin',
    'hour_tmax',
]

class KNeighborsCorrector:
    """
    K-Nearest Neighbors corrector for non-modified weather variables.

    After adjusting temperature and precipitation to monthly predictions,
    adapts wind, humidity, and pressure by searching for the most similar
    historical days in temperature/precipitation and copying (K=1) or
    weighted-averaging (K>1) their values.

    Args:
        k: Number of neighbors (1 = direct copy, >1 = weighted average).
        month_weight: Weight for seasonality columns (month sin/cos).
                  0 = no seasonal preference, high values prioritize
                  the same month.
    """

    def __init__(self, k: int = 3, month_weight: float = 0.25):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.month_weight = month_weight

    # ================================================================
    # Main correction function
    # ================================================================

    def correct(
        self,
        adjusted_data: List[Dict],
        historical_data: List[Dict],
    ) -> List[Dict]:
        """
        Correct non-modified variables in adjusted data using KNN.

        Args:
            adjusted_data:   Generated data with temperature/precipitation
                             already adjusted to monthly predictions.
            historical_data: Original historical data (KNN reference).

        Returns:
            List of records with corrected non-modified variables.
            Temperature and precipitation stay unchanged.
        """
        # -- 1. Build historical features + fit scaler ---------------
        hist_matrix, hist_valid_records, scaler, active_ref_vars = \
            self._build_feature_matrix(historical_data)

        if hist_matrix is None or len(hist_matrix) == 0:
            print("  ⚠️  KNN: Not enough historical data, skipping correction")
            return adjusted_data

        # -- 2. Train NearestNeighbors on historical data -------------
        k_fit = min(self.k, len(hist_matrix))
        nn = NearestNeighbors(n_neighbors=k_fit, metric='euclidean')
        nn.fit(hist_matrix)

        # -- 3. Build queries for all adjusted days -------------------
        adj_raw, adj_indices = self._build_query_matrix(adjusted_data, active_ref_vars)

        if len(adj_raw) == 0:
            print("  ⚠️  KNN: No adjusted day has valid features")
            return adjusted_data

        # Scale and weight month columns (same transform as historical data)
        adj_scaled = scaler.transform(adj_raw)
        adj_scaled[:, -2] *= self.month_weight
        adj_scaled[:, -1] *= self.month_weight

        # -- 4. Search neighbors for all days at once (vectorized) ----
        distances, neighbor_idx = nn.kneighbors(adj_scaled)

        # -- 5. Correct each day --------------------------------------
        corrected = [record.copy() for record in adjusted_data]

        for idx_corrected, (pos, rec_idx) in enumerate(enumerate(adj_indices)):
            new_record = corrected[rec_idx]
            nbr_dists = distances[pos]      # shape (k_fit,)
            nbr_idxs = neighbor_idx[pos]    # shape (k_fit,) — indices in hist_valid_records

            # -- Keep values before correction -------------------------
            record_before = {
                'humidity_min': new_record.get('humidity_min'),
                'humidity_max': new_record.get('humidity_max'),
                'humidity_mean': new_record.get('humidity_mean'),
                'wind_speed_mean': new_record.get('wind_speed_mean'),
                'wind_speed_max': new_record.get('wind_speed_max'),
                'wind_direction': new_record.get('wind_direction'),
                'pressure_min': new_record.get('pressure_min'),
                'pressure_max': new_record.get('pressure_max'),
            }

            # -- Assign variables from neighbor(s) ---------------------
            nearest_record = hist_valid_records[nbr_idxs[0]]
            if k_fit == 1:
                self._copy_target_vars(new_record, nearest_record)
            else:
                self._weighted_average_target_vars(
                    new_record, hist_valid_records, nbr_idxs, nbr_dists
                )

            # -- Final physical constraints ----------------------------
            self._apply_physical_constraints(new_record)

            # -- Log: show every 60 corrected days ---------------------
            if idx_corrected % 460 == 0:
                nearest = hist_valid_records[nbr_idxs[0]]
                
                # Temperature and precipitation data
                t_min_gen = new_record.get('temperature_min', 'N/A')
                t_max_gen = new_record.get('temperature_max', 'N/A')
                t_mean_gen = new_record.get('temperature_mean', 'N/A')
                p_gen = new_record.get('precipitation', 'N/A')
                
                t_min_nearest = nearest.get('temperature_min', 'N/A')
                t_max_nearest = nearest.get('temperature_max', 'N/A')
                t_mean_nearest = nearest.get('temperature_mean', 'N/A')
                p_nearest = nearest.get('precipitation', 'N/A')
                
                
                print(f"\n  📍 Day {idx_corrected+1}: {new_record.get('date', 'N/A')} "
                        f"(neighbor: {nearest.get('date', 'N/A')}) ")
                print(f"     Temperature (Gen):      Tmin={t_min_gen}°C, Tmax={t_max_gen}°C, Tmean={t_mean_gen}°C, P={p_gen}mm")
                print(f"     Temperature (KNeighbor): Tmin={t_min_nearest}°C, Tmax={t_max_nearest}°C, Tmean={t_mean_nearest}°C, P={p_nearest}mm")
                print(f"     Humidity max:            {record_before['humidity_max']} → {new_record.get('humidity_max')}")
                print(f"     Humidity min:            {record_before['humidity_min']} → {new_record.get('humidity_min')}")
                print(f"     Humidity mean:           {record_before['humidity_mean']} → {new_record.get('humidity_mean')}")
                print(f"     Wind mean:               {record_before['wind_speed_mean']} → {new_record.get('wind_speed_mean')}")
                print(f"     Wind max:                {record_before['wind_speed_max']} → {new_record.get('wind_speed_max')}")
                print(f"     Direction:               {record_before['wind_direction']} → {new_record.get('wind_direction')}")
                print(f"     Pressure max:            {record_before['pressure_max']} → {new_record.get('pressure_max')}")
                print(f"     Pressure min:            {record_before['pressure_min']} → {new_record.get('pressure_min')}")

        return corrected

    # ================================================================
    # Feature building (StandardScaler + cyclic encoding)
    # ================================================================

    def _build_feature_matrix(self, data: List[Dict]):
        """
        Build the standardized historical feature matrix.

        For each record, all REFERENCE_VARS are extracted. Columns whose values
        are all None (variable fully absent in historical data) are detected
        and removed from the matrix. If 3 or more variables are fully empty,
        the algorithm is canceled.

        Active features: [vars_with_data..., sin_month, cos_month]
        They are standardized with StandardScaler and then month columns are
        multiplied by ``month_weight``.

        Returns:
            (scaled_matrix, valid_records, fitted_scaler, active_ref_vars)
            or (None, None, None, None) if canceled.
        """
        valid_records = []
        raw_rows = []

        for record in data:
            month = self._get_month(record)
            if month is None:
                continue

            sin_m = sin(2 * pi * month / 12)
            cos_m = cos(2 * pi * month / 12)

            row = {v: record.get(v) for v in REFERENCE_VARS}
            row['_sin'] = sin_m
            row['_cos'] = cos_m
            raw_rows.append(row)
            valid_records.append(record)

        if len(raw_rows) < 2:
            return None, None, None, None

        # -- Detect fully-null columns --------------------------------
        all_null_vars = [
            v for v in REFERENCE_VARS
            if all(row[v] is None for row in raw_rows)
        ]

        if len(all_null_vars) >= 3:
            print(f"  ⚠️  KNN: {len(all_null_vars)} reference variables without data "
                f"({', '.join(all_null_vars)}), canceling correction")
            return None, None, None, None

        active_ref_vars = [v for v in REFERENCE_VARS if v not in all_null_vars]

        if all_null_vars:
            print(f"  ℹ️  KNN: Ignored reference variables (no data): "
                  f"{', '.join(all_null_vars)}")

        # -- Build matrix with active variables ------------------------
        matrix = np.array(
            [
                [row[v] if row[v] is not None else np.nan
                 for v in active_ref_vars]
                + [row['_sin'], row['_cos']]
                for row in raw_rows
            ],
            dtype=np.float64,
        )

        # Standarize with StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

        # Apply weight to month columns (last 2)
        scaled[:, -2] *= self.month_weight
        scaled[:, -1] *= self.month_weight

        return scaled, valid_records, scaler, active_ref_vars

    def _build_query_matrix(self, data: List[Dict], active_ref_vars: List[str]):
        """
        Build the *unscaled* adjusted-data feature matrix,
        using only active reference variables (those with data in historical
        records).

        Args:
            data:            Adjusted records to query.
            active_ref_vars: Subset of REFERENCE_VARS with available data,
                             returned by _build_feature_matrix.

        Returns:
            (raw_matrix, valid_indices) where valid_indices are the positions
            in ``data`` which could build features.
        """
        raw_features = []
        valid_indices = []

        for i, record in enumerate(data):
            vals = [record.get(v) for v in active_ref_vars]

            month = self._get_month(record)
            if month is None:
                continue

            sin_m = sin(2 * pi * month / 12)
            cos_m = cos(2 * pi * month / 12)

            raw_features.append(vals + [sin_m, cos_m])
            valid_indices.append(i)

        if not raw_features:
            return np.empty((0, len(active_ref_vars) + 2)), []

        return np.array(raw_features, dtype=np.float64), valid_indices

    # ================================================================
    # Target variable assignment
    # ================================================================

    def _copy_target_vars(self, target: Dict, source: Dict):
        """Copy all target variables from the nearest neighbor."""
        for var in TARGET_VARS_NUMERIC:
            if source.get(var) is not None:
                target[var] = source[var]

        for var in TARGET_VARS_CATEGORICAL:
            if source.get(var) is not None:
                target[var] = source[var]

        for field in HOUR_FIELDS:
            if source.get(field) is not None:
                target[field] = source[field]

        #for field in HOUR_REFERENCE_FIELDS:
        #    if source.get(field) is not None:
        #        target[field] = source[field]

    def _weighted_average_target_vars(
        self, target: Dict, hist_records: List[Dict],
        indices: np.ndarray, distances: np.ndarray
    ):
        """
        Inverse-distance weighted average for numeric variables.
        Categorical variables and hour fields are taken from nearest neighbor.
        """
        # Weights: inverse distance (epsilon avoids div/0)
        epsilon = 1e-10
        inv_distances = 1.0 / (distances + epsilon)
        weights = inv_distances / inv_distances.sum()

        # Numeric variables: weighted average
        for var in TARGET_VARS_NUMERIC:
            values = []
            valid_weights = []
            for i, idx in enumerate(indices):
                val = hist_records[idx].get(var)
                if val is not None:
                    values.append(float(val))
                    valid_weights.append(weights[i])

            if values:
                w = np.array(valid_weights)
                w = w / w.sum()  # re-normalize in case NaNs were discarded
                weighted_val = np.dot(values, w)

                # Round based on variable type
                if 'humidity' in var:
                    target[var] = int(round(weighted_val))
                else:
                    target[var] = round(weighted_val, 1)

        # Categorical variables: weighted majority vote
        for var in TARGET_VARS_CATEGORICAL:
            vote_weights: Dict[str, float] = {}
            for i, idx in enumerate(indices):
                val = hist_records[idx].get(var)
                if val is not None:
                    vote_weights[val] = vote_weights.get(val, 0) + weights[i]
            if vote_weights:
                target[var] = max(vote_weights, key=vote_weights.get)

        # Hour fields: from nearest neighbor (averaging hours is meaningless)
        nearest = hist_records[indices[0]]
        for field in HOUR_FIELDS:
            if nearest.get(field) is not None:
                target[field] = nearest[field]

    # ================================================================
    # Physical constraints
    # ================================================================

    @staticmethod
    def _apply_physical_constraints(record: Dict):
        """
        Apply physical constraints to corrected variables:
        - Humidity: [0, 100] and max >= mean >= min
        - Pressure: max >= min, both > 0
        - Wind: >= 0, max >= mean
        """
        # -- Humidity --
        for hv in ('humidity_min', 'humidity_max', 'humidity_mean'):
            val = record.get(hv)
            if val is not None:
                record[hv] = max(0, min(100, int(round(val))))

        h_min = record.get('humidity_min')
        h_max = record.get('humidity_max')
        h_mean = record.get('humidity_mean')

        if h_min is not None and h_max is not None and h_min > h_max:
            record['humidity_min'], record['humidity_max'] = h_max, h_min

        if h_mean is not None:
            if h_min is not None and h_mean < h_min:
                record['humidity_mean'] = h_min
            if h_max is not None and h_mean > h_max:
                record['humidity_mean'] = h_max

        # -- Pressure --
        for pv in ('pressure_min', 'pressure_max'):
            val = record.get(pv)
            if val is not None:
                record[pv] = round(max(0, val), 1)

        p_min = record.get('pressure_min')
        p_max = record.get('pressure_max')
        if p_min is not None and p_max is not None and p_min > p_max:
            record['pressure_min'], record['pressure_max'] = p_max, p_min

        # -- Wind --
        for wv in ('wind_speed_mean', 'wind_speed_max'):
            val = record.get(wv)
            if val is not None:
                record[wv] = round(max(0, val), 1)

        w_mean = record.get('wind_speed_mean')
        w_max = record.get('wind_speed_max')
        if w_mean is not None and w_max is not None and w_mean > w_max:
            record['wind_speed_mean'], record['wind_speed_max'] = w_max, w_mean

    # ================================================================
    # Utilities
    # ================================================================

    @staticmethod
    def _get_month(record: Dict) -> Optional[int]:
        """Extract month from a record ('date' field YYYY-MM-DD)."""
        date_str = record.get('date')
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').month
        except (ValueError, TypeError):
            return None
