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
# Variables de referencia (usadas para calcular distancia)
# ---------------------------------------------------------------------------
REFERENCE_VARS = [
    'temperature_min',
    'temperature_max',
    'temperature_mean',
    'precipitation',
]

# ---------------------------------------------------------------------------
# Variables objetivo numéricas (corregidas por KNN)
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
# Variables objetivo categóricas (se copian del vecino más cercano)
# ---------------------------------------------------------------------------
TARGET_VARS_CATEGORICAL = [
    'wind_direction',
]

# ---------------------------------------------------------------------------
# Campos de hora asociados (se copian del vecino más cercano siempre)
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
    Corrector K-Vecinos para variables meteorológicas no modificadas.

    Tras ajustar temperatura y precipitación a predicciones mensuales,
    adapta viento, humedad y presión buscando los días históricos más
    parecidos en temperatura/precipitación y copiando (K=1) o promediando
    ponderadamente (K>1) sus valores.

    Args:
        k: Número de vecinos (1 = copia directa, >1 = promedio ponderado).
        month_weight: Peso de las columnas de estacionalidad (sin/cos del mes).
                      0 = sin preferencia estacional, valores altos priorizan
                      mismo mes.
    """

    def __init__(self, k: int = 3, month_weight: float = 0.25):
        if k < 1:
            raise ValueError("k debe ser >= 1")
        self.k = k
        self.month_weight = month_weight

    # ================================================================
    # Función principal de corrección
    # ================================================================

    def correct(
        self,
        adjusted_data: List[Dict],
        historical_data: List[Dict],
    ) -> List[Dict]:
        """
        Corrige las variables no modificadas de los datos ajustados usando KNN.

        Args:
            adjusted_data:   Datos generados con temperatura/precipitación ya
                             ajustadas a predicciones mensuales.
            historical_data: Datos históricos originales (referencia para KNN).

        Returns:
            Lista de registros con las variables no modificadas corregidas.
            Temperatura y precipitación se mantienen intactas.
        """
        # -- 1. Preparar features históricos + ajustar scaler --------
        hist_matrix, hist_valid_records, scaler, active_ref_vars = \
            self._build_feature_matrix(historical_data)

        if hist_matrix is None or len(hist_matrix) == 0:
            print("  ⚠️  KNN: No hay suficientes datos históricos, se omite corrección")
            return adjusted_data

        # -- 2. Entrenar NearestNeighbors sobre el histórico ----------
        k_fit = min(self.k, len(hist_matrix))
        nn = NearestNeighbors(n_neighbors=k_fit, metric='euclidean')
        nn.fit(hist_matrix)

        # -- 3. Preparar queries de todos los días ajustados ----------
        adj_raw, adj_indices = self._build_query_matrix(adjusted_data, active_ref_vars)

        if len(adj_raw) == 0:
            print("  ⚠️  KNN: Ningún día ajustado tiene features válidos")
            return adjusted_data

        # Escalar y ponderar mes (misma transformación que el histórico)
        adj_scaled = scaler.transform(adj_raw)
        adj_scaled[:, -2] *= self.month_weight
        adj_scaled[:, -1] *= self.month_weight

        # -- 4. Buscar vecinos de todos los días de golpe (vectorizado)
        distances, neighbor_idx = nn.kneighbors(adj_scaled)

        # -- 5. Corregir cada día ------------------------------------
        corrected = [record.copy() for record in adjusted_data]

        for idx_corrected, (pos, rec_idx) in enumerate(enumerate(adj_indices)):
            new_record = corrected[rec_idx]
            nbr_dists = distances[pos]      # shape (k_fit,)
            nbr_idxs = neighbor_idx[pos]    # shape (k_fit,) — índices en hist_valid_records

            # -- Guardar valores antes de la corrección ---------------
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

            # -- Asignar variables del/los vecino(s) ------------------
            nearest_record = hist_valid_records[nbr_idxs[0]]
            if k_fit == 1:
                self._copy_target_vars(new_record, nearest_record)
            else:
                self._weighted_average_target_vars(
                    new_record, hist_valid_records, nbr_idxs, nbr_dists
                )

            # -- Restricciones físicas finales ------------------------
            self._apply_physical_constraints(new_record)

            # -- Log: mostrar cada 60 días corregidos ----------------
            if idx_corrected % 460 == 0:
                nearest = hist_valid_records[nbr_idxs[0]]
                
                # Datos de temperatura y precipitación
                t_min_gen = new_record.get('temperature_min', 'N/A')
                t_max_gen = new_record.get('temperature_max', 'N/A')
                t_mean_gen = new_record.get('temperature_mean', 'N/A')
                p_gen = new_record.get('precipitation', 'N/A')
                
                t_min_nearest = nearest.get('temperature_min', 'N/A')
                t_max_nearest = nearest.get('temperature_max', 'N/A')
                t_mean_nearest = nearest.get('temperature_mean', 'N/A')
                p_nearest = nearest.get('precipitation', 'N/A')
                
                
                print(f"\n  📍 Día {idx_corrected+1}: {new_record.get('date', 'N/A')} "
                      f"(vecino: {nearest.get('date', 'N/A')}) ")
                print(f"     Temperatura (Gen):      Tmin={t_min_gen}°C, Tmax={t_max_gen}°C, Tmean={t_mean_gen}°C, P={p_gen}mm")
                print(f"     Temperatura (KVecino):  Tmin={t_min_nearest}°C, Tmax={t_max_nearest}°C, Tmean={t_mean_nearest}°C, P={p_nearest}mm")
                print(f"     Humedad máx:             {record_before['humidity_max']} → {new_record.get('humidity_max')}")
                print(f"     Humedad mín:             {record_before['humidity_min']} → {new_record.get('humidity_min')}")
                print(f"     Humedad media:           {record_before['humidity_mean']} → {new_record.get('humidity_mean')}")
                print(f"     Viento medio:            {record_before['wind_speed_mean']} → {new_record.get('wind_speed_mean')}")
                print(f"     Viento máximo:           {record_before['wind_speed_max']} → {new_record.get('wind_speed_max')}")
                print(f"     Dirección:               {record_before['wind_direction']} → {new_record.get('wind_direction')}")
                print(f"     Presión máx:             {record_before['pressure_max']} → {new_record.get('pressure_max')}")
                print(f"     Presión mín:             {record_before['pressure_min']} → {new_record.get('pressure_min')}")

        return corrected

    # ================================================================
    # Construcción de features (StandardScaler + codificación cíclica)
    # ================================================================

    def _build_feature_matrix(self, data: List[Dict]):
        """
        Construye la matriz de features estandarizada del histórico.

        Para cada registro se extraen todas las REFERENCE_VARS. Se detectan
        qué columnas tienen todos los valores a None (variable completamente
        ausente en el histórico) y se eliminan de la matriz. Si 3 o más
        variables están completamente vacías el algoritmo se cancela.

        Features activas: [vars_con_datos..., sin_month, cos_month]
        Se estandarizan con StandardScaler y luego se multiplican las columnas
        de mes por ``month_weight``.

        Returns:
            (scaled_matrix, valid_records, fitted_scaler, active_ref_vars)
            o (None, None, None, None) si se cancela.
        """
        valid_records = []
        raw_rows = []   # lista de dicts con valores crudos por registro

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

        # -- Detectar columnas completamente nulas --------------------
        all_null_vars = [
            v for v in REFERENCE_VARS
            if all(row[v] is None for row in raw_rows)
        ]

        if len(all_null_vars) >= 3:
            print(f"  ⚠️  KNN: {len(all_null_vars)} variables de referencia sin datos "
                  f"({', '.join(all_null_vars)}), se cancela la corrección")
            return None, None, None, None

        active_ref_vars = [v for v in REFERENCE_VARS if v not in all_null_vars]

        if all_null_vars:
            print(f"  ℹ️  KNN: Variables de referencia ignoradas (sin datos): "
                  f"{', '.join(all_null_vars)}")

        # -- Construir matriz con variables activas -------------------
        matrix = np.array(
            [
                [row[v] if row[v] is not None else np.nan
                 for v in active_ref_vars]
                + [row['_sin'], row['_cos']]
                for row in raw_rows
            ],
            dtype=np.float64,
        )

        # Estandarizar con StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)

        # Aplicar peso a columnas de mes (últimas 2)
        scaled[:, -2] *= self.month_weight
        scaled[:, -1] *= self.month_weight

        return scaled, valid_records, scaler, active_ref_vars

    def _build_query_matrix(self, data: List[Dict], active_ref_vars: List[str]):
        """
        Construye la matriz de features *sin escalar* de los datos ajustados,
        usando únicamente las variables de referencia activas (las que tienen
        datos en el histórico).

        Args:
            data:            Registros ajustados a consultar.
            active_ref_vars: Subconjunto de REFERENCE_VARS con datos disponibles,
                             devuelto por _build_feature_matrix.

        Returns:
            (raw_matrix, valid_indices) donde valid_indices son las posiciones
            en ``data`` que pudieron construir features.
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
    # Asignación de variables objetivo
    # ================================================================

    def _copy_target_vars(self, target: Dict, source: Dict):
        """Copia todas las variables objetivo del vecino más cercano."""
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
        Promedio ponderado por inversa de distancia para variables numéricas.
        Variables categóricas y campos de hora se toman del vecino más cercano.
        """
        # Pesos: inversa de la distancia (epsilon para evitar div/0)
        epsilon = 1e-10
        inv_distances = 1.0 / (distances + epsilon)
        weights = inv_distances / inv_distances.sum()

        # Variables numéricas: promedio ponderado
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
                w = w / w.sum()  # renormalizar por si se descartaron NaN
                weighted_val = np.dot(values, w)

                # Redondear según tipo de variable
                if 'humidity' in var:
                    target[var] = int(round(weighted_val))
                else:
                    target[var] = round(weighted_val, 1)

        # Variables categóricas: voto mayoritario ponderado
        for var in TARGET_VARS_CATEGORICAL:
            vote_weights: Dict[str, float] = {}
            for i, idx in enumerate(indices):
                val = hist_records[idx].get(var)
                if val is not None:
                    vote_weights[val] = vote_weights.get(val, 0) + weights[i]
            if vote_weights:
                target[var] = max(vote_weights, key=vote_weights.get)

        # Campos de hora: del vecino más cercano (promediar horas no tiene sentido)
        nearest = hist_records[indices[0]]
        for field in HOUR_FIELDS:
            if nearest.get(field) is not None:
                target[field] = nearest[field]

    # ================================================================
    # Restricciones físicas
    # ================================================================

    @staticmethod
    def _apply_physical_constraints(record: Dict):
        """
        Aplica restricciones físicas a las variables corregidas:
        - Humedad: [0, 100] y max >= mean >= min
        - Presión: max >= min, ambas > 0
        - Viento: >= 0, max >= mean
        """
        # -- Humedad --
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

        # -- Presión --
        for pv in ('pressure_min', 'pressure_max'):
            val = record.get(pv)
            if val is not None:
                record[pv] = round(max(0, val), 1)

        p_min = record.get('pressure_min')
        p_max = record.get('pressure_max')
        if p_min is not None and p_max is not None and p_min > p_max:
            record['pressure_min'], record['pressure_max'] = p_max, p_min

        # -- Viento --
        for wv in ('wind_speed_mean', 'wind_speed_max'):
            val = record.get(wv)
            if val is not None:
                record[wv] = round(max(0, val), 1)

        w_mean = record.get('wind_speed_mean')
        w_max = record.get('wind_speed_max')
        if w_mean is not None and w_max is not None and w_mean > w_max:
            record['wind_speed_mean'], record['wind_speed_max'] = w_max, w_mean

    # ================================================================
    # Utilidades
    # ================================================================

    @staticmethod
    def _get_month(record: Dict) -> Optional[int]:
        """Extrae el mes de un registro (campo 'date' YYYY-MM-DD)."""
        date_str = record.get('date')
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').month
        except (ValueError, TypeError):
            return None
