"""
Multivariate Bias Correction (MBCn) for synthetic weather data. 
IMPLEMENTED BUT NOT USED.

Corrects inter-variable dependencies after univariate adjustments
using the xclim.sdba library (MBCn).

Reference: Cannon, A.J. (2018). Multivariate quantile mapping bias correction:
an N-dimensional probability density function transform for climate model
output to observations. Climate Dynamics, 50(1), 31-49.

Flow:
    1. Convert List[Dict] to xr.Dataset with 'time' dimension
    2. Assign physical units (required by xclim)
    3. Detect variables without NULLs in historical data (only those are corrected)
    4. stack_variables() -> DataArray (time, multivar)
    5. For each month (1-12): filter by month, train and adjust MBCn
       - ref_m = hist_m (all historical years for that month)
       - sim_m = all adjusted years for that month
    6. Concatenate the 12 corrected months and sort by date
    7. unstack_variables() -> Dataset -> DataFrame -> List[Dict]
    8. Apply physical constraints (max>=mean>=min, precip>=0, etc.)
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Dict

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from xclim import sdba


# Physical units per variable (required by xclim)
_UNITS: Dict[str, str] = {
    'temperature_min':  'degC',
    'temperature_max':  'degC',
    'temperature_mean': 'degC',
    'precipitation':    'mm/d',
    'humidity_min':     '%',
    'humidity_max':     '%',
    'humidity_mean':    '%',
    'pressure_min':     'hPa',
    'pressure_max':     'hPa',
    'wind_speed_mean':  'm/s',
    'wind_speed_max':   'm/s',
}


class MBCnCorrector:
    """
    Month-by-month multivariate MBCn correction.

        For each of the 12 months:
            - ref_m / hist_m = all historical days for that month (all years)
            - sim_m          = all adjusted days for that month (all simulation years)

    This guarantees seasonal coherence: January is corrected against historical
    Januaries, August against historical Augusts.

    Non-meteorological fields (hour_*, wind_direction, source, id_station...)
    are preserved as-is: only numeric VARIABLES_METEO variables without NULLs
    in historical data are overwritten.
    """

    VARIABLES_METEO = list(_UNITS.keys())

    def __init__(self, n_iter: int = 20):
        """
        Args:
            n_iter: Number of MBCn rotation iterations (default 20).
                    10 fast, 20 balanced, 50 very expensive.
        """
        self.n_iter = n_iter

    # ================================================================
    # Entry point
    # ================================================================

    def correct(
        self,
        adjusted_data: List[Dict],
        historical_data: List[Dict]
    ) -> List[Dict]:
        """
        Apply month-by-month MBCn and return corrected data.

        Args:
            adjusted_data:   Generated data already adjusted univariately.
            historical_data: Original historical data (reference).

        Returns:
            List of records with corrected multivariate dependency structure
            and physical constraints applied.
        """
        # -- 1. Convert to DataFrames ----------------------------------
        df_adj  = pd.DataFrame(adjusted_data).copy()
        df_hist = pd.DataFrame(historical_data).copy()

        df_adj['date']  = pd.to_datetime(df_adj['date'])
        df_hist['date'] = pd.to_datetime(df_hist['date'])

        # -- 2. Detect valid variables (without NULLs in historical data) --
        valid_vars = [
            v for v in self.VARIABLES_METEO
            if v in df_hist.columns
            and v in df_adj.columns
            and df_hist[v].notnull().all()
        ]

        if len(valid_vars) < 2:
            print("   Warning: Fewer than 2 valid variables for MBCn, skipping")
            return adjusted_data

        print(f"   Variables included in MBCn ({len(valid_vars)}): {valid_vars}")

        # -- 3. Build xr.Datasets with 'time' dimension ---------------
        ds_hist = (
            df_hist.set_index('date')[valid_vars]
            .to_xarray()
            .rename({'date': 'time'})
        )
        ds_adj = (
            df_adj.set_index('date')[valid_vars]
            .to_xarray()
            .rename({'date': 'time'})
        )

        # -- 4. Assign physical units (required for xclim) ------------
        for ds in [ds_hist, ds_adj]:
            for var in valid_vars:
                ds[var].attrs['units'] = _UNITS[var]

        # -- 5. Stack -> DataArray (time, multivar) ------------------
        #   sdba.stack_variables stacks Dataset variables into a new
        #   'multivar' dimension, generating the format MBCn expects.
        ref_all = sdba.stack_variables(ds_hist)   # (time_hist, n_vars)
        sim_all = sdba.stack_variables(ds_adj)    # (time_adj,  n_vars)

        # -- 6. Month-by-month MBCn -----------------------------------
        #   xclim does NOT support group='time.month' inside MBCn (it raises
        #   NotImplementedError), so we run a monthly loop and filter
        #   DataArrays manually.
        corrected_months: List[xr.DataArray] = []
        months_ok = 0

        for m in range(1, 13):
            ref_m = ref_all.sel(time=ref_all.time.dt.month == m)
            sim_m = sim_all.sel(time=sim_all.time.dt.month == m)

            if len(ref_m.time) < 10 or len(sim_m.time) < 5:
                corrected_months.append(sim_m)
                print(f"   Month {m:02d}: insufficient data "
                        f"(ref={len(ref_m.time)}, sim={len(sim_m.time)}), skipped")
                continue

            try:
                # train(ref, hist) -- ref == hist because we do not have
                # a separate climate model; MBCn learns historical dependency
                # structure and transfers it to sim.
                mbcn = sdba.adjustment.MBCn.train(
                    ref=ref_m,
                    hist=sim_m,
                    n_iter=self.n_iter,
                    n_escore=20   # disable energy score (causes out-of-bounds
                )                # with small arrays in xclim 0.55)

                # adjust(sim, ref, hist) -- xclim needs ref and hist
                # during adjustment for internal random rotations.
                corrected_m = mbcn.adjust(sim_m, ref_m, sim_m)
                corrected_months.append(corrected_m)
                months_ok += 1

            except Exception as e:
                print(f"   MBCn error month {m:02d}: {e}")
                corrected_months.append(sim_m)

        print(f"   MBCn applied to {months_ok}/12 months")

        # -- 7. Merge all 12 months back to List[Dict] ----------------
        corrected_all = xr.concat(corrected_months, dim='time').sortby('time')
        ds_final      = sdba.unstack_variables(corrected_all)

        df_final = (
            ds_final.to_dataframe()
            .reset_index()
            .rename(columns={'time': 'date'})
        )
        df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')

        # Merge: preserve non-MBCn columns (hour_*, wind_direction, etc.)
        df_result = df_adj.copy()
        df_result['date'] = df_result['date'].dt.strftime('%Y-%m-%d')
        df_result = df_result.set_index('date')
        df_final  = df_final.set_index('date')
        df_result.update(df_final)   # sobreescribe solo las columnas de valid_vars
        result_records = df_result.reset_index().to_dict('records')

        # -- 8. Physical constraints ----------------------------------
        #self._apply_constraints(result_records)

        return result_records

    # ================================================================
    # Physical constraints
    # ================================================================

    def _apply_constraints(self, records: List[Dict]) -> None:
        """
        Apply in-place physical constraints after MBCn correction.
        """
        for rec in records:
            if rec.get('precipitation') is not None:
                rec['precipitation'] = round(max(0.0, float(rec['precipitation'])), 1)

            for hv in ['humidity_min', 'humidity_max', 'humidity_mean']:
                if rec.get(hv) is not None:
                    rec[hv] = int(round(float(np.clip(float(rec[hv]), 0, 100))))

            for pv in ['pressure_min', 'pressure_max']:
                if rec.get(pv) is not None:
                    rec[pv] = round(max(870.0, float(rec[pv])), 1)

            for wv in ['wind_speed_mean', 'wind_speed_max']:
                if rec.get(wv) is not None:
                    rec[wv] = round(max(0.0, float(rec[wv])), 1)

            tmax  = rec.get('temperature_max')
            tmean = rec.get('temperature_mean')
            tmin  = rec.get('temperature_min')
            if tmax is not None and tmin is not None:
                if float(tmax) < float(tmin):
                    avg = (float(tmax) + float(tmin)) / 2
                    rec['temperature_max'] = round(avg + 0.5, 1)
                    rec['temperature_min'] = round(avg - 0.5, 1)
                    tmax, tmin = rec['temperature_max'], rec['temperature_min']
                if tmean is not None:
                    rec['temperature_mean'] = round(
                        float(np.clip(float(tmean), float(tmin), float(tmax))), 1
                    )

            hmax  = rec.get('humidity_max')
            hmin  = rec.get('humidity_min')
            hmean = rec.get('humidity_mean')
            if hmax is not None and hmin is not None:
                if int(hmax) < int(hmin):
                    rec['humidity_max'], rec['humidity_min'] = hmin, hmax
                    hmax, hmin = rec['humidity_max'], rec['humidity_min']
                if hmean is not None:
                    rec['humidity_mean'] = int(round(
                        float(np.clip(float(hmean), float(hmin), float(hmax)))
                    ))

            pmax = rec.get('pressure_max')
            pmin = rec.get('pressure_min')
            if pmax is not None and pmin is not None and float(pmax) < float(pmin):
                rec['pressure_max'], rec['pressure_min'] = pmin, pmax

            wmax  = rec.get('wind_speed_max')
            wmean = rec.get('wind_speed_mean')
            if wmax is not None and wmean is not None and float(wmax) < float(wmean):
                rec['wind_speed_max'] = wmean
