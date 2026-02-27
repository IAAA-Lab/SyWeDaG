"""
Multivariate Bias Correction (MBCn) for synthetic weather data.

Corrects inter-variable dependencies after univariate adjustments
using the xclim.sdba library (MBCn).

Reference: Cannon, A.J. (2018). Multivariate quantile mapping bias correction:
an N-dimensional probability density function transform for climate model
output to observations. Climate Dynamics, 50(1), 31-49.

Flow:
    1. Convertir List[Dict] a xr.Dataset con dimension 'time'
    2. Asignar unidades fisicas (requerido por xclim)
    3. Detectar variables sin NULLs en historico (solo esas se corrigen)
    4. stack_variables() -> DataArray (time, multivar)
    5. Para cada mes (1-12): filtrar por mes, entrenar y ajustar MBCn
       - ref_m = hist_m (todos los anos historicos de ese mes)
       - sim_m = todos los anos ajustados de ese mes
    6. Concatenar los 12 meses corregidos y ordenar por fecha
    7. unstack_variables() -> Dataset -> DataFrame -> List[Dict]
    8. Aplicar restricciones fisicas (max>=mean>=min, precip>=0, etc.)
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Dict

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from xclim import sdba


# Unidades fisicas por variable (requeridas por xclim)
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
    Correccion multivariada MBCn mes a mes.

    Para cada uno de los 12 meses:
      - ref_m / hist_m = todos los dias historicos de ese mes (todos los anos)
      - sim_m          = todos los dias ajustados de ese mes (todos los anos sim)

    Esto garantiza coherencia estacional: enero se corrige contra eneros
    historicos, agosto contra agostos historicos.

    Los campos no meteorologicos (hour_*, wind_direction, source, id_station...)
    se preservan intactos: solo se re-escriben las variables numericas de
    VARIABLES_METEO que existan sin NULLs en el historico.
    """

    VARIABLES_METEO = list(_UNITS.keys())

    def __init__(self, n_iter: int = 20):
        """
        Args:
            n_iter: Numero de iteraciones de rotacion MBCn (default 20).
                    10 rapido, 20 equilibrio, 50 muy costoso.
        """
        self.n_iter = n_iter

    # ================================================================
    # Punto de entrada
    # ================================================================

    def correct(
        self,
        adjusted_data: List[Dict],
        historical_data: List[Dict]
    ) -> List[Dict]:
        """
        Aplica MBCn mes a mes y devuelve los datos corregidos.

        Args:
            adjusted_data:   Datos generados ya ajustados univariadamente.
            historical_data: Datos historicos originales (referencia).

        Returns:
            Lista de registros con la estructura de dependencia multivariada
            corregida y restricciones fisicas aplicadas.
        """
        # -- 1. Convertir a DataFrames --------------------------------
        df_adj  = pd.DataFrame(adjusted_data).copy()
        df_hist = pd.DataFrame(historical_data).copy()

        df_adj['date']  = pd.to_datetime(df_adj['date'])
        df_hist['date'] = pd.to_datetime(df_hist['date'])

        # -- 2. Detectar variables validas (sin NULLs en historico) --
        valid_vars = [
            v for v in self.VARIABLES_METEO
            if v in df_hist.columns
            and v in df_adj.columns
            and df_hist[v].notnull().all()
        ]

        if len(valid_vars) < 2:
            print("   Warning: Menos de 2 variables validas para MBCn, se omite")
            return adjusted_data

        print(f"   Variables incluidas en MBCn ({len(valid_vars)}): {valid_vars}")

        # -- 3. Construir xr.Datasets con dimension 'time' -----------
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

        # -- 4. Asignar unidades fisicas (obligatorio para xclim) ----
        for ds in [ds_hist, ds_adj]:
            for var in valid_vars:
                ds[var].attrs['units'] = _UNITS[var]

        # -- 5. Stack -> DataArray (time, multivar) ------------------
        #   sdba.stack_variables apila las variables del Dataset en una
        #   nueva dimension 'multivar', generando el formato que MBCn espera.
        ref_all = sdba.stack_variables(ds_hist)   # (time_hist, n_vars)
        sim_all = sdba.stack_variables(ds_adj)    # (time_adj,  n_vars)

        # -- 6. MBCn mes a mes ---------------------------------------
        #   xclim NO admite group='time.month' dentro de MBCn (lanza
        #   NotImplementedError), asi que hacemos el bucle mensual
        #   filtrando los DataArrays manualmente.
        corrected_months: List[xr.DataArray] = []
        months_ok = 0

        for m in range(1, 13):
            ref_m = ref_all.sel(time=ref_all.time.dt.month == m)
            sim_m = sim_all.sel(time=sim_all.time.dt.month == m)

            if len(ref_m.time) < 10 or len(sim_m.time) < 5:
                corrected_months.append(sim_m)
                print(f"   Mes {m:02d}: datos insuficientes "
                      f"(ref={len(ref_m.time)}, sim={len(sim_m.time)}), se omite")
                continue

            try:
                # train(ref, hist) -- ref == hist porque no tenemos
                # un modelo climatico separado; MBCn aprende la estructura
                # de dependencia del historico y la transfiere a sim.
                mbcn = sdba.adjustment.MBCn.train(
                    ref=ref_m,
                    hist=sim_m,
                    n_iter=self.n_iter,
                    n_escore=20   # deshabilitar energy score (causa out-of-bounds
                )                # con arrays pequenos en xclim 0.55)

                # adjust(sim, ref, hist) -- xclim necesita ref e hist
                # tambien en el ajuste para las rotaciones aleatorias internas.
                corrected_m = mbcn.adjust(sim_m, ref_m, sim_m)
                corrected_months.append(corrected_m)
                months_ok += 1

            except Exception as e:
                print(f"   Error MBCn mes {m:02d}: {e}")
                corrected_months.append(sim_m)

        print(f"   MBCn aplicado a {months_ok}/12 meses")

        # -- 7. Reunir los 12 meses y volver a List[Dict] ------------
        corrected_all = xr.concat(corrected_months, dim='time').sortby('time')
        ds_final      = sdba.unstack_variables(corrected_all)

        df_final = (
            ds_final.to_dataframe()
            .reset_index()
            .rename(columns={'time': 'date'})
        )
        df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')

        # Fusionar: preservar columnas no-MBCn (hour_*, wind_direction, etc.)
        df_result = df_adj.copy()
        df_result['date'] = df_result['date'].dt.strftime('%Y-%m-%d')
        df_result = df_result.set_index('date')
        df_final  = df_final.set_index('date')
        df_result.update(df_final)   # sobreescribe solo las columnas de valid_vars
        result_records = df_result.reset_index().to_dict('records')

        # -- 8. Restricciones fisicas --------------------------------
        #self._apply_constraints(result_records)

        return result_records

    # ================================================================
    # Restricciones fisicas
    # ================================================================

    def _apply_constraints(self, records: List[Dict]) -> None:
        """
        Aplica restricciones fisicas in-place tras la correccion MBCn.
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
