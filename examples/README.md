# Offline demo

`run_offline_demo.py` runs the full SyWeDaG generation pipeline (cycle -> adjust
-> interpolate -> verify) without a database, an AEMET API key, or network
access. It is meant as a quick-start tutorial and as an executable check that
the generation pipeline still works end to end.

## What it does

1. Loads `sample_historical_data.csv`, a 3-year (2018-2020) bundled daily
   dataset shaped like the records the app fetches from a real data source.
2. Loads one year (2021) of monthly predictions from
   `../sample_pred_excels/combined_data.csv`.
3. Feeds both into `SyntheticWeatherGenerator.generate()` directly, bypassing
   the database and data-source layers (the only production line touched is
   the generation pipeline itself).
4. Writes the resulting hourly series to `output/generated_hourly_2021.csv`.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
python examples/run_offline_demo.py
```

The console output includes the pipeline's own consistency checks (daily vs.
monthly predictions, hourly vs. daily), so you can see how SyWeDaG reports
discrepancies between the requested climate targets and the generated series.

## Adapting it

To try a different station or period, replace `sample_historical_data.csv`
with your own daily records (same columns) or point `HISTORICAL_CSV` in the
script at a CSV exported from the app's own database via the results page.
