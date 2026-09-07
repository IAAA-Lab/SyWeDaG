# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Automated test suite (`pytest`) covering the generation pipeline (daily
  cycling, monthly adjustment, hourly interpolation, verification), the
  KNN/XGBoost secondary-variable correctors, the SQLite persistence layer,
  the AEMET/Open-Meteo adapters (network calls mocked), and the results
  import/export ZIP round trip.
- Continuous integration (`.github/workflows/ci.yml`): the test suite runs
  on Linux, Windows, and macOS across Python 3.11-3.12, plus a smoke test
  that launches the app headless and confirms it responds on each OS.
- Offline, network-free tutorial (`examples/run_offline_demo.py`) with a
  bundled sample dataset, also run in CI as an integration test.
- `CONTRIBUTING.md` with development setup, test instructions, code
  conventions, and versioning policy.
- `src/_version.py` as the single source of truth for the app version,
  surfaced in the Settings page.

### Fixed

- `SyntheticWeatherGenerator._verify_hourly_vs_daily` and
  `_verify_daily_vs_predictions` now actually fail (`all_pass = False`) when
  a discrepancy exceeds the documented tolerance; previously several checks
  only printed a warning and always returned `True`.

## [0.1.0]

Baseline captured when this changelog was introduced. Reflects the
functionality already present on `main`:

### Added

- Streamlit-based desktop/web application for generating synthetic hourly
  weather time series from historical daily data.
- AEMET and Open-Meteo data sources behind a common `BaseWeatherSource`
  interface, selected via `config/config.json`.
- Synthetic generation pipeline: historical-year cycling (annual or
  monthly mode), monthly-prediction adjustment for temperature and
  precipitation, secondary-variable correction via K-Nearest Neighbors or
  XGBoost, and continuous hourly interpolation.
- Interactive map for selecting a location and geocoding by name.
- Results page with CSV/ZIP export and import of generated hourly data.
- Local SQLite persistence for historical and generated data.
- Windows desktop build via `streamlit-desktop-app` and PyInstaller
  (`build_desktop.bat`).
