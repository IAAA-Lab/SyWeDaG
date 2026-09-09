# SyWeDaG: Synthetic Weather Data Generator

[![CI](https://github.com/IAAA-Lab/SyWeDaG/actions/workflows/ci.yml/badge.svg)](https://github.com/IAAA-Lab/SyWeDaG/actions/workflows/ci.yml)

A desktop application for generating and visualizing synthetic meteorological scenarios using historical weather data from multiple sources (AEMET for Spain, extensible to other countries).

## Project Structure

```
├── assets/                          # Images and resources
├── config/ 
│   └── config.json                  # Application and data-source configuration
├── data/                            # Local SQLite database files
├── examples/                        # Offline, network-free tutorial (see Testing below)
├── sample_pred_excels/              # Sample prediction Excel files
├── tests/                           # pytest suite, mirrors the src/ layout
├── src/ 
│   ├── _version.py                  # Single source of truth for the app version
│   ├── main.py                      # Streamlit entry point
│   ├── application/                 # Application/business logic (UI-independent)
│   │   ├── map_services.py          # Geocoding + GeoJSON coverage logic
│   │   └── config_services.py       # Validation + fetch/generate orchestration
│   ├── ui/                          # Presentation layer (Streamlit/Folium)
│   │   ├── styles/                  # UI styles per page/component
│   │   ├── map_component.py         # Interactive map page/component
│   │   ├── config_page.py           # Data/generation configuration page
│   │   └── results_page.py          # Results and visualization page
│   ├── data_sources/                # Weather source adapters
│   │   ├── base_source.py           # Common source interface/models
│   │   ├── aemet_source.py          # AEMET implementation
│   │   └── source_selector.py       # Source factory/selector
│   ├── generators/                  # Synthetic data generation logic
│   │   ├── synthetic_generator.py   # Main orchestration for daily/hourly generation
│   │   ├── daily_correctors/        # Secondary-variable correction models
│   │   │   ├── k_neighbors.py
│   │   │   ├── xgboost_model.py
│   │   │   └── mbc_correction.py
│   │   ├── monthly_adjustments/     # Monthly prediction adjustment logic
│   │   │   ├── temperature_adjuster.py
│   │   │   └── precipitation_adjuster.py
│   │   └── hourly_generation/       # Daily-to-hourly interpolation helpers
│   │       └── hourly_interpolator.py
│   ├── database/
│   │   └── sqliteDB.py              # DB schema and persistence helpers
│   ├── modelValidation              # Instructions and utilities for validating the supported models
│   └── utils/                       # Shared utility helpers
│       ├── data_parsing.py
│       ├── geospatial.py
│       ├── historical_data_treatment.py
│       └── system_utils.py
├── .github/workflows/ci.yml         # CI: test matrix + app startup smoke test
├── build_desktop.bat                # Desktop build script
├── SyWeDaG.spec              # PyInstaller spec (generated/used in builds)
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt             # Additional dependencies for running tests
├── CONTRIBUTING.md                  # Development setup, conventions, versioning policy
├── CHANGELOG.md                     # Notable changes, per Keep a Changelog
└── README.md
```

## Features

- **Interactive Map**: Select geographical points in Spain using OpenStreetMap
- **Search Functionality**: Search for locations by name
- **Zoom Controls**: Navigate the map with custom zoom buttons
- **Data Source Highlighting**: Visual indication of areas with available data
- **Modular Design**: Easy to add new data sources for other countries

## Installation

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

From the `src` directory:

```bash
streamlit run main.py
```

For desktop mode (from root directory):
```bash
build_desktop.bat
```

This will create a standalone executable in the `dist` folder.

### Try it without an API key

`examples/run_offline_demo.py` runs the full generation pipeline on a bundled
sample dataset, no AEMET API key or network access required. See
[`examples/README.md`](examples/README.md).

## Configuration

Edit `config/config.json` to:
- Add new data sources
- Modify default map settings
- Configure data source geographical boundaries

## Technologies

- **Streamlit**: Web framework for the UI
- **Folium**: Interactive maps
- **SQLite**: Local data storage
- **Pandas/NumPy**: Data manipulation
- **Plotly**: Data visualization

## Testing

```bash
pip install -r requirements-dev.txt
pytest --cov=src --cov-report=term-missing
```

The suite focuses on the generation pipeline's scientific properties rather
than just execution: monthly adjustment invariants (e.g. `Tmin <= Tmean <=
Tmax` after adjustment, monthly means matching predictions within
tolerance), hourly interpolation consistency against daily aggregates, and
the SQLite persistence and ZIP export/import round trips. Network calls to
AEMET and Open-Meteo are mocked, so no test requires internet access or an
API key.

The Streamlit UI layer (`src/ui/`) is not unit-tested; it is instead covered
by a CI job that launches the packaged app and confirms it responds. The
MBCn corrector (`generators/daily_correctors/mbc_correction.py`) is
implemented but not wired into the generation pipeline, and is untested
accordingly.

CI (`.github/workflows/ci.yml`) runs the full suite on Linux, Windows, and
macOS across Python 3.11-3.12 on every push and pull request (`numpy` 2.3.5,
pinned in `requirements.txt`, requires Python >= 3.11).

## Versioning

SyWeDaG follows [Semantic Versioning](https://semver.org/). The current
version is defined in `src/_version.py`; see [`CHANGELOG.md`](CHANGELOG.md)
for the history of notable changes.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, test
instructions, code conventions, and how to report issues.
