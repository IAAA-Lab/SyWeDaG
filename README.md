# SyWeDaG: Synthetic Weather Data Generator

A desktop application for generating and visualizing synthetic meteorological scenarios using historical weather data from multiple sources (AEMET for Spain, extensible to other countries).

## Project Structure

```
├── assets/                          # Images and resources
├── config/ 
│   └── config.json                  # Application and data-source configuration
├── data/                            # Local SQLite database files
├── sample_pred_excels/              # Sample prediction Excel files
├── src/ 
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
│   └── utils/                       # Shared utility helpers
│       ├── data_parsing.py
│       ├── geospatial.py
│       ├── historical_data_treatment.py
│       └── system_utils.py
├── build_desktop.bat                # Desktop build script
├── MeteoSynthetic.spec              # PyInstaller spec (generated/used in builds)
├── requirements.txt                 # Python dependencies
└── README.md
```

## Features

- **Interactive Map**: Select geographical points in Spain using OpenStreetMap
- **Search Functionality**: Search for locations by name
- **Zoom Controls**: Navigate the map with custom zoom buttons
- **Data Source Highlighting**: Visual indication of areas with available data
- **Modular Design**: Easy to add new data sources for other countries

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Additional dependency for point-in-polygon detection:
```bash
pip install shapely
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
