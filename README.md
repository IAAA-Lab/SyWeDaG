# meteoZar - Synthetic Weather Scenario Generator

#TODO: Update this README

A desktop application for generating and visualizing synthetic meteorological scenarios using historical weather data from multiple sources (AEMET for Spain, extensible to other countries).

## Project Structure

```
├── assets/                     # Images and resources
├── config/
│   └── config.json             # Application configuration
├── data/                       # Local SQLite database
├── sample_pred_excels/         # Prediction sample Excel files
├── src/
│   ├── main.py                 # Main entry point
│   ├── ui/                     # User interface components
│   │   ├── styles/             # Custom UI styles
│   │   ├── config_page.py      # Configuration page for data generation
│   │   ├── map_component.py    # Map with search and zoom
│   │   └── results_page.py     # Page with graphics for generated scenarios
│   ├── data_sources/           # Modular data source adapters
│   │   ├── base_source.py      # Abstract base class
│   │   └── aemet_source.py     # AEMET implementation
│   ├── database/               # SQLite database management
│   │   └── sqliteDB.py
│   ├── generators/             # Synthetic data generation
│   │   └── synthetic_generator.py
│   └── utils/                  # Utility functions
│       └── geocoding.py
├── build_desktop.bat           # Build script for desktop application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
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

## License

TFG Project - Escuela de Ingenieria y Arquitectura de Zaragoza (EINA) - Universidad de Zaragoza (UNIZAR) - 2026
