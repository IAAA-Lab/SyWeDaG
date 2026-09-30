#!/bin/bash

# Build script for SyWeDaG - Desktop App with streamlit-desktop-app
echo "Building SyWeDaG desktop application..."
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
if [ -d "dist" ]; then
    echo "Removing dist folder..."
    rm -rf dist
    sleep 2
fi

if [ -d "build" ]; then
    echo "Removing build folder..."
    rm -rf build
fi

if [ -f "SyWeDaG.spec" ]; then
    echo "Removing spec file..."
    rm -f SyWeDaG.spec
fi

echo "Resetting data folder..."
rm -rf data
mkdir -p data
echo ""

# Check Python 3.12 on Linux
if ! command -v python3.12 &> /dev/null; then
        echo "ERROR: Python 3.12 is not installed or is not available."
        read -p "Press [Enter] key to resume..."
        exit 1
fi

echo "Starting compilation..."
echo ""

# Only options PyInstaller can't work out on its own are listed here.
# main.py is added as data (not analysed), so our own modules must be hidden
# imports; their third-party imports are then followed automatically.
streamlit-desktop-app build src/main.py \
  --name SyWeDaG \
  --pyinstaller-options \
    --paths src \
    --collect-all streamlit_folium \
    --collect-all xgboost \
    --collect-all xclim \
    --collect-data yamale \
    --copy-metadata streamlit \
    --copy-metadata streamlit-folium \
    --copy-metadata folium \
    --copy-metadata numpy \
    --copy-metadata requests \
    --copy-metadata openmeteo-requests \
    --copy-metadata requests-cache \
    --copy-metadata retry-requests \
    --copy-metadata geopy \
    --copy-metadata shapely \
    --copy-metadata scikit-learn \
    --copy-metadata xgboost \
    --copy-metadata python-dotenv \
    --copy-metadata xclim \
    --copy-metadata xarray \
    --copy-metadata pint \
    --add-data "config:config" \
    --add-data "assets:assets" \
    --add-data "data:data" \
    --hidden-import _version \
    --hidden-import dotenv \
    --hidden-import ui.map_component \
    --hidden-import ui.config_page \
    --hidden-import ui.results_page \
    --hidden-import ui.settings_page \
    --hidden-import ui.styles.base_styles \
    --hidden-import ui.styles.map_styles \
    --hidden-import ui.styles.config_styles \
    --hidden-import ui.styles.results_styles \
    --hidden-import ui.styles.settings_styles \
    --hidden-import data_sources.aemet_source \
    --hidden-import data_sources.open_meteo_source \
    --hidden-import data_sources.base_source \
    --hidden-import data_sources.source_selector \
    --hidden-import generators.daily_correctors.k_neighbors \
    --hidden-import generators.daily_correctors.xgboost_model \
    --hidden-import generators.monthly_adjustments.temperature_adjuster \
    --hidden-import generators.monthly_adjustments.precipitation_adjuster \
    --hidden-import generators.hourly_generation.hourly_interpolator \
    --hidden-import generators.synthetic_generator \
    --hidden-import generators.daily_correctors.mbc_correction \
    --hidden-import application.map_services \
    --hidden-import application.config_services \
    --hidden-import application.results_services \
    --hidden-import database.sqliteDB \
    --hidden-import utils.data_parsing \
    --hidden-import utils.geospatial \
    --hidden-import utils.historical_data_treatment \
    --hidden-import utils.system_utils \
    --hidden-import PyQt6.QtWebEngineCore \
    --hidden-import PyQt6.QtWebEngineWidgets \
    --hidden-import webview.platforms.qt \
    --onedir \
    --noconsole \
    --noconfirm

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed. See the output above."
    read -p "Press [Enter] key to continue..."
    exit 1
fi

echo ""
echo "Build complete! Your desktop app is ready."
echo ""
read -p "Press [Enter] key to continue..."
