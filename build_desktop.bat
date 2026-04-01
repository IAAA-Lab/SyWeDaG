@echo off
REM Build script for MeteoSynthetic - Desktop App with streamlit-desktop-app
echo Building MeteoSynthetic desktop application...
echo.

REM Limpiar builds anteriores
echo Cleaning previous builds...
if exist dist (
    echo Removing dist folder...
    rmdir /s /q dist 2>nul
    timeout /t 2 /nobreak >nul
)
if exist build (
    echo Removing build folder...
    rmdir /s /q build 2>nul
)
if exist MeteoSynthetic.spec (
    echo Removing spec file...
    del /q MeteoSynthetic.spec 2>nul
)
if exist data (
    echo Cleaning data folder...
    rmdir /s /q data 2>nul
    timeout /t 1 /nobreak >nul
    mkdir data
)
echo.

py -3.12 --version >nul 2>&1
if errorlevel 1 (
        echo ERROR: Python 3.12 is not installed or is not available with launcher ^(py -3.12^).
        pause
        exit /b 1
)

echo Starting compilation...
echo.

streamlit-desktop-app build src/main.py ^
  --name MeteoSynthetic ^
  --pyinstaller-options ^
    --collect-all streamlit ^
    --collect-all streamlit_folium ^
    --collect-all folium ^
    --collect-all numpy ^
    --collect-all requests ^
    --collect-all geopy ^
    --collect-all shapely ^
    --collect-all sklearn ^
    --collect-all xgboost ^
    --copy-metadata streamlit ^
    --copy-metadata streamlit-folium ^
    --copy-metadata folium ^
    --copy-metadata numpy ^
    --copy-metadata requests ^
    --copy-metadata geopy ^
    --copy-metadata shapely ^
    --copy-metadata scikit-learn ^
    --copy-metadata xgboost ^
    --add-data "config;config" ^
    --add-data "assets;assets" ^
    --add-data "assets/geocountries;assets/geocountries" ^
    --add-data "data;data" ^
    --add-data "src/ui;ui" ^
    --add-data "src/ui/styles;ui/styles" ^
    --add-data "src/data_sources;data_sources" ^
    --add-data "src/database;database" ^
    --add-data "src/generators;generators" ^
    --add-data "src/application;application" ^
    --add-data "src/utils;utils" ^
    --hidden-import database.sqliteDB ^
    --hidden-import sqlite3 ^
    --hidden-import ui.map_component ^
    --hidden-import ui.config_page ^
    --hidden-import ui.results_page ^
    --hidden-import ui.settings_page ^
    --hidden-import ui.styles.base_styles ^
    --hidden-import ui.styles.map_styles ^
    --hidden-import ui.styles.config_styles ^
    --hidden-import ui.styles.results_styles ^
    --hidden-import ui.styles.settings_styles ^
    --hidden-import data_sources ^
    --hidden-import data_sources.aemet_source ^
    --hidden-import data_sources.base_source ^
    --hidden-import data_sources.source_selector ^
    --hidden-import data_sources.__init__ ^
    --hidden-import generators ^
    --hidden-import generators.daily_correctors ^
    --hidden-import generators.daily_correctors.k_neighbors ^
    --hidden-import generators.daily_correctors.xgboost_model ^
    --hidden-import generators.monthly_adjustments ^
    --hidden-import generators.monthly_adjustments.temperature_adjuster ^
    --hidden-import generators.monthly_adjustments.precipitation_adjuster ^
    --hidden-import generators.hourly_generation ^
    --hidden-import generators.hourly_generation.hourly_interpolator ^
    --hidden-import generators.synthetic_generator ^
    --hidden-import generators.daily_correctors.mbc_correction ^
    --hidden-import application ^
    --hidden-import application.map_services ^
    --hidden-import application.config_services ^
    --hidden-import application.results_services ^
    --hidden-import sklearn ^
    --hidden-import sklearn.neighbors ^
    --hidden-import sklearn.preprocessing ^
    --hidden-import sklearn.linear_model ^
    --hidden-import sklearn.impute ^
    --hidden-import sklearn.multioutput ^
    --hidden-import xgboost ^
    --hidden-import joblib ^
    --hidden-import multiprocessing ^
    --hidden-import multiprocessing.managers ^
    --hidden-import multiprocessing.pool ^
    --hidden-import concurrent.futures ^
    --hidden-import folium ^
    --hidden-import folium.plugins ^
    --hidden-import geopy ^
    --hidden-import geopy.geocoders ^
    --hidden-import geopy.geocoders.nominatim ^
    --hidden-import geopy.exc ^
    --hidden-import shapely ^
    --hidden-import shapely.geometry ^
    --hidden-import shapely.geometry.point ^
    --hidden-import shapely.geometry.polygon ^
    --hidden-import shapely.geometry.shape ^
    --hidden-import branca ^
    --hidden-import branca.element ^
    --hidden-import jinja2 ^
    --hidden-import numpy ^
    --hidden-import numpy.core ^
    --hidden-import numpy.core._multiarray_umath ^
    --hidden-import requests ^
    --hidden-import requests.adapters ^
    --hidden-import requests.structures ^
    --hidden-import urllib3 ^
    --hidden-import urllib3.util ^
    --hidden-import certifi ^
    --hidden-import charset_normalizer ^
    --hidden-import idna ^
    --hidden-import json ^
    --hidden-import pathlib ^
    --hidden-import datetime ^
    --hidden-import math ^
    --hidden-import time ^
    --hidden-import typing ^
    --hidden-import utils.data_parsing ^
    --hidden-import utils.geospatial ^
    --hidden-import utils.historical_data_treatment ^
    --hidden-import utils.system_utils ^
    --hidden-import base64 ^
    --hidden-import random ^
    --hidden-import sys ^
    --onedir ^
    --noconsole ^
    --noconfirm

echo.
echo Build complete! Your desktop app is ready.
echo.
pause
