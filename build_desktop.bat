@echo off
REM Build script for SyWeDaG - Desktop App with streamlit-desktop-app
echo Building SyWeDaG desktop application...
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
if exist SyWeDaG.spec (
    echo Removing spec file...
    del /q SyWeDaG.spec 2>nul
)
REM data\ is gitignored, so it may not exist on a fresh clone; PyInstaller's
REM --add-data "data;data" fails if it is missing. Always start from an empty one.
echo Resetting data folder...
if exist data (
    rmdir /s /q data 2>nul
    timeout /t 1 /nobreak >nul
)
mkdir data
echo.

py -3.12 --version >nul 2>&1
if errorlevel 1 (
        echo ERROR: Python 3.12 is not installed or is not available with launcher ^(py -3.12^).
        pause
        exit /b 1
)

echo Starting compilation...
echo.

REM Only options PyInstaller can't work out on its own are listed here.
REM main.py is added as data (not analysed), so our own modules must be hidden
REM imports; their third-party imports are then followed automatically.
streamlit-desktop-app build src/main.py ^
  --name SyWeDaG ^
  --pyinstaller-options ^
    --paths src ^
    --collect-all streamlit_folium ^
    --collect-all xgboost ^
    --collect-all xclim ^
    --collect-data yamale ^
    --copy-metadata streamlit ^
    --copy-metadata streamlit-folium ^
    --copy-metadata folium ^
    --copy-metadata numpy ^
    --copy-metadata requests ^
    --copy-metadata openmeteo-requests ^
    --copy-metadata requests-cache ^
    --copy-metadata retry-requests ^
    --copy-metadata geopy ^
    --copy-metadata shapely ^
    --copy-metadata scikit-learn ^
    --copy-metadata xgboost ^
    --copy-metadata python-dotenv ^
    --copy-metadata xclim ^
    --copy-metadata xarray ^
    --copy-metadata pint ^
    --add-data "config;config" ^
    --add-data "assets;assets" ^
    --add-data "data;data" ^
    --hidden-import _version ^
    --hidden-import dotenv ^
    --hidden-import ui.map_component ^
    --hidden-import ui.config_page ^
    --hidden-import ui.results_page ^
    --hidden-import ui.settings_page ^
    --hidden-import ui.styles.base_styles ^
    --hidden-import ui.styles.map_styles ^
    --hidden-import ui.styles.config_styles ^
    --hidden-import ui.styles.results_styles ^
    --hidden-import ui.styles.settings_styles ^
    --hidden-import data_sources.aemet_source ^
    --hidden-import data_sources.open_meteo_source ^
    --hidden-import data_sources.base_source ^
    --hidden-import data_sources.source_selector ^
    --hidden-import generators.daily_correctors.k_neighbors ^
    --hidden-import generators.daily_correctors.xgboost_model ^
    --hidden-import generators.monthly_adjustments.temperature_adjuster ^
    --hidden-import generators.monthly_adjustments.precipitation_adjuster ^
    --hidden-import generators.hourly_generation.hourly_interpolator ^
    --hidden-import generators.synthetic_generator ^
    --hidden-import generators.daily_correctors.mbc_correction ^
    --hidden-import application.map_services ^
    --hidden-import application.config_services ^
    --hidden-import application.results_services ^
    --hidden-import database.sqliteDB ^
    --hidden-import utils.data_parsing ^
    --hidden-import utils.geospatial ^
    --hidden-import utils.historical_data_treatment ^
    --hidden-import utils.system_utils ^
    --onedir ^
    --noconsole ^
    --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: Build failed. See the output above.
    pause
    exit /b 1
)

echo.
echo Build complete! Your desktop app is ready.
echo.
pause
