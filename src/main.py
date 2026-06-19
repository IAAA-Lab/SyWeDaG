"""
SyWeDaG - Synthetic Weather Data Generator
Main entry point for the Streamlit application
"""

import streamlit as st
import json
from dotenv import load_dotenv
from utils.system_utils import get_resource_path

# Load environment variables from .env file
env_path = get_resource_path(".env")
load_dotenv(dotenv_path=str(env_path))

# Page configuration
st.set_page_config(
    page_title="SyWeDaG",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load configuration
config_path = get_resource_path("config/config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Import UI components
from ui.map_component import render_map
from ui.config_page import render_config_page
from ui.results_page import render_results_page
from ui.settings_page import render_settings_page
from ui.styles.base_styles import apply_base_styles
from application.results_services import (
    validate_hourly_data_import_zip,
    persist_imported_hourly_package,
)
from data_sources.base_source import WeatherStation

# Import database initialization
from database.sqliteDB import init_database


@st.dialog("Import generated results", width="large")
def show_import_results_modal():
    """Display ZIP import dialog and validate package structure/content."""
    st.markdown(
        "Upload a ZIP package containing `metadata.json` and `hourly_data.csv`."
    )

    uploaded_zip = st.file_uploader(
        "Select results package (.zip)",
        type=["zip"],
        key="import_results_zip_uploader",
    )

    if uploaded_zip is None:
        return

    try:
        metadata, dataframe = validate_hourly_data_import_zip(uploaded_zip.getvalue())

        nearest_station = metadata.get("nearest_station", {}) or {}
        station_name = nearest_station.get("name") or "N/A"
        station_region = nearest_station.get("region") or ""
        station_display = station_name if not station_region else f"{station_name}, {station_region}"

        st.success(f"✅ Valid package. {len(dataframe):,} hourly rows detected.")
        st.markdown(
            f"""
            - **Source**: {metadata.get('data_source', 'N/A')}
            - **Station**: {station_display}
            - **Simulated period**: {metadata.get('periods', {}).get('generated_start', 'N/A')} → {metadata.get('periods', {}).get('generated_end', 'N/A')}
            """
        )

        if st.button("Load imported data", key="load_imported_data_button", type="primary"):
            try:
                with st.spinner("Loading imported data..."):
                    job_id, imported_count = persist_imported_hourly_package(metadata, dataframe)

                    location = metadata.get("location", {}) or {}
                    nearest_station = metadata.get("nearest_station", {}) or {}
                    station_source = metadata.get("data_source") or "N/A"

                    station_for_session = WeatherStation(
                        source=str(station_source),
                        id_station=nearest_station.get("id_station"),
                        name=str(nearest_station.get("name") or "N/A"),
                        region=nearest_station.get("region"),
                        latitude=float(location.get("latitude")),
                        longitude=float(location.get("longitude")),
                        height=0,
                    )

                    st.session_state.job_id = job_id
                    st.session_state.nearest_station = station_for_session
                    st.session_state.records_count = imported_count
                    st.session_state.current_page = 'results'
                    st.rerun()
            except Exception as error:
                st.error(f"❌ Error loading imported data: {error}")
    except ValueError as error:
        st.error(f"❌ Invalid package: {error}")
    except Exception as error:
        st.error(f"❌ Unexpected error validating package: {error}")

def main():
    """Main application function"""
    
    # Initialize database if it doesn't exist
    db_path = get_resource_path("data/weather.db")
    if not db_path.exists():
        init_database()
    
    # Apply base styles
    apply_base_styles()
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = config
    if 'selected_point' not in st.session_state:
        st.session_state.selected_point = None
    if 'selected_data_source' not in st.session_state:
        st.session_state.selected_data_source = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'map'

    current_config = st.session_state.config
    
    # Menu button in header (visible on all pages)
    if st.button("☰", key="header_menu_button"):
        st.session_state.current_page = 'settings'
        st.rerun()
    
    # Import button in header (only on map page)
    if st.session_state.current_page == 'map':
        if st.button("📥", key="header_import_button"):
            show_import_results_modal()

    # Home button in header (visible on all pages except map)
    if st.session_state.current_page != 'map':
        if st.button("←", key="header_home_button"):
            st.session_state.current_page = 'map'
            st.session_state.selected_point = None
            st.session_state.selected_data_source = None
            st.rerun()
    
    # Blue Header with logo
    logo_path = get_resource_path("assets/Portada.png")
    if logo_path.exists():
        st.markdown(f'''
        <div class="blue-header">
            <img src="data:image/png;base64,{get_base64_image(str(logo_path))}" class="logo" />
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="blue-header">
            <h1 class="header-title">SyWeDaG</h1>
        </div>
        ''', unsafe_allow_html=True)
    
    # Render the appropriate page
    if st.session_state.current_page == 'map':
        render_map(current_config)
    elif st.session_state.current_page == 'config':
        render_config_page(current_config)
    elif st.session_state.current_page == 'results':
        render_results_page()
    elif st.session_state.current_page == 'settings':
        render_settings_page(current_config)

def get_base64_image(image_path):
    """Convert image to base64 for embedding"""
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

if __name__ == "__main__":
    main()
