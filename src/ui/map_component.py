"""
Map component for location selection
Includes search functionality and zoom controls
"""

import os
import streamlit as st
import folium
from streamlit_folium import st_folium
from application.map_services import (
    load_geojson_files,
    geocode_location,
    get_data_sources_for_point,
)
from ui.styles.map_styles import apply_map_styles
from utils.system_utils import safe_print


def _selected_source_missing_api_key(config):
    """
    Check whether currently selected data source requires an API key but has it empty.
    Checks environment variables based on config specification.

    Args:
        config (dict): Configuration dictionary

    Returns:
        tuple[bool, str | None]: (is_missing, source_name)
    """
    selected_source_name = st.session_state.get("selected_data_source")
    if not selected_source_name:
        return False, None

    for source in config.get("data_sources", []):
        source_name = source.get("name")
        if source_name == selected_source_name:
            # Check if source requires an API key
            env_var_name = source.get("api_key_env_var")
            if env_var_name:
                # API key is required - check if it's set in environment
                env_api_key = os.getenv(env_var_name)
                
                if not env_api_key or not str(env_api_key).strip():
                    # API key is required but missing
                    return True, source_name
            
            return False, source_name

    return False, None


@st.dialog("API Key Required", width="large")
def show_api_key_required_modal(source_name):
    """Show modal dialog when selected source requires an API key."""
    st.markdown(
        f"The data source **{source_name}** requires an API key to continue."
    )
    st.markdown("Go to **Settings** and enter an API key.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("GO TO SETTINGS", key="go_settings_from_api_modal", use_container_width=True):
            st.session_state.current_page = "settings"
            st.rerun()

    with col2:
        if st.button("CLOSE", key="close_api_modal", use_container_width=True):
            st.rerun()


@st.dialog("Select Data Source", width="large")
def show_data_source_selection_modal(candidate_sources):
    """Show modal to select a data source when multiple match the clicked point."""
    st.markdown("More than one data source is available for this location. Select one:")

    source_names = [source.get("name", "Unknown") for source in candidate_sources]
    selected_name = st.radio(
        "Data source",
        options=source_names,
        key="selected_data_source_radio_modal",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("CONFIRM", key="confirm_data_source_modal", use_container_width=True):
            st.session_state.selected_data_source = selected_name
            st.session_state.source_selection_open = False
            st.session_state.source_selection_candidates = []
            st.rerun()

    with col2:
        if st.button("CANCEL", key="cancel_data_source_modal", use_container_width=True):
            st.session_state.selected_data_source = None
            st.session_state.source_selection_open = False
            st.session_state.source_selection_candidates = []
            st.rerun()

def add_country_polygons(m, config):
    """
    Add semitransparent country polygons to the map from local GeoJSON files
    Uses cached loading for performance
    
    Args:
        m (folium.Map): Folium map object
        config (dict): Configuration dictionary with data sources
    """
    try:
        # Load cached GeoJSON files
        geojson_cache = load_geojson_files()
        
        # Group by GeoJSON path to avoid drawing the same polygon multiple times.
        polygons_by_path = {}
        for source in config.get("data_sources", []):
            source_name = source.get("name", source.get("country", "Unknown"))
            country_name = source.get("country", "Unknown")
            color = source.get("color", "#3388ff")
            geojson_path = source.get("geojson_path") or source_name

            if not source_name or source_name not in geojson_cache:
                continue

            if geojson_path not in polygons_by_path:
                polygons_by_path[geojson_path] = {
                    "country": country_name,
                    "color": color,
                    "sources": [source_name],
                    "geojson": geojson_cache[source_name],
                }
            else:
                polygons_by_path[geojson_path]["sources"].append(source_name)

        for polygon_info in polygons_by_path.values():
            try:
                tooltip_text = (
                    f"{polygon_info['country']} "
                    f"({', '.join(polygon_info['sources'])})"
                )
                folium.GeoJson(
                    polygon_info["geojson"],
                    name=tooltip_text,
                    style_function=lambda x, c=polygon_info["color"]: {
                        'fillColor': c,
                        'color': '#333333',
                        'weight': 2,
                        'fillOpacity': 0.4
                    },
                    tooltip=tooltip_text
                ).add_to(m)

                safe_print(f"✅ Polygon loaded for {', '.join(polygon_info['sources'])}")
            except Exception as e:
                safe_print(f"⚠️ Error loading polygon: {e}")
    
    except Exception as e:
        safe_print(f"⚠️ Error adding country polygons: {e}")

def create_map(config, center=None, zoom=None):
    """
    Create a Folium map without overlays
    
    Args:
        config (dict): Configuration dictionary
        center (list): Map center [lat, lon]
        zoom (int): Zoom level
        
    Returns:
        folium.Map: Configured map object
    """
    if center is None:
        center = config.get("default_map_center", [40.4168, -3.7038])
    if zoom is None:
        zoom = config.get("default_zoom", 6)
    
    # Create base map
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        zoom_control=False,
        scrollWheelZoom=True,
        dragging=True,
        attribution_control=True
    )
    
    # Add country polygons
    add_country_polygons(m, config)
    
    return m

def render_map(config):
    """
    Render the main map interface with search overlay
    
    Args:
        config (dict): Configuration dictionary
    """
    
    # Apply map-specific styles
    apply_map_styles()

    if st.session_state.get("source_selection_open", False):
        candidates = st.session_state.get("source_selection_candidates", [])
        if candidates:
            show_data_source_selection_modal(candidates)
    
    # Search input
    search_query = st.text_input(
        "search",
        key="search_input",
        placeholder="Search...",
        label_visibility="collapsed"
    )
    
    # Handle search
    if search_query and search_query != st.session_state.get('last_search', ''):
        coords = geocode_location(search_query)
        if coords:
            st.session_state.map_center = list(coords)
            st.session_state.map_zoom = 12
            st.session_state.last_search = search_query
            st.rerun()

    # SIMPLE SELECT BUTTON - at the bottom
    has_selection = (st.session_state.get('selected_point') is not None and 
                     st.session_state.get('selected_data_source') is not None)
    
    if st.button("SELECT", 
                 key="select_button", 
                 disabled=not has_selection,
                 use_container_width=False):
        missing_api_key, source_name = _selected_source_missing_api_key(config)
        if missing_api_key:
            show_api_key_required_modal(source_name)
        else:
            st.session_state.current_page = 'config'
            st.rerun()
    
    # Get current map state
    center = st.session_state.get('map_center', config.get("default_map_center"))
    zoom = st.session_state.get('map_zoom', config.get("default_zoom"))
    
    # Create and display map
    m = create_map(config, center, zoom)
    
    # Add marker for selected point
    if st.session_state.selected_point:
        folium.Marker(
            location=st.session_state.selected_point,
            popup="Selected Location",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
    
    # Render map (full screen, no borders)
    map_data = st_folium(
        m,
        width="100%",
        height=800,
        key="main_map",
        returned_objects=["last_clicked", "zoom", "center"]
    )
    
    # Handle map clicks - ONLY rerun on actual clicks
    if map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        
        # Check if we need to update (avoid infinite rerun loop)
        previous_point = st.session_state.get('last_processed_click')
        
        # Only rerun if this is a genuinely new click
        if previous_point is None or abs(previous_point[0] - lat) > 0.00001 or abs(previous_point[1] - lon) > 0.00001:
            # Update selected point
            st.session_state.selected_point = [lat, lon]
            st.session_state.last_processed_click = [lat, lon]
            
            # Save the CURRENT camera position (center) to restore after rerun
            if map_data.get("center"):
                st.session_state.map_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
            
            # Save the current zoom level to restore after rerun
            if map_data.get("zoom"):
                st.session_state.map_zoom = map_data["zoom"]
            
            # Check which data source(s) cover this point
            try:
                candidate_sources = get_data_sources_for_point(lat, lon, config)

                if len(candidate_sources) == 1:
                    st.session_state.selected_data_source = candidate_sources[0]["name"]
                    st.session_state.source_selection_open = False
                    st.session_state.source_selection_candidates = []
                elif len(candidate_sources) > 1:
                    st.session_state.selected_data_source = None
                    st.session_state.source_selection_open = True
                    st.session_state.source_selection_candidates = candidate_sources
                else:
                    st.session_state.selected_data_source = None
                    st.session_state.source_selection_open = False
                    st.session_state.source_selection_candidates = []
            except ImportError:
                pass
            
            # Force rerun to immediately show the marker
            st.rerun()
    

