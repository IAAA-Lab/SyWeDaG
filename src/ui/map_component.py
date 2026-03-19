"""
Map component for location selection
Includes search functionality and zoom controls
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import json
from ui.styles.map_styles import apply_map_styles
from utils.system_utils import get_resource_path

@st.cache_resource
def load_geojson_files():
    """
    Cache loaded GeoJSON files to avoid reloading on every click
    
    Returns:
        dict: Dictionary with source names as keys and GeoJSON data as values
    """
    geojson_cache = {}
    
    try:
        config_path = get_resource_path("config/config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for source in config.get("data_sources", []):
            geojson_path = source.get("geojson_path")
            source_name = source.get("name")
            
            if geojson_path and source_name:
                full_path = get_resource_path(geojson_path)
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            geojson_cache[source_name] = json.load(f)
                    except Exception as e:
                        print(f"⚠️ Error loading {geojson_path}: {e}")
    
    except Exception as e:
        print(f"⚠️ Error in load_geojson_files: {e}")
    
    return geojson_cache

def geocode_location(search_query):
    """
    Geocode a search query to coordinates
    
    Args:
        search_query (str): Location to search for
        
    Returns:
        tuple: (latitude, longitude) or None if not found
    """
    try:
        geolocator = Nominatim(user_agent="meteoZar")
        location = geolocator.geocode(search_query, timeout=10)
        if location:
            return (location.latitude, location.longitude)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.error(f"Geocoding error: {str(e)}")
    return None

def get_data_source_for_point(lat, lon, config):
    """
    Determine which data source covers a given point using GeoJSON boundaries
    Uses cached GeoJSON for performance
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        config (dict): Configuration dictionary
        
    Returns:
        dict: Data source configuration or None
    """
    from shapely.geometry import Point, shape
    
    point = Point(lon, lat)
    
    # Load cached GeoJSON files
    geojson_cache = load_geojson_files()
    
    for source in config.get("data_sources", []):
        source_name = source.get("name")
        
        if source_name and source_name in geojson_cache:
            geojson_data = geojson_cache[source_name]
            
            try:
                # Check each feature in the GeoJSON
                for feature in geojson_data.get("features", []):
                    geometry = feature.get("geometry")
                    if geometry:
                        # Convert GeoJSON geometry to Shapely shape
                        shapely_geom = shape(geometry)
                        
                        # Check if point is within this geometry
                        if shapely_geom.contains(point):
                            return source
            
            except Exception as e:
                print(f"⚠️ Error checking GeoJSON {source_name}: {e}")
    
    return None

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
        
        # Add each data source country polygon
        for source in config.get("data_sources", []):
            source_name = source.get("name", source.get("country", "Unknown"))
            country_name = source.get("country", "Unknown")
            color = source.get("color", "#3388ff")  # Default blue if not specified
            
            if source_name and source_name in geojson_cache:
                geojson_data = geojson_cache[source_name]
                
                try:
                    # Add to map with custom style and tooltip
                    tooltip_text = f"{country_name} ({source_name})"
                    folium.GeoJson(
                        geojson_data,
                        name=source_name,
                        style_function=lambda x, c=color: {
                            'fillColor': c,
                            'color': '#333333',
                            'weight': 2,
                            'fillOpacity': 0.4
                        },
                        tooltip=tooltip_text
                    ).add_to(m)
                    
                    print(f"✅ Polygon loaded for {source_name}")
                
                except Exception as e:
                    print(f"⚠️ Error loading polygon for {source_name}: {e}")
    
    except Exception as e:
        print(f"⚠️ Error adding country polygons: {e}")

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
                 #type="primary" if has_selection else "secondary",
                 use_container_width=False):
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
            
            # Check which data source covers this point
            try:
                from shapely.geometry import Point, Polygon
                data_source = get_data_source_for_point(lat, lon, config)
                if data_source:
                    st.session_state.selected_data_source = data_source["name"]
                else:
                    st.session_state.selected_data_source = None
            except ImportError:
                pass
            
            # Force rerun to immediately show the marker
            st.rerun()
    

