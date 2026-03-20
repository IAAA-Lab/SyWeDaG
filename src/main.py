"""
meteoZar - Synthetic Weather Scenario Generator
Main entry point for the Streamlit application
"""

import streamlit as st
import json
from utils.system_utils import get_resource_path

# Page configuration
st.set_page_config(
    page_title="meteoZar",
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
from ui.styles.base_styles import apply_base_styles

# Import database initialization
from database.sqliteDB import init_database

def main():
    """Main application function"""
    
    # Initialize database if it doesn't exist
    db_path = get_resource_path("data/weather.db")
    if not db_path.exists():
        init_database()
    
    # Apply base styles
    apply_base_styles()
    
    # Initialize session state
    if 'selected_point' not in st.session_state:
        st.session_state.selected_point = None
    if 'selected_data_source' not in st.session_state:
        st.session_state.selected_data_source = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'map'
    
    # Menu button in header (visible on all pages)
    if st.button("☰", key="header_menu_button"):
        pass  # TODO: Implement menu functionality
    
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
            <h1 class="header-title">meteoZar</h1>
        </div>
        ''', unsafe_allow_html=True)
    
    # Render the appropriate page
    if st.session_state.current_page == 'map':
        render_map(config)
    elif st.session_state.current_page == 'config':
        render_config_page(config)
    elif st.session_state.current_page == 'results':
        render_results_page()

def get_base64_image(image_path):
    """Convert image to base64 for embedding"""
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

if __name__ == "__main__":
    main()
