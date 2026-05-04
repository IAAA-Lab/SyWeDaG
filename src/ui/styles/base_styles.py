"""
Base styles shared across all pages
"""

import streamlit as st

def apply_base_styles():
    """Apply base styles (header, common elements)"""
    st.markdown("""
    <style>
    /* Base page layout: allow vertical scrolling */
    html, body {
        margin: 0 !important;
        padding: 0 !important;
        overflow: auto !important;
        height: 100vh !important;
        width: 100vw !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    [data-testid="stHeader"] {display: none !important;}
    
    /* Main container */
    .main {
        background-color: transparent;
        padding: 0 !important;
        margin: 0 !important;
        overflow: auto !important;
        height: 100vh !important;
    }
    
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        overflow: auto !important;
    }
    
    .stApp {
        margin: 0 !important;
        padding: 0 !important;
        overflow: auto !important;
    }
    
    [data-testid="stAppViewContainer"] {
        padding: 0 !important;
        overflow: auto !important;
    }
    
    /* Blue Header */
    .blue-header {
        background-color: #0088FF !important;
        padding: 0 !important;
        text-align: center !important;
        position: fixed !important;
        z-index: 111 !important;
        height: 10% !important;
        width: 100% !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .blue-header .logo {
        height: 200%;
        max-width: 100%;
        object-fit: contain;
    }
    
    .header-title {
        color: white;
        font-size: 48px;
        font-weight: bold;
        margin: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Home button in header */
    .st-key-header_home_button {
        position: fixed !important;
        top: 5% !important;
        left: 15px !important;
        transform: translateY(-50%) !important;
        z-index: 112 !important;
        width: auto !important;
        height: auto !important;
    }
    
    .st-key-header_home_button button {
        background-color: white !important;
        color: #0088FF !important;
        border: none !important;
        padding: 0 !important;
        font-size: 3.5vh !important;
        font-weight: bold !important;
        border-radius: 50% !important;
        width: 6vh !important;
        height: 6vh !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .st-key-header_home_button button p {
        font-size: 3vh !important;
    }
    
    .st-key-header_home_button button:hover {
        background-color: #f0f0f0 !important;
        opacity: 0.95 !important;
    }

    /* Import button in header (map page only) */
    .st-key-header_import_button {
        position: fixed !important;
        top: 5% !important;
        left: 15px !important;
        transform: translateY(-50%) !important;
        z-index: 112 !important;
        width: auto !important;
        height: auto !important;
    }

    .st-key-header_import_button button {
        background-color: white !important;
        color: #0088FF !important;
        border: none !important;
        padding: 0 !important;
        font-size: 3.5vh !important;
        font-weight: bold !important;
        border-radius: 50% !important;
        width: 6vh !important;
        height: 6vh !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .st-key-header_import_button button p {
        font-size: 3vh !important;
    }

    .st-key-header_import_button button:hover {
        background-color: #f0f0f0 !important;
        opacity: 0.95 !important;
    }
    
    /* Menu button in header */
    .st-key-header_menu_button {
        position: fixed !important;
        top: 5% !important;
        right: 15px !important;
        transform: translateY(-50%) !important;
        z-index: 112 !important;
        width: auto !important;
        height: auto !important;
    }
    
    .st-key-header_menu_button button {
        background-color: white !important;
        color: #0088FF !important;
        border: none !important;
        padding: 0 !important;
        font-size: 3.5vh !important;
        font-weight: bold !important;
        border-radius: 50% !important;
        width: 6vh !important;
        height: 6vh !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    .st-key-header_menu_button button p {
        font-size: 3vh !important;
    }
    
    .st-key-header_menu_button button:hover {
        background-color: #f0f0f0 !important;
        opacity: 0.95 !important;
    }
    </style>
    """, unsafe_allow_html=True)
