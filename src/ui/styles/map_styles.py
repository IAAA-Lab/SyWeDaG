"""
Styles specific to the map page
"""

import streamlit as st

def apply_map_styles():
    """Apply map-specific styles"""
    st.markdown("""
    <style>
    /* Search input styling */
    .stTextInput {
        display: block !important;
        background: transparent !important;
        position: fixed !important;
        top: calc(10vh + 1.5vh) !important;
        left: 1.5vw !important;
        z-index: 1000 !important;
        width: 20vw !important;
        min-width: 180px !important;
        max-width: 300px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stTextInput > div {
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }
    
    .stTextInput > div > div {
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }
    
    .stTextInput > div > div > input {
        background-color: white !important;
        border-radius: 25px !important;
        border: 2px solid #0088FF !important;
        padding: 0.8vh 1.5vw !important;
        font-size: clamp(12px, 1.2vw, 14px) !important;
        color: #333 !important;
        width: 100% !important;
        margin: 0 !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999 !important;
        opacity: 1 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid #0088FF !important;
        box-shadow: 0 2px 12px rgba(0, 136, 255, 0.3) !important;
        outline: none !important;
    }
    
    .stTextInput > label {
        display: none !important;
    }
    
    /* SELECT button styling */
    .st-key-select_button {
        position: fixed !important;
        bottom: 20px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        z-index: 1000 !important;
        width: auto !important;
    }
    
    .st-key-select_button button {
        z-index: 1000 !important;
        background-color: #0088FF !important;
        color: white !important;
        border: none !important;
        padding: 10px 30px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 25px !important;
        min-width: 150px !important;
        max-width: 200px !important;        
        text-align: center !important;
        display: block !important;
        margin: 0 auto !important;    
    }
    
    .st-key-select_button button:hover {
        background-color: #0088FF !important;
        color: white !important;
        opacity: 0.9 !important;
    }
    
    .st-key-select_button button:disabled {
        background-color: #999999 !important;
        color: white !important;
        opacity: 1 !important;
        cursor: not-allowed !important;
    }
    
    .st-key-select_button button:disabled:hover {
        background-color: #999999 !important;
        color: white !important;
        opacity: 1 !important;
    }
    
    /* Map container - full screen */
    .element-container iframe {
        border: none !important;
        position: fixed !important;
        top: 10vh !important;
        left: 0 !important;
        height: 90vh !important;
        width: 100vw !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 10 !important;
    }
    
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
