"""
Styles specific to the configuration page
"""

import streamlit as st

def apply_config_styles():
    """Apply config page specific styles"""
    st.markdown("""
    <style>
    /* White background */
    .stApp {
        background-color: white !important;
    }
    
    /* Text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stNumberInput label {
        color: black !important;
    }
    
    /* Fix main container position */
    .stMain {
        position: fixed !important;
        top: 10% !important;
        padding-top: 0 !important;
        padding-left: 1% !important;
        padding-right: 1% !important;
        width: 100% !important;
        height: 90% !important;
        overflow-y: auto !important;
    }
    
    .main .block-container {
        padding-top: 8px !important;
        margin-top: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        max-width: none !important;
    }
    
    /* Content wrapper with margins */
    .config-wrapper {
        margin-left: 20px !important;
        margin-right: 20px !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    .config-wrapper > * {
        width: auto !important;
    }
    
    /* Reduce number input width */
    .stNumberInput {
        width: 150px !important;
    }
    
    .stNumberInput > div {
        width: 150px !important;
    }
    
    .stNumberInput input {
        width: 150px !important;
    }
    
    /* GENERATE button styling */
    .st-key-generate_button {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 5px auto !important;
    }
    
    .st-key-generate_button button {
        background-color: #0088FF !important;
        color: white !important;
        border: none !important;
        padding: 8px 20px !important;
        border-radius: 20px !important;
        min-width: 150px !important;
    }
    
    .st-key-generate_button button:hover {
        opacity: 0.9 !important;
    }
    
    /* Modal overlay */
    .modal-overlay {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background-color: rgba(0, 0, 0, 0.7) !important;
        z-index: 9999 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    /* Remove gap from emotion cache */
    .st-emotion-cache-tn0cau {
        gap: 0 !important;
    }
    
    /* Center vertical block content */
    .st-emotion-cache-wfksaw {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .modal-content {
        background-color: white !important;
        border-radius: 15px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important;
        width: 500px !important;
        max-width: 90vw !important;
        overflow: hidden !important;
        padding-bottom: 20px !important;
    }
    
    .modal-header {
        background-color: #0088FF !important;
        color: white !important;
        padding: 20px !important;
        text-align: center !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    .modal-body {
        padding: 30px 40px !important;
        text-align: center !important;
        color: black !important;
    }
    
    .modal-body p {
        font-size: 16px !important;
        margin: 5px 0 !important;
        color: black !important;
    }
    
    .spinner {
        border: 4px solid #f3f3f3 !important;
        border-top: 4px solid #0088FF !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        animation: spin 1s linear infinite !important;
        margin: 10px auto 20px auto !important;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Template Modal Dialog Styles */
    [data-testid="stDialog"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stDialog"] .stMarkdown p {
        color: white !important;
    }
    
    [data-testid="stDialog"] h4 {
        color: white !important;
    }
    
    [data-testid="stDialog"] h3 {
        color: white !important;
    }
    
    [data-testid="stDialog"] .stTabs [role="tab"] {
        color: white !important;
    }
    
    [data-testid="stDialog"] .stDataFrame {
        background-color: #1f1f1f !important;
        color: white !important;
    }
    
    [data-testid="stDialog"] .stDataFrame th {
        color: white !important;
        background-color: #2f2f2f !important;
    }
    
    [data-testid="stDialog"] .stDataFrame td {
        color: white !important;
    }
    
    /* File Uploader Text Styling */
    .stFileUploader {
        max-width: 80% !important;
    }
    
    .stFileUploader label {
        color: black !important;
    }
    
    /* Reduce markdown separator (---) space */
    .stMarkdown hr {
        margin: 5px 0 !important;
    }
    
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    
    /* View Template Button Styling */
    .st-key-view_template_button {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 50% !important;
        min-height: 100px !important;
    }
    
    .st-key-view_template_button button {
        transition: all 0.2s ease !important;
    }
    
    .st-key-view_template_button button:hover {
        opacity: 0.85 !important;
        background-color: #005799 !important;
    }
    
    /* Reduce spacing in info boxes */
    .stInfo {
        padding: 8px 12px !important;
        margin: 5px 0 !important;
    }
    
    /* Reduce spacing in radio buttons */
    .stRadio {
        margin: 5px 0 !important;
    }
    
    /* Reduce column spacing */
    .stColumns [data-testid="column"] {
        gap: 0 !important;
    }
    
    /* Radio button text color */
    .st-key-correction_method_radio label {
        color: black !important;
    }
    
    .st-key-correction_method_radio [role="radio"] {
        color: black !important;
    }
    
    .st-key-correction_method_radio span {
        color: black !important;
    }
    
    .st-key-correction_method_radio div {
        color: black !important;
    }
    
    /* Error Modal Styles */
    .error-modal-container {
        background-color: #e78888 !important;
        border: 1px solid #ef4444 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 16px !important;
        color: #991b1b !important;
    }
    
    .error-modal-container h4 {
        color: #b91c1c !important;
        margin-top: 0 !important;
        margin-bottom: 10px !important;
    }
    
    [data-testid="stDialog"] .error-modal-container p {
        color: #ffffff !important;
        margin-bottom: 0 !important;
    }
                
    </style>
    """, unsafe_allow_html=True)
