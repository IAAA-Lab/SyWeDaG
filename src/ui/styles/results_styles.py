"""
Styles specific to the results page
"""

import streamlit as st

def apply_results_styles():
    """Apply results page specific styles"""
    st.markdown("""
    <style>
    /* White background */
    .stApp {
        background-color: white !important;
    }
    
    /* Text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: black !important;
    }
    
    /* Main container position*/
    .stMain {
        position: fixed !important;
        top: 11% !important;
        padding-top: 0 !important;
        padding-left: 1% !important;
        padding-right: 1% !important;
        padding-bottom: 6% !important;
        overflow-y: auto !important;
    }
    
    .main .block-container {
        padding-top: 20px !important;
        margin-top: 0 !important;
        max-width: 1100px !important;
    }
                
    /* Remove gap from emotion cache */
    .st-emotion-cache-tn0cau {
        gap: 0 !important;
    }
    
    /* ───── Info cards ───── */
    .info-card {
        background: #f8f9fa;
        border: 1px solid #e0e4e8;
        border-radius: 12px;
        padding: 18px 22px 14px 22px;
        margin-bottom: 6px;
        margin-left: 12px !important;
        margin-right: 12px !important;
    }
    
    .info-card-header {
        font-size: 17px;
        font-weight: 700;
        color: #0088FF;
        margin-bottom: 10px;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid #eee;
    }
    
    .info-row:last-child {
        border-bottom: none;
    }
    
    .info-label {
        font-weight: 600;
        color: #333;
        font-size: 14px;
    }
    
    .info-value {
        color: #555;
        font-size: 14px;
        text-align: right;
    }
    
    /* ───── Date inputs & selectbox ───── */
    .stDateInput, .stSelectbox {
        background-color: #f8f9fa !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    
    .stDateInput label, .stSelectbox label {
        color: black !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    .stDateInput input, .stSelectbox [data-baseweb="select"] {
        color: black !important;
        background-color: white !important;
    }
    
    /* ───── Spinner ───── */
    .stSpinner > div {
        color: #0088FF !important;
    }
    
    /* Info messages styling */
    .stAlert {
        color: black !important;
    }
    
    /* Warning text color - make it more visible */
    .st-emotion-cache-2fgyt4.eg78z5t0 p {
        color: #856404 !important;
        font-weight: 600 !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda !important;
        border-color: #c3e6cb !important;
        color: #155724 !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        border-color: #bee5eb !important;
        color: #0c5460 !important;
    }
    
    /* ───── Spacers ───── */
    .spacer-lg {
        height: 30px !important;
    }
    
    .spacer-sm {
        height: 10px !important;
    }
    
    /* ───── Chart info label ───── */
    .chart-info {
        color: #555 !important;
        font-size: 13px !important;
        margin-bottom: 2px !important;
    }
    
    /* ───── Modifications card scroll ───── */
    .info-card-scroll {
        max-height: 150px;
        overflow-y: auto;
        padding-right: 6px;
    }
    
    .info-card-scroll::-webkit-scrollbar {
        width: 6px;
    }
    
    .info-card-scroll::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .info-card-scroll::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 3px;
    }
    
    .info-card-scroll::-webkit-scrollbar-thumb:hover {
        background: #999;
    }
    
    .modification-item {
        font-size: 13px;
        color: #555;
        padding: 6px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .modification-item:last-child {
        border-bottom: none;
    }
    </style>
    """, unsafe_allow_html=True)
