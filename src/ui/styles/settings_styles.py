"""
Styles specific to the settings page
"""

import streamlit as st


def apply_settings_styles():
    """Apply settings page specific styles"""
    st.markdown("""
    <style>
    .stApp {
        background-color: white !important;
    }

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

    .st-emotion-cache-tn0cau {
        gap: 0 !important;
    }

    .settings-wrapper {
        margin-left: 20px !important;
        margin-right: 20px !important;
        position: relative !important;
        z-index: 1 !important;
    }

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6,
    .stMarkdown li,
    .stMarkdown span,
    .stMarkdown strong,
    .stMarkdown em,
    .stMarkdown code,
    .settings-wrapper label,
    .stTextInput label {
        color: black !important;
    }

    .settings-wrapper .stTextInput label,
    .settings-wrapper [data-testid="stTextInput"] label,
    .settings-wrapper [data-testid="stTextInput"] [data-testid="stWidgetLabel"],
    .settings-wrapper [data-testid="stTextInput"] [data-testid="stWidgetLabel"] p,
    .settings-wrapper [data-testid="stWidgetLabel"],
    .settings-wrapper [data-testid="stWidgetLabel"] p {
        color: black !important;
    }

    .stForm {
        border: 1px solid #e6e6e6 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }

    .st-key-settings_api_keys_form button[kind="primary"] {
        background-color: #0088FF !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 8px 20px !important;
    }

    .st-key-settings_api_keys_form button[kind="primary"]:hover {
        opacity: 0.9 !important;
    }

    .stLinkButton button {
        border-radius: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)