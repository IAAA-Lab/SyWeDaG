"""
Settings page component for SyWeDaG
Allows configuring API keys for data sources
"""

import json
import os
import streamlit as st
from dotenv import load_dotenv, set_key

from ui.styles.settings_styles import apply_settings_styles
from utils.system_utils import get_resource_path
from database.sqliteDB import clear_all_data


def _save_api_keys_to_env(config: dict, api_keys: dict) -> tuple[bool, str]:
    """Save API keys to .env file based on config specifications."""
    env_path = get_resource_path(".env")
    
    try:
        # Create .env if it doesn't exist
        if not env_path.exists():
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.touch()
        
        # Get environment variable names from config
        for source in config.get("data_sources", []):
            source_name = source.get("name")
            
            # Only save if config specifies an env var and user provided a key
            if "api_key_env_var" in source and source_name in api_keys:
                env_var_name = source.get("api_key_env_var")
                # Use set_key to update or add the key
                set_key(str(env_path), env_var_name, api_keys[source_name])
        
        # Reload environment variables
        load_dotenv(dotenv_path=str(env_path), override=True)
        
        return True, "Settings saved successfully"
    except Exception as error:
        return False, f"Error saving settings: {error}"


def render_settings_page(config: dict):
    """
    Render settings page to manage data source API keys.

    Args:
        config (dict): Current configuration dictionary
    """
    apply_settings_styles()

    st.markdown('<div class="settings-wrapper">', unsafe_allow_html=True)
    st.markdown("### Settings")
    st.markdown("Configure API keys for available data sources.")

    data_sources = config.get("data_sources", [])
    sources_with_keys = [
        source for source in data_sources
        if "api_key_env_var" in source or "key_url" in source
    ]

    if not sources_with_keys:
        st.info("No data sources with API key configuration were found.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    with st.form("settings_api_keys_form"):
        updated_api_keys = {}

        for source in sources_with_keys:
            source_name = source.get("name", "Unknown Source")
            source_country = source.get("country", "")
            source_label = source_name if not source_country else f"{source_name} ({source_country})"

            #st.markdown("---")
            st.markdown(f"#### {source_label}")

            # Get current API key from environment variable
            env_var_name = source.get("api_key_env_var")
            if env_var_name:
                current_api_key = os.getenv(env_var_name, "")
                
                api_key_input = st.text_input(
                    f"{source_name} API Key",
                    value=current_api_key,
                    type="password",
                    key=f"settings_api_key_{source_name}",
                    help="Enter the API key used to access this data source. Stored securely in .env"
                )
                updated_api_keys[source_name] = api_key_input.strip()

            key_url = source.get("key_url")
            if key_url:
                st.link_button(
                    f"Get {source_name} API Key",
                    key_url,
                    use_container_width=False,
                )

        save_clicked = st.form_submit_button("SAVE SETTINGS")

    if save_clicked:
        # Save API keys to .env instead of config.json
        saved_ok, message = _save_api_keys_to_env(config, updated_api_keys)
        if saved_ok:
            st.success(message)
        else:
            st.error(message)

    st.markdown("### Database Maintenance")

    if st.button(
        "CLEAR DATABASE",
        key="settings_clear_db_button",
        type="secondary",
        help="Deletes all rows in all database tables.",
    ):
        try:
            clear_all_data(reset_sequences=True)

            # Clear session data that may reference removed records.
            st.session_state.pop("job_id", None)
            st.session_state.pop("nearest_station", None)
            st.session_state.pop("records_count", None)

            st.success(f"Database cleaned successfully.")
        except Exception as error:
            st.error(f"Error cleaning database: {error}")

    st.markdown('</div>', unsafe_allow_html=True)
    