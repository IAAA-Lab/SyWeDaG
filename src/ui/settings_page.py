"""
Settings page component for MeteoSynthetic
Allows configuring API keys for data sources
"""

from pathlib import Path
import json
import streamlit as st

from ui.styles.settings_styles import apply_settings_styles
from utils.system_utils import get_resource_path


def _get_config_file_path() -> Path:
    """Resolve config file path for save operations."""
    return get_resource_path("config/config.json")


def _save_config(updated_config: dict) -> tuple[bool, str]:
    """Save config dictionary to config JSON file."""
    config_path = _get_config_file_path()

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(updated_config, config_file, indent=2, ensure_ascii=False)
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
        if "api_key" in source or "key_url" in source
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

            current_api_key = source.get("api_key", "")
            api_key_input = st.text_input(
                f"{source_name} API Key",
                value=current_api_key,
                type="password",
                key=f"settings_api_key_{source_name}",
                help="Enter the API key used to access this data source."
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
        updated_config = dict(config)
        updated_data_sources = []

        for source in data_sources:
            source_copy = dict(source)
            source_name = source_copy.get("name")

            if source_name in updated_api_keys:
                source_copy["api_key"] = updated_api_keys[source_name]

            updated_data_sources.append(source_copy)

        updated_config["data_sources"] = updated_data_sources

        saved_ok, message = _save_config(updated_config)
        if saved_ok:
            st.session_state.config = updated_config
            st.success(message)
        else:
            st.error(message)

    st.markdown('</div>', unsafe_allow_html=True)