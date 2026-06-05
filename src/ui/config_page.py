"""
Configuration page component for MeteoSynthetic
"""

import streamlit as st
from datetime import datetime
import pandas as pd
from ui.styles.config_styles import apply_config_styles
from utils.system_utils import save_bytes_to_downloads, safe_print
from application.config_services import (
    validate_predictions,
    get_mandatory_weather_data,
    compute_generation_dates,
    generate_synthetic_data,
    METHOD_OPTIONS_MAP,
)

# Build template example DataFrame and variables DataFrame
def _build_template_example_dataframe():
    """Build the example DataFrame used in the template."""
    return pd.DataFrame({
        'Year': [2026, 2026, 2026, 2026, 2026, 2026, 2026, 2026, 2026, 2026],
        'Month': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        'Variable': ['temperature_max', 'temperature_mean', 'temperature_min', 'precipitation', 'number_days_rain',
                     'temperature_max', 'temperature_mean', 'temperature_min', 'precipitation', 'number_days_rain'],
        'Minimum': [15.0, 10.0, 5.0, 20.0, 3.0, 16.0, 11.0, 6.0, 25.0, 4.0],
        'Mean': [18.0, 13.0, 8.0, 35.0, 6.0, 19.0, 14.0, 9.0, 40.0, 7.0],
        'Maximum': [22.0, 16.0, 11.0, 50.0, 10.0, 23.0, 17.0, 12.0, 55.0, 11.0]
    })

def _create_template_example_csv_bytes():
    """Create a .csv template with example predictions and return bytes."""
    csv_content = _build_template_example_dataframe().to_csv(index=False)
    return csv_content.encode('utf-8')

def _build_variables_dataframe():
    """Build a DataFrame with variables information."""
    return pd.DataFrame({
        'Variable': ['temperature_max', 'temperature_min', 'temperature_mean', 'precipitation', 'number_days_rain'],
        'Description': [
            'Maximum daily temperature of the month',
            'Minimum daily temperature of the month',
            'Mean daily temperature of the month',
            'Mean daily precipitation of the month',
            'Number of rainy days in the month'
        ],
        'Unit': ['°C', '°C', '°C', 'mm/day', 'days'],
        'Type': ['temperature', 'temperature', 'temperature', 'precipitation', 'precipitation']
    })

@st.dialog("Template Information", width="large")
def show_template_modal_dialog():
    """Modal dialog with two tabs: data example and variable descriptions"""
    # Create two tabs
    tab1, tab2 = st.tabs(["📋 Data Example", "📊 Variables Description"])
    
    with tab1:
        st.markdown("#### Example Data Format")
        st.markdown("Your predictions file should follow this structure:")
        example_df = _build_template_example_dataframe()
        st.table(example_df)

        template_bytes = _create_template_example_csv_bytes()
        template_filename = "monthly_predictions_template.csv"

        if st.button("Save template to Downloads", key="save_template_to_downloads"):
            try:
                saved_path = save_bytes_to_downloads(template_filename, template_bytes)
                st.success(f"Saved: {saved_path}")
            except Exception as error:
                st.error(f"Error saving template file: {error}")
        
        st.markdown("""
        - **Year**: The year for the prediction
        - **Month**: Month number (1-12)
        - **Variable**: One of the four meteorological variables
        - **Minimum**: Minimum expected value for the month
        - **Mean**: Mean expected value for the month
        - **Maximum**: Maximum expected value for the month
        """)
    
    with tab2:
        st.markdown("#### Meteorological Variables")
        st.markdown("Description of all variables used in predictions:")
        variables_df = _build_variables_dataframe()
        st.table(variables_df)
        
        st.warning(
            "⚠️ **Important**: To modify temperature data, you must provide predictions for "
            "**all three temperature variables** (`temperature_min`, `temperature_mean`, and "
            "`temperature_max`) for each month and year. If any temperature variable is missing, "
            "the temperature adjustment will be skipped for that period."
        )

@st.dialog("Data Source Error")
def show_error_modal(error_message):
    """Show modal dialog when an error occurs during data generation."""
    st.markdown(f"""
    <div class="error-modal-container">
        <h4>❌ Error Occurred</h4>
        <p>{error_message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Close", key="close_error_modal", use_container_width=True):
        st.session_state.show_error_modal = False
        st.session_state.error_message = None
        st.rerun()


def render_config_page(config):
    """
    Render the configuration page
    
    Args:
        config (dict): Configuration dictionary
    """
    
    # Apply config page styles
    apply_config_styles()
    
    # Check if we need to show an error modal
    if st.session_state.get('show_error_modal', False):
        show_error_modal(st.session_state.get('error_message', 'An unknown error occurred.'))
    
    # Start wrapper div with margins
    st.markdown('<div class="config-wrapper">', unsafe_allow_html=True)
    
    # Initialize predictions_df variable (not stored in session)
    predictions_df = None
    
    # Split page into left and right halves
    col_left, col_right = st.columns(2)
    
    # Left column: Year selection
    with col_left:
        st.markdown("### From which range of years should we collect the data?")
        
        current_year = datetime.now().year
        
        # Create horizontal layout for year inputs
        year_col1, year_col2 = st.columns(2)
        
        with year_col1:
            start_year = st.number_input(
                "Start Year",
                min_value=1900,
                max_value=current_year,
                value=current_year - 5,
                step=1,
                key="data_start_year"
            )
        
        with year_col2:
            end_year = st.number_input(
                "End Year",
                min_value=1900,
                max_value=current_year,
                value=current_year,
                step=1,
                key="data_end_year"
            )
        
        if start_year > end_year:
            st.error("⚠️ Start year must be less than or equal to end year")
    
    # Right column: Generation year range
    with col_right:
        st.markdown("### For which range of years should we generate the data?")
        
        # Create horizontal layout for generation year inputs
        gen_col1, gen_col2 = st.columns(2)
        
        with gen_col1:
            gen_start_year = st.number_input(
                "Generation Start Year",
                min_value=current_year,
                max_value=current_year + 150,
                value=current_year,
                step=1,
                key="gen_start_year"
            )
        
        with gen_col2:
            gen_end_year = st.number_input(
                "Generation End Year",
                min_value=current_year,
                max_value=current_year + 150,
                value=current_year + 5,
                step=1,
                key="gen_end_year"
            )
        
        if gen_start_year > gen_end_year:
            st.error("⚠️ Generation start year must be less than or equal to end year")

    # ── Monthly Predictions (Excel Upload) ─────────────────────────────
    st.markdown("---")
    st.markdown("### Monthly Predictions (Optional)")
    st.markdown(
        "Upload a CSV file (.csv) with monthly predictions. "
        "The generated data will be adjusted to match these predictions."
    )
    
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        # Button to show template modal
        if st.button("View Template Information", key="view_template_button"):
            show_template_modal_dialog()
    
    with pred_col2:
        uploaded_file = st.file_uploader(
            "Upload filled predictions",
            type=['csv'],
            help="Upload the filled CSV template with monthly predictions"
        )
    
    if uploaded_file is not None:
        try:
            # sep=None lets pandas infer comma/semicolon delimiters.
            predictions_df = pd.read_csv(uploaded_file, sep=None, engine='python')
            validate_predictions(predictions_df)
            st.success(f"✅ Predictions loaded: {len(predictions_df)} entries")
        except ValueError as e:
            st.error(f"❌ Invalid file: {e}")
            predictions_df = None
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            predictions_df = None
    else:
        predictions_df = None

    # ── Secondary Variables Method ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### Secondary Variables Method")

    if predictions_df is not None:
        _METHOD_DESCRIPTIONS = {
            "K-Nearest Neighbors":
                "K-Nearest Neighbors (KNN) — Finds the K most similar historical days (by temperature and precipitation) and copies/averages their wind, humidity and pressure values. ✔️ Fast.",
            "Machine Learning (XGBoost)":
                "Machine Learning — XGBoost Sliding Window — Trains a gradient-boosting model on rolling windows of 5 consecutive days. ✔️ Captures inter-day dependencies. ❌ Requires ≥5 days.",
        }
        
        selected_method_label = st.radio(
            "Select method:",
            options=list(METHOD_OPTIONS_MAP.keys()),
            index=1,
            key="correction_method_radio",
            horizontal=True,
            help="Hover over options above for details"
        )
        
        # Show tooltip-like description with HTML
        st.markdown(f"""
        <div style="background-color: #e8f4f8; border-left: 4px solid #0088FF; padding: 10px 12px; margin: 5px 0; border-radius: 4px; font-size: 13px;">
            {_METHOD_DESCRIPTIONS[selected_method_label]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(
            "Upload a valid monthly predictions file above to enable this selector. "
            "Without predictions, all variables "
            "are taken directly from the historical record with no correction applied."
        )

    if st.button("GENERATE", key="generate_button"):
        st.session_state.generating = True
        st.rerun()
    
    # Close wrapper div
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show modal overlay when generating
    if st.session_state.get('generating', False):
        # Show the modal with loading status
        modal_placeholder = st.empty()
        
        with modal_placeholder.container():
            st.markdown("""
            <div class="modal-overlay">
                <div class="modal-content">
                    <div class="modal-header">
                        Data Generation
                    </div>
                    <div class="modal-body">
                        <div class="spinner"></div>
                        <p>Loading station information...</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Get station and weather information
        selected_point = st.session_state.get('selected_point')
        nearest_station = None
        
        weather_data = None
        error_message = None
        generated_data = None
        job_id = None
        
        if selected_point:
            try:
                # selected_point is a list [lat, lon]
                latitude = selected_point[0]
                longitude = selected_point[1]
                
                # Get data using modular function
                nearest_station, weather_data = get_mandatory_weather_data(
                    latitude=latitude,
                    longitude=longitude,
                    start_year=start_year,
                    end_year=end_year,
                    config=config,
                    selected_data_source=st.session_state.get('selected_data_source')
                )
                
                # Save selected data source to session for later use
                selected_source = st.session_state.get('selected_data_source')
                
                actual_historical_end, actual_generation_start = compute_generation_dates(
                    weather_data=weather_data,
                    end_year=end_year,
                    gen_start_year=gen_start_year,
                )
                
                # Update modal: Generating data
                with modal_placeholder.container():
                    st.markdown("""
                    <div class="modal-overlay">
                        <div class="modal-content">
                            <div class="modal-header">
                                Data Generation
                            </div>
                            <div class="modal-body">
                                <div class="spinner"></div>
                                <p>Generating synthetic data...</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate synthetic data
                selected_label = st.session_state.get(
                    'correction_method_radio', 'K-Nearest Neighbors'
                )

                job_id, rows_inserted, generated_data = generate_synthetic_data(
                    source=selected_source,
                    station_id=nearest_station.id_station,
                    start_year=start_year,
                    gen_end_year=gen_end_year,
                    actual_historical_end=actual_historical_end,
                    actual_generation_start=actual_generation_start,
                    latitude=latitude,
                    longitude=longitude,
                    predictions_df=predictions_df,
                    correction_method_label=selected_label,
                )
                
                # Clear modal
                modal_placeholder.empty()
                
                # Save only necessary information to session (job_id and station)
                st.session_state.job_id = job_id
                st.session_state.nearest_station = nearest_station
                st.session_state.records_count = rows_inserted
                
                # Switch to results page
                st.session_state.current_page = 'results'
                st.session_state.generating = False
                st.rerun()
                
            except ValueError as e:
                error_message = str(e)
                safe_print(f"Error: {error_message}")
                modal_placeholder.empty()
                st.session_state.generating = False
                st.session_state.show_error_modal = True
                st.session_state.error_message = error_message
                st.rerun()
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                error_message = f"Unexpected error: {str(e)}"
                safe_print(f"Error getting mandatory data: {e}")
                safe_print(error_msg)
                modal_placeholder.empty()
                st.session_state.generating = False
                st.session_state.show_error_modal = True
                st.session_state.error_message = error_message
                st.rerun()

