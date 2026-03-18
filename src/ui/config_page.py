"""
Configuration page component for meteoZar
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import pandas as pd
from ui.styles.config_styles import apply_config_styles
from data_sources.aemet_source import AemetWeatherSource
from generators.synthetic_generator import SyntheticWeatherGenerator

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = Path(__file__).parent.parent.parent
    
    return Path(base_path) / relative_path

def create_template_example_dataframe():
    """Create an example DataFrame for the template"""
    return pd.DataFrame({
        'Year': [2026, 2026, 2026, 2026, 2026, 2026, 2026, 2026],
        'Month': [1, 1, 1, 1, 2, 2, 2, 2],
        'Variable': ['temperature_max', 'temperature_mean', 'temperature_min', 'precipitation',
                     'temperature_max', 'temperature_mean', 'temperature_min', 'precipitation'],
        'Minimum': [15.0, 10.0, 5.0, 20.0, 16.0, 11.0, 6.0, 25.0],
        'Mean': [18.0, 13.0, 8.0, 35.0, 19.0, 14.0, 9.0, 40.0],
        'Maximum': [22.0, 16.0, 11.0, 50.0, 23.0, 17.0, 12.0, 55.0]
    })

def create_variables_dataframe():
    """Create a DataFrame with variables information"""
    return pd.DataFrame({
        'Variable': ['temperature_max', 'temperature_min', 'temperature_mean', 'precipitation'],
        'Description': [
            'Maximum daily temperature of the month',
            'Minimum daily temperature of the month',
            'Mean daily temperature of the month',
            'Mean daily precipitation of the month'
        ],
        'Unit': ['°C', '°C', '°C', 'mm/day'],
        'Type': ['temperature', 'temperature', 'temperature', 'precipitation']
    })

@st.dialog("Template Information", width="large")
def show_template_modal_dialog():
    """Modal dialog with two tabs: data example and variable descriptions"""
    # Create two tabs
    tab1, tab2 = st.tabs(["📋 Data Example", "📊 Variables Description"])
    
    with tab1:
        st.markdown("#### Example Data Format")
        st.markdown("Your predictions file should follow this structure:")
        example_df = create_template_example_dataframe()
        st.dataframe(example_df, width='stretch', hide_index=True)
        
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
        variables_df = create_variables_dataframe()
        st.dataframe(variables_df, width='stretch', hide_index=True)
        
        st.warning(
            "⚠️ **Important**: To modify temperature data, you must provide predictions for "
            "**all three temperature variables** (`temperature_min`, `temperature_mean`, and "
            "`temperature_max`) for each month and year. If any temperature variable is missing, "
            "the temperature adjustment will be skipped for that period."
        )


def get_data_source_instance(source_name: str, config: dict):
    """
    Get a data source instance based on name.
    
    Args:
        source_name: Name of the data source (e.g. 'AEMET')
        config: Configuration dictionary
        
    Returns:
        Data source instance, or None if not found
    """
    # Find the data source configuration
    source_config = next(
        (s for s in config.get('data_sources', []) if s['name'] == source_name),
        None
    )
    
    if not source_config:
        return None
    
    # Instantiate based on name
    if source_name == 'AEMET':
        return AemetWeatherSource(source_config)
    
    # Add more sources here in the future
    # elif source_name == 'OTHER_SOURCE':
    #     return OtherWeatherSource(source_config)
    
    return None

def get_mandatory_weather_data(latitude: float, longitude: float, start_year: int, end_year: int, config: dict):
    """
    Get the nearest station and required meteorological data.
    
    Args:
        latitude: Latitude of the point
        longitude: Longitude of the point
        start_year: Start year
        end_year: End year
        config: Configuration dictionary
        
    Returns:
        Tuple (WeatherStation, WeatherData) or (None, None) if there's an error
        
    Raises:
        ValueError: If no data source is available
    """
    selected_data_source = st.session_state.get('selected_data_source')
    
    if selected_data_source is None:
        raise ValueError("No data source available for this location")
    
    # Get data source instance
    data_source = get_data_source_instance(selected_data_source, config)
    
    if data_source is None:
        raise ValueError(f"Data source '{selected_data_source}' not configured")
    
    # Get data using the modular method
    try:
        nearest_station, weather_data = data_source.get_mandatory_data(
            latitude=latitude,
            longitude=longitude,
            start_year=start_year,
            end_year=end_year
        )
        return nearest_station, weather_data
    except Exception as e:
        print(f"Error getting data: {e}")
        raise

def _validate_predictions(df: pd.DataFrame):
    """
    Validate that the predictions DataFrame has the correct format.
    
    Raises:
        ValueError: If the format is invalid
    """
    required_columns = {'Year', 'Month', 'Variable', 'Minimum', 'Mean', 'Maximum'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    valid_variables = {'precipitation', 'temperature_max', 'temperature_mean', 'temperature_min','number_days_rain'}
    invalid = set(df['Variable'].unique()) - valid_variables
    if invalid:
        raise ValueError(f"Invalid variables: {invalid}. Valid: {valid_variables}")
    
    if not all(df['Month'].between(1, 12)):
        raise ValueError("Month values must be between 1 and 12")
    
    if df['Year'].isna().any():
        raise ValueError("Year column cannot contain empty values")
    
    # Check for missing values in Minimum, Mean, Maximum columns
    value_columns = ['Minimum', 'Mean', 'Maximum']
    for col in value_columns:
        if df[col].isna().any():
            missing_rows = df[df[col].isna()]
            raise ValueError(
                f"Missing values in '{col}' column. "
                f"Found empty cells in rows with Year={missing_rows['Year'].iloc[0]}, "
                f"Month={missing_rows['Month'].iloc[0]}, Variable={missing_rows['Variable'].iloc[0]}"
            )
    
    # Check that Minimum <= Mean <= Maximum for each row
    invalid_rows = df[(df['Minimum'] > df['Mean']) | (df['Mean'] > df['Maximum'])]
    if not invalid_rows.empty:
        first_invalid = invalid_rows.iloc[0]
        raise ValueError(
            f"Invalid value ranges for Year={first_invalid['Year']}, Month={first_invalid['Month']}, "
            f"Variable={first_invalid['Variable']}: "
            f"Minimum ({first_invalid['Minimum']}) must be <= Mean ({first_invalid['Mean']}) "
            f"must be <= Maximum ({first_invalid['Maximum']})"
        )


def render_config_page(config):
    """
    Render the configuration page
    
    Args:
        config (dict): Configuration dictionary
    """
    
    # Apply config page styles
    apply_config_styles()
    
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
        "Upload an Excel file (.xlsx) with monthly predictions. "
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
            type=['xlsx'],
            help="Upload the filled Excel template with monthly predictions"
        )
    
    if uploaded_file is not None:
        try:
            predictions_df = pd.read_excel(uploaded_file, engine='openpyxl')
            _validate_predictions(predictions_df)
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
        _METHOD_OPTIONS = {
            "K-Nearest Neighbors": "knn",
            "Machine Learning (XGBoost)": "xgboost",
        }

        _METHOD_DESCRIPTIONS = {
            "K-Nearest Neighbors":
                "K-Nearest Neighbors (KNN) — Finds the K most similar historical days (by temperature and precipitation) and copies/averages their wind, humidity and pressure values. ✔️ Fast.",
            "Machine Learning (XGBoost)":
                "Machine Learning — XGBoost Sliding Window — Trains a gradient-boosting model on rolling windows of 5 consecutive days. ✔️ Captures inter-day dependencies. ❌ Requires ≥5 days.",
        }
        
        selected_method_label = st.radio(
            "Select method:",
            options=list(_METHOD_OPTIONS.keys()),
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
                
                # Load config
                config_path = get_resource_path("config/config.json")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Get data using modular function
                nearest_station, weather_data = get_mandatory_weather_data(
                    latitude=latitude,
                    longitude=longitude,
                    start_year=start_year,
                    end_year=end_year,
                    config=config
                )
                
                # Save selected data source to session for later use
                selected_source = st.session_state.get('selected_data_source')
                
                # Extract the last real date from obtained historical data
                if weather_data and weather_data.daily_records:
                    last_record = weather_data.daily_records[-1]
                    last_historical_date = datetime.strptime(last_record.date, '%Y-%m-%d').date()
                    actual_historical_end = last_historical_date.isoformat()
                else:
                    actual_historical_end = f'{end_year}-12-31'
                    last_historical_date = datetime(end_year, 12, 31).date()
                
                # Adjust generation start date if it's the current year
                actual_generation_start = f'{gen_start_year}-01-01'
                if gen_start_year == datetime.now().year and gen_start_year == end_year:
                    # If we're generating in the current year and historical data is also from current year,
                    # start the day after the last historical record
                    next_day = last_historical_date + timedelta(days=1)
                    actual_generation_start = next_day.isoformat()
                    print(f"📅 Adjusting generation start to: {actual_generation_start}")
                
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
                generator = SyntheticWeatherGenerator(
                    source=selected_source,
                    id_station=nearest_station.id_station,
                    historical_start=f'{start_year}-01-01',
                    historical_end=actual_historical_end,
                    generation_start=actual_generation_start,
                    generation_end=f'{gen_end_year}-12-31'
                )
                
                _METHOD_OPTIONS_MAP = {
                    "K-Nearest Neighbors": "knn",
                    "Machine Learning (XGBoost)": "xgboost",
                }
                selected_label = st.session_state.get(
                    'correction_method_radio', 'K-Nearest Neighbors'
                )
                correction_method = _METHOD_OPTIONS_MAP.get(selected_label, 'knn')

                job_id, rows_inserted, generated_data = generator.generate_and_save(
                    latitude,
                    longitude,
                    predictions_df,
                    correction_method
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
                print(f"Error: {error_message}")
                modal_placeholder.empty()
                st.error(f"❌ {error_message}")
                st.session_state.generating = False
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                error_message = f"Unexpected error: {str(e)}"
                print(f"Error getting mandatory data: {e}")
                print(error_msg)
                modal_placeholder.empty()
                st.error(f"❌ {error_message}")
                st.session_state.generating = False

