"""
Results page component for MeteoSynthetic
Displays the results of the synthetic weather data generation
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from ui.styles.results_styles import apply_results_styles
from database.sqliteDB import get_generated_hourly_data, get_generation_job_info, get_monthly_predictions

# Parameter display config
PARAMETER_OPTIONS = {
    'temperature': {'label': 'Temperature (°C)', 'color': '#FF6B35', 'unit': '°C'},
    'precipitation': {'label': 'Precipitation (mm)', 'color': '#0088FF', 'unit': 'mm'},
    'wind_speed': {'label': 'Wind Speed (km/h)', 'color': '#2ECC71', 'unit': 'km/h'},
    'humidity': {'label': 'Humidity (%)', 'color': '#9B59B6', 'unit': '%'},
    'pressure': {'label': 'Pressure (hPa)', 'color': '#E74C3C', 'unit': 'hPa'},
}

@st.cache_data(show_spinner=False)
def load_generated_data(job_id: int) -> pd.DataFrame:
    """Load generated hourly data from DB and return as DataFrame, cached."""
    data = get_generated_hourly_data(job_id)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'].str.replace('Z', '', regex=False))
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def render_results_page():
    """Render the results page"""
    
    apply_results_styles()
    
    # Get data from session
    job_id = st.session_state.get('job_id')
    nearest_station = st.session_state.get('nearest_station')
    records_count = st.session_state.get('records_count')
    
    if not job_id:
        st.warning("No generation data available.")
        return
    
    # Get job details from database
    job_info = get_generation_job_info(job_id)
    if not job_info:
        st.error("Job information not found in database.")
        return
    
    # Extract data
    latitude = job_info['latitude']
    longitude = job_info['longitude']
    hist_start = job_info['historicalStartDate']
    hist_end = job_info['historicalEndDate']
    gen_start = job_info['generatedStartDate']
    gen_end = job_info['generatedEndDate']
    data_source = nearest_station.source if nearest_station else 'N/A'
    
    # ── Job information in three columns ──────────────────────────────────
    station_name = nearest_station.name if nearest_station else 'N/A'
    station_region = nearest_station.region if nearest_station and nearest_station.region else ''
    station_display = f"{station_name}, {station_region}" if station_region else station_name
    
    def _fmt(date_str: str) -> str:
        """Format YYYY-MM-DD to DD/MM/YYYY"""
        try:
            return datetime.strptime(date_str[:10], '%Y-%m-%d').strftime('%d/%m/%Y')
        except Exception:
            return date_str
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-header">Location & Source</div>
            <div class="info-row"><span class="info-label">Coordinates</span> 
                <span class="info-value">Lat {latitude:.4f}, Lon {longitude:.4f}</span></div>
            <div class="info-row"><span class="info-label">Data source</span> 
                <span class="info-value">{data_source}</span></div>
            <div class="info-row"><span class="info-label">Nearest station</span> 
                <span class="info-value">{station_display}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-header">Generation Summary</div>
            <div class="info-row"><span class="info-label">Historical period</span> 
                <span class="info-value">{_fmt(hist_start)} – {_fmt(hist_end)}</span></div>
            <div class="info-row"><span class="info-label">Simulated period</span> 
                <span class="info-value">{_fmt(gen_start)} – {_fmt(gen_end)}</span></div>
            <div class="info-row"><span class="info-label">Records generated</span> 
                <span class="info-value">{records_count:,}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='spacer-lg'></div>", unsafe_allow_html=True)
    
    # ── Load data ───────────────────────────────────────────────────────
    with st.spinner("Loading generated data…"):
        df = load_generated_data(job_id)
    
    if df.empty:
        st.error("No generated data found for this job.")
        return
    
    # ── Chart controls ──────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    
    min_date = df['datetime'].min().date()
    max_date = df['datetime'].max().date()
    
    with ctrl1:
        chart_start = st.date_input(
            "Chart start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="chart_start"
        )
    with ctrl2:
        chart_end = st.date_input(
            "Chart end date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="chart_end"
        )
    with ctrl3:
        param_key = st.selectbox(
            "Parameter",
            options=list(PARAMETER_OPTIONS.keys()),
            format_func=lambda k: PARAMETER_OPTIONS[k]['label'],
            key="chart_param"
        )
    
    if chart_start > chart_end:
        st.error("⚠️ Start date must be before end date")
        return
    
    # ── Filter & render chart ───────────────────────────────────────────
    mask = (df['datetime'].dt.date >= chart_start) & (df['datetime'].dt.date <= chart_end)
    df_filtered = df.loc[mask].copy()
    
    if df_filtered.empty:
        st.info("No data in the selected date range.")
    else:
        param_cfg = PARAMETER_OPTIONS[param_key]
        
        # Build a daily aggregation for cleaner charts when range is large
        days_span = (chart_end - chart_start).days
        if days_span > 365:
            # Show daily averages for large ranges
            plot_df = (
                df_filtered
                .set_index('datetime')
                .resample('D')[param_key]
                .mean()
                .dropna()
                .reset_index()
            )
            chart_note = "Daily average"
        else:
            plot_df = df_filtered[['datetime', param_key]].dropna(subset=[param_key])
            chart_note = "Hourly data"
        
        # Verificar si hay datos válidos para la variable seleccionada
        if plot_df.empty:
            st.warning(
                f"⚠️ No data available for {param_cfg['label'].lower()} in the selected period. "
                f"It was not possible to collect data of this type in the selected area."
            )
        else:
            st.markdown(
                f"<p class='chart-info'>"
                f"{param_cfg['label']} · {chart_note} · {len(plot_df):,} points</p>",
                unsafe_allow_html=True
            )
            
            st.line_chart(
                plot_df.rename(columns={param_key: param_cfg['label']}).set_index('datetime'),
                color=param_cfg['color'],
                width='stretch',
            )
    
    # ── End of results ──────────────────────────────────────────────────
    st.markdown("<div class='spacer-sm'></div>", unsafe_allow_html=True)
