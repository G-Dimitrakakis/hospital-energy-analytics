import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hospital Energy Analytics & Fractal Analysis",
    page_icon="⚡",
    layout="wide"
)

# --- CUSTOM CSS FOR SIDEBAR TOGGLE & VISIBILITY ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Make Top Header transparent but KEEP Sidebar Collapse/Expand Button */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            z-index: 99999 !important;
        }
        
        /* Hide right-side header menu options, keep left-side sidebar toggle */
        [data-testid="stHeader"] > div:nth-child(2) {
            display: none !important;
        }

        /* Style the Sidebar Toggle Button explicitly */
        [data-testid="stSidebarCollapseButton"], button[aria-label="Expand sidebar"], button[aria-label="Collapse sidebar"] {
            visibility: visible !important;
            display: flex !important;
            color: #1C1C1C !important;
            background-color: #EFECE1 !important;
            border: 1px solid #D8D4C5 !important;
            border-radius: 6px !important;
        }

        .main .block-container {
            padding-top: 2rem !important;
        }

        /* Global Typography & Colors */
        html, body, [class*="css"], .stApp {
            font-family: 'Inter', sans-serif;
            background-color: #F8F4E9 !important;
            color: #1C1C1C !important;
        }
        
        /* Headings & Text Elements */
        h1, h2, h3, h4, h5, h6, p, label, span, div, caption {
            color: #1C1C1C !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #EFECE1 !important;
            border-right: 1px solid #D8D4C5;
        }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
            color: #1C1C1C !important;
            font-weight: 500;
        }

        /* Input Fields, Selectboxes & Dropdowns Fix */
        input, select, textarea, div[role="combobox"], div[data-baseweb="select"] {
            color: #1C1C1C !important;
            background-color: #FFFFFF !important;
            border-radius: 6px !important;
        }
        
        div[data-baseweb="select"] * {
            color: #1C1C1C !important;
            background-color: #FFFFFF !important;
        }
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
            background-color: #FFFFFF !important;
        }
        li[role="option"] span, li[role="option"] {
            color: #1C1C1C !important;
            background-color: #FFFFFF !important;
        }

        /* Fix for File Uploader "Browse files" Button */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF !important;
            border: 1px dashed #A7C957 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        [data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
        }
        [data-testid="stFileUploader"] button {
            background-color: #A7C957 !important;
            color: #1C1C1C !important;
            border: none !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background-color: #96B846 !important;
            color: #1C1C1C !important;
        }
        [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
            color: #1C1C1C !important;
        }

        /* Fix for Calendar / Date Input Popover */
        div[data-baseweb="calendar"], div[data-baseweb="calendar"] * {
            background-color: #FFFFFF !important;
            color: #1C1C1C !important;
        }
        div[aria-roledescription="calendar"] {
            background-color: #FFFFFF !important;
        }
        div[role="gridcell"] {
            color: #1C1C1C !important;
        }

        /* Info & Success Message Boxes */
        .stAlert > div {
            color: #1C1C1C !important;
            background-color: #E6EFE0 !important;
            border: 1px solid #A7C957 !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: #EFECE1;
            border-radius: 6px 6px 0px 0px;
            color: #1C1C1C !important;
            font-weight: 500;
            padding-left: 16px;
            padding-right: 16px;
            border: 1px solid #D8D4C5;
        }
        .stTabs [aria-selected="true"] {
            background-color: #A7C957 !important;
            color: #1C1C1C !important;
            font-weight: 700 !important;
            border-bottom: none !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #A7C957 !important;
            color: #1C1C1C !important;
            border: none;
            font-weight: 600;
            border-radius: 6px;
        }
        .stButton>button:hover {
            background-color: #96B846 !important;
            color: #1C1C1C !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- HIGUCHI & HURST FUNCTION ---
def estimate_higuchi_hurst(ts, minimal=11):
    """
    Calculates Hurst Exponent & Fractal Dimension using the Higuchi Method
    based on signal integration (cumsum) for fGn time series.
    """
    ts = np.asarray(ts, dtype=np.float64)
    N = len(ts)
    
    if N < minimal * 2:
        return 0.5, 1.5, np.array([1]), np.array([1]), 0, 0

    # Cumulative Sum (fGn -> fBm)
    Y = np.cumsum(ts - np.mean(ts))
    
    Lm, Cm = [], []
    
    for j in range(1, minimal):
        m = j
        Cm.append(m)
        k = N // m
        
        Xsub = np.reshape(Y[N % m:], [k, m])
        Lk = np.abs(Xsub[1:] - Xsub[:-1])
        
        Lm.append(np.mean(Lk) * (N - 1) / (m ** 2))
        
    log_Cm = np.log(1.0 / np.array(Cm))
    log_Lm = np.log(np.array(Lm))
    
    slope, intercept = np.polyfit(log_Cm, log_Lm, 1)
    
    fractal_dimension = slope
    hurst_exponent = 2.0 - fractal_dimension
    
    return hurst_exponent, fractal_dimension, log_Cm, log_Lm, slope, intercept

# --- HEADER & LOGO ---
st.title("⚡ Hospital Energy Analytics & Fractal Analysis")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Parameters & Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# SAMPLE DATA GENERATOR
@st.cache_data
def load_sample_data():
    dates = pd.date_range(start="2024-01-01", periods=24*90, freq="h")
    base = 300
    sin_wave = np.sin(np.pi * (dates.hour - 6) / 12)
    daily_pattern = 100 * np.clip(sin_wave, a_min=0, a_max=None)
    noise = np.random.normal(0, 20, len(dates))
    load = base + daily_pattern + noise
    return pd.DataFrame({"timestamp": dates, "load_kwh": load})

# --- DATA LOADING & COLUMN MAPPING ---
if uploaded_file is not None:
    st.sidebar.success("File uploaded successfully.")
    
    try:
        df_raw = pd.read_csv(uploaded_file)
        if len(df_raw.columns) == 1:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=';')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()
    
    df_raw.columns = df_raw.columns.str.strip()
    cols = list(df_raw.columns)
    
    # Αυτόματος εντοπισμός προεπιλεγμένων ευρετηρίων (Default Indices)
    default_time_idx = next((i for i, c in enumerate(cols) if any(k in c.lower() for k in ['time', 'date', 'ώρα', 'ημερομηνία', 'timestamp'])), 0)
    default_load_idx = next((i for i, c in enumerate(cols) if any(k in c.lower() for k in ['load', 'kwh', 'kw', 'κατανάλωση', 'energy', 'power', 'val']) and i != default_time_idx), min(1, len(cols) - 1))
    
    # Πτυσσόμενο πλαίσιο επιλογής στηλών (Expander)
    with st.sidebar.expander("Column Mapping Configuration", expanded=False):
        selected_time_col = st.selectbox("Timestamp Column:", options=cols, index=default_time_idx)
        selected_load_col = st.selectbox("Load/Consumption Column:", options=cols, index=default_load_idx)

    # Δημιουργία επεξεργασμένου DataFrame
    df = df_raw.rename(columns={selected_time_col: 'timestamp', selected_load_col: 'load_kwh'}).copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['load_kwh'] = pd.to_numeric(df['load_kwh'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'load_kwh'])

else:
    st.info("Sample data loaded automatically. Upload your CSV from the sidebar.")
    df = load_sample_data()

df = df.sort_values('timestamp').reset_index(drop=True)

# Financial & Environmental Inputs in Sidebar
st.sidebar.subheader("Financial & Environmental Parameters")
cost_per_kwh = st.sidebar.number_input("Electricity Cost (€/kWh)", value=0.15, step=0.01)
co2_per_kwh = st.sidebar.number_input("CO2 Factor (kg/kWh)", value=0.42, step=0.01)

# Higuchi Parameters in Sidebar
st.sidebar.subheader("Higuchi Parameters")
minimal_k = st.sidebar.slider("Minimal Scale (kmax / Minimal)", min_value=5, max_value=30, value=11, step=1)

# Plotly Shared Layout Config
PLOT_COLOR_LINE = "#1C1C1C"
PLOT_COLOR_ACCENT = "#A7C957"

def apply_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1C1C1C", family="Inter"),
        title_font=dict(color="#1C1C1C", size=16),
        xaxis=dict(title_font=dict(color="#1C1C1C"), tickfont=dict(color="#1C1C1C"), gridcolor="#E0DDD0"),
        yaxis=dict(title_font=dict(color="#1C1C1C"), tickfont=dict(color="#1C1C1C"), gridcolor="#E0DDD0"),
        legend=dict(font=dict(color="#1C1C1C"))
    )
    return fig

# --- TABS LAYOUT ---
tab_kpi, tab_ts, tab_fractal, tab_advanced = st.tabs([
    "Overview & KPIs", 
    "Time Series & Profiles", 
    "Fractal Analysis", 
    "Heatmaps & LDC"
])

# === TAB 1: OVERVIEW & KPIS ===
with tab_kpi:
    st.subheader("Key Performance Indicators (KPIs)")
    
    total_energy = df['load_kwh'].sum()
    peak_demand = df['load_kwh'].max()
    peak_time = df.loc[df['load_kwh'].idxmax(), 'timestamp']
    mean_load = df['load_kwh'].mean()
    base_load = df['load_kwh'].quantile(0.05)
    load_factor = (mean_load / peak_demand) if peak_demand > 0 else 0
    total_cost = total_energy * cost_per_kwh
    total_co2 = total_energy * co2_per_kwh

    def format_large_number(val, unit="", is_currency=False):
        prefix = "€" if is_currency else ""
        if val >= 1_000_000_000:
            return f"{prefix}{val/1_000_000_000:,.2f} B {unit}".strip()
        elif val >= 1_000_000:
            return f"{prefix}{val/1_000_000:,.2f} M {unit}".strip()
        elif val >= 100_000:
            return f"{prefix}{val/1_000:,.1f} k {unit}".strip()
        else:
            return f"{prefix}{val:,.2f} {unit}".strip() if is_currency else f"{val:,.0f} {unit}".strip()

    def render_metric_card(label, main_value, sub_text=""):
        st.markdown(f"""
            <div style="
                background-color: #EFECE1;
                border: 1px solid #D8D4C5;
                border-radius: 8px;
                padding: 12px 14px;
                text-align: center;
                box-shadow: 0px 1px 3px rgba(0,0,0,0.05);
                margin-bottom: 10px;
            ">
                <div style="color: #555555; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px;">
                    {label}
                </div>
                <div style="color: #1C1C1C; font-size: clamp(1.1rem, 1.8vw, 1.5rem); font-weight: 700; word-break: break-word; line-height: 1.2;">
                    {main_value}
                </div>
                <div style="color: #666666; font-size: 0.75rem; margin-top: 4px; font-weight: 500;">
                    {sub_text}
                </div>
            </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        render_metric_card(
            "Total Consumption", 
            format_large_number(total_energy, "kWh"), 
            f"Exact: {total_energy:,.0f} kWh" if total_energy >= 1_000_000 else "Full Period Sum"
        )
    with col2:
        render_metric_card(
            "Peak Demand", 
            f"{peak_demand:,.1f} kW", 
            f"At: {peak_time.strftime('%Y-%m-%d %H:%M')}"
        )
    with col3:
        render_metric_card(
            "Load Factor", 
            f"{load_factor:.2%}", 
            f"Base Load: {base_load:,.1f} kW"
        )
    with col4:
        render_metric_card(
            "Estimated Cost", 
            format_large_number(total_cost, is_currency=True), 
            f"Exact: €{total_cost:,.2f}" if total_cost >= 1_000_000 else f"@ €{cost_per_kwh:.2f}/kWh"
        )
    with col5:
        render_metric_card(
            "CO₂ Emissions", 
            format_large_number(total_co2 / 1000, "t"), 
            f"Exact: {total_co2/1000:,.1f} tons" if (total_co2/1000) >= 1_000_000 else f"@ {co2_per_kwh:.2f} kg/kWh"
        )

    st.divider()
    
    st.subheader("Executive Summary")
    st.write(f"""
    - **Total Analyzed Period:** {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')} ({len(df):,} hourly observations)
    - **Average Hourly Consumption:** {mean_load:,.2f} kWh/h
    - **Baseload Share:** The continuous base load of {base_load:,.1f} kW accounts for approximately {((base_load * len(df)) / total_energy) * 100:.1f}% of total energy consumption.
    - **Total Cost:** €{total_cost:,.2f} | **Total CO₂ Footprint:** {total_co2/1000:,.2f} metric tons.
    """)

# === TAB 2: TIME SERIES & PROFILES ===
with tab_ts:
    st.subheader("Interactive Time Series")
    
    file_min_date = df['timestamp'].min().date()
    file_max_date = df['timestamp'].max().date()

    st.write("**Quick Range Selectors (Relative to Dataset):**")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    
    if 'start_date' not in st.session_state or 'end_date' not in st.session_state:
        st.session_state.start_date = file_min_date
        st.session_state.end_date = file_max_date

    if btn_col1.button("Full Dataset"):
        st.session_state.start_date = file_min_date
        st.session_state.end_date = file_max_date
    if btn_col2.button("Last Day of Dataset"):
        st.session_state.start_date = file_max_date
        st.session_state.end_date = file_max_date
    if btn_col3.button("Last 7 Days of Dataset"):
        st.session_state.start_date = max(file_min_date, file_max_date - timedelta(days=7))
        st.session_state.end_date = file_max_date
    if btn_col4.button("Last 30 Days of Dataset"):
        st.session_state.start_date = max(file_min_date, file_max_date - timedelta(days=30))
        st.session_state.end_date = file_max_date

    selected_dates = st.date_input(
        "Custom Date Range",
        value=[st.session_state.start_date, st.session_state.end_date],
        min_value=file_min_date,
        max_value=file_max_date
    )
    
    if len(selected_dates) == 2:
        mask = (df['timestamp'].dt.date >= selected_dates[0]) & (df['timestamp'].dt.date <= selected_dates[1])
        filtered_df = df.loc[mask]
    else:
        filtered_df = df

    fig_ts = px.line(filtered_df, x='timestamp', y='load_kwh', title="Hourly Energy Load (kWh)",
                     labels={'timestamp': 'Timestamp', 'load_kwh': 'Load (kWh)'})
    fig_ts.update_traces(line_color=PLOT_COLOR_LINE)
    fig_ts = apply_plotly_theme(fig_ts)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("Average 24-Hour Profile (Weekdays vs Weekends)")
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy['hour'] = filtered_df_copy['timestamp'].dt.hour
    filtered_df_copy['is_weekend'] = filtered_df_copy['timestamp'].dt.dayofweek >= 5
    filtered_df_copy['day_type'] = filtered_df_copy['is_weekend'].map({True: 'Weekend', False: 'Weekday'})

    profile = filtered_df_copy.groupby(['hour', 'day_type'])['load_kwh'].mean().reset_index()

    fig_profile = px.line(profile, x='hour', y='load_kwh', color='day_type',
                          title="Average Daily Load Pattern",
                          labels={'hour': 'Hour of Day (0-23)', 'load_kwh': 'Average Load (kWh)'},
                          color_discrete_map={'Weekday': PLOT_COLOR_LINE, 'Weekend': PLOT_COLOR_ACCENT},
                          markers=True)
    fig_profile = apply_plotly_theme(fig_profile)
    st.plotly_chart(fig_profile, use_container_width=True)

# === TAB 3: FRACTAL ANALYSIS ===
with tab_fractal:
    st.subheader("Higuchi Fractal Dimension (HFD) & Hurst Exponent")
    
    hurst, fd, log_Cm, log_Lm, slope, intercept = estimate_higuchi_hurst(filtered_df['load_kwh'], minimal=minimal_k)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.metric("Higuchi Fractal Dimension (FD)", f"{fd:.4f}")
        st.info("""
        **FD Interpretation:**
        - **FD ≈ 1.0:** Highly smooth & periodic signal (predictable).
        - **FD ≈ 1.5:** Brownian Motion behavior / random walk.
        - **FD ≈ 2.0:** Highly volatile & irregular signal (noisy).
        """)
    
    with col_f2:
        st.metric("Hurst Exponent (H)", f"{hurst:.4f}")
        st.info("""
        **Hurst Exponent Interpretation:**
        - **H > 0.5:** Persistent time series (trend-reinforcing).
        - **H = 0.5:** True Random Walk.
        - **H < 0.5:** Anti-persistent time series (mean-reverting).
        """)

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Log-Log Fit Diagram (Higuchi Method)")
        fit_y = slope * log_Cm + intercept
        
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=log_Cm, y=log_Lm, mode='markers', name='Data L(m)',
                                     marker=dict(size=10, color=PLOT_COLOR_LINE)))
        fig_fit.add_trace(go.Scatter(x=log_Cm, y=fit_y, mode='lines', name=f'Linear Fit (Slope = {slope:.4f})',
                                     line=dict(color=PLOT_COLOR_ACCENT, width=3, dash='dash')))
        
        fig_fit.update_layout(
            title="Log-Log Fit: ln(1/m) vs ln(Lm)",
            xaxis_title="ln(1/m)",
            yaxis_title="ln(Lm)"
        )
        fig_fit = apply_plotly_theme(fig_fit)
        st.plotly_chart(fig_fit, use_container_width=True)

    with col_g2:
        st.subheader("Sensitivity Analysis: FD & Hurst vs. kmax")
        
        k_range = list(range(5, 31))
        fd_list, hurst_list = [], []

        for k_val in k_range:
            h_val, f_val, _, _, _, _ = estimate_higuchi_hurst(filtered_df['load_kwh'], minimal=k_val)
            fd_list.append(f_val)
            hurst_list.append(h_val)

        fig_sens = go.Figure()
        
        fig_sens.add_trace(go.Scatter(
            x=k_range, y=fd_list, mode='lines+markers', name='Fractal Dimension (FD)',
            line=dict(color=PLOT_COLOR_LINE, width=2), marker=dict(size=6)
        ))
        
        fig_sens.add_trace(go.Scatter(
            x=k_range, y=hurst_list, mode='lines+markers', name='Hurst Exponent (H)',
            line=dict(color=PLOT_COLOR_ACCENT, width=2, dash='dot'), marker=dict(size=6),
            yaxis='y2'
        ))

        fig_sens.update_layout(
            title="Parameter Sensitivity across Scales",
            xaxis=dict(title="Minimal Scale (kmax)"),
            yaxis=dict(
                title="Fractal Dimension (FD)",
                title_font=dict(color=PLOT_COLOR_LINE),
                tickfont=dict(color=PLOT_COLOR_LINE)
            ),
            yaxis2=dict(
                title="Hurst Exponent (H)",
                title_font=dict(color=PLOT_COLOR_ACCENT),
                tickfont=dict(color=PLOT_COLOR_ACCENT),
                overlaying='y',
                side='right'
            ),
            showlegend=False
        )
        fig_sens = apply_plotly_theme(fig_sens)
        st.plotly_chart(fig_sens, use_container_width=True)

# === TAB 4: HEATMAPS & LDC ===
with tab_advanced:
    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        st.subheader("Consumption Heatmap (24h vs Date)")
        df_heatmap = filtered_df.copy()
        df_heatmap['date'] = df_heatmap['timestamp'].dt.date
        df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
        pivot_df = df_heatmap.pivot(index='hour', columns='date', values='load_kwh')
        
        fig_heat = px.imshow(pivot_df, labels=dict(x="Date", y="Hour of Day", color="kWh"),
                             aspect="auto", color_continuous_scale="Viridis")
        fig_heat = apply_plotly_theme(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_h2:
        st.subheader("Load Duration Curve (LDC)")
        sorted_load = np.sort(filtered_df['load_kwh'])[::-1]
        percentiles = np.linspace(0, 100, len(sorted_load))
        
        fig_ldc = px.line(x=percentiles, y=sorted_load,
                          labels={'x': 'Time Percentage (%)', 'y': 'Load (kWh/kW)'},
                          title="Load Duration Curve")
        fig_ldc.update_traces(line_color=PLOT_COLOR_LINE, fill='tozeroy', fillcolor='rgba(167, 201, 87, 0.4)')
        fig_ldc = apply_plotly_theme(fig_ldc)
        st.plotly_chart(fig_ldc, use_container_width=True)