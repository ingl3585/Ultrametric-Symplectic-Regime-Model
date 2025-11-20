#!/usr/bin/env python3
"""
Real-Time Phase Space Dashboard

Streamlit dashboard for monitoring model behavior in real-time.

Usage:
    streamlit run visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time


# Page configuration
st.set_page_config(
    page_title="Symplectic Model Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_latest_data(hours=24):
    """Load recent phase space data."""
    log_dir = Path("logs")

    # Try to find recent log files
    cutoff_time = datetime.now() - timedelta(hours=hours)
    all_data = []

    for log_file in sorted(log_dir.glob("phase_space_*.csv"), reverse=True):
        try:
            df = pd.read_csv(log_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Only include recent data
            df = df[df['timestamp'] >= cutoff_time]

            if len(df) > 0:
                all_data.append(df)
        except Exception as e:
            st.sidebar.warning(f"Error loading {log_file.name}: {e}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp')

    return combined


def create_phase_space_plot(df, window_size=100):
    """Create phase space scatter plot."""
    # Use last N points
    df_window = df.tail(window_size)

    fig = go.Figure()

    # Color map
    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    labels = {-1: 'Short', 0: 'Flat', 1: 'Long'}

    # Plot by direction
    for direction in [-1, 0, 1]:
        mask = df_window['direction'] == direction
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df_window.loc[mask, 'q'],
                y=df_window.loc[mask, 'pi'],
                mode='markers',
                marker=dict(color=colors[direction], size=10, opacity=0.7),
                name=labels[direction],
                text=[f"{t.strftime('%H:%M')}" for t in df_window.loc[mask, 'timestamp']],
                hovertemplate='<b>%{text}</b><br>q=%{x:.4f}<br>π=%{y:.4f}<extra></extra>'
            ))

    # Add trajectory line
    fig.add_trace(go.Scatter(
        x=df_window['q'],
        y=df_window['pi'],
        mode='lines',
        line=dict(color='lightgray', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Add energy contours
    if len(df_window) > 0:
        kappa = df_window['kappa'].iloc[0]

        q_range = df_window['q'].max() - df_window['q'].min()
        pi_range = df_window['pi'].max() - df_window['pi'].min()

        q_min = df_window['q'].min() - 0.2 * q_range
        q_max = df_window['q'].max() + 0.2 * q_range
        pi_min = df_window['pi'].min() - 0.2 * pi_range
        pi_max = df_window['pi'].max() + 0.2 * pi_range

        q_grid = np.linspace(q_min, q_max, 50)
        pi_grid = np.linspace(pi_min, pi_max, 50)
        Q, PI = np.meshgrid(q_grid, pi_grid)

        H_grid = 0.5 * PI**2 + 0.5 * kappa * Q**2

        fig.add_trace(go.Contour(
            x=q_grid, y=pi_grid, z=H_grid,
            showscale=False,
            contours=dict(coloring='lines'),
            line=dict(color='lightblue', width=1),
            opacity=0.3,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="Phase Space Trajectory (Recent)",
        xaxis_title="q (position/volume)",
        yaxis_title="π (momentum/return)",
        height=500,
        hovermode='closest'
    )

    return fig


def create_energy_plot(df, window_size=100):
    """Create energy monitoring plot."""
    df_window = df.tail(window_size)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Hamiltonian', 'Energy Drift'),
        vertical_spacing=0.15
    )

    # Energy
    fig.add_trace(
        go.Scatter(
            x=df_window['timestamp'],
            y=df_window['H_before'],
            mode='lines',
            name='H',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # Energy drift
    fig.add_trace(
        go.Scatter(
            x=df_window['timestamp'],
            y=df_window['energy_drift'],
            mode='lines',
            name='|ΔH|',
            line=dict(color='purple'),
            fill='tozeroy'
        ),
        row=2, col=1
    )

    # Add threshold line
    mean_drift = df_window['energy_drift'].mean()
    fig.add_hline(
        y=mean_drift,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Mean: {mean_drift:.6f}",
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Energy", row=1, col=1)
    fig.update_yaxes(title_text="|ΔH|", row=2, col=1)

    fig.update_layout(height=500, showlegend=True)

    return fig


def create_forecast_plot(df, window_size=100):
    """Create forecast analysis plot."""
    df_window = df.tail(window_size)

    fig = go.Figure()

    # Forecast magnitude
    fig.add_trace(go.Scatter(
        x=df_window['timestamp'],
        y=df_window['forecast'].abs(),
        mode='lines+markers',
        name='|Forecast|',
        line=dict(color='darkblue', width=2),
        marker=dict(size=6)
    ))

    # Color bars by direction
    colors = df_window['direction'].map({-1: 'rgba(255,0,0,0.3)', 0: 'rgba(128,128,128,0.3)', 1: 'rgba(0,255,0,0.3)'})

    fig.add_trace(go.Bar(
        x=df_window['timestamp'],
        y=df_window['forecast'].abs(),
        marker_color=colors,
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="Forecast Magnitude Over Time",
        xaxis_title="Time",
        yaxis_title="|π_next|",
        height=400
    )

    return fig


def create_performance_metrics(df):
    """Create performance metrics cards."""
    # Recent window
    recent = df.tail(50)

    total_signals = len(recent)
    long_signals = sum(recent['direction'] == 1)
    short_signals = sum(recent['direction'] == -1)
    flat_signals = sum(recent['direction'] == 0)

    avg_energy = recent['H_before'].mean()
    avg_drift = recent['energy_drift'].mean()
    avg_forecast = recent['forecast'].abs().mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Signals (50)",
            total_signals,
            delta=None
        )
        st.caption(f"🟢 Long: {long_signals} | 🔴 Short: {short_signals} | ⚪ Flat: {flat_signals}")

    with col2:
        st.metric(
            "Avg Energy",
            f"{avg_energy:.6f}",
            delta=None
        )
        st.caption(f"κ = {recent['kappa'].iloc[-1]:.4f}")

    with col3:
        drift_status = "🟢 Good" if avg_drift < 0.001 else "🟡 Fair" if avg_drift < 0.01 else "🔴 High"
        st.metric(
            "Avg Energy Drift",
            f"{avg_drift:.6f}",
            delta=None
        )
        st.caption(f"Status: {drift_status}")

    with col4:
        st.metric(
            "Avg |Forecast|",
            f"{avg_forecast:.4f}",
            delta=None
        )
        st.caption(f"Range: [{recent['forecast'].min():.4f}, {recent['forecast'].max():.4f}]")


def create_signal_distribution(df):
    """Create signal distribution plot."""
    fig = go.Figure()

    signal_counts = df['direction'].value_counts().to_dict()

    colors_map = {-1: 'red', 0: 'gray', 1: 'green'}
    labels_map = {-1: 'Short', 0: 'Flat', 1: 'Long'}

    directions = [-1, 0, 1]
    counts = [signal_counts.get(d, 0) for d in directions]
    colors = [colors_map[d] for d in directions]
    labels = [labels_map[d] for d in directions]

    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        marker_color=colors,
        text=counts,
        textposition='auto'
    ))

    fig.update_layout(
        title="Signal Distribution",
        xaxis_title="Direction",
        yaxis_title="Count",
        height=300,
        showlegend=False
    )

    return fig


# Main App
def main():
    # Header
    st.markdown('<div class="main-header">📈 Symplectic Model Dashboard</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # Refresh controls
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10, 300, 60)

    # Data window
    hours_back = st.sidebar.slider("Hours of data to show", 1, 168, 24)
    plot_window = st.sidebar.slider("Plot window size (points)", 20, 500, 100)

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()

    # Load data
    df = load_latest_data(hours=hours_back)

    if df is None or len(df) == 0:
        st.warning("⚠️ No data available. Make sure the server is running and generating signals.")
        st.info("Log files should be in: `logs/phase_space_YYYYMMDD.csv`")
        return

    # Show last update time
    last_signal = df['timestamp'].max()
    st.sidebar.success(f"✅ Last signal: {last_signal.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.info(f"📊 Total points: {len(df)}")

    # Performance Metrics
    st.subheader("📊 Performance Metrics")
    create_performance_metrics(df)

    # Main visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Phase Space", "⚡ Energy", "📈 Forecast", "📊 Statistics"])

    with tab1:
        st.plotly_chart(create_phase_space_plot(df, window_size=plot_window), use_container_width=True)
        st.caption("**Interpretation:** Points show (q, π) state colored by signal direction. Contours show energy levels. Trajectory should respect energy conservation (follow contours).")

    with tab2:
        st.plotly_chart(create_energy_plot(df, window_size=plot_window), use_container_width=True)
        st.caption("**Interpretation:** Top: Hamiltonian over time (should be relatively stable). Bottom: Energy drift (should be small for good symplectic integration).")

    with tab3:
        st.plotly_chart(create_forecast_plot(df, window_size=plot_window), use_container_width=True)
        st.caption("**Interpretation:** Forecast magnitude |π_next| over time. Bars colored by signal direction. Larger values = stronger conviction.")

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_signal_distribution(df), use_container_width=True)

        with col2:
            st.write("**Recent Statistics:**")
            recent = df.tail(100)
            stats_df = pd.DataFrame({
                'Metric': ['q (position)', 'π (momentum)', 'Energy (H)', 'Energy Drift', 'Forecast'],
                'Mean': [
                    f"{recent['q'].mean():.4f}",
                    f"{recent['pi'].mean():.4f}",
                    f"{recent['H_before'].mean():.6f}",
                    f"{recent['energy_drift'].mean():.6f}",
                    f"{recent['forecast'].abs().mean():.4f}"
                ],
                'Std': [
                    f"{recent['q'].std():.4f}",
                    f"{recent['pi'].std():.4f}",
                    f"{recent['H_before'].std():.6f}",
                    f"{recent['energy_drift'].std():.6f}",
                    f"{recent['forecast'].abs().std():.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Recent signals table
    with st.expander("📝 Recent Signals (Last 10)", expanded=False):
        recent_signals = df.tail(10)[['timestamp', 'direction', 'size_factor', 'q', 'pi', 'forecast', 'H_before', 'energy_drift']]
        recent_signals = recent_signals.sort_values('timestamp', ascending=False)
        st.dataframe(recent_signals, use_container_width=True, hide_index=True)

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
