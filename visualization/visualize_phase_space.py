#!/usr/bin/env python3
"""
Offline Phase Space Visualization Tool

Loads phase space logs and creates visualizations to understand model behavior.

Usage:
    python visualization/visualize_phase_space.py [--date YYYYMMDD]
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse


def load_phase_space_data(date=None):
    """Load phase space CSV log for specified date (or today)."""
    log_dir = Path("logs")

    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    log_file = log_dir / f"phase_space_{date}.csv"

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        print(f"\nAvailable log files:")
        for f in sorted(log_dir.glob("phase_space_*.csv")):
            print(f"  {f.name}")
        return None

    df = pd.read_csv(log_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} signals from {log_file.name}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Directions: Long={sum(df['direction']==1)}, Short={sum(df['direction']==-1)}, Flat={sum(df['direction']==0)}")

    return df


def plot_phase_space(df):
    """Create phase space scatter plot with energy contours."""
    fig = go.Figure()

    # Color map for directions
    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    labels = {-1: 'Short', 0: 'Flat', 1: 'Long'}

    # Plot points by direction
    for direction in [-1, 0, 1]:
        mask = df['direction'] == direction
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df.loc[mask, 'q'],
                y=df.loc[mask, 'pi'],
                mode='markers',
                marker=dict(
                    color=colors[direction],
                    size=8,
                    opacity=0.6
                ),
                name=labels[direction],
                text=[f"t={i}<br>forecast={f:.4f}<br>H={h:.6f}"
                      for i, f, h in zip(
                          df.loc[mask].index,
                          df.loc[mask, 'forecast'],
                          df.loc[mask, 'H_before']
                      )],
                hovertemplate='<b>%{text}</b><br>q=%{x:.4f}<br>π=%{y:.4f}<extra></extra>'
            ))

    # Add energy contours
    if len(df) > 0:
        kappa = df['kappa'].iloc[0]

        # Create grid for contours
        q_range = df['q'].max() - df['q'].min()
        pi_range = df['pi'].max() - df['pi'].min()

        q_min = df['q'].min() - 0.2 * q_range
        q_max = df['q'].max() + 0.2 * q_range
        pi_min = df['pi'].min() - 0.2 * pi_range
        pi_max = df['pi'].max() + 0.2 * pi_range

        q_grid = np.linspace(q_min, q_max, 100)
        pi_grid = np.linspace(pi_min, pi_max, 100)
        Q, PI = np.meshgrid(q_grid, pi_grid)

        # H = 0.5*π² + 0.5*κ*q²
        H_grid = 0.5 * PI**2 + 0.5 * kappa * Q**2

        # Add contours
        fig.add_trace(go.Contour(
            x=q_grid,
            y=pi_grid,
            z=H_grid,
            showscale=False,
            contours=dict(
                coloring='lines',
                showlabels=True
            ),
            line=dict(color='lightblue', width=1),
            name='Energy Contours',
            opacity=0.4
        ))

    fig.update_layout(
        title="Phase Space Trajectory<br><sub>Color = Signal Direction | Contours = Energy Levels</sub>",
        xaxis_title="q (position/volume)",
        yaxis_title="π (momentum/return)",
        showlegend=True,
        hovermode='closest',
        height=600
    )

    return fig


def plot_energy_analysis(df):
    """Plot energy drift and conservation over time."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Hamiltonian Over Time', 'Energy Drift'),
        vertical_spacing=0.15
    )

    # Energy over time
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['H_before'],
            mode='lines+markers',
            name='H (before)',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['H_after'],
            mode='lines+markers',
            name='H (after)',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )

    # Energy drift
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['energy_drift'],
            mode='lines+markers',
            name='|ΔH|',
            line=dict(color='purple'),
            fill='tozeroy'
        ),
        row=2, col=1
    )

    # Add horizontal line at mean drift
    mean_drift = df['energy_drift'].mean()
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

    fig.update_layout(
        title="Energy Conservation Analysis",
        height=700,
        showlegend=True
    )

    return fig


def plot_forecast_analysis(df):
    """Analyze forecast quality (requires actual returns to be filled in later)."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Forecast Distribution',
            'Forecast by Direction',
            'Forecast Magnitude',
            'Signal Confidence'
        )
    )

    # Forecast distribution
    fig.add_trace(
        go.Histogram(
            x=df['forecast'],
            nbinsx=30,
            name='Forecast',
            marker_color='steelblue'
        ),
        row=1, col=1
    )

    # Forecast by direction
    for direction, color, label in [(-1, 'red', 'Short'), (1, 'green', 'Long')]:
        mask = df['direction'] == direction
        if mask.sum() > 0:
            fig.add_trace(
                go.Box(
                    y=df.loc[mask, 'forecast'],
                    name=label,
                    marker_color=color
                ),
                row=1, col=2
            )

    # Forecast magnitude over time
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['forecast'].abs(),
            mode='lines+markers',
            name='|forecast|',
            line=dict(color='darkblue')
        ),
        row=2, col=1
    )

    # Signal confidence (size_factor)
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['size_factor'],
            mode='lines+markers',
            name='Size Factor',
            line=dict(color='purple')
        ),
        row=2, col=2
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Forecast", row=1, col=2)
    fig.update_yaxes(title_text="|π_next|", row=2, col=1)
    fig.update_yaxes(title_text="Size Factor", row=2, col=2)

    fig.update_layout(
        title="Forecast Analysis",
        height=700,
        showlegend=True
    )

    return fig


def plot_volume_momentum_relationship(df):
    """Analyze relationship between q (volume) and π (momentum) for Encoding A."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Volume (q) Distribution', 'Momentum (π) Distribution')
    )

    # Volume distribution
    fig.add_trace(
        go.Histogram(
            x=df['q'],
            nbinsx=30,
            name='q (volume)',
            marker_color='lightblue'
        ),
        row=1, col=1
    )

    # Momentum distribution
    fig.add_trace(
        go.Histogram(
            x=df['pi'],
            nbinsx=30,
            name='π (momentum)',
            marker_color='lightcoral'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="q", row=1, col=1)
    fig.update_xaxes(title_text="π", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        title="Phase Space Variable Distributions",
        height=400,
        showlegend=False
    )

    return fig


def generate_summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("PHASE SPACE ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nTotal Signals: {len(df)}")
    print(f"  Long:  {sum(df['direction'] == 1)}")
    print(f"  Short: {sum(df['direction'] == -1)}")
    print(f"  Flat:  {sum(df['direction'] == 0)}")

    print(f"\nPhase Space Statistics:")
    print(f"  q (position):  mean={df['q'].mean():.4f}, std={df['q'].std():.4f}, range=[{df['q'].min():.4f}, {df['q'].max():.4f}]")
    print(f"  π (momentum):  mean={df['pi'].mean():.4f}, std={df['pi'].std():.4f}, range=[{df['pi'].min():.4f}, {df['pi'].max():.4f}]")

    print(f"\nEnergy Statistics:")
    print(f"  H (before):    mean={df['H_before'].mean():.6f}, std={df['H_before'].std():.6f}")
    print(f"  Energy drift:  mean={df['energy_drift'].mean():.6f}, max={df['energy_drift'].max():.6f}")
    print(f"  Energy conservation quality: {'Good' if df['energy_drift'].mean() < 0.001 else 'Fair' if df['energy_drift'].mean() < 0.01 else 'Poor'}")

    print(f"\nForecast Statistics:")
    print(f"  Forecast (π_next): mean={df['forecast'].mean():.4f}, std={df['forecast'].std():.4f}")
    print(f"  |Forecast|:        mean={df['forecast'].abs().mean():.4f}, max={df['forecast'].abs().max():.4f}")

    print(f"\nModel Parameters:")
    print(f"  κ (kappa):  {df['kappa'].iloc[0]:.4f}")
    print(f"  Encoding:   {df['encoding'].iloc[0]}")

    print("="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize phase space data from logs')
    parser.add_argument('--date', type=str, help='Date in YYYYMMDD format (default: today)')
    parser.add_argument('--save', action='store_true', help='Save plots as HTML files')
    args = parser.parse_args()

    # Load data
    df = load_phase_space_data(args.date)
    if df is None:
        return

    # Generate summary stats
    generate_summary_stats(df)

    # Create visualizations
    print("\nGenerating visualizations...")

    figs = {
        'phase_space': plot_phase_space(df),
        'energy': plot_energy_analysis(df),
        'forecast': plot_forecast_analysis(df),
        'distributions': plot_volume_momentum_relationship(df)
    }

    # Show or save plots
    if args.save:
        output_dir = Path("visualization/output")
        output_dir.mkdir(exist_ok=True)
        date_str = args.date if args.date else datetime.now().strftime('%Y%m%d')

        for name, fig in figs.items():
            output_file = output_dir / f"{name}_{date_str}.html"
            fig.write_html(output_file)
            print(f"Saved: {output_file}")
    else:
        print("\nOpening interactive plots in browser...")
        for name, fig in figs.items():
            fig.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
