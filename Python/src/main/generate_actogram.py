#!/usr/bin/env python3
"""
Generate Actogram Chart for Fly Activity

This script creates an inverted bar actogram showing a fly's activity pattern over time.
Each row represents one day, with activity bars extending downward. The chart shows
circadian rhythms and activity patterns across light/dark cycles.

An actogram is a type of chart used to show how an animal's activity changes over time
- sort of like a daily timeline of movement. Each row represents one day, the x-axis
shows Zeitgeber Time (ZT), and bars show when the animal was active.

Usage:
    python generate_actogram.py [monitor] [channel] [input_file] [output_file]
    
Examples:
    python generate_actogram.py 5 16
    python generate_actogram.py 5 16 dam_data_MT.csv actogram_5_16.png
    python generate_actogram.py 6 23 ../../data/processed/dam_data_MT.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import os
import sys


# --- Settings ---
lights_on = 9  # Hour when lights turn on (ZT0)
bin_length = 1  # Minutes per bin


def calculate_zt(datetime_series, lights_on):
    """
    Calculate Zeitgeber Time (ZT) from datetime.
    
    ZT = (hour + minute/60 - lights_on) %% 24
    
    Args:
        datetime_series: Series of datetime objects
        lights_on: Hour when lights turn on (default 9)
        
    Returns:
        Series of ZT values (0-24)
    """
    hours = datetime_series.dt.hour
    minutes = datetime_series.dt.minute
    zt = (hours + minutes / 60 - lights_on) % 24
    return zt


def prepare_actogram_data(df, monitor, channel, lights_on, bin_length):
    """
    Prepare data for actogram plotting.
    
    Args:
        df: DataFrame with MT data
        monitor: Monitor number to plot
        channel: Channel number to plot
        lights_on: Hour when lights turn on
        bin_length: Minutes per bin
        
    Returns:
        DataFrame with Date, ZT, Activity, and Activity_Scaled columns
    """
    # Filter for specific fly
    fly_data = df[
        (df['monitor'] == monitor) & 
        (df['channel'] == channel)
    ].copy()
    
    if len(fly_data) == 0:
        raise ValueError(f"No data found for Monitor {monitor}, Channel {channel}")
    
    # Ensure datetime is datetime type
    fly_data['datetime'] = pd.to_datetime(fly_data['datetime'])
    
    # Create Date column
    fly_data['Date'] = fly_data['datetime'].dt.date
    
    # Floor datetime to bin_length minutes
    fly_data['BIN_TIME'] = fly_data['datetime'].dt.floor(f'{bin_length}min')
    
    # Calculate ZT (Zeitgeber Time)
    fly_data['ZT'] = calculate_zt(fly_data['BIN_TIME'], lights_on)
    
    # Group by Date and ZT, sum activity
    # Handle both 'value' and 'activity_count' column names
    activity_col = 'value' if 'value' in fly_data.columns else 'activity_count'
    
    act_data = fly_data.groupby(['Date', 'ZT'], as_index=False).agg({
        activity_col: 'sum'
    }).rename(columns={activity_col: 'Activity'})
    
    # Sort by Date and ZT
    act_data = act_data.sort_values(['Date', 'ZT']).reset_index(drop=True)
    
    # Convert Date to categorical to preserve order (ordered=True for min/max operations)
    unique_dates = sorted(act_data['Date'].unique())
    act_data['Date'] = pd.Categorical(act_data['Date'], categories=unique_dates, ordered=True)
    
    # Scale activity within each day (0 to 0.8)
    act_data['Activity_Scaled'] = act_data.groupby('Date', observed=True)['Activity'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) * 0.8 if x.max() > x.min() else 0
    )
    
    return act_data


def plot_actogram(act_data, monitor, channel, output_file=None):
    """
    Create inverted bar actogram plot.
    
    Args:
        act_data: DataFrame with Date, ZT, Activity_Scaled columns
        monitor: Monitor number (for title)
        channel: Channel number (for title)
        output_file: Path to save plot (if None, displays plot)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(act_data['Date'].unique()) * 0.3)))
    
    # Get unique dates in order (handle categorical)
    if isinstance(act_data['Date'].dtype, pd.CategoricalDtype):
        unique_dates = list(act_data['Date'].cat.categories)
    else:
        unique_dates = sorted(act_data['Date'].unique())
    date_to_num = {date: i + 1 for i, date in enumerate(unique_dates)}
    
    # Add dark phase shading (ZT 12-24)
    ax.axvspan(12, 24, alpha=0.5, color='grey', zorder=0)
    
    # Plot bars for each data point
    for _, row in act_data.iterrows():
        date_num = date_to_num[row['Date']]
        zt = row['ZT']
        activity_scaled = row['Activity_Scaled']
        
        # Draw segment extending downward from day line
        # y position: date_num - 0.5 (center of day row)
        # extends downward by activity_scaled
        y_start = date_num - 0.5
        y_end = y_start - activity_scaled
        
        ax.plot([zt, zt], [y_start, y_end], 
                color='black', linewidth=0.9, solid_capstyle='round')
    
    # Set y-axis (reversed, so earliest day at top)
    ax.set_ylim(len(unique_dates) + 0.5, 0.5)
    ax.set_yticks(range(1, len(unique_dates) + 1))
    ax.set_yticklabels([str(date) for date in unique_dates])
    
    # Set x-axis (ZT 0-24)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 6))
    ax.set_xticks(range(0, 25, 3), minor=True)
    ax.set_xlabel('ZT (h)', fontsize=13)
    ax.set_ylabel('Day', fontsize=13)
    
    # Add grid
    ax.grid(True, which='major', axis='x', color='grey', linewidth=0.3, alpha=0.8)
    ax.grid(True, which='minor', axis='x', color='grey', linewidth=0.2, 
            linestyle='dotted', alpha=0.6)
    
    # Title
    ax.set_title(f'Actogram — Fly {monitor}-{channel}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved actogram to {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to generate actogram."""
    print("=" * 60)
    print("Actogram Generator")
    print("=" * 60)
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python generate_actogram.py <monitor> <channel> [input_file] [output_file]")
        print("\nExamples:")
        print("  python generate_actogram.py 5 16")
        print("  python generate_actogram.py 5 16 dam_data_MT.csv actogram_5_16.png")
        sys.exit(1)
    
    try:
        monitor = int(sys.argv[1])
        channel = int(sys.argv[2])
    except ValueError:
        print("ERROR: Monitor and channel must be integers")
        sys.exit(1)
    
    # Determine input file
    if len(sys.argv) > 3:
        input_file = sys.argv[3]
    else:
        # Default: look for MT file
        default_paths = [
            '../../data/processed/dam_data_MT.csv'
            # '../../data/processed/dam_data_with_flies.csv',
            # '../../data/processed/dam_data_merged.csv'
        ]
        input_file = None
        for path in default_paths:
            if os.path.exists(path):
                input_file = path
                break
        
        if input_file is None:
            print("ERROR: No input file specified and no default file found.")
            print("Please specify an input file or ensure one of these exists:")
            for path in default_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    # Determine output file
    if len(sys.argv) > 4:
        output_file = sys.argv[4]
    else:
        output_file = f'../../data/processed/actogram_{monitor}_{channel}.png'
    
    print(f"\n📊 Input file: {input_file}")
    print(f"📈 Monitor: {monitor}, Channel: {channel}")
    print(f"📄 Output file: {output_file}")
    print()
    
    # Read data
    print("Reading data...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Handle reading column if present (filter to MT)
    if 'reading' in df.columns:
        df = df[df['reading'] == 'MT'].copy()
        print(f"✓ Filtered to MT readings: {len(df):,} rows")
    
    # Handle channel format (remove 'ch' prefix if present)
    if df['channel'].dtype == 'object':
        df['channel'] = df['channel'].str.replace('^ch', '', regex=True).astype(int)
    
    # Prepare data
    print(f"\nPreparing actogram data...")
    try:
        act_data = prepare_actogram_data(df, monitor, channel, lights_on, bin_length)
        unique_dates = sorted(act_data['Date'].unique())
        print(f"✓ Prepared data for {len(unique_dates)} days")
        print(f"  Date range: {unique_dates[0]} to {unique_dates[-1]}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Create plot
    print(f"\nGenerating actogram...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plot_actogram(act_data, monitor, channel, output_file)
    
    print(f"\n✅ Actogram generation complete!")
    print()


if __name__ == '__main__':
    main()

