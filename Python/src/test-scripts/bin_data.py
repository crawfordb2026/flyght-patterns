#!/usr/bin/env python3
"""
Bin DAM Data into Time Intervals

This script bins DAM data (MT, CT, or Pn) into specified time intervals by summing values.
Works with any of the split reading type files and automatically detects the reading type
from the filename to create appropriately named output files.

Usage:
    python src/bin_data.py <input_file> [interval_minutes]
    
Examples:
    python src/bin_data.py data/processed/dam_data_MT.csv 5
    python src/bin_data.py data/processed/dam_data_CT.csv 30
    python src/bin_data.py data/processed/dam_data_Pn.csv
"""

import pandas as pd
import os
import re
import sys
from datetime import datetime


def extract_reading_type(filepath):
    """
    Extract reading type from filename using regex.
    
    Args:
        filepath (str): Path to input file
        
    Returns:
        str: Reading type (MT, CT, or Pn)
        
    Raises:
        ValueError: If reading type cannot be extracted from filename
    """
    basename = os.path.basename(filepath)
    match = re.search(r'dam_data_([A-Z][A-Za-z]+)\.csv', basename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract reading type from: {basename}\n"
                        f"Filename must match pattern: dam_data_XX.csv\n"
                        f"Examples: dam_data_MT.csv, dam_data_CT.csv, dam_data_Pn.csv")


def create_output_filename(input_filepath, interval_minutes, output_dir):
    """
    Create output filename based on input filename and bin size.
    
    Args:
        input_filepath (str): Path to input file
        interval_minutes (int): Bin size in minutes
        output_dir (str): Output directory
        
    Returns:
        str: Full path to output file
    """
    reading_type = extract_reading_type(input_filepath)
    output_filename = f'dam_data_{reading_type}_{interval_minutes}min.csv'
    return os.path.join(output_dir, output_filename)


def bin_to_intervals(df, interval_minutes=5):
    """
    Bin data to specified time intervals by summing values.
    
    Args:
        df (pd.DataFrame): Input data with columns [datetime, monitor, channel, value, 
                          fly_id, genotype, sex, treatment]
        interval_minutes (int): Bin size in minutes
        
    Returns:
        pd.DataFrame: Binned data with summed values
    """
    df = df.copy()
    
    # Convert datetime to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Round down to nearest interval (floor)
    # Example: 11:47:00 with 5min bins → 11:45:00
    df['datetime_binned'] = df['datetime'].dt.floor(f'{interval_minutes}min')
    
    # Handle NaN values in metadata columns by filling with 'Unknown'
    # This prevents flies from being dropped during grouping
    metadata_cols = ['genotype', 'sex', 'treatment']
    for col in metadata_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    # Group by all metadata columns and sum the values
    binned = df.groupby(
        ['datetime_binned', 'monitor', 'channel', 'fly_id', 'genotype', 'sex', 'treatment'],
        as_index=False
    )['value'].sum()
    
    # Rename back to datetime for consistency
    binned.rename(columns={'datetime_binned': 'datetime'}, inplace=True)
    
    # Sort by datetime, monitor, channel for better organization
    binned = binned.sort_values(['datetime', 'monitor', 'channel']).reset_index(drop=True)
    
    return binned


def validate_input_data(df):
    """
    Validate that input DataFrame has required columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ['datetime', 'monitor', 'channel', 'value', 'fly_id', 'genotype', 'sex', 'treatment']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}\n"
                        f"Required columns: {required_columns}")


def print_binning_example(original_df, binned_df, fly_id, interval_minutes):
    """
    Print an example of how binning works for a specific fly.
    
    Args:
        original_df (pd.DataFrame): Original 1-minute data
        binned_df (pd.DataFrame): Binned data
        fly_id (str): Fly ID to use for example
        interval_minutes (int): Bin size in minutes
    """
    # Find a good example time period
    fly_original = original_df[original_df['fly_id'] == fly_id].copy()
    fly_original['datetime'] = pd.to_datetime(fly_original['datetime'])
    fly_original = fly_original.sort_values('datetime')
    
    if len(fly_original) == 0:
        return
    
    # Find first 5-minute period with data
    start_time = fly_original['datetime'].iloc[0]
    bin_start = start_time.floor(f'{interval_minutes}min')
    bin_end = bin_start + pd.Timedelta(minutes=interval_minutes)
    
    # Get original data in this bin
    bin_original = fly_original[
        (fly_original['datetime'] >= bin_start) & 
        (fly_original['datetime'] < bin_end)
    ]
    
    # Get binned data
    bin_binned = binned_df[
        (binned_df['fly_id'] == fly_id) & 
        (binned_df['datetime'] == bin_start)
    ]
    
    if len(bin_original) > 0 and len(bin_binned) > 0:
        print(f"\nExample binning for fly {fly_id} at {bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}:")
        print(f"  Before ({interval_minutes}-min data):")
        
        for _, row in bin_original.head(5).iterrows():
            print(f"    {row['datetime'].strftime('%H:%M:%S')}  value={row['value']}")
        
        if len(bin_original) > 5:
            print(f"    ... ({len(bin_original)-5} more rows)")
        
        total_original = bin_original['value'].sum()
        print(f"    Total: {total_original}")
        
        print(f"\n  After ({interval_minutes}-min bin):")
        print(f"    {bin_start.strftime('%H:%M:%S')}  value={bin_binned['value'].iloc[0]}")


def main(input_file, interval_minutes=5, output_dir='../../data/processed'):
    """
    Main function to load, bin, and save data.
    
    Args:
        input_file (str): Path to input CSV file
        interval_minutes (int): Bin size in minutes
        output_dir (str): Output directory
    """
    print("=" * 60)
    print("=== BINNING DATA TO TIME INTERVALS ===")
    print("=" * 60)
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return
    
    # Validate interval
    if interval_minutes <= 0:
        print(f"❌ Error: Interval must be positive integer, got: {interval_minutes}")
        return
    
    try:
        # Extract reading type from filename
        reading_type = extract_reading_type(input_file)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"\nInput file: {os.path.basename(input_file)}")
    print(f"Reading type detected: {reading_type}")
    print(f"Bin size: {interval_minutes} minutes")
    
    # Load input data
    print(f"\n📂 Loading input data...")
    try:
        original_df = pd.read_csv(input_file)
        original_df['datetime'] = pd.to_datetime(original_df['datetime'])
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Validate input data
    try:
        validate_input_data(original_df)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return
    
    # Print input data summary
    print(f"\nInput data:")
    print(f"- Total rows: {len(original_df):,}")
    print(f"- Date range: {original_df['datetime'].min()} to {original_df['datetime'].max()}")
    print(f"- Unique flies: {original_df['fly_id'].nunique()}")
    print(f"- Time resolution: 1 minute")
    print(f"- Average rows per fly: {len(original_df) // original_df['fly_id'].nunique():,}")
    
    # Bin the data
    print(f"\n🔧 Binning data...")
    try:
        binned_df = bin_to_intervals(original_df, interval_minutes)
    except Exception as e:
        print(f"❌ Error during binning: {e}")
        return
    
    # Create output filename
    output_file = create_output_filename(input_file, interval_minutes, output_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binned data
    print(f"💾 Saving binned data...")
    try:
        binned_df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return
    
    # Calculate file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    
    # Print output data summary
    print(f"\nOutput data:")
    print(f"- Total rows: {len(binned_df):,}")
    print(f"- Date range: {binned_df['datetime'].min()} to {binned_df['datetime'].max()}")
    print(f"- Unique flies: {binned_df['fly_id'].nunique()}")
    print(f"- Time resolution: {interval_minutes} minutes")
    print(f"- Average rows per fly: {len(binned_df) // binned_df['fly_id'].nunique():,}")
    
    # Calculate reduction percentage
    reduction_pct = ((len(original_df) - len(binned_df)) / len(original_df)) * 100
    print(f"- Row reduction: {reduction_pct:.1f}%")
    
    # Print binning example
    if len(original_df) > 0:
        example_fly = original_df['fly_id'].iloc[0]
        print_binning_example(original_df, binned_df, example_fly, interval_minutes)
    
    print(f"\nOutput saved: {os.path.basename(output_file)}")
    print(f"File size: ~{file_size:.1f} MB")
    
    # Validation checks
    print(f"\n🔍 Running validation checks...")
    
    # Verify all flies are present
    original_flies = set(original_df['fly_id'].unique())
    binned_flies = set(binned_df['fly_id'].unique())
    
    if original_flies == binned_flies:
        print("✓ All flies present in binned data")
    else:
        print("❌ Error: Some flies missing from binned data!")
        missing = original_flies - binned_flies
        if missing:
            print(f"  Missing flies: {missing}")
        return
    
    # Verify no data loss (within rounding)
    try:
        original_totals = original_df.groupby('fly_id')['value'].sum().sort_index()
        binned_totals = binned_df.groupby('fly_id')['value'].sum().sort_index()
        
        if original_totals.equals(binned_totals):
            print("✓ No data loss: Value totals match exactly")
        else:
            print("⚠️  Warning: Value totals don't match exactly (may be due to rounding)")
            # Show a few examples of differences
            diff = (original_totals - binned_totals).abs()
            if diff.max() > 0:
                print(f"  Max difference: {diff.max()}")
    except Exception as e:
        print(f"⚠️  Could not verify value totals: {e}")
    
    print(f"\n✅ Binning complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/bin_data.py <input_file> [interval_minutes]")
        print("\nExamples:")
        print("  python src/bin_data.py data/processed/dam_data_MT.csv 5")
        print("  python src/bin_data.py data/processed/dam_data_CT.csv 30")
        print("  python src/bin_data.py data/processed/dam_data_Pn.csv")
        print("\nDefault interval: 5 minutes")
        sys.exit(1)
    
    input_file = sys.argv[1]
    interval_minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    print(f"Processing: {input_file}")
    print(f"Interval: {interval_minutes} minutes\n")
    
    main(input_file, interval_minutes)
