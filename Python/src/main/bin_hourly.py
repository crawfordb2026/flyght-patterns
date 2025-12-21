#!/usr/bin/env python3
"""
Bin DAM Data to Hourly Intervals

This script bins raw DAM data (from split reading type files) into hourly intervals.

Usage:
    python bin_hourly.py <input_file> [<input_file2> ...]
    
Examples:
    python bin_hourly.py dam_data_MT.csv
    python bin_hourly.py dam_data_MT.csv dam_data_CT.csv dam_data_Pn.csv
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta


def bin_to_hourly(df):
    """
    Bin raw DAM data to hourly intervals.
    
    Args:
        df (pd.DataFrame): Input data with columns including datetime, value, etc.
        
    Returns:
        pd.DataFrame: Hourly binned data with one row per fly per hour
    """
    df = df.copy()
    
    # Convert datetime to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Store the column that contains activity count (could be 'value' or 'activity_count')
    activity_col = None
    for col in ['activity_count', 'value']:
        if col in df.columns:
            activity_col = col
            break
    
    if activity_col is None:
        raise ValueError("No activity count column found (expected 'activity_count' or 'value')")
    
    # Identify metadata columns (those that should be constant per fly)
    # These are: monitor, channel, fly_id, genotype, sex, treatment, reading
    metadata_cols = ['monitor', 'channel', 'fly_id', 'genotype', 'sex', 'treatment']
    if 'reading' in df.columns:
        metadata_cols.append('reading')
    
    # Keep only columns that exist
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    
    # Floor datetime to hour
    df['datetime_hour'] = df['datetime'].dt.floor('H')
    
    # Group by hour and fly metadata
    # Aggregate: sum value
    agg_dict = {
        activity_col: 'sum'  # Sum activity values
    }
    
   
    
    # Group by datetime_hour and all metadata columns
    binned = df.groupby(['datetime_hour'] + metadata_cols, as_index=False).agg(agg_dict)
    
    # Rename datetime_hour back to datetime
    binned.rename(columns={'datetime_hour': 'datetime'}, inplace=True)
    
    return binned


def fill_missing_hours(df, min_hour, max_hour):
    """
    Fill in missing hours for each fly to ensure complete hourly coverage.
    
    Args:
        df (pd.DataFrame): Binned data with datetime floored to hours
        min_hour (pd.Timestamp): Earliest hour in dataset
        max_hour (pd.Timestamp): Latest hour in dataset
        
    Returns:
        pd.DataFrame: Data with all missing hours filled in
    """
    # Create complete hourly range
    all_hours = pd.date_range(start=min_hour, end=max_hour, freq='H')
    
    # Get unique flies
    metadata_cols = [col for col in ['monitor', 'channel', 'fly_id', 'genotype', 'sex', 'treatment', 'reading'] 
                     if col in df.columns]
    
    unique_flys = df[metadata_cols].drop_duplicates()
    
    # Create complete index (all hours × all flies)
    complete_index = []
    for hour in all_hours:
        for _, fly in unique_flys.iterrows():
            row = {'datetime': hour}
            row.update(fly.to_dict())
            complete_index.append(row)
    
    complete_df = pd.DataFrame(complete_index)
    
    # Merge with binned data
    # For hours with no data, value will be NaN - fill with 0
    activity_col = 'value' if 'value' in df.columns else 'activity_count'
    
    merged = complete_df.merge(df, on=['datetime'] + metadata_cols, how='left', suffixes=('', '_y'))
    
    # Fill NaN values in activity count with 0
    if activity_col + '_y' in merged.columns:
        merged[activity_col] = merged[activity_col + '_y'].fillna(0)
        merged = merged.drop(columns=[activity_col + '_y'])
    else:
        merged[activity_col] = 0
    
    return merged


def process_file(input_file):
    """
    Process a single file to create hourly binned output.
    
    Args:
        input_file (str): Path to input file
        
    Returns:
        str: Path to output file
    """
    print("\n" + "=" * 60)
    print("=== BINNING TO HOURLY INTERVALS ===")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        return None
    
    # Determine reading type from filename
    basename = os.path.basename(input_file)
    reading_type = None
    
    if 'MT' in basename:
        reading_type = 'MT'
    elif 'CT' in basename:
        reading_type = 'CT'
    elif 'Pn' in basename:
        reading_type = 'Pn'
    
    print(f"\n📂 Processing: {basename}")
    if reading_type:
        print(f"   Reading type: {reading_type}")
    
    # Load data
    print(f"   Loading data...")
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Store statistics
    input_rows = len(df)
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()
    num_flies = df['fly_id'].nunique()
    
    print(f"   Loaded {input_rows:,} rows")
    print(f"   Date range: {min_date} to {max_date}")
    print(f"   Number of flies: {num_flies}")
    
    # Bin to hourly intervals
    print(f"\n🔧 Binning to hourly intervals...")
    binned = bin_to_hourly(df)
    
    # Fill missing hours
    print(f"   Filling missing hours...")
    binned = fill_missing_hours(binned, min_date.floor('H'), max_date.floor('H'))
    
    # Create output filename
    output_file = input_file.replace('.csv', '_hourly.csv')
    
    # Save output
    print(f"💾 Saving to: {os.path.basename(output_file)}")
    binned.to_csv(output_file, index=False)
    
    # Calculate statistics
    output_rows = len(binned)
    avg_per_hour = input_rows / output_rows if output_rows > 0 else 0
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Input file: {basename}")
    print(f"   Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"   Number of flies: {num_flies}")
    print(f"   Original rows (raw readings): {input_rows:,}")
    print(f"   Output rows (hourly bins): {output_rows:,}")
    print(f"   Average readings per hour: {avg_per_hour:.1f}")
    print(f"   Output file: {os.path.basename(output_file)}")
    
    return output_file


def main():
    """
    Main function to process one or more files.
    """
    if len(sys.argv) < 2:
        print("Usage: python bin_hourly.py <input_file> [<input_file2> ...]")
        print("\nExamples:")
        print("  python bin_hourly.py data/processed/dam_data_MT.csv")
        print("  python bin_hourly.py data/processed/dam_data_MT.csv data/processed/dam_data_CT.csv")
        sys.exit(1)
    
    input_files = sys.argv[1:]
    
    print("=" * 60)
    print("HOURLY BINNING FOR DAM DATA")
    print("=" * 60)
    print(f"\nProcessing {len(input_files)} file(s)...")
    
    output_files = []
    
    for input_file in input_files:
        output_file = process_file(input_file)
        if output_file:
            output_files.append(output_file)
    
    if output_files:
        print(f"\n✅ Successfully processed {len(output_files)} file(s)!")
        print(f"\nOutput files:")
        for output_file in output_files:
            print(f"  - {output_file}")
    else:
        print(f"\n❌ No files were successfully processed.")


if __name__ == "__main__":
    main()

