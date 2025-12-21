#!/usr/bin/env python3
"""
Fly Sleep Behavior Database Creator

This script creates a SINGLE merged table in LONG format containing all data.

Structure: data/processed/dam_data_merged.csv
| datetime | monitor | channel | reading | value | fly_id | genotype | sex | treatment |

Where:
- Each timestamp has 3 ROWS per channel (MT, CT, Pn as separate rows)
- reading column contains: "MT", "CT", or "Pn"
- value column contains the corresponding measurement
- Metadata (fly_id, genotype, sex, treatment) is included in every row

Benefits:
- Single file for all analysis
- Long format ideal for plotting and analysis
- Complete information in every row
- Easy to filter by reading type
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys


def parse_details(filepath):
    """
    Parse details.txt to extract fly metadata.
    
    Args:
        filepath (str): Path to details.txt file
        
    Returns:
        pd.DataFrame: fly_metadata with columns:
            monitor, channel, fly_id, genotype, sex, treatment
    """
    print(f"📋 Parsing metadata from {filepath}...")
    
    # Read the details file
    df = pd.read_csv(filepath, sep='\t')
    
    # Clean up the data
    df['monitor'] = df['Monitor'].astype(int)
    df['channel'] = df['Channel'].str.replace('ch', '').astype(int)
    df['genotype'] = df['Genotype']
    df['sex'] = df['Sex']
    df['treatment'] = df['Treatment']
    
    # Create fly_id: M{monitor}_Ch{channel:02d}
    df['fly_id'] = df.apply(lambda row: f"M{row['monitor']}_Ch{row['channel']:02d}", axis=1)
    
    # Select and reorder columns
    fly_metadata = df[['monitor', 'channel', 'fly_id', 'genotype', 'sex', 'treatment']].copy()
    
    # Remove rows with NA values (empty channels)
    fly_metadata = fly_metadata[fly_metadata['genotype'] != 'NA'].copy()
    
    print(f"✅ Parsed {len(fly_metadata)} flies from metadata")
    print(f"   Monitors: {sorted(fly_metadata['monitor'].unique())}")
    print(f"   Genotypes: {list(fly_metadata['genotype'].unique())}")
    print(f"   Treatments: {list(fly_metadata['treatment'].unique())}")
    
    return fly_metadata


def parse_monitor_file(filepath, monitor_num):
    """
    Parse one Monitor*.txt file to extract time-series data in LONG format.
    
    Creates long format data where each timestamp has 3 rows per channel:
    - One row for MT, one for CT, one for Pn
    
    Args:
        filepath (str): Path to Monitor*.txt file
        monitor_num (int): Monitor number (5 or 6)
        
    Returns:
        pd.DataFrame: time_series_data with columns:
            datetime, monitor, channel, reading, value
    """
    print(f"📊 Parsing time-series data from {filepath} (Monitor {monitor_num})...")
    
    # Read the monitor file
    df = pd.read_csv(filepath, sep='\t', header=None)
    
    # Define column names based on the data structure
    # Columns: ID, date, time, port, [unknowns], movement_type, 0, 0, [32 channel values]
    columns = ['id', 'date', 'time', 'port', 'unknown1', 'unknown2', 'unknown3', 'movement_type', 'zero1', 'zero2']
    # Add 32 channel columns (channels 1-32)
    for i in range(1, 33):
        columns.append(f'ch{i}')
    
    df.columns = columns
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d %b %y %H:%M:%S')
    
    # Filter for the three movement types: MT, CT, Pn
    movement_types = ['MT', 'CT', 'Pn']
    df_filtered = df[df['movement_type'].isin(movement_types)].copy()
    
    print(f"   Found {len(df_filtered)} rows with movement data")
    print(f"   Date range: {df_filtered['datetime'].min()} to {df_filtered['datetime'].max()}")
    print(f"   Movement types: {df_filtered['movement_type'].unique()}")
    
    # Create long format data
    # Each timestamp has 3 rows (MT, CT, Pn), we want one row per reading type per channel
    time_series_list = []
    
    # Group by timestamp (id, date, time)
    for (timestamp_id, date, time), group in df_filtered.groupby(['id', 'date', 'time']):
        datetime_val = group['datetime'].iloc[0]
        
        # Get the three movement types for this timestamp
        mt_data = group[group['movement_type'] == 'MT'].iloc[0] if 'MT' in group['movement_type'].values else None
        ct_data = group[group['movement_type'] == 'CT'].iloc[0] if 'CT' in group['movement_type'].values else None
        pn_data = group[group['movement_type'] == 'Pn'].iloc[0] if 'Pn' in group['movement_type'].values else None
        
        # Extract channel values (columns 10-41, which are ch1-ch32)
        for channel in range(1, 33):
            channel_col = f'ch{channel}'
            
            # Get values for each movement type
            mt_val = mt_data[channel_col] if mt_data is not None else 0
            ct_val = ct_data[channel_col] if ct_data is not None else 0
            pn_val = pn_data[channel_col] if pn_data is not None else 0
            
            # Only include if at least one value is non-zero (active channel)
            if mt_val > 0 or ct_val > 0 or pn_val > 0:
                # Create 3 rows per channel: one for each reading type
                for reading, value in [('MT', mt_val), ('CT', ct_val), ('Pn', pn_val)]:
                    time_series_list.append({
                        'datetime': datetime_val,
                        'monitor': monitor_num,
                        'channel': channel,
                        'reading': reading,
                        'value': int(value)
                    })
    
    time_series_df = pd.DataFrame(time_series_list)
    
    print(f"✅ Created {len(time_series_df)} time-series records for Monitor {monitor_num}")
    print(f"   Channels with data: {list(time_series_df['channel'].unique())}")
    print(f"   Date range: {time_series_df['datetime'].min()} to {time_series_df['datetime'].max()}")
    print(f"   Reading types: {list(time_series_df['reading'].unique())}")
    
    return time_series_df


def main():
    """
    Main function to create the single merged database in LONG format.
    
    Creates one CSV file:
    data/processed/dam_data_merged.csv - Complete merged data with all information
    
    Structure: datetime, monitor, channel, reading, value, fly_id, genotype, sex, treatment
    """
    print("🚀 Starting Fly Sleep Behavior Database Creation")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs('../../data/processed', exist_ok=True)
    
    # PART 1: Parse metadata
    print("\n📋 PART 1: Parsing fly metadata")
    print("-" * 40)
    
    fly_metadata = parse_details('../../details.txt')
    
    # PART 2: Parse time-series data in long format
    print("\n📊 PART 2: Parsing time-series data in long format")
    print("-" * 40)
    
    # Parse both monitor files
    monitor5_data = parse_monitor_file('../../Monitor5.txt', 5)
    monitor6_data = parse_monitor_file('../../Monitor6.txt', 6)
    
    # Combine into single time-series table
    time_series_data = pd.concat([monitor5_data, monitor6_data], ignore_index=True)
    
    # Sort by datetime for better organization
    time_series_data = time_series_data.sort_values(['datetime', 'monitor', 'channel', 'reading']).reset_index(drop=True)
    
    # PART 3: Merge with metadata
    print("\n🔗 PART 3: Merging with fly metadata")
    print("-" * 40)
    
    # Merge time-series data with metadata using (monitor, channel)
    merged_data = time_series_data.merge(fly_metadata, on=['monitor', 'channel'])
    
    # Reorder columns to match desired structure
    final_columns = ['datetime', 'monitor', 'channel', 'reading', 'value', 'fly_id', 'genotype', 'sex', 'treatment']
    merged_data = merged_data[final_columns]
    
    # Save merged data
    merged_path = '../../data/processed/dam_data_merged.csv'
    merged_data.to_csv(merged_path, index=False)
    print(f"💾 Saved merged data to {merged_path}")
    
    # PART 4: Validation and Summary
    print("\n📈 PART 4: Validation and Summary")
    print("-" * 40)
    
    print(f"Total rows: {len(merged_data):,}")
    print(f"Unique timestamps: {merged_data['datetime'].nunique():,}")
    print(f"Unique flies: {merged_data['fly_id'].nunique()}")
    print(f"Date range: {merged_data['datetime'].min()} to {merged_data['datetime'].max()}")
    print(f"File size: {os.path.getsize(merged_path) / (1024*1024):.1f} MB")
    
    print(f"\nRows per reading type:")
    reading_counts = merged_data['reading'].value_counts().sort_index()
    for reading, count in reading_counts.items():
        print(f"  {reading}: {count:,} rows")
    
    print(f"\nUnique flies per monitor:")
    flies_per_monitor = merged_data.groupby('monitor')['fly_id'].nunique()
    for monitor, count in flies_per_monitor.items():
        print(f"  Monitor {monitor}: {count} flies")
    
    # Show example data for one fly at one timestamp
    print(f"\nExample data - All three reading types for one fly at one timestamp:")
    example_fly = merged_data['fly_id'].iloc[0]
    example_timestamp = merged_data['datetime'].iloc[0]
    example_data = merged_data[
        (merged_data['fly_id'] == example_fly) & 
        (merged_data['datetime'] == example_timestamp)
    ].sort_values('reading')
    print(example_data[['datetime', 'monitor', 'channel', 'reading', 'value', 'fly_id', 'genotype', 'sex', 'treatment']])
    
    print(f"\n✅ Database creation complete!")
    print(f"   File created: {merged_path}")
    print(f"\n💡 To use in analysis:")
    print(f"   data = pd.read_csv('{merged_path}')")
    print(f"   # Filter by reading type: data[data['reading'] == 'MT']")
    print(f"   # Filter by genotype: data[data['genotype'] == 'SSS']")
    print(f"   # Group by fly: data.groupby('fly_id')")


if __name__ == "__main__":
    main()
