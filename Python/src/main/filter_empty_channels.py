#!/usr/bin/env python3
"""
Filter Empty Channels from DAM Data

This script removes rows from the merged dataset that correspond to empty channels
(channels that don't have actual flies). Empty channels are identified by having
null/empty metadata or no corresponding entry in details.txt.
"""

import pandas as pd
import os
import sys


def filter_empty_channels(input_file, output_file):
    """
    Filter out empty channels (channels without actual flies).
    
    Args:
        input_file (str): Path to dam_data_merged.csv
        output_file (str): Path to save dam_data_with_flies.csv
    """
    print("=" * 60)
    print("=== FILTERING EMPTY CHANNELS ===")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Please run create_database.py first.")
        sys.exit(1)
    
    print(f"\n📂 Loading data from: {os.path.basename(input_file)}")
    
    # Load the merged data
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique channels: {len(df.groupby(['monitor', 'channel']).size())}")
    
    # Identify empty channels
    # Empty channels have:
    # 1. Missing or null fly_id
    # 2. Missing genotype/sex/treatment (all NA/null)
    empty_mask = (
        df['fly_id'].isna() | 
        (df['fly_id'].astype(str).str.strip() == '') |
        df['genotype'].isna() |
        (df['genotype'].astype(str).str.strip().str.upper() == 'NA')
    )
    
    empty_channels = df[empty_mask].groupby(['monitor', 'channel']).size()
    
    print(f"\n🔍 Found {len(empty_channels)} empty channels")
    
    if len(empty_channels) > 0:
        print("\n   Empty channels:")
        for (monitor, channel), count in empty_channels.items():
            print(f"     Monitor {monitor}, Channel {channel}: {count:,} rows")
    
    # Filter to keep only channels with actual flies
    df_filtered = df[~empty_mask].copy()
    
    rows_removed = len(df) - len(df_filtered)
    channels_removed = len(empty_channels)
    channels_remaining = len(df_filtered.groupby(['monitor', 'channel']).size())
    
    # Save filtered data
    print(f"\n💾 Saving filtered data to: {os.path.basename(output_file)}")
    df_filtered.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("=== FILTERING SUMMARY ===")
    print("=" * 60)
    print(f"\nRemoved {rows_removed:,} rows from {channels_removed} empty channels.")
    print(f"Remaining: {len(df_filtered):,} rows from {channels_remaining} channels.")
    
    print(f"\nColumns preserved:")
    for col in df_filtered.columns:
        print(f"  - {col}")
    
    # Verify LIKELY_DEAD column is preserved
    if 'LIKELY_DEAD' in df_filtered.columns:
        print(f"\n✓ LIKELY_DEAD column preserved")
        true_count = df_filtered['LIKELY_DEAD'].sum()
        print(f"  Rows with LIKELY_DEAD=True: {true_count:,}")
    
    print(f"\n✅ Empty channel filtering complete!")
    
    return df_filtered


def main():
    """
    Main function to filter empty channels.
    """
    input_file = '../../data/processed/dam_data_merged.csv'
    output_file = '../../data/processed/dam_data_with_flies.csv'
    
    # Ensure output directory exists
    os.makedirs('../../data/processed', exist_ok=True)
    
    filter_empty_channels(input_file, output_file)


if __name__ == "__main__":
    main()

