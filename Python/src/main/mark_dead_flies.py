#!/usr/bin/env python3
"""
Mark Dead Flies in DAM Data

This script identifies flies that have been inactive for 24+ consecutive hours
and marks them as LIKELY_DEAD (sets LIKELY_DEAD=True for all their rows).

Unlike the old detect_dead_flies.py, this script:
- Uses 24-hour threshold (instead of 12 hours)
- DOES NOT remove any rows
- Only marks rows with LIKELY_DEAD flag
- Sets LIKELY_DEAD=True for all rows of a fly once it meets the threshold
  (but leaves earlier time points as False)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys


def mark_fly_as_dead(fly_data):
    """
    Detect if a fly died during the experiment and return the time of death.
    
    Args:
        fly_data (pd.DataFrame): Data for ONE fly (all timestamps, all reading types)
                                Must have columns: datetime, reading, value, fly_id
        
    Returns:
        datetime or None: Time of death if detected, None if fly survived
    """
    # Filter to MT (movement) readings only
    mt_data = fly_data[fly_data['reading'] == 'MT'].copy()
    
    if len(mt_data) == 0:
        print(f"⚠️  Warning: No MT data found for fly {fly_data['fly_id'].iloc[0]}")
        return None
    
    # Sort by datetime
    mt_data = mt_data.sort_values('datetime').reset_index(drop=True)
    
    fly_id = mt_data['fly_id'].iloc[0]
    experiment_start = mt_data['datetime'].min()
    
    # Find consecutive periods of inactivity (MT = 0)
    mt_data['is_zero'] = (mt_data['value'] == 0)
    mt_data['zero_group'] = (mt_data['is_zero'] != mt_data['is_zero'].shift()).cumsum()
    
    # Calculate duration of each inactive period
    inactive_periods = []
    
    for group_id in mt_data['zero_group'].unique():
        group_data = mt_data[mt_data['zero_group'] == group_id]
        
        if group_data['is_zero'].iloc[0]:  # Only process zero groups
            start_time = group_data['datetime'].min()
            end_time = group_data['datetime'].max()
            
            # Calculate duration in minutes
            duration_minutes = (end_time - start_time).total_seconds() / 60 + 1
            
            inactive_periods.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': duration_minutes
            })
    
    # Check if any inactive period lasted 1440+ minutes (24 hours)
    death_periods = [p for p in inactive_periods if p['duration_minutes'] >= 1440]
    
    if death_periods:
        # Use the first death period (earliest 24+ hour inactivity)
        death_period = min(death_periods, key=lambda x: x['start_time'])
        
        time_of_death = death_period['start_time']
        
        return time_of_death
    
    return None


def main():
    """
    Main function to mark dead flies in the dataset.
    """
    print("=" * 60)
    print("=== MARKING DEAD FLIES ===")
    print("=" * 60)
    
    input_file = '../../data/processed/dam_data_with_flies.csv'
    output_file = '../../data/processed/dam_data_marked.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        print("   Please run filter_empty_channels.py first.")
        sys.exit(1)
    
    print(f"\n📂 Loading data from: {os.path.basename(input_file)}")
    
    # Load the data
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"   Loaded {len(df):,} rows for {df['fly_id'].nunique()} flies")
    
    # Ensure LIKELY_DEAD column exists
    if 'LIKELY_DEAD' not in df.columns:
        df['LIKELY_DEAD'] = False
        print("   Added LIKELY_DEAD column (default=False)")
    
    # Get list of unique flies
    unique_flies = df['fly_id'].unique()
    total_flies = len(unique_flies)
    
    print(f"\n🔍 Analyzing {total_flies} flies for death detection...")
    print(f"   Threshold: 24 consecutive hours (1440 minutes) of MT=0")
    
    # Track which flies are marked as dead
    marked_dead = []
    
    # Process each fly
    for idx, fly_id in enumerate(unique_flies):
        if (idx + 1) % 10 == 0:
            print(f"   Processing fly {idx + 1}/{total_flies}...", end='\r')
        
        fly_data = df[df['fly_id'] == fly_id]
        
        # Detect if this fly died
        time_of_death = mark_fly_as_dead(fly_data)
        
        if time_of_death:
            # Mark all rows for this fly from death time onwards
            death_mask = (df['fly_id'] == fly_id) & (df['datetime'] >= time_of_death)
            df.loc[death_mask, 'LIKELY_DEAD'] = True
            
            # Count rows marked
            rows_marked = death_mask.sum()
            
            marked_dead.append({
                'fly_id': fly_id,
                'time_of_death': time_of_death,
                'rows_marked': rows_marked
            })
    
    print()  # New line after progress indicator
    
    # Save marked data
    print(f"\n💾 Saving marked data to: {os.path.basename(output_file)}")
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("=== MARKING SUMMARY ===")
    print("=" * 60)
    
    alive_flies = total_flies - len(marked_dead)
    
    print(f"\nTotal flies: {total_flies}")
    print(f"Marked as likely dead: {len(marked_dead)}")
    print(f"Flies remaining alive: {alive_flies}")
    
    if marked_dead:
        print(f"\nFlies marked as likely dead:")
        for fly_info in marked_dead:
            hours_survived = (fly_info['time_of_death'] - df['datetime'].min()).total_seconds() / 3600
            print(f"  {fly_info['fly_id']}: died at {fly_info['time_of_death']} ({hours_survived:.1f}h)")
            print(f"    Marked {fly_info['rows_marked']:,} rows as LIKELY_DEAD=True")
    
    # Statistics about marked rows
    rows_marked_total = df['LIKELY_DEAD'].sum()
    rows_alive_total = (~df['LIKELY_DEAD']).sum()
    
    print(f"\nRow statistics:")
    print(f"  Rows with LIKELY_DEAD=True: {rows_marked_total:,}")
    print(f"  Rows with LIKELY_DEAD=False: {rows_alive_total:,}")
    print(f"  Total rows: {len(df):,}")
    
    # Verify no data loss
    if len(df) == len(pd.read_csv(input_file)):
        print(f"\n✓ Verification passed: No data loss (same number of rows)")
    else:
        print(f"\n⚠️  Warning: Row count changed during marking!")
    
    print(f"\n✅ Dead fly marking complete!")
    print(f"   Files:")
    print(f"   - Input:  {os.path.basename(input_file)}")
    print(f"   - Output: {os.path.basename(output_file)}")


if __name__ == "__main__":
    main()

