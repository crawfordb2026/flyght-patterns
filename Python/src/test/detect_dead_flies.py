#!/usr/bin/env python3
"""
Dead Fly Detection and Filtering Script

This script detects flies that have been inactive for 24+ consecutive hours
and filters them out of the dataset to prevent analysis bias.

DEAD FLY DEFINITION:
A fly is considered dead after 24 consecutive hours (1440 minutes) of inactivity.
Inactivity = MT (movement) value of 0 for 1440+ consecutive minutes.

The script uses only MT (movement) readings for death detection as CT and Pn
are not reliable indicators when flies are dead.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys


def detect_death(fly_data):
    """
    Detect if a fly died during the experiment based on 24+ hours of inactivity.
    
    Args:
        fly_data (pd.DataFrame): Data for ONE fly (all timestamps, all reading types)
                                Must have columns: datetime, reading, value, fly_id
        
    Returns:
        dict or None: Death information if detected, None if fly survived
        Format: {'fly_id': str, 'time_of_death': datetime, 'hours_survived': float}
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
    mt_data['is_inactive'] = (mt_data['value'] == 0)
    
    # Create groups of consecutive inactive periods
    mt_data['inactive_group'] = (mt_data['is_inactive'] != mt_data['is_inactive'].shift()).cumsum()
    
    # Calculate duration of each inactive period
    inactive_periods = []
    
    for group_id in mt_data['inactive_group'].unique():
        group_data = mt_data[mt_data['inactive_group'] == group_id]
        
        if group_data['is_inactive'].iloc[0]:  # Only process inactive groups
            start_time = group_data['datetime'].min()
            end_time = group_data['datetime'].max()
            
            # Calculate duration in minutes
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            # Add 1 minute to account for the minute itself (if MT=0 at 11:46, that minute counts)
            duration_minutes += 1
            
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
        hours_survived = (time_of_death - experiment_start).total_seconds() / 3600
        
        return {
            'fly_id': fly_id,
            'time_of_death': time_of_death,
            'hours_survived': round(hours_survived, 1)
        }
    
    return None


def create_death_report(dam_data):
    """
    Create a report of all flies that died during the experiment.
    
    Args:
        dam_data (pd.DataFrame): Full dam_data_merged.csv
        
    Returns:
        pd.DataFrame: Death report with columns:
            fly_id, genotype, sex, treatment, time_of_death, hours_survived, total_recording_hours
    """
    print("🔍 Analyzing flies for death detection...")
    
    # Get experiment start and end times
    experiment_start = dam_data['datetime'].min()
    experiment_end = dam_data['datetime'].max()
    total_recording_hours = (experiment_end - experiment_start).total_seconds() / 3600
    
    print(f"   Experiment duration: {total_recording_hours:.1f} hours")
    print(f"   Analyzing {dam_data['fly_id'].nunique()} flies...")
    
    death_reports = []
    
    # Group by fly_id and analyze each fly
    for fly_id, fly_data in dam_data.groupby('fly_id'):
        print(f"   Analyzing {fly_id}...", end=' ')
        
        death_info = detect_death(fly_data)
        
        if death_info:
            # Get fly metadata
            fly_metadata = fly_data.iloc[0]
            
            death_reports.append({
                'fly_id': fly_id,
                'genotype': fly_metadata['genotype'],
                'sex': fly_metadata['sex'],
                'treatment': fly_metadata['treatment'],
                'time_of_death': death_info['time_of_death'],
                'hours_survived': death_info['hours_survived'],
                'total_recording_hours': round(total_recording_hours, 1)
            })
            print(f"DEAD at {death_info['time_of_death']} ({death_info['hours_survived']:.1f}h)")
        else:
            print("ALIVE")
    
    # Create DataFrame and sort by time of death
    if death_reports:
        death_df = pd.DataFrame(death_reports)
        death_df = death_df.sort_values('time_of_death').reset_index(drop=True)
    else:
        # Create empty DataFrame with correct columns
        death_df = pd.DataFrame(columns=[
            'fly_id', 'genotype', 'sex', 'treatment', 
            'time_of_death', 'hours_survived', 'total_recording_hours'
        ])
    
    print(f"✅ Death analysis complete: {len(death_df)} flies died")
    
    return death_df


def filter_dead_flies(dam_data, death_report):
    """
    Filter out ALL data from dead flies (both before and after death time).
    
    Args:
        dam_data (pd.DataFrame): Full dam_data_merged.csv
        death_report (pd.DataFrame): Death report from create_death_report
        
    Returns:
        pd.DataFrame: Filtered data with ALL dead fly data removed
    """
    print("🔧 Filtering out ALL data from dead flies...")
    
    if len(death_report) == 0:
        print("   No dead flies to filter - returning original data")
        return dam_data.copy()
    
    # Create a copy to avoid modifying original data
    filtered_data = dam_data.copy()
    
    # For each dead fly, remove ALL data (before and after death)
    for _, death_row in death_report.iterrows():
        fly_id = death_row['fly_id']
        time_of_death = death_row['time_of_death']
        
        # Count rows to be removed
        rows_before = len(filtered_data[filtered_data['fly_id'] == fly_id])
        
        # Remove ALL data for this fly (both before and after death)
        mask = (filtered_data['fly_id'] == fly_id)
        filtered_data = filtered_data[~mask]
        
        rows_after = len(filtered_data[filtered_data['fly_id'] == fly_id])
        rows_removed = rows_before - rows_after
        
        print(f"   {fly_id}: Removed ALL {rows_removed} rows (died at {time_of_death})")
    
    print(f"✅ Filtering complete")
    
    return filtered_data


def main():
    """
    Main function to detect dead flies and create filtered dataset.
    """
    print("🪰 Dead Fly Detection and Filtering")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs('../../data/processed', exist_ok=True)
    
    # Load the merged data
    print("\n📂 Loading data...")
    data_path = '../../data/processed/dam_data_merged.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found!")
        print("   Please run create_database.py first to generate the merged data.")
        sys.exit(1)
    
    dam_data = pd.read_csv(data_path)
    dam_data['datetime'] = pd.to_datetime(dam_data['datetime'])
    
    print(f"   Loaded {len(dam_data):,} rows for {dam_data['fly_id'].nunique()} flies")
    
    # Create death report
    print("\n🔍 Creating death report...")
    death_report = create_death_report(dam_data)
    
    # Save death report
    death_report_path = '../../data/processed/dead_flies_report.csv'
    death_report.to_csv(death_report_path, index=False)
    print(f"💾 Saved death report to {death_report_path}")
    
    # Filter dead flies
    print("\n🔧 Filtering dead flies from dataset...")
    filtered_data = filter_dead_flies(dam_data, death_report)
    
    # Save filtered data
    filtered_path = '../../data/processed/dam_data_filtered.csv'
    filtered_data.to_csv(filtered_path, index=False)
    print(f"💾 Saved filtered data to {filtered_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("=== DEAD FLY DETECTION SUMMARY ===")
    print("=" * 50)
    
    total_flies = dam_data['fly_id'].nunique()
    dead_flies = len(death_report)
    alive_flies = total_flies - dead_flies
    
    print(f"\nTotal flies analyzed: {total_flies}")
    print(f"Dead flies detected: {dead_flies}")
    print(f"Alive flies: {alive_flies}")
    
    if dead_flies > 0:
        print(f"\nDead Flies Report:")
        print("| fly_id   | genotype | time_of_death       | hours_survived |")
        print("|----------|----------|---------------------|----------------|")
        
        for _, row in death_report.iterrows():
            print(f"| {row['fly_id']:<8} | {row['genotype']:<8} | {row['time_of_death']} | {row['hours_survived']:>14.1f} |")
        
        # Find earliest and latest deaths
        earliest_death = death_report.loc[death_report['time_of_death'].idxmin()]
        latest_death = death_report.loc[death_report['time_of_death'].idxmax()]
        
        print(f"\nEarliest death: {earliest_death['fly_id']} at {earliest_death['time_of_death']}")
        print(f"Latest death: {latest_death['fly_id']} at {latest_death['time_of_death']}")
    
    # Dataset statistics
    rows_before = len(dam_data)
    rows_after = len(filtered_data)
    rows_removed = rows_before - rows_after
    
    print(f"\nFiltered dataset:")
    print(f"- Rows before filtering: {rows_before:,}")
    print(f"- Rows after filtering: {rows_after:,}")
    print(f"- Rows removed: {rows_removed:,}")
    
    print(f"\nFiles saved:")
    print(f"✓ {death_report_path}")
    print(f"✓ {filtered_path}")
    
    print(f"\n✅ Dead fly detection and filtering complete!")


if __name__ == "__main__":
    main()
