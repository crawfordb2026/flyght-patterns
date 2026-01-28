#!/usr/bin/env python3
"""
Fly Health Report Generator (MT-based)

This script generates a health report for flies based on MT (movement) data,
classifying each fly as Alive, Unhealthy, Dead, or QC_Fail.

The script can read from either:
- dam_data_MT.csv (MT-only data, no "reading" column) - preferred for efficiency
- dam_data_marked.csv or dam_data_merged.csv (all readings, has "reading" column)

Usage:
    python generate_health_report.py [input_file] [output_file]
    
Examples:
    python generate_health_report.py
    python generate_health_report.py ../../data/processed/dam_data_MT.csv
    python generate_health_report.py ../../data/processed/dam_data_marked.csv health_report.csv
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path


# --- Settings ---
lights_on = 9
lights_off = 21
bin_length_min = 1
exclude_days = [1, 7]
ref_day = 4
decline_threshold = 0.5
death_threshold = 0.2
transition_window = 10  # minutes

# --- Thresholds ---
THRESHOLDS = {
    "A1": 12 * 60 / bin_length_min,  # 720 (12 hours in bins)
    "A2": 24 * 60 / bin_length_min,  # 1440 (24 hours in bins)
    "ACTIVITY_LOW": 50,
    "INDEX_LOW": 0.02,
    "SLEEP_MAX": 1300,
    "SLEEP_BOUT": 720,
    "MISSING_MAX": 0.10
}


def rle(seq):
    """
    Run-length encoding.
    
    Returns:
        tuple: (values, lengths) where values are the unique values and
               lengths are the run lengths
    """
    if len(seq) == 0:
        return np.array([]), np.array([])
    
    # Find where values change
    changes = np.diff(seq) != 0
    change_indices = np.where(changes)[0] + 1
    
    # Add start and end
    indices = np.concatenate(([0], change_indices, [len(seq)]))
    
    # Calculate run lengths
    lengths = np.diff(indices)
    
    # Get values at start of each run
    values = seq[indices[:-1]]
    
    return values, lengths


def longest_zero_run(counts, bin_length_min):
    """
    Calculate longest consecutive zero-activity period using run-length encoding.
    
    Matches R logic: rle(COUNTS > 0), then max(runs$lengths[!runs$values])
    
    Args:
        counts: Series or array of activity counts
        bin_length_min: Length of each bin in minutes
        
    Returns:
        Longest zero run in minutes
    """
    if len(counts) == 0:
        return 0
    
    # Create boolean: True if counts > 0, False if counts == 0
    has_activity = (counts > 0).astype(int)
    
    # Run-length encoding
    values, lengths = rle(has_activity)
    
    # Find runs where has_activity is False (i.e., counts == 0)
    zero_runs = lengths[values == 0]
    
    if len(zero_runs) == 0:
        return 0
    
    return max(zero_runs) * bin_length_min


def total_sleep_minutes(counts, bin_length_min):
    """
    Calculate total sleep minutes (immobility ≥5 min bouts) using run-length encoding.
    
    Matches R logic: rle(COUNTS > 0), then sum(runs$lengths[!runs$values & runs$lengths >= 5])
    
    Args:
        counts: Series or array of activity counts
        bin_length_min: Length of each bin in minutes
        
    Returns:
        Total sleep minutes
    """
    if len(counts) == 0:
        return 0
    
    # Create boolean: True if counts > 0, False if counts == 0
    has_activity = (counts > 0).astype(int)
    
    # Run-length encoding
    values, lengths = rle(has_activity)
    
    # Find runs where has_activity is False (i.e., counts == 0) AND length >= 5
    sleep_runs = lengths[(values == 0) & (lengths >= 5)]
    
    return sum(sleep_runs) * bin_length_min


def is_ok(x):
    """
    Check if a value is valid (not NA, not empty, not "na").
    
    Matches R logic: !is.na(x) & x != "" & tolower(as.character(x)) != "na"
    """
    if pd.isna(x):
        return False
    x_str = str(x).lower()
    return x_str != "" and x_str != "na"


def calculate_exp_day(df):
    """
    Calculate Exp_Day from datetime if it doesn't exist.
    Day 1 = first date in dataset.
    
    Args:
        df: DataFrame with datetime column
        
    Returns:
        Series with Exp_Day values
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Date'] = df['datetime'].dt.date
    
    # Get unique dates sorted
    unique_dates = sorted(df['Date'].unique())
    date_to_day = {date: day for day, date in enumerate(unique_dates, start=1)}
    
    return df['Date'].map(date_to_day)


def prep_data(df):
    """
    Prepare and filter data for health report analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Prepared DataFrame with MT data only
    """
    df = df.copy()
    
    # Auto-detect format: check if "reading" column exists
    if 'reading' in df.columns:
        # Filter to MT only
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
        print(f"✓ Filtered to MT readings only")
    else:
        print(f"✓ Data is already MT-only")
    
    # Ensure datetime is datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Calculate Exp_Day if missing
    if 'Exp_Day' not in df.columns:
        print("✓ Calculating Exp_Day from datetime...")
        df['Exp_Day'] = calculate_exp_day(df)
    else:
        print("✓ Exp_Day column found")
    
    # Exclude specified days
    if 'Exp_Day' in df.columns:
        before = len(df)
        df = df[~df['Exp_Day'].isin(exclude_days)].copy()
        after = len(df)
        print(f"✓ Excluded days {exclude_days}: {before - after} rows removed")
    
    # Clean channel names: remove "ch" prefix if present, convert to integer
    if df['channel'].dtype == 'object':
        df['channel'] = df['channel'].str.replace('^ch', '', regex=True)
    df['channel'] = df['channel'].astype(int)
    
    # Rename value to COUNTS
    df = df.rename(columns={'value': 'COUNTS'})
    
    # Standardize column names (handle case differences)
    column_mapping = {
        'monitor': 'Monitor',
        'channel': 'Channel',
        'genotype': 'Genotype',
        'sex': 'Sex',
        'treatment': 'Treatment'
    }
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    return df


def calculate_daily_metrics(dam_activity):
    """
    Calculate daily metrics per fly per day.
    
    Args:
        dam_activity: Prepared DataFrame with MT data
        
    Returns:
        DataFrame with daily summary metrics
    """
    # Add Date column
    dam_activity = dam_activity.copy()
    dam_activity['Date'] = dam_activity['datetime'].dt.date
    
    # Group by Monitor, Channel, Date, Exp_Day, Genotype, Sex, Treatment
    group_cols = ['Monitor', 'Channel', 'Date', 'Exp_Day', 'Genotype', 'Sex', 'Treatment']
    group_cols = [col for col in group_cols if col in dam_activity.columns]
    
    def calc_metrics(group):
        return pd.Series({
            'TOTAL_ACTIVITY': group['COUNTS'].sum(skipna=True),
            'ACTIVITY_INDEX': (group['COUNTS'] > 0).mean(),
            'LONGEST_ZERO': longest_zero_run(group['COUNTS'], bin_length_min),
            'TOTAL_SLEEP': total_sleep_minutes(group['COUNTS'], bin_length_min),
            'MISSING_FRAC': group['COUNTS'].isna().mean()
        })
    
    daily_summary = dam_activity.groupby(group_cols, dropna=False).apply(calc_metrics).reset_index()
    
    return daily_summary


def normalize_to_ref_day(daily_summary, ref_day):
    """
    Normalize activity to reference day.
    
    Args:
        daily_summary: DataFrame with daily metrics
        ref_day: Reference day number
        
    Returns:
        DataFrame with REF_ACTIVITY, REL_ACTIVITY, and DECLINE_STATUS
    """
    daily_summary = daily_summary.copy()
    
    # Get reference activity per fly (one value per Monitor-Channel)
    ref_activity = daily_summary[daily_summary['Exp_Day'] == ref_day][
        ['Monitor', 'Channel', 'TOTAL_ACTIVITY']
    ].rename(columns={'TOTAL_ACTIVITY': 'REF_ACTIVITY'})
    
    # Merge reference activity back to all rows
    daily_summary = daily_summary.merge(
        ref_activity, on=['Monitor', 'Channel'], how='left'
    )
    
    # Calculate relative activity
    daily_summary['REL_ACTIVITY'] = daily_summary['TOTAL_ACTIVITY'] / daily_summary['REF_ACTIVITY']
    
    # Classify decline status
    def classify_decline(row):
        if pd.isna(row['REF_ACTIVITY']):
            return "No Reference"
        if row['REL_ACTIVITY'] < death_threshold:
            return "Dead (by decline)"
        if row['REL_ACTIVITY'] < decline_threshold:
            return "Unhealthy (by decline)"
        return "Stable"
    
    daily_summary['DECLINE_STATUS'] = daily_summary.apply(classify_decline, axis=1)
    
    return daily_summary


def startle_test(dam_activity, lights_on, lights_off, transition_window):
    """
    Test for startle response at light transitions.
    
    Args:
        dam_activity: Prepared DataFrame with MT data
        lights_on: Hour when lights turn on
        lights_off: Hour when lights turn off
        transition_window: Window in minutes around transitions
        
    Returns:
        DataFrame with TRANSITION_COUNTS and NO_STARTLE per fly per day
    """
    dam_activity = dam_activity.copy()
    
    # Calculate hour of day
    dam_activity['HOUR'] = dam_activity['datetime'].dt.hour + dam_activity['datetime'].dt.minute / 60
    
    # Identify transition windows
    dam_activity['IS_TRANSITION'] = (
        (np.abs(dam_activity['HOUR'] - lights_on) <= transition_window / 60) |
        (np.abs(dam_activity['HOUR'] - lights_off) <= transition_window / 60)
    )
    
    # Group by Monitor, Channel, Date
    dam_activity['Date'] = dam_activity['datetime'].dt.date
    
    # Calculate transition counts per fly per day
    def calc_transition_counts(group):
        transition_rows = group[group['IS_TRANSITION']]
        return pd.Series({
            'Date': group['Date'].iloc[0],
            'TRANSITION_COUNTS': transition_rows['COUNTS'].sum(skipna=True)
        })
    
    transition_data = dam_activity.groupby(['Monitor', 'Channel', 'Date'], group_keys=False).apply(
        calc_transition_counts
    ).reset_index()
    
    transition_data['NO_STARTLE'] = transition_data['TRANSITION_COUNTS'] == 0
    
    return transition_data


def classify_status(daily_summary, transition_data):
    """
    Classify fly status using decision tree.
    
    Args:
        daily_summary: DataFrame with daily metrics and decline status
        transition_data: DataFrame with startle test results
        
    Returns:
        DataFrame with STATUS column
    """
    # Merge with transition data
    fly_status = daily_summary.merge(
        transition_data, on=['Monitor', 'Channel', 'Date'], how='left'
    )
    
    # Fill missing NO_STARTLE with False (conservative)
    fly_status['NO_STARTLE'] = fly_status['NO_STARTLE'].fillna(False)
    
    # Calculate flags
    fly_status['FLAG_A2'] = fly_status['LONGEST_ZERO'] >= THRESHOLDS['A2']
    fly_status['FLAG_A1'] = fly_status['LONGEST_ZERO'] >= THRESHOLDS['A1']
    fly_status['FLAG_LOW_ACTIVITY'] = (
        (fly_status['TOTAL_ACTIVITY'] <= THRESHOLDS['ACTIVITY_LOW']) |
        (fly_status['ACTIVITY_INDEX'] <= THRESHOLDS['INDEX_LOW'])
    )
    fly_status['FLAG_SLEEP'] = (
        (fly_status['TOTAL_SLEEP'] >= THRESHOLDS['SLEEP_MAX']) |
        (fly_status['LONGEST_ZERO'] >= THRESHOLDS['SLEEP_BOUT'])
    )
    fly_status['FLAG_NO_STARTLE'] = fly_status['NO_STARTLE']
    fly_status['FLAG_MISSING'] = fly_status['MISSING_FRAC'] > THRESHOLDS['MISSING_MAX']
    
    # Classify status using decision tree (in order)
    def classify_row(row):
        if row['FLAG_A2']:
            return "Dead"
        if row['FLAG_A1'] and row['FLAG_NO_STARTLE']:
            return "Dead"
        if row['DECLINE_STATUS'] == "Dead (by decline)":
            return "Dead"
        if (row['FLAG_LOW_ACTIVITY'] or row['FLAG_SLEEP']) and row['FLAG_NO_STARTLE']:
            return "Unhealthy"
        if row['DECLINE_STATUS'] == "Unhealthy (by decline)":
            return "Unhealthy"
        if row['FLAG_MISSING']:
            return "QC_Fail"
        return "Alive"
    
    fly_status['STATUS'] = fly_status.apply(classify_row, axis=1)
    
    return fly_status


def apply_irreversible_death(fly_status):
    """
    Apply irreversible death rule: once Dead, always Dead.
    
    Args:
        fly_status: DataFrame with STATUS column
        
    Returns:
        DataFrame with STATUS updated to enforce irreversible death
    """
    fly_status = fly_status.copy()
    fly_status = fly_status.sort_values(['Monitor', 'Channel', 'Exp_Day']).reset_index(drop=True)
    
    def mark_permanent_death(group):
        group = group.copy()
        dead_found = False
        new_status = []
        for status in group['STATUS']:
            if dead_found:
                new_status.append("Dead")
            elif status == "Dead":
                dead_found = True
                new_status.append("Dead")
            else:
                new_status.append(status)
        group['STATUS'] = new_status
        return group
    
    fly_status = fly_status.groupby(['Monitor', 'Channel'], group_keys=False).apply(mark_permanent_death)
    
    return fly_status.reset_index(drop=True)


def generate_summary(fly_status):
    """
    Generate per-fly summary table.
    
    Args:
        fly_status: DataFrame with daily STATUS for each fly
        
    Returns:
        DataFrame with summary statistics per fly
    """
    # Filter missing metadata
    mask = (
        fly_status['Genotype'].apply(is_ok) &
        fly_status['Sex'].apply(is_ok) &
        fly_status['Treatment'].apply(is_ok)
    )
    fly_status_clean = fly_status[mask].copy()
    
    # Group by Monitor, Channel, Genotype, Sex, Treatment
    def agg_func(group):
        return pd.Series({
            'DAYS_ANALYZED': len(group),
            'DAYS_ALIVE': (group['STATUS'] == 'Alive').sum(),
            'DAYS_UNHEALTHY': (group['STATUS'] == 'Unhealthy').sum(),
            'DAYS_DEAD': (group['STATUS'] == 'Dead').sum(),
            'DAYS_QC_FAIL': (group['STATUS'] == 'QC_Fail').sum(),
            'FIRST_UNHEALTHY_DAY': group.loc[group['STATUS'] == 'Unhealthy', 'Exp_Day'].min() if (group['STATUS'] == 'Unhealthy').any() else np.nan,
            'FIRST_DEAD_DAY': group.loc[group['STATUS'] == 'Dead', 'Exp_Day'].min() if (group['STATUS'] == 'Dead').any() else np.nan,
            'LAST_ALIVE_DAY': group.loc[group['STATUS'] == 'Alive', 'Exp_Day'].max() if (group['STATUS'] == 'Alive').any() else np.nan,
            'FINAL_STATUS': group['STATUS'].iloc[-1]  # Last status
        })
    
    health_report = fly_status_clean.groupby(
        ['Monitor', 'Channel', 'Genotype', 'Sex', 'Treatment'], dropna=False
    ).apply(agg_func).reset_index()
    
    # Sort by Monitor, Channel
    health_report = health_report.sort_values(['Monitor', 'Channel']).reset_index(drop=True)
    
    return health_report


def main():
    """Main function to generate health report."""
    print("=" * 60)
    print("Fly Health Report Generator (MT-based)")
    print("=" * 60)
    
    # Determine input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default: look for dam_data_MT.csv, then dam_data_marked.csv
        default_paths = [
            '../../data/processed/dam_data_MT.csv',
            '../../data/processed/dam_data_marked.csv',
            '../../data/processed/dam_data_merged.csv'
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
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = '../../data/processed/health_report.csv'
    
    print(f"\n📊 Input file: {input_file}")
    print(f"📄 Output file: {output_file}")
    print()
    
    # Read data
    print("Reading data...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} rows")
    
    # Step 1: Prep data
    print("\n[Step 1] Preparing data...")
    dam_activity = prep_data(df)
    print(f"✓ {len(dam_activity):,} rows after preparation")
    
    # Step 2: Calculate daily metrics
    print("\n[Step 2] Calculating daily metrics...")
    daily_summary = calculate_daily_metrics(dam_activity)
    print(f"✓ Calculated metrics for {len(daily_summary):,} fly-days")
    
    # Step 3: Normalize to reference day
    print("\n[Step 3] Normalizing to reference day...")
    daily_summary = normalize_to_ref_day(daily_summary, ref_day)
    print(f"✓ Normalized to day {ref_day}")
    
    # Step 4: Filter missing metadata
    print("\n[Step 4] Filtering missing metadata...")
    before = len(daily_summary)
    mask = (
        daily_summary['Genotype'].apply(is_ok) &
        daily_summary['Sex'].apply(is_ok) &
        daily_summary['Treatment'].apply(is_ok)
    )
    daily_summary = daily_summary[mask].copy()
    after = len(daily_summary)
    print(f"✓ Removed {before - after} rows with missing metadata")
    
    # Step 5: Startle test
    print("\n[Step 5] Testing startle response...")
    transition_data = startle_test(dam_activity, lights_on, lights_off, transition_window)
    print(f"✓ Tested {len(transition_data):,} fly-days")
    
    # Step 6: Classify status
    print("\n[Step 6] Classifying fly status...")
    fly_status = classify_status(daily_summary, transition_data)
    print(f"✓ Classified {len(fly_status):,} fly-days")
    
    # Step 7: Apply irreversible death
    print("\n[Step 7] Applying irreversible death rule...")
    fly_status = apply_irreversible_death(fly_status)
    print("✓ Applied irreversible death rule")
    
    # Step 8: Generate summary
    print("\n[Step 8] Generating summary table...")
    health_report = generate_summary(fly_status)
    print(f"✓ Generated report for {len(health_report):,} flies")
    
    # Save output
    print(f"\n💾 Saving health report to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    health_report.to_csv(output_file, index=False)
    print(f"✓ Saved successfully")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Excluded days: {exclude_days}")
    print(f"Reference day: {ref_day}")
    print(f"Total flies: {len(health_report)}")
    print(f"Dead: {(health_report['FINAL_STATUS'] == 'Dead').sum()}")
    print(f"Unhealthy: {(health_report['FINAL_STATUS'] == 'Unhealthy').sum()}")
    print(f"Alive: {(health_report['FINAL_STATUS'] == 'Alive').sum()}")
    print(f"QC Fail: {(health_report['FINAL_STATUS'] == 'QC_Fail').sum()}")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()

