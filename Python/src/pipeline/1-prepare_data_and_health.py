#!/usr/bin/env python3
"""
Pipeline Step 1: Prepare Data and Generate Health Report

This script:
1. Loads raw DAM files (Monitor5.txt, Monitor6.txt) and metadata (details.txt)
2. Merges data into long format
3. Calculates time variables: Date, Time, ZT (Zeitgeber Time), Phase (Light/Dark)
4. Optionally filters by date range
5. Calculates Exp_Day (experimental day) using global experiment start
6. Generates health report (using in-memory data, no file I/O)

Output: dam_data_prepared.csv, health_report.csv

This is the first step in the pipeline. The output is used by:
- Step 2: Fly removal (optional)
- Step 3: Feature extraction
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, date
from pathlib import Path


# ============================================================
#   DATA LOADING FUNCTIONS
# ============================================================

def parse_details(filepath):
    """
    Parse details.txt to extract fly metadata.
    
    Handles space-separated or tab-separated values. Treatment (last column)
    can contain spaces (e.g., "2mM Arg", "2mM His").
    
    Args:
        filepath (str): Path to details.txt file
        
    Returns:
        pd.DataFrame: fly_metadata with columns:
            monitor, channel, fly_id, genotype, sex, treatment
    """
    print(f"📋 Parsing metadata from {filepath}...")
    
    # Read file and parse manually to handle spaces in treatment field
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        
        # Split on whitespace (handles both tabs and spaces)
        parts = line.split()
        
        if len(parts) < 4:
            continue  # Skip malformed lines
        
        # First 4 parts are: Monitor, Channel, Genotype, Sex
        # Everything after is Treatment (can contain spaces)
        monitor = parts[0]
        channel = parts[1]
        genotype = parts[2]
        sex = parts[3]
        treatment = ' '.join(parts[4:]) if len(parts) > 4 else ''
        
        rows.append({
            'Monitor': monitor,
            'Channel': channel,
            'Genotype': genotype,
            'Sex': sex,
            'Treatment': treatment
        })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
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
    fly_metadata = fly_metadata[fly_metadata['genotype'].apply(is_ok)].copy()
    
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


# ============================================================
#   USER CONFIGURATION
# ============================================================

# Default file paths (relative to script location)
def get_default_monitor_files():
    """Get all monitor files from Monitors_date_filtered folder."""
    script_dir = Path(__file__).parent
    monitors_dir = script_dir.parent.parent / 'Monitors_date_filtered'
    monitor_files = sorted(monitors_dir.glob('Monitor*.txt'))
    # Use relative path from script_dir (../../Monitors_date_filtered/filename.txt)
    return [f'../../Monitors_date_filtered/{f.name}' for f in monitor_files] if monitor_files else []

DEFAULT_DAM_FILES = get_default_monitor_files()
DEFAULT_META_PATH = '../../details.txt'
DEFAULT_OUTPUT_DATA = 'data/processed/dam_data_prepared.csv'
DEFAULT_OUTPUT_HEALTH = 'data/processed/health_report.csv'

# Light cycle settings
DEFAULT_LIGHTS_ON = 9
DEFAULT_LIGHTS_OFF = 21

# Date filtering (optional)
DEFAULT_APPLY_DATE_FILTER = False
DEFAULT_EXP_START = None  # None = auto-detect from data
DEFAULT_EXP_END = None    # None = auto-detect from data

# Health report settings
DEFAULT_BIN_LENGTH_MIN = 1
DEFAULT_EXCLUDE_DAYS = [1, 7]
DEFAULT_REF_DAY = 4
DEFAULT_DECLINE_THRESHOLD = 0.5
DEFAULT_DEATH_THRESHOLD = 0.2
DEFAULT_TRANSITION_WINDOW = 10  # minutes

# Thresholds for health classification
THRESHOLDS = {
    "A1": 12 * 60 / DEFAULT_BIN_LENGTH_MIN,  # 720 (12 hours in bins)
    "A2": 24 * 60 / DEFAULT_BIN_LENGTH_MIN,  # 1440 (24 hours in bins)
    "ACTIVITY_LOW": 50,
    "INDEX_LOW": 0.02,
    "SLEEP_MAX": 1300,
    "SLEEP_BOUT": 720,
    "MISSING_MAX": 0.10
}


# ============================================================
#   DATA PREPARATION FUNCTIONS
# ============================================================

def calculate_zt_phase(datetime_series, lights_on):
    """
    Calculate ZT (Zeitgeber Time) and Phase (Light/Dark).
    
    ZT is truncated to integer (0-23) based on hour boundaries.
    Each ZT value spans a full hour: 9:00-9:59 → ZT0, 10:00-10:59 → ZT1, etc.
    Phase is "Light" if ZT < 12, "Dark" otherwise.
    
    Args:
        datetime_series: Series of datetime objects
        lights_on: Hour when lights turn on (default 9)
        
    Returns:
        tuple: (ZT, Phase) where ZT is integer 0-23, Phase is "Light" or "Dark"
    """
    hours = datetime_series.dt.hour
    minutes = datetime_series.dt.minute
    hour_local = hours + minutes / 60
    
    # Calculate ZT_raw
    zt_raw = (hour_local - lights_on) % 24
    
    # Truncate to integer (floor), handle 24 -> 0
    zt = np.floor(zt_raw).astype(int)
    zt = np.where(zt == 24, 0, zt)
    
    # Calculate Phase
    phase = np.where(zt_raw < 12, "Light", "Dark")
    
    return zt, phase


def apply_date_filter(df, apply_filter, exp_start, exp_end):
    """
    Apply optional date filtering to the dataset.
    
    Args:
        df: DataFrame with datetime column
        apply_filter: Boolean, whether to apply date filter
        exp_start: Start date (date object or None)
        exp_end: End date (date object or None)
        
    Returns:
        tuple: (filtered_df, actual_exp_start, actual_exp_end)
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['datetime']).dt.date
    
    # Auto-detect date range if not specified
    all_dates = sorted(df['Date'].unique())
    auto_start = min(all_dates) if all_dates else None
    auto_end = max(all_dates) if all_dates else None
    
    if exp_start is None:
        actual_exp_start = auto_start
    else:
        actual_exp_start = exp_start if isinstance(exp_start, date) else pd.to_datetime(exp_start).date()
    
    if exp_end is None:
        actual_exp_end = auto_end
    else:
        actual_exp_end = exp_end if isinstance(exp_end, date) else pd.to_datetime(exp_end).date()
    
    print(f"Experiment start: {actual_exp_start}")
    print(f"Experiment end:   {actual_exp_end}")
    
    if apply_filter:
        before = len(df)
        df = df[(df['Date'] >= actual_exp_start) & (df['Date'] <= actual_exp_end)].copy()
        after = len(df)
        print(f"✓ Date filtering active: {before - after:,} rows removed")
    else:
        print("✓ Date filtering disabled; keeping all rows")
    
    return df, actual_exp_start, actual_exp_end


def calculate_exp_day_global(df, exp_start):
    """
    Calculate Exp_Day using global experiment start date (matching R behavior).
    
    Args:
        df: DataFrame with Date column
        exp_start: Experiment start date (date object)
        
    Returns:
        Series with Exp_Day values
    """
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df['datetime']).dt.date
    
    # Calculate days since start
    exp_start_pd = pd.to_datetime(exp_start)
    df['Exp_Day'] = (pd.to_datetime(df['Date']) - exp_start_pd).dt.days + 1
    
    return df['Exp_Day']


# ============================================================
#   HEALTH REPORT FUNCTIONS
# ============================================================

def rle(seq):
    """Run-length encoding."""
    if len(seq) == 0:
        return np.array([]), np.array([])
    
    # Convert to numpy array to avoid pandas indexing issues
    if isinstance(seq, pd.Series):
        seq_array = seq.values
    else:
        seq_array = np.asarray(seq)
    
    changes = np.diff(seq_array) != 0
    change_indices = np.where(changes)[0] + 1
    indices = np.concatenate(([0], change_indices, [len(seq_array)]))
    lengths = np.diff(indices)
    values = seq_array[indices[:-1]]
    
    return values, lengths


def longest_zero_run(counts, bin_length_min):
    """Calculate longest consecutive zero-activity period."""
    if len(counts) == 0:
        return 0
    
    has_activity = (counts > 0).astype(int)
    values, lengths = rle(has_activity)
    
    zero_runs = lengths[values == 0]
    if len(zero_runs) == 0:
        return 0
    
    return max(zero_runs) * bin_length_min


def total_sleep_minutes(counts, bin_length_min):
    """Calculate total sleep minutes (5+ minute inactivity periods)."""
    if len(counts) == 0:
        return 0
    
    has_activity = (counts > 0).astype(int)
    values, lengths = rle(has_activity)
    
    sleep_runs = lengths[(values == 0) & (lengths >= 5)]
    return sum(sleep_runs) * bin_length_min


def is_ok(x):
    """Check if a value is valid (not NA, not empty, not "na", not "nan")."""
    if pd.isna(x):
        return False
    x_str = str(x).lower()
    return x_str != "" and x_str != "na" and x_str != "nan"


def prep_data_for_health(df, exclude_days, bin_length_min):
    """Prepare and filter data for health report analysis."""
    df = df.copy()
    
    # Filter to MT only
    if 'reading' in df.columns:
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
        print(f"✓ Filtered to MT readings only")
    else:
        print(f"✓ Data is already MT-only")
    
    # Ensure datetime is datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Exclude specified days
    if 'Exp_Day' in df.columns:
        before = len(df)
        df = df[~df['Exp_Day'].isin(exclude_days)].copy()
        after = len(df)
        print(f"✓ Excluded days {exclude_days}: {before - after:,} rows removed")
    
    # Clean channel names
    if df['channel'].dtype == 'object':
        df['channel'] = df['channel'].str.replace('^ch', '', regex=True)
    df['channel'] = df['channel'].astype(int)
    
    # Rename value to COUNTS
    df = df.rename(columns={'value': 'COUNTS'})
    
    # Standardize column names
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


def calculate_daily_metrics(dam_activity, bin_length_min):
    """Calculate daily metrics per fly per day."""
    dam_activity = dam_activity.copy()
    dam_activity['Date'] = dam_activity['datetime'].dt.date
    
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
    
    daily_summary = dam_activity.groupby(group_cols, dropna=False).apply(calc_metrics, include_groups=False).reset_index()
    return daily_summary


def normalize_to_ref_day(daily_summary, ref_day, decline_threshold, death_threshold):
    """Normalize activity to reference day."""
    daily_summary = daily_summary.copy()
    
    ref_activity = daily_summary[daily_summary['Exp_Day'] == ref_day][
        ['Monitor', 'Channel', 'TOTAL_ACTIVITY']
    ].rename(columns={'TOTAL_ACTIVITY': 'REF_ACTIVITY'})
    
    daily_summary = daily_summary.merge(ref_activity, on=['Monitor', 'Channel'], how='left')
    daily_summary['REL_ACTIVITY'] = daily_summary['TOTAL_ACTIVITY'] / daily_summary['REF_ACTIVITY']
    
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
    """Test for startle response at light transitions."""
    dam_activity = dam_activity.copy()
    dam_activity['HOUR'] = dam_activity['datetime'].dt.hour + dam_activity['datetime'].dt.minute / 60
    dam_activity['IS_TRANSITION'] = (
        (np.abs(dam_activity['HOUR'] - lights_on) <= transition_window / 60) |
        (np.abs(dam_activity['HOUR'] - lights_off) <= transition_window / 60)
    )
    dam_activity['Date'] = dam_activity['datetime'].dt.date
    
    def calc_transition_counts(group):
        transition_rows = group[group['IS_TRANSITION']]
        return pd.Series({
            'TRANSITION_COUNTS': transition_rows['COUNTS'].sum(skipna=True)
        })
    
    transition_data = dam_activity.groupby(['Monitor', 'Channel', 'Date'], group_keys=False).apply(
        calc_transition_counts, include_groups=False
    ).reset_index()
    
    transition_data['NO_STARTLE'] = transition_data['TRANSITION_COUNTS'] == 0
    return transition_data


def classify_status(daily_summary, transition_data, thresholds):
    """Classify fly status using decision tree."""
    fly_status = daily_summary.merge(
        transition_data, on=['Monitor', 'Channel', 'Date'], how='left'
    )
    fly_status['NO_STARTLE'] = fly_status['NO_STARTLE'].fillna(False)
    
    fly_status['FLAG_A2'] = fly_status['LONGEST_ZERO'] >= thresholds['A2']
    fly_status['FLAG_A1'] = fly_status['LONGEST_ZERO'] >= thresholds['A1']
    fly_status['FLAG_LOW_ACTIVITY'] = (
        (fly_status['TOTAL_ACTIVITY'] <= thresholds['ACTIVITY_LOW']) |
        (fly_status['ACTIVITY_INDEX'] <= thresholds['INDEX_LOW'])
    )
    fly_status['FLAG_SLEEP'] = (
        (fly_status['TOTAL_SLEEP'] >= thresholds['SLEEP_MAX']) |
        (fly_status['LONGEST_ZERO'] >= thresholds['SLEEP_BOUT'])
    )
    fly_status['FLAG_NO_STARTLE'] = fly_status['NO_STARTLE']
    fly_status['FLAG_MISSING'] = fly_status['MISSING_FRAC'] > thresholds['MISSING_MAX']
    
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
    """Apply irreversible death rule: once Dead, always Dead."""
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
    """Generate per-fly summary table."""
    mask = (
        fly_status['Genotype'].apply(is_ok) &
        fly_status['Sex'].apply(is_ok) &
        fly_status['Treatment'].apply(is_ok)
    )
    fly_status_clean = fly_status[mask].copy()
    
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
            'FINAL_STATUS': group['STATUS'].iloc[-1]
        })
    
    health_report = fly_status_clean.groupby(
        ['Monitor', 'Channel', 'Genotype', 'Sex', 'Treatment'], dropna=False
    ).apply(agg_func).reset_index()
    
    health_report = health_report.sort_values(['Monitor', 'Channel']).reset_index(drop=True)
    return health_report


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def prepare_data_and_health(
    dam_files=None,
    meta_path=None,
    lights_on=DEFAULT_LIGHTS_ON,
    lights_off=DEFAULT_LIGHTS_OFF,
    apply_date_filter_flag=DEFAULT_APPLY_DATE_FILTER,
    exp_start=DEFAULT_EXP_START,
    exp_end=DEFAULT_EXP_END,
    output_data_file=None,
    output_health_file=None,
    bin_length_min=DEFAULT_BIN_LENGTH_MIN,
    exclude_days=DEFAULT_EXCLUDE_DAYS,
    ref_day=DEFAULT_REF_DAY,
    decline_threshold=DEFAULT_DECLINE_THRESHOLD,
    death_threshold=DEFAULT_DEATH_THRESHOLD,
    transition_window=DEFAULT_TRANSITION_WINDOW
):
    """
    Main function to prepare data and generate health report.
    
    Args:
        dam_files: List of Monitor*.txt file paths
        meta_path: Path to details.txt metadata file
        lights_on: Hour when lights turn on
        lights_off: Hour when lights turn off
        apply_date_filter_flag: Whether to apply date filtering
        exp_start: Experiment start date (None = auto-detect)
        exp_end: Experiment end date (None = auto-detect)
        output_data_file: Output CSV file path for prepared data
        output_health_file: Output CSV file path for health report
        bin_length_min: Length of each time bin in minutes
        exclude_days: List of days to exclude from health analysis
        ref_day: Reference day for normalization
        decline_threshold: Threshold for unhealthy classification
        death_threshold: Threshold for death classification
        transition_window: Window in minutes around light transitions
        
    Returns:
        tuple: (prepared_data, health_report)
    """
    print("=" * 60)
    print("PIPELINE STEP 1: PREPARE DATA AND GENERATE HEALTH REPORT")
    print("=" * 60)
    
    # Set defaults
    if dam_files is None:
        dam_files = DEFAULT_DAM_FILES
    if meta_path is None:
        meta_path = DEFAULT_META_PATH
    if output_data_file is None:
        output_data_file = DEFAULT_OUTPUT_DATA
    if output_health_file is None:
        output_health_file = DEFAULT_OUTPUT_HEALTH
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dam_files = [os.path.join(script_dir, f) if not os.path.isabs(f) else f for f in dam_files]
    meta_path = os.path.join(script_dir, meta_path) if not os.path.isabs(meta_path) else meta_path
    output_data_file = os.path.join(script_dir, output_data_file) if not os.path.isabs(output_data_file) else output_data_file
    output_health_file = os.path.join(script_dir, output_health_file) if not os.path.isabs(output_health_file) else output_health_file
    
    # ============================================================
    # PART 1: DATA PREPARATION
    # ============================================================
    print("\n[Part 1] Data Preparation")
    print("-" * 60)
    
    # STEP 1.1: Load and merge data
    print("\n[Step 1.1] Loading and merging data...")
    
    # Parse metadata
    fly_metadata = parse_details(meta_path)
    
    # Parse monitor files
    time_series_list = []
    for dam_file in dam_files:
        if not os.path.exists(dam_file):
            print(f"ERROR: File not found: {dam_file}")
            sys.exit(1)
        
        # Extract monitor number from filename
        # Handle both "Monitor51.txt" and "Monitor51_06_20_25.txt" formats
        filename = Path(dam_file).stem
        # Remove "Monitor" prefix and extract first number before underscore or end
        if filename.startswith('Monitor'):
            monitor_str = filename[7:]  # Remove "Monitor" (7 chars)
            # Extract digits until underscore or end
            monitor_num_str = ''
            for char in monitor_str:
                if char.isdigit():
                    monitor_num_str += char
                elif char == '_':
                    break
                else:
                    break
            monitor_num = int(monitor_num_str) if monitor_num_str else int(''.join(filter(str.isdigit, filename)))
        else:
            # Fallback: extract all digits (old behavior)
            monitor_num = int(''.join(filter(str.isdigit, filename)))
        
        monitor_data = parse_monitor_file(dam_file, monitor_num)
        time_series_list.append(monitor_data)
    
    # Combine time-series data
    time_series_data = pd.concat(time_series_list, ignore_index=True)
    time_series_data = time_series_data.sort_values(['datetime', 'monitor', 'channel', 'reading']).reset_index(drop=True)
    
    # Merge with metadata
    dam_merged = time_series_data.merge(fly_metadata, on=['monitor', 'channel'])
    
    # Reorder columns
    final_columns = ['datetime', 'monitor', 'channel', 'reading', 'value', 'fly_id', 'genotype', 'sex', 'treatment']
    dam_merged = dam_merged[final_columns]
    
    print(f"✓ Merged data: {len(dam_merged):,} rows")
    print(f"   Unique flies: {dam_merged['fly_id'].nunique()}")
    print(f"   Date range: {dam_merged['datetime'].min()} to {dam_merged['datetime'].max()}")
    
    # STEP 1.2: Calculate time variables
    print("\n[Step 1.2] Calculating time variables...")
    
    dam_merged['Date'] = pd.to_datetime(dam_merged['datetime']).dt.date
    dam_merged['Time'] = pd.to_datetime(dam_merged['datetime']).dt.strftime('%H:%M:%S')
    
    zt, phase = calculate_zt_phase(dam_merged['datetime'], lights_on)
    dam_merged['ZT'] = zt
    dam_merged['Phase'] = phase
    
    print(f"✓ Calculated ZT (0-23) and Phase (Light/Dark)")
    print(f"   ZT range: {dam_merged['ZT'].min()} to {dam_merged['ZT'].max()}")
    print(f"   Phase distribution: {dam_merged['Phase'].value_counts().to_dict()}")
    
    # STEP 1.3: Optional date filtering
    print("\n[Step 1.3] Applying date filter (if enabled)...")
    
    dam_merged, actual_exp_start, actual_exp_end = apply_date_filter(
        dam_merged, apply_date_filter_flag, exp_start, exp_end
    )
    
    # STEP 1.4: Calculate Exp_Day
    print("\n[Step 1.4] Calculating Exp_Day...")
    
    dam_merged['Exp_Day'] = calculate_exp_day_global(dam_merged, actual_exp_start)
    
    print(f"✓ Calculated Exp_Day (Day 1 = {actual_exp_start})")
    print(f"   Exp_Day range: {dam_merged['Exp_Day'].min()} to {dam_merged['Exp_Day'].max()}")
    
    # STEP 1.5: Save prepared data
    print("\n[Step 1.5] Saving prepared data...")
    
    # Reorder columns for final output
    output_columns = ['datetime', 'Date', 'Time', 'monitor', 'channel', 'reading', 'value', 
                      'fly_id', 'genotype', 'sex', 'treatment', 'ZT', 'Phase', 'Exp_Day']
    dam_merged = dam_merged[output_columns]
    
    # Save output
    os.makedirs(os.path.dirname(output_data_file), exist_ok=True)
    dam_merged.to_csv(output_data_file, index=False)
    
    print(f"✓ Saved prepared data to {output_data_file}")
    print(f"   Final rows: {len(dam_merged):,}")
    print(f"   Final flies: {dam_merged.groupby(['monitor', 'channel']).ngroups}")
    
    # ============================================================
    # PART 2: HEALTH REPORT GENERATION (using in-memory data)
    # ============================================================
    print("\n" + "=" * 60)
    print("[Part 2] Health Report Generation")
    print("-" * 60)
    
    # Update thresholds based on bin_length_min
    thresholds = {
        "A1": 12 * 60 / bin_length_min,
        "A2": 24 * 60 / bin_length_min,
        "ACTIVITY_LOW": THRESHOLDS["ACTIVITY_LOW"],
        "INDEX_LOW": THRESHOLDS["INDEX_LOW"],
        "SLEEP_MAX": THRESHOLDS["SLEEP_MAX"],
        "SLEEP_BOUT": THRESHOLDS["SLEEP_BOUT"],
        "MISSING_MAX": THRESHOLDS["MISSING_MAX"]
    }
    
    # STEP 2.1: Prepare data for health analysis
    print("\n[Step 2.1] Preparing data for health analysis...")
    
    dam_activity = prep_data_for_health(dam_merged, exclude_days, bin_length_min)
    print(f"✓ Prepared {len(dam_activity):,} rows for analysis")
    
    # STEP 2.2: Calculate daily metrics
    print("\n[Step 2.2] Calculating daily metrics...")
    
    daily_summary = calculate_daily_metrics(dam_activity, bin_length_min)
    print(f"✓ Calculated metrics for {len(daily_summary)} fly-days")
    
    # STEP 2.3: Normalize to reference day
    print("\n[Step 2.3] Normalizing to reference day...")
    
    daily_summary = normalize_to_ref_day(daily_summary, ref_day, decline_threshold, death_threshold)
    print(f"✓ Normalized activity to reference day {ref_day}")
    
    # STEP 2.4: Startle test
    print("\n[Step 2.4] Testing startle response...")
    
    transition_data = startle_test(dam_activity, lights_on, lights_off, transition_window)
    print(f"✓ Tested startle response for {len(transition_data)} fly-days")
    
    # STEP 2.5: Classify status
    print("\n[Step 2.5] Classifying fly status...")
    
    fly_status = classify_status(daily_summary, transition_data, thresholds)
    print(f"✓ Classified status for {len(fly_status)} fly-days")
    
    # STEP 2.6: Apply irreversible death
    print("\n[Step 2.6] Applying irreversible death rule...")
    
    fly_status = apply_irreversible_death(fly_status)
    print(f"✓ Applied irreversible death rule")
    
    # STEP 2.7: Generate summary
    print("\n[Step 2.7] Generating health report summary...")
    
    health_report = generate_summary(fly_status)
    print(f"✓ Generated health report for {len(health_report)} flies")
    
    # Print summary statistics
    print("\nHealth Status Summary:")
    print(f"   Alive: {(health_report['FINAL_STATUS'] == 'Alive').sum()}")
    print(f"   Unhealthy: {(health_report['FINAL_STATUS'] == 'Unhealthy').sum()}")
    print(f"   Dead: {(health_report['FINAL_STATUS'] == 'Dead').sum()}")
    print(f"   QC Fail: {(health_report['FINAL_STATUS'] == 'QC_Fail').sum()}")
    
    # STEP 2.8: Save health report
    print(f"\n[Step 2.8] Saving health report to {output_health_file}...")
    
    os.makedirs(os.path.dirname(output_health_file), exist_ok=True)
    health_report.to_csv(output_health_file, index=False)
    print(f"✓ Saved health report successfully")
    
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  1. Prepared data: {output_data_file}")
    print(f"  2. Health report: {output_health_file}")
    print(f"\nNext steps:")
    print(f"  1. Review the health report: {output_health_file}")
    print(f"  2. Decide which flies to remove (optional)")
    print(f"  3. Run: python 2-remove_flies.py (optional)")
    
    return dam_merged, health_report


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Pipeline Step 1: Prepare data and generate health report',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data input
    parser.add_argument('--dam-files', nargs='+', default=None,
                       help='List of Monitor*.txt files (default: Monitor5.txt, Monitor6.txt)')
    parser.add_argument('--meta-path', type=str, default=None,
                       help='Metadata file path (default: details.txt)')
    
    # Settings
    parser.add_argument('--lights-on', type=int, default=DEFAULT_LIGHTS_ON,
                       help=f'Hour when lights turn on (default: {DEFAULT_LIGHTS_ON})')
    parser.add_argument('--lights-off', type=int, default=DEFAULT_LIGHTS_OFF,
                       help=f'Hour when lights turn off (default: {DEFAULT_LIGHTS_OFF})')
    
    # Date filtering
    parser.add_argument('--apply-date-filter', action='store_true',
                       help='Enable date filtering')
    parser.add_argument('--exp-start', type=str, default=None,
                       help='Experiment start date (YYYY-MM-DD, default: auto-detect)')
    parser.add_argument('--exp-end', type=str, default=None,
                       help='Experiment end date (YYYY-MM-DD, default: auto-detect)')
    
    # Health report settings
    parser.add_argument('--ref-day', type=int, default=DEFAULT_REF_DAY,
                       help=f'Reference day for normalization (default: {DEFAULT_REF_DAY})')
    parser.add_argument('--exclude-days', nargs='+', type=int, default=DEFAULT_EXCLUDE_DAYS,
                       help=f'Days to exclude from health analysis (default: {DEFAULT_EXCLUDE_DAYS})')
    
    # Output
    parser.add_argument('--output-data', type=str, default=None,
                       help='Output file path for prepared data (default: dam_data_prepared.csv)')
    parser.add_argument('--output-health', type=str, default=None,
                       help='Output file path for health report (default: health_report.csv)')
    
    args = parser.parse_args()
    
    # Parse dates if provided
    exp_start = pd.to_datetime(args.exp_start).date() if args.exp_start else None
    exp_end = pd.to_datetime(args.exp_end).date() if args.exp_end else None
    
    # Run pipeline
    prepare_data_and_health(
        dam_files=args.dam_files,
        meta_path=args.meta_path,
        lights_on=args.lights_on,
        lights_off=args.lights_off,
        apply_date_filter_flag=args.apply_date_filter,
        exp_start=exp_start,
        exp_end=exp_end,
        output_data_file=args.output_data,
        output_health_file=args.output_health,
        exclude_days=args.exclude_days,
        ref_day=args.ref_day
    )


if __name__ == "__main__":
    main()

