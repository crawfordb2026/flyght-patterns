#!/usr/bin/env python3
"""
Combined Pipeline Script - Data Processing, Health Report, and Fly Removal

This script combines the functionality of:
- create_database.py (data loading and merging)
- generate_health_report.py (health classification)
- remove_flies.py (fly removal)

Matches the workflow of pipeline.r in R.

Usage:
    python pipeline.py [options]
    
Examples:
    python pipeline.py
    python pipeline.py --apply-date-filter --exp-start 2025-09-20
    python pipeline.py --flies-to-remove "6-ch23,6-ch5" --output dam_clean.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import importlib.util
from datetime import datetime, date
from pathlib import Path

# Add current directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import functions from existing scripts
try:
    from create_database import parse_details, parse_monitor_file
except ImportError:
    print("Warning: Could not import from create_database.py. Make sure it's in the same directory.")
    sys.exit(1)

try:
    from remove_flies import create_fly_key, remove_flies, is_ok
except ImportError:
    print("Warning: Could not import from remove_flies.py. Make sure it's in the same directory.")
    sys.exit(1)


# ============================================================
#   USER CONFIGURATION (Edit these directly in the script)
# ============================================================

# Fly removal settings - set these to remove flies directly in the script
# Leave as None or empty lists if you want to use command-line arguments instead
REMOVE_FLIES = None  # Example: ["6-ch23", "6-ch5", "5-ch7"]
REMOVE_GENOTYPES = None  # Example: ["mutant1", "control"]
REMOVE_SEXES = None  # Example: ["Male", "Female"]
REMOVE_TREATMENTS = None  # Example: ["drug1", "vehicle"]
REMOVE_PER_FLY_DAYS = None  # Example: {"5-ch7": [1, 2], "6-ch18": [3]}

# Date filtering
APPLY_DATE_FILTER = False  # Set to True to enable date filtering
EXP_START = None  # Example: date(2025, 9, 20) or None for auto-detect
EXP_END = None  # Example: date(2025, 9, 30) or None for auto-detect

# Light cycle
LIGHTS_ON = 9
LIGHTS_OFF = 21

# Health report settings
BIN_LENGTH_MIN = 1
EXCLUDE_DAYS = [1, 7]
REF_DAY = 4
DECLINE_THRESHOLD = 0.5
DEATH_THRESHOLD = 0.2
TRANSITION_WINDOW = 10  # minutes

# ============================================================
#   SETTINGS (Advanced - usually don't need to change)
# ============================================================

# Default settings (can be overridden by command-line arguments)
DEFAULT_LIGHTS_ON = LIGHTS_ON
DEFAULT_LIGHTS_OFF = LIGHTS_OFF
DEFAULT_BIN_LENGTH_MIN = BIN_LENGTH_MIN
DEFAULT_EXCLUDE_DAYS = EXCLUDE_DAYS
DEFAULT_REF_DAY = REF_DAY
DEFAULT_DECLINE_THRESHOLD = DECLINE_THRESHOLD
DEFAULT_DEATH_THRESHOLD = DEATH_THRESHOLD
DEFAULT_TRANSITION_WINDOW = TRANSITION_WINDOW

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
#   HELPER FUNCTIONS
# ============================================================

def calculate_zt_phase(datetime_series, lights_on):
    """
    Calculate ZT (Zeitgeber Time) and Phase (Light/Dark).
    
    ZT is rounded to integer (0-23) to match R behavior.
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
    
    # Round to integer, handle 24 -> 0
    zt = np.round(zt_raw).astype(int)
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
#   HEALTH REPORT FUNCTIONS (adapted from generate_health_report.py)
# ============================================================

def rle(seq):
    """Run-length encoding."""
    if len(seq) == 0:
        return np.array([]), np.array([])
    
    changes = np.diff(seq) != 0
    change_indices = np.where(changes)[0] + 1
    indices = np.concatenate(([0], change_indices, [len(seq)]))
    lengths = np.diff(indices)
    values = seq[indices[:-1]]
    
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


def calculate_daily_metrics_health(dam_activity, bin_length_min):
    """Calculate daily metrics for health report."""
    dam_activity = dam_activity.copy()
    dam_activity['Date'] = pd.to_datetime(dam_activity['datetime']).dt.date
    
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


def normalize_to_ref_day_health(daily_summary, ref_day, decline_threshold, death_threshold):
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


def startle_test_health(dam_activity, lights_on, lights_off, transition_window):
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
            'Date': group['Date'].iloc[0],
            'TRANSITION_COUNTS': transition_rows['COUNTS'].sum(skipna=True)
        })
    
    transition_data = dam_activity.groupby(['Monitor', 'Channel', 'Date'], group_keys=False).apply(
        calc_transition_counts
    ).reset_index()
    
    transition_data['NO_STARTLE'] = transition_data['TRANSITION_COUNTS'] == 0
    return transition_data


def classify_status_health(daily_summary, transition_data, thresholds):
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


def apply_irreversible_death_health(fly_status):
    """Apply irreversible death rule."""
    fly_status = fly_status.copy()
    fly_status = fly_status.sort_values(['Monitor', 'Channel', 'Exp_Day'])
    
    def apply_rule(group):
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
        return pd.Series(new_status, index=group.index)
    
    fly_status['STATUS'] = fly_status.groupby(['Monitor', 'Channel'], group_keys=False).apply(
        lambda g: apply_rule(g)
    ).values
    
    return fly_status


def generate_health_report_summary(fly_status):
    """Generate health report summary table (matching generate_health_report.py format)."""
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


# ============================================================
#   MAIN PIPELINE WORKFLOW
# ============================================================

def run_pipeline(
    dam_files=None,
    meta_path=None,
    apply_date_filter=None,  # None means use config, False/True override
    exp_start=None,
    exp_end=None,
    lights_on=None,  # None means use config
    lights_off=None,
    bin_length_min=None,
    exclude_days=None,
    ref_day=None,
    decline_threshold=None,
    death_threshold=None,
    transition_window=None,
    flies_to_remove=None,  # None means use config, empty list means no removal
    per_fly_remove=None,
    genotypes_to_remove=None,
    sexes_to_remove=None,
    treatments_to_remove=None,
    output_file=None,
    use_config=True  # If True, use hardcoded config values when args are None
):
    """
    Run the complete pipeline workflow.
    
    Returns:
        tuple: (dam_clean, health_report)
    """
    # Use hardcoded config values if arguments are None and use_config is True
    if use_config:
        if apply_date_filter is None:
            apply_date_filter = APPLY_DATE_FILTER
        if exp_start is None:
            exp_start = EXP_START
        if exp_end is None:
            exp_end = EXP_END
        if lights_on is None:
            lights_on = LIGHTS_ON
        if lights_off is None:
            lights_off = LIGHTS_OFF
        if bin_length_min is None:
            bin_length_min = BIN_LENGTH_MIN
        if exclude_days is None:
            exclude_days = EXCLUDE_DAYS
        if ref_day is None:
            ref_day = REF_DAY
        if decline_threshold is None:
            decline_threshold = DECLINE_THRESHOLD
        if death_threshold is None:
            death_threshold = DEATH_THRESHOLD
        if transition_window is None:
            transition_window = TRANSITION_WINDOW
        if flies_to_remove is None:
            flies_to_remove = REMOVE_FLIES
        if genotypes_to_remove is None:
            genotypes_to_remove = REMOVE_GENOTYPES
        if sexes_to_remove is None:
            sexes_to_remove = REMOVE_SEXES
        if treatments_to_remove is None:
            treatments_to_remove = REMOVE_TREATMENTS
        if per_fly_remove is None:
            per_fly_remove = REMOVE_PER_FLY_DAYS
    
    # Fallback to defaults if still None
    if lights_on is None:
        lights_on = DEFAULT_LIGHTS_ON
    if lights_off is None:
        lights_off = DEFAULT_LIGHTS_OFF
    if bin_length_min is None:
        bin_length_min = DEFAULT_BIN_LENGTH_MIN
    if exclude_days is None:
        exclude_days = DEFAULT_EXCLUDE_DAYS
    if ref_day is None:
        ref_day = DEFAULT_REF_DAY
    if decline_threshold is None:
        decline_threshold = DEFAULT_DECLINE_THRESHOLD
    if death_threshold is None:
        death_threshold = DEFAULT_DEATH_THRESHOLD
    if transition_window is None:
        transition_window = DEFAULT_TRANSITION_WINDOW
    if apply_date_filter is None:
        apply_date_filter = False
    
    print("=" * 60)
    print("FLY SLEEP ANALYSIS PIPELINE")
    print("=" * 60)
    
    # ============================================================
    # STEP 1: Load and merge data
    # ============================================================
    print("\n[Step 1] Loading and merging data...")
    
    # Default file paths
    if dam_files is None:
        dam_files = ['../../Monitor5.txt', '../../Monitor6.txt']
    if meta_path is None:
        meta_path = '../../details.txt'
    
    # Parse metadata
    fly_metadata = parse_details(meta_path)
    
    # Parse monitor files
    time_series_list = []
    for dam_file in dam_files:
        # Extract monitor number from filename
        monitor_num = int(''.join(filter(str.isdigit, Path(dam_file).stem)))
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
    
    # ============================================================
    # STEP 2: Calculate time variables (Date, Time, ZT, Phase)
    # ============================================================
    print("\n[Step 2] Calculating time variables...")
    
    dam_merged['Date'] = pd.to_datetime(dam_merged['datetime']).dt.date
    dam_merged['Time'] = pd.to_datetime(dam_merged['datetime']).dt.strftime('%H:%M:%S')
    
    zt, phase = calculate_zt_phase(dam_merged['datetime'], lights_on)
    dam_merged['ZT'] = zt
    dam_merged['Phase'] = phase
    
    print(f"✓ Calculated ZT (0-23) and Phase (Light/Dark)")
    
    # ============================================================
    # STEP 3: Optional date filtering
    # ============================================================
    print("\n[Step 3] Applying date filter (if enabled)...")
    
    dam_merged, actual_exp_start, actual_exp_end = apply_date_filter(
        dam_merged, apply_date_filter, exp_start, exp_end
    )
    
    # ============================================================
    # STEP 4: Calculate Exp_Day
    # ============================================================
    print("\n[Step 4] Calculating Exp_Day...")
    
    dam_merged['Exp_Day'] = calculate_exp_day_global(dam_merged, actual_exp_start)
    
    print(f"✓ Calculated Exp_Day (Day 1 = {actual_exp_start})")
    
    # ============================================================
    # STEP 5: Generate health report
    # ============================================================
    print("\n[Step 5] Generating health report...")
    
    # Filter to MT data for health report
    dam_mt = dam_merged[dam_merged['reading'] == 'MT'].copy()
    dam_mt = dam_mt[~dam_mt['Exp_Day'].isin(exclude_days)].copy()
    
    # Standardize column names for health report functions
    dam_mt['Monitor'] = dam_mt['monitor']
    dam_mt['Channel'] = dam_mt['channel'].astype(int) if dam_mt['channel'].dtype == 'object' else dam_mt['channel']
    dam_mt['Genotype'] = dam_mt['genotype']
    dam_mt['Sex'] = dam_mt['sex']
    dam_mt['Treatment'] = dam_mt['treatment']
    dam_mt['COUNTS'] = dam_mt['value']
    
    # Filter missing metadata
    dam_mt = dam_mt[
        dam_mt['Genotype'].apply(is_ok) &
        dam_mt['Sex'].apply(is_ok) &
        dam_mt['Treatment'].apply(is_ok)
    ].copy()
    
    # Calculate daily metrics
    daily_summary = calculate_daily_metrics_health(dam_mt, bin_length_min)
    
    # Normalize to reference day
    daily_summary = normalize_to_ref_day_health(daily_summary, ref_day, decline_threshold, death_threshold)
    
    # Startle test
    transition_data = startle_test_health(dam_mt, lights_on, lights_off, transition_window)
    
    # Classify status
    fly_status = classify_status_health(daily_summary, transition_data, THRESHOLDS)
    
    # Apply irreversible death
    fly_status = apply_irreversible_death_health(fly_status)
    
    # Generate summary
    health_report = generate_health_report_summary(fly_status)
    
    print(f"✓ Generated health report for {len(health_report)} flies")
    print(f"   Alive: {(health_report['FINAL_STATUS'] == 'Alive').sum()}")
    print(f"   Unhealthy: {(health_report['FINAL_STATUS'] == 'Unhealthy').sum()}")
    print(f"   Dead: {(health_report['FINAL_STATUS'] == 'Dead').sum()}")
    print(f"   QC Fail: {(health_report['FINAL_STATUS'] == 'QC_Fail').sum()}")
    
    # Save health report
    health_report_path = '../../data/processed/health_report.csv'
    os.makedirs(os.path.dirname(health_report_path), exist_ok=True)
    health_report.to_csv(health_report_path, index=False)
    print(f"✓ Saved health report to {health_report_path}")
    
    # ============================================================
    # STEP 6: Remove flies
    # ============================================================
    print("\n[Step 6] Removing flies...")
    
    # Convert removal lists (handle both string and list inputs)
    if flies_to_remove is None:
        flies_list = None
    elif isinstance(flies_to_remove, str):
        flies_list = flies_to_remove.split(',') if flies_to_remove else None
    else:
        flies_list = flies_to_remove  # Already a list
    
    if genotypes_to_remove is None:
        genotypes_list = None
    elif isinstance(genotypes_to_remove, str):
        genotypes_list = genotypes_to_remove.split(',') if genotypes_to_remove else None
    else:
        genotypes_list = genotypes_to_remove
    
    if sexes_to_remove is None:
        sexes_list = None
    elif isinstance(sexes_to_remove, str):
        sexes_list = sexes_to_remove.split(',') if sexes_to_remove else None
    else:
        sexes_list = sexes_to_remove
    
    if treatments_to_remove is None:
        treatments_list = None
    elif isinstance(treatments_to_remove, str):
        treatments_list = treatments_to_remove.split(',') if treatments_to_remove else None
    else:
        treatments_list = treatments_to_remove
    
    # Parse per_fly_remove if provided (JSON string or dict)
    per_fly_dict = None
    if per_fly_remove:
        if isinstance(per_fly_remove, str):
            import json
            per_fly_dict = json.loads(per_fly_remove)
        else:
            per_fly_dict = per_fly_remove  # Already a dict
    
    dam_clean, removals_summary = remove_flies(
        dam_merged,
        flies_to_remove=flies_list,
        per_fly_remove=per_fly_dict,
        genotypes_to_remove=genotypes_list,
        sexes_to_remove=sexes_list,
        treatments_to_remove=treatments_list
    )
    
    print(f"✓ Removed flies: {dam_merged.shape[0] - dam_clean.shape[0]:,} rows removed")
    
    # ============================================================
    # STEP 7: Save output
    # ============================================================
    print("\n[Step 7] Saving output...")
    
    if output_file is None:
        output_file = '../../data/processed/dam_data_cleaned.csv'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dam_clean.to_csv(output_file, index=False)
    
    print(f"✓ Saved cleaned data to {output_file}")
    print(f"   Final rows: {len(dam_clean):,}")
    print(f"   Final flies: {dam_clean.groupby(['monitor', 'channel']).ngroups}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return dam_clean, health_report


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Combined pipeline for fly sleep analysis: data loading, health report, and fly removal'
    )
    
    # Data input
    parser.add_argument('--dam-files', nargs='+', default=None,
                       help='List of Monitor*.txt files (default: Monitor5.txt, Monitor6.txt)')
    parser.add_argument('--meta-path', type=str, default=None,
                       help='Metadata file path (default: details.txt)')
    
    # Date filtering
    parser.add_argument('--apply-date-filter', action='store_true',
                       help='Enable date filtering')
    parser.add_argument('--exp-start', type=str, default=None,
                       help='Experiment start date (YYYY-MM-DD, default: auto-detect)')
    parser.add_argument('--exp-end', type=str, default=None,
                       help='Experiment end date (YYYY-MM-DD, default: auto-detect)')
    
    # Settings
    parser.add_argument('--lights-on', type=int, default=DEFAULT_LIGHTS_ON,
                       help=f'Hour when lights turn on (default: {DEFAULT_LIGHTS_ON})')
    parser.add_argument('--lights-off', type=int, default=DEFAULT_LIGHTS_OFF,
                       help=f'Hour when lights turn off (default: {DEFAULT_LIGHTS_OFF})')
    parser.add_argument('--ref-day', type=int, default=DEFAULT_REF_DAY,
                       help=f'Reference day for normalization (default: {DEFAULT_REF_DAY})')
    
    # Fly removal
    parser.add_argument('--flies-to-remove', type=str, default=None,
                       help='Comma-separated fly IDs (e.g., "6-ch23,6-ch5")')
    parser.add_argument('--genotypes-to-remove', type=str, default=None,
                       help='Comma-separated genotypes')
    parser.add_argument('--sexes-to-remove', type=str, default=None,
                       help='Comma-separated sexes')
    parser.add_argument('--treatments-to-remove', type=str, default=None,
                       help='Comma-separated treatments')
    parser.add_argument('--per-fly-remove', type=str, default=None,
                       help='JSON string mapping fly IDs to days to remove (e.g., \'{"5-ch7": [1, 2]}\')')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: dam_data_cleaned.csv)')
    
    args = parser.parse_args()
    
    # Parse dates if provided
    exp_start = pd.to_datetime(args.exp_start).date() if args.exp_start else None
    exp_end = pd.to_datetime(args.exp_end).date() if args.exp_end else None
    
    # Run pipeline
    run_pipeline(
        dam_files=args.dam_files,
        meta_path=args.meta_path,
        apply_date_filter=args.apply_date_filter,
        exp_start=exp_start,
        exp_end=exp_end,
        lights_on=args.lights_on,
        lights_off=args.lights_off,
        flies_to_remove=args.flies_to_remove,
        per_fly_remove=args.per_fly_remove,
        genotypes_to_remove=args.genotypes_to_remove,
        sexes_to_remove=args.sexes_to_remove,
        treatments_to_remove=args.treatments_to_remove,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

