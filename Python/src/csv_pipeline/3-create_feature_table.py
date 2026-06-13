#!/usr/bin/env python3
"""
Pipeline Step 3: Create Feature Table

This script:
1. Reads cleaned data (from Step 1 or Step 2)
2. Extracts RHYTHM features (circadian):
   - Calculates hourly totals per fly per day per ZT
   - Runs daily cosinor regression and Lomb-Scargle periodogram (per fly per day)
   - Detects activity onset/offset and interdaily stability
   - Aggregates to per-fly means and SDs
3. Extracts SLEEP features:
   - Calculates daily sleep metrics per fly per day
   - Aggregates to per-fly means
4. Merges rhythm + sleep features into final ML_features table

Output: ML_features.csv

This is the final step in the pipeline. The output is ready for ML analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from scipy import stats
from scipy.signal import lombscargle
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ============================================================
#   USER CONFIGURATION
# ============================================================

# Default file paths
DEFAULT_INPUT = 'data/processed/dam_data_cleaned.csv'  # Try cleaned first, fallback to prepared
DEFAULT_OUTPUT = 'data/processed/ML_features.csv'

# Feature extraction settings
DEFAULT_EXCLUDE_DAYS = [1, 7]
DEFAULT_SLEEP_THRESHOLD_MIN = 5  # minutes of inactivity defining sleep
DEFAULT_BIN_LENGTH_MIN = 1  # MT resolution in minutes
DEFAULT_PERIOD = 24  # Circadian period in hours
DEFAULT_LIGHTS_ON = 9  # Hour when lights turn on (ZT0)


# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def is_ok(x):
    """Check if a value is valid (not NA, not empty, not "na")."""
    if pd.isna(x):
        return False
    x_str = str(x).lower()
    return x_str != "" and x_str != "na"


# ============================================================
#   RHYTHM FEATURES (CIRCADIAN)
# ============================================================

def prepare_rhythm_data(dam_clean, exclude_days):
    """Prepare data for rhythm analysis (MT only)."""
    df = dam_clean.copy()
    
    # Filter to MT only
    if 'reading' in df.columns:
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
    else:
        print("✓ Data is already MT-only")
    
    # Normalize column names
    col_mapping = {
        'Monitor': 'monitor', 'Channel': 'channel', 'Value': 'value',
        'Genotype': 'genotype', 'Sex': 'sex', 'Treatment': 'treatment'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Exclude days
    if 'Exp_Day' in df.columns:
        before = len(df)
        df = df[~df['Exp_Day'].isin(exclude_days)].copy()
        after = len(df)
        print(f"✓ Excluded days {exclude_days}: {before - after:,} rows removed")
    
    # Drop rows with missing metadata or ZT
    before = len(df)
    df = df.dropna(subset=['genotype', 'sex', 'treatment', 'ZT', 'Exp_Day'])
    after = len(df)
    if before != after:
        print(f"✓ Removed {before - after:,} rows with missing data")
    
    # Create fly_id
    df['fly_id'] = df['monitor'].astype(str) + '-' + df['channel'].astype(str)
    
    # Convert ZT to numeric
    df['ZT'] = pd.to_numeric(df['ZT'], errors='coerce')
    df = df.dropna(subset=['ZT'])
    
    return df


def calculate_hourly_totals(dam_rhythm):
    """Calculate hourly totals per fly per day per ZT."""
    hourly_day = dam_rhythm.groupby(
        ['fly_id', 'genotype', 'sex', 'treatment', 'Exp_Day', 'ZT'],
        as_index=False
    )['value'].sum().rename(columns={'value': 'hourly_MT'})
    
    print(f"✓ Calculated hourly totals for {hourly_day['fly_id'].nunique()} flies")
    return hourly_day


def run_daily_cosinor(hourly_data, period=24):
    """
    Run cosinor regression for one fly-day.
    
    Model: hourly_MT ~ Mesor + A*cos(2π*ZT/period) + B*sin(2π*ZT/period)
    
    Returns:
        Series with fly_id, Exp_Day, Mesor, Amp, Phase, Cos_p
    """
    df = hourly_data.copy()
    
    # Create cos/sin terms
    df['rad'] = 2 * np.pi * df['ZT'] / period
    df['cos_term'] = np.cos(df['rad'])
    df['sin_term'] = np.sin(df['rad'])
    
    # Fit linear regression
    X = df[['cos_term', 'sin_term']].values
    y = df['hourly_MT'].values
    
    if len(y) < 3:
        # Not enough data points
        return pd.Series({
            'fly_id': df['fly_id'].iloc[0] if len(df) > 0 else None,
            'Exp_Day': df['Exp_Day'].iloc[0] if len(df) > 0 else None,
            'Mesor': np.nan,
            'Amp': np.nan,
            'Phase': np.nan,
            'Cos_p': np.nan
        })
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    intercept = model.intercept_
    cos_coef = model.coef_[0]
    sin_coef = model.coef_[1]
    
    # Calculate amplitude
    amplitude = np.sqrt(cos_coef**2 + sin_coef**2)
    
    # Calculate phase (in hours)
    phase_rad = np.arctan2(-sin_coef, cos_coef)
    phase_hours = (period * phase_rad / (2 * np.pi)) % period
    
    # Calculate p-value using F-test
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    n = len(y)
    p = 2  # number of predictors (cos, sin)
    
    if ss_tot > 0 and n > p:
        f_stat = ((ss_tot - ss_res) / p) / (ss_res / (n - p - 1))
        p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
    else:
        p_value = np.nan
    
    return pd.Series({
        'fly_id': df['fly_id'].iloc[0],
        'Exp_Day': df['Exp_Day'].iloc[0],
        'Mesor': intercept,
        'Amp': amplitude,
        'Phase': phase_hours,
        'Cos_p': p_value
    })


def run_daily_periodogram(hourly_data):
    """
    Compute Lomb-Scargle periodogram to detect dominant period and rhythm strength.
    Returns Series with periodogram_period and periodogram_power (both NaN if insufficient data).
    """
    df = hourly_data.copy()
    zt_hours = df['ZT'].values
    activity_hourly = df['hourly_MT'].values

    if len(activity_hourly) < 3 or np.isnan(activity_hourly).any() or np.isnan(zt_hours).any():
        return pd.Series({
            'fly_id': df['fly_id'].iloc[0] if len(df) > 0 else None,
            'Exp_Day': df['Exp_Day'].iloc[0] if len(df) > 0 else None,
            'periodogram_period': np.nan,
            'periodogram_power': np.nan
        })

    y = activity_hourly - np.mean(activity_hourly)
    if np.var(y) == 0:
        return pd.Series({
            'fly_id': df['fly_id'].iloc[0],
            'Exp_Day': df['Exp_Day'].iloc[0],
            'periodogram_period': np.nan,
            'periodogram_power': np.nan
        })

    t = zt_hours * 2 * np.pi / 24
    periods = np.linspace(18, 30, 1000)
    freqs = 2 * np.pi / periods
    power = lombscargle(t, y, freqs, normalize=True)
    max_idx = np.argmax(power)
    return pd.Series({
        'fly_id': df['fly_id'].iloc[0],
        'Exp_Day': df['Exp_Day'].iloc[0],
        'periodogram_period': periods[max_idx],
        'periodogram_power': power[max_idx]
    })


def run_daily_onset_offset(hourly_data, threshold_sd=1.0, min_bins=2):
    """
    Detect activity onset and offset for one fly-day.
    Onset: first sustained crossing above threshold after the daily minimum.
    Offset: last above-threshold bin followed by a sustained below-threshold period.
    """
    df = hourly_data.sort_values('ZT').copy()
    zt = df['ZT'].values
    y = df['hourly_MT'].values.astype(float)

    null = pd.Series({
        'fly_id': df['fly_id'].iloc[0] if len(df) > 0 else None,
        'Exp_Day': df['Exp_Day'].iloc[0] if len(df) > 0 else None,
        'daily_activity_onset_zt': np.nan,
        'daily_activity_offset_zt': np.nan,
    })
    if len(y) < 3 or np.isnan(y).any():
        return null

    smoothed = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    threshold = np.percentile(smoothed, 10) + threshold_sd * np.std(smoothed)
    above = smoothed > threshold

    onset_zt = np.nan
    min_idx = int(np.argmin(smoothed))
    for i in range(min_idx, len(above) - min_bins + 1):
        if np.all(above[i:i + min_bins]):
            onset_zt = zt[i]
            break

    offset_zt = np.nan
    for i in range(len(above) - min_bins - 1, -1, -1):
        if above[i] and np.all(~above[i + 1:i + 1 + min_bins]):
            offset_zt = zt[i]
            break

    return pd.Series({
        'fly_id': df['fly_id'].iloc[0],
        'Exp_Day': df['Exp_Day'].iloc[0],
        'daily_activity_onset_zt': onset_zt,
        'daily_activity_offset_zt': offset_zt,
    })


def compute_interdaily_stability(hourly_day):
    """
    Compute interdaily stability (IS) per fly from the full multi-day hourly time series.
    IS = [n * sum_h(mean_h - grand_mean)^2] / [p * sum_i(x_i - grand_mean)^2]
    """
    results = []
    for fly_id, group in hourly_day.groupby('fly_id'):
        x = group['hourly_MT'].values.astype(float)
        if len(x) < 2 or np.isnan(x).any():
            results.append({'fly_id': fly_id, 'interdaily_stability': np.nan})
            continue
        grand_mean = np.mean(x)
        denom = np.sum((x - grand_mean) ** 2)
        if denom == 0:
            results.append({'fly_id': fly_id, 'interdaily_stability': np.nan})
            continue
        hourly_means = group.groupby('ZT')['hourly_MT'].mean()
        p = len(hourly_means)
        n = len(x)
        is_val = (n * np.sum((hourly_means.values - grand_mean) ** 2)) / (p * denom)
        results.append({'fly_id': fly_id, 'interdaily_stability': is_val})
    return pd.DataFrame(results)


def compute_rhythm_features(dam_clean, exclude_days, period):
    """Compute per-fly rhythm features (daily cosinor, then aggregate)."""
    print("\n[Step 3.1] Computing rhythm features...")
    
    # Prepare data
    dam_rhythm = prepare_rhythm_data(dam_clean, exclude_days)

    # Calculate hourly totals
    hourly_day = calculate_hourly_totals(dam_rhythm)

    # Compute interdaily stability from full cross-day series (one value per fly)
    is_features = compute_interdaily_stability(hourly_day)

    # Run daily cosinor, periodogram, and onset/offset for each fly-day
    print("  Running daily cosinor regression, periodogram, and onset/offset...")
    daily_cosinor_list = []
    daily_periodogram_list = []
    daily_onset_offset_list = []

    for (fly_id, exp_day), group in hourly_day.groupby(['fly_id', 'Exp_Day']):
        daily_cosinor_list.append(run_daily_cosinor(group, period))
        daily_periodogram_list.append(run_daily_periodogram(group))
        daily_onset_offset_list.append(run_daily_onset_offset(group))

    daily_cosinor = pd.DataFrame(daily_cosinor_list)
    daily_periodogram = pd.DataFrame(daily_periodogram_list)
    daily_onset_offset = pd.DataFrame(daily_onset_offset_list)
    print(f"✓ Computed daily features for {len(daily_cosinor)} fly-days")

    # Merge all daily results
    daily_features = daily_cosinor.merge(
        daily_periodogram[['fly_id', 'Exp_Day', 'periodogram_period', 'periodogram_power']],
        on=['fly_id', 'Exp_Day'], how='left'
    ).merge(
        daily_onset_offset[['fly_id', 'Exp_Day', 'daily_activity_onset_zt', 'daily_activity_offset_zt']],
        on=['fly_id', 'Exp_Day'], how='left'
    )

    # Get metadata
    metadata = dam_rhythm[['fly_id', 'genotype', 'sex', 'treatment']].drop_duplicates()
    daily_features = daily_features.merge(metadata, on='fly_id', how='left')

    # Aggregate to per-fly means and SDs
    print("  Aggregating to per-fly features...")
    cosinor_features = daily_features.groupby('fly_id').agg({
        'genotype': 'first',
        'sex': 'first',
        'treatment': 'first',
        'Mesor': ['mean', 'std'],
        'Amp': ['mean', 'std'],
        'Phase': ['mean', 'std'],
        'Cos_p': lambda x: (x < 0.05).sum(),
        'periodogram_period': ['mean', 'std'],
        'periodogram_power': 'mean',
        'daily_activity_onset_zt': ['mean', 'std'],
        'daily_activity_offset_zt': ['mean', 'std'],
    }).reset_index()

    # Flatten column names (lowercase, matching sql_db_pipeline convention)
    cosinor_features.columns = [
        'fly_id', 'genotype', 'sex', 'treatment',
        'mesor_mean', 'mesor_sd', 'amplitude_mean', 'amplitude_sd',
        'phase_mean', 'phase_sd', 'rhythmic_days',
        'periodogram_period_mean', 'periodogram_period_sd', 'periodogram_power_mean',
        'activity_onset_zt_mean', 'activity_onset_zt_sd',
        'activity_offset_zt_mean', 'activity_offset_zt_sd',
    ]

    # Merge interdaily stability
    cosinor_features = cosinor_features.merge(is_features, on='fly_id', how='left')

    print(f"✓ Computed rhythm features for {len(cosinor_features)} flies")
    return cosinor_features


# ============================================================
#   SLEEP FEATURES
# ============================================================

def prepare_sleep_data(dam_clean, exclude_days):
    """Prepare data for sleep analysis (MT only)."""
    df = dam_clean.copy()
    
    # Filter to MT only
    if 'reading' in df.columns:
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
    
    # Normalize column names
    col_mapping = {
        'Monitor': 'monitor', 'Channel': 'channel', 'Value': 'value',
        'Genotype': 'genotype', 'Sex': 'sex', 'Treatment': 'treatment'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Exclude days
    if 'Exp_Day' in df.columns:
        df = df[~df['Exp_Day'].isin(exclude_days)].copy()
    
    # Create fly_id and ZT_num
    df['fly_id'] = df['monitor'].astype(str) + '-' + df['channel'].astype(str)
    df['ZT_num'] = pd.to_numeric(df['ZT'], errors='coerce')
    
    # Rename value to movement
    df = df.rename(columns={'value': 'movement'})
    
    # Sort by fly_id, Exp_Day, datetime
    df = df.sort_values(['fly_id', 'Exp_Day', 'datetime']).reset_index(drop=True)
    
    return df


def compute_sleep_features_daily(df, bin_length_min, sleep_threshold_min):
    """
    Compute daily sleep features for one fly-day.
    """
    df = df.copy().reset_index(drop=True)
    
    # Detect inactivity and sleep
    df['inactive'] = (df['movement'] == 0)
    df['run_id'] = (df['inactive'] != df['inactive'].shift(1, fill_value=False)).cumsum()
    df['run_len_min'] = df.groupby('run_id')['inactive'].transform('count') * bin_length_min
    df['sleep'] = df['inactive'] & (df['run_len_min'] >= sleep_threshold_min)
    df['is_day'] = df['ZT_num'] < 12
    
    # Detect sleep bout starts
    df['start'] = df['sleep'] & (~df['sleep'].shift(1, fill_value=False))
    df['bout_id'] = (df['start'].cumsum() * df['sleep']).replace(0, np.nan)
    
    # Extract bouts
    # Use as_index=False to keep grouping columns as regular columns (avoids reset_index conflict)
    bouts = df[df['sleep'] & df['bout_id'].notna()].groupby(['fly_id', 'Exp_Day', 'bout_id'], as_index=False).agg({
        'bout_id': 'count',  # Count bins per bout (bout_len_min will be calculated)
        'is_day': 'first'
    })
    # Rename the aggregated bout_id count to avoid name conflict with grouping column
    bouts = bouts.rename(columns={'bout_id': 'bout_count'})
    bouts['bout_len_min'] = bouts['bout_count'] * bin_length_min
    
    # Calculate metrics
    total_bouts = len(bouts)
    day_bouts = bouts['is_day'].sum()
    night_bouts = (~bouts['is_day']).sum()
    
    total_sleep_min = df['sleep'].sum() * bin_length_min
    day_sleep_min = (df['sleep'] & df['is_day']).sum() * bin_length_min
    night_sleep_min = (df['sleep'] & ~df['is_day']).sum() * bin_length_min
    
    total_hours = len(df) * bin_length_min / 60
    
    mean_bout_min = bouts['bout_len_min'].mean() if total_bouts > 0 else np.nan
    max_bout_min = bouts['bout_len_min'].max() if total_bouts > 0 else np.nan
    
    mean_day_bout_min = bouts[bouts['is_day']]['bout_len_min'].mean() if day_bouts > 0 else np.nan
    max_day_bout_min = bouts[bouts['is_day']]['bout_len_min'].max() if day_bouts > 0 else np.nan
    
    mean_night_bout_min = bouts[~bouts['is_day']]['bout_len_min'].mean() if night_bouts > 0 else np.nan
    max_night_bout_min = bouts[~bouts['is_day']]['bout_len_min'].max() if night_bouts > 0 else np.nan
    
    fragmentation_hour = total_bouts / total_hours if total_hours > 0 else np.nan
    fragmentation_min_sleep = total_bouts / total_sleep_min if total_sleep_min > 0 else np.nan
    
    # Transition probabilities
    sleep_vec = df['sleep'].values
    N = len(sleep_vec)
    N_S = sleep_vec.sum()
    N_W = (~sleep_vec).sum()
    
    if N > 1:
        N_S_to_W = ((~sleep_vec[1:]) & sleep_vec[:-1]).sum()
        N_W_to_S = (sleep_vec[1:] & (~sleep_vec[:-1])).sum()
    else:
        N_S_to_W = 0
        N_W_to_S = 0
    
    P_wake = N_S_to_W / N_S if N_S > 0 else np.nan
    P_doze = N_W_to_S / N_W if N_W > 0 else np.nan
    
    # Sleep latency and WASO (from dark phase)
    dark_df = df[(df['ZT_num'] >= 12) & (df['ZT_num'] < 24)].reset_index(drop=True)
    
    if len(dark_df) > 0 and dark_df['sleep'].any():
        idx = dark_df['sleep'].idxmax()
        sleep_latency_min = idx * bin_length_min
        WASO_min = (~dark_df.loc[idx:, 'sleep']).sum() * bin_length_min
    else:
        sleep_latency_min = np.nan
        WASO_min = np.nan
    
    # Mean wake bout length
    df['wake'] = ~df['sleep']
    df['wake_run'] = (df['wake'] != df['wake'].shift(1, fill_value=False)).cumsum()
    wake_bouts = df[df['wake']].groupby('wake_run').size() * bin_length_min
    mean_wake_bout_min = wake_bouts.mean() if len(wake_bouts) > 0 else np.nan
    
    return pd.Series({
        'fly_id': df['fly_id'].iloc[0],
        'Exp_Day': df['Exp_Day'].iloc[0],
        'total_sleep_min': total_sleep_min,
        'day_sleep_min': day_sleep_min,
        'night_sleep_min': night_sleep_min,
        'total_bouts': total_bouts,
        'day_bouts': day_bouts,
        'night_bouts': night_bouts,
        'mean_bout_min': mean_bout_min,
        'max_bout_min': max_bout_min,
        'mean_day_bout_min': mean_day_bout_min,
        'max_day_bout_min': max_day_bout_min,
        'mean_night_bout_min': mean_night_bout_min,
        'max_night_bout_min': max_night_bout_min,
        'fragmentation_bouts_per_hour': fragmentation_hour,
        'fragmentation_bouts_per_min_sleep': fragmentation_min_sleep,
        'P_wake': P_wake,
        'P_doze': P_doze,
        'sleep_latency_min': sleep_latency_min,
        'WASO_min': WASO_min,
        'mean_wake_bout_min': mean_wake_bout_min
    })


def compute_sleep_features(dam_clean, exclude_days, bin_length_min, sleep_threshold_min):
    """Compute per-fly sleep features (daily metrics, then aggregate)."""
    print("\n[Step 3.2] Computing sleep features...")
    
    # Prepare data
    mt_data = prepare_sleep_data(dam_clean, exclude_days)
    print(f"✓ Prepared data for {mt_data['fly_id'].nunique()} flies")
    
    # Compute daily sleep features
    print("  Computing daily sleep metrics...")
    daily_sleep_list = []
    
    for (fly_id, exp_day), group in mt_data.groupby(['fly_id', 'Exp_Day']):
        result = compute_sleep_features_daily(group, bin_length_min, sleep_threshold_min)
        daily_sleep_list.append(result)
    
    daily_sleep_features = pd.DataFrame(daily_sleep_list)
    print(f"✓ Computed daily sleep features for {len(daily_sleep_features)} fly-days")
    
    # Get metadata
    metadata = mt_data[['fly_id', 'genotype', 'sex', 'treatment']].drop_duplicates()
    daily_sleep_features = daily_sleep_features.merge(metadata, on='fly_id', how='left')
    
    # Aggregate to per-fly means
    print("  Aggregating to per-fly features...")
    sleep_ML_features = daily_sleep_features.groupby('fly_id').agg({
        'genotype': 'first',
        'sex': 'first',
        'treatment': 'first',
        'total_sleep_min': 'mean',
        'day_sleep_min': 'mean',
        'night_sleep_min': 'mean',
        'total_bouts': 'mean',
        'day_bouts': 'mean',
        'night_bouts': 'mean',
        'mean_bout_min': 'mean',
        'max_bout_min': 'mean',
        'mean_day_bout_min': 'mean',
        'max_day_bout_min': 'mean',
        'mean_night_bout_min': 'mean',
        'max_night_bout_min': 'mean',
        'fragmentation_bouts_per_hour': 'mean',
        'fragmentation_bouts_per_min_sleep': 'mean',
        'mean_wake_bout_min': 'mean',
        'P_wake': 'mean',
        'P_doze': 'mean',
        'sleep_latency_min': 'mean',
        'WASO_min': 'mean',
        'Exp_Day': 'count'  # n_days
    }).reset_index()
    
    # Rename columns (lowercase, matching sql_db_pipeline convention)
    sleep_ML_features = sleep_ML_features.rename(columns={
        'total_sleep_min': 'total_sleep_mean',
        'day_sleep_min': 'day_sleep_mean',
        'night_sleep_min': 'night_sleep_mean',
        'total_bouts': 'total_bouts_mean',
        'day_bouts': 'day_bouts_mean',
        'night_bouts': 'night_bouts_mean',
        'mean_bout_min': 'mean_bout_mean',
        'max_bout_min': 'max_bout_mean',
        'mean_day_bout_min': 'mean_day_bout_mean',
        'max_day_bout_min': 'max_day_bout_mean',
        'mean_night_bout_min': 'mean_night_bout_mean',
        'max_night_bout_min': 'max_night_bout_mean',
        'fragmentation_bouts_per_hour': 'frag_bouts_per_hour_mean',
        'fragmentation_bouts_per_min_sleep': 'frag_bouts_per_min_sleep_mean',
        'mean_wake_bout_min': 'mean_wake_bout_mean',
        'P_wake': 'p_wake_mean',
        'P_doze': 'p_doze_mean',
        'sleep_latency_min': 'sleep_latency_mean',
        'WASO_min': 'waso_mean',
        'Exp_Day': 'n_days'
    })
    
    print(f"✓ Computed sleep features for {len(sleep_ML_features)} flies")
    
    return sleep_ML_features


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def create_feature_table(
    input_file=None,
    output_file=None,
    exclude_days=DEFAULT_EXCLUDE_DAYS,
    sleep_threshold_min=DEFAULT_SLEEP_THRESHOLD_MIN,
    bin_length_min=DEFAULT_BIN_LENGTH_MIN,
    period=DEFAULT_PERIOD
):
    """
    Main function to create ML feature table.
    
    Args:
        input_file: Path to cleaned/prepared data
        output_file: Path to save ML_features
        exclude_days: List of days to exclude
        sleep_threshold_min: Minimum minutes of inactivity for sleep
        bin_length_min: Length of each time bin in minutes
        period: Circadian period in hours
        
    Returns:
        ML_features DataFrame
    """
    print("=" * 60)
    print("PIPELINE STEP 3: CREATE FEATURE TABLE")
    print("=" * 60)
    
    # Set defaults
    if input_file is None:
        input_file = DEFAULT_INPUT
    if output_file is None:
        output_file = DEFAULT_OUTPUT
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, input_file) if not os.path.isabs(input_file) else input_file
    output_file = os.path.join(script_dir, output_file) if not os.path.isabs(output_file) else output_file
    
    # Try cleaned data first, fallback to prepared
    if not os.path.exists(input_file):
        # Try prepared data instead
        prepared_file = input_file.replace('dam_data_cleaned.csv', 'dam_data_prepared.csv')
        if os.path.exists(prepared_file):
            print(f"Note: Using prepared data instead: {prepared_file}")
            input_file = prepared_file
        else:
            print(f"ERROR: Input file not found: {input_file}")
            print("Please run Step 1 first: python 1-prepare_data_and_health.py")
            sys.exit(1)
    
    # ============================================================
    # STEP 1: Load data
    # ============================================================
    print(f"\n[Step 3.0] Loading data from {input_file}...")
    
    dam_clean = pd.read_csv(input_file)
    print(f"✓ Loaded {len(dam_clean):,} rows")
    print(f"   Unique flies: {dam_clean.groupby(['monitor', 'channel']).ngroups}")
    
    # ============================================================
    # STEP 2: Compute rhythm features
    # ============================================================
    cosinor_features = compute_rhythm_features(dam_clean, exclude_days, period)
    
    # ============================================================
    # STEP 3: Compute sleep features
    # ============================================================
    sleep_features = compute_sleep_features(dam_clean, exclude_days, bin_length_min, sleep_threshold_min)
    
    # ============================================================
    # STEP 3: Merge features
    # ============================================================
    print("\n[Step 3.3] Merging rhythm and sleep features...")
    
    ML_features = cosinor_features.merge(
        sleep_features,
        on=['fly_id', 'genotype', 'sex', 'treatment'],
        how='inner'
    )
    
    print(f"✓ Merged features for {len(ML_features)} flies")
    print(f"   Features: {len(ML_features.columns)} columns")
    
    # ============================================================
    # STEP 4: Sort and save output
    # ============================================================
    print(f"\n[Step 3.4] Sorting and saving ML features to {output_file}...")
    
    # Sort by monitor and channel (extract from fly_id for proper numeric sorting)
    def extract_monitor_channel(fly_id):
        """Extract monitor and channel from fly_id for sorting."""
        parts = fly_id.split('-')
        monitor = int(parts[0])
        channel = int(parts[1])
        return (monitor, channel)
    
    ML_features['_sort_monitor'] = ML_features['fly_id'].apply(lambda x: extract_monitor_channel(x)[0])
    ML_features['_sort_channel'] = ML_features['fly_id'].apply(lambda x: extract_monitor_channel(x)[1])
    ML_features = ML_features.sort_values(['_sort_monitor', '_sort_channel']).drop(columns=['_sort_monitor', '_sort_channel'])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    ML_features.to_csv(output_file, index=False)
    print(f"✓ Saved ML features successfully")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE TABLE SUMMARY")
    print("=" * 60)
    print(f"Total flies: {len(ML_features)}")
    print(f"Total features: {len(ML_features.columns)}")
    print(f"\nRhythm features: mesor_mean/sd, amplitude_mean/sd, phase_mean/sd, rhythmic_days, periodogram_period_mean/sd, periodogram_power_mean, activity_onset/offset_zt_mean/sd, interdaily_stability")
    print(f"Sleep features: {len([c for c in ML_features.columns if 'sleep' in c.lower() or 'bout' in c.lower() or 'p_wake' in c or 'p_doze' in c or 'waso' in c or 'latency' in c.lower()])} columns")
    print(f"\nGenotypes: {sorted(ML_features['genotype'].unique())}")
    print(f"Sexes: {sorted(ML_features['sex'].unique())}")
    print(f"Treatments: {sorted(ML_features['treatment'].unique())}")
    
    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    print(f"\nPipeline complete! ML features ready for analysis:")
    print(f"  {output_file}")
    
    return ML_features


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Pipeline Step 3: Create ML feature table',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', type=str, default=None,
                       help='Input cleaned/prepared data file (default: dam_data_cleaned.csv)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ML features file (default: ML_features.csv)')
    parser.add_argument('--exclude-days', nargs='+', type=int, default=DEFAULT_EXCLUDE_DAYS,
                       help=f'Days to exclude (default: {DEFAULT_EXCLUDE_DAYS})')
    parser.add_argument('--sleep-threshold', type=int, default=DEFAULT_SLEEP_THRESHOLD_MIN,
                       help=f'Minimum minutes of inactivity for sleep (default: {DEFAULT_SLEEP_THRESHOLD_MIN})')
    
    args = parser.parse_args()
    
    create_feature_table(
        input_file=args.input,
        output_file=args.output,
        exclude_days=args.exclude_days,
        sleep_threshold_min=args.sleep_threshold
    )


if __name__ == "__main__":
    main()

