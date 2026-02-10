#!/usr/bin/env python3
"""
Pipeline Step 3: Create Feature Table

This script:
1. Reads cleaned data (from Step 1 or Step 2)
2. Extracts RHYTHM features (circadian):
   - Calculates hourly totals per fly per day per ZT
   - Runs daily cosinor regression (per fly per day)
   - Aggregates to per-fly means and SDs
3. Extracts SLEEP features:
   - Calculates daily sleep metrics per fly per day
   - Aggregates to per-fly means
4. Merges rhythm + sleep features into final ML_features table

Output: Saved to database (features table)

This is the final step in the pipeline. The output is ready for ML analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path
from importlib import import_module

# Allow importing config from parent (db-pipeline) when run from death-classification-pipeline
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    import psycopg2
    from psycopg2.extras import execute_values
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    USE_DATABASE = False

# ============================================================
#   USER CONFIGURATION
# ============================================================


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
    return x_str != "" and x_str != "na" and x_str != "nan"


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
    
    # Normalize column names
    col_mapping = {
        'Monitor': 'monitor', 'Channel': 'channel', 'Value': 'value',
        'Genotype': 'genotype', 'Sex': 'sex', 'Treatment': 'treatment'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Exclude days
    if 'exp_day' in df.columns:
        df = df[~df['exp_day'].isin(exclude_days)].copy()
    
    # Drop rows with missing metadata or zt
    df = df.dropna(subset=['genotype', 'sex', 'treatment', 'zt', 'exp_day'])
    
    # Use fly_id from database if available, otherwise create it (format: M{monitor}_Ch{channel:02d})
    if 'fly_id' not in df.columns or df['fly_id'].isna().all():
        df['fly_id'] = 'M' + df['monitor'].astype(str) + '_Ch' + df['channel'].astype(str).str.zfill(2)
    
    # Convert zt to numeric
    df['zt'] = pd.to_numeric(df['zt'], errors='coerce')
    df = df.dropna(subset=['zt'])
    
    return df


def calculate_hourly_totals(dam_rhythm):
    """Calculate hourly totals per fly per day per ZT."""
    hourly_day = dam_rhythm.groupby(
        ['fly_id', 'genotype', 'sex', 'treatment', 'exp_day', 'zt'],
        as_index=False
    )['value'].sum().rename(columns={'value': 'hourly_MT'})
    
    # Merge back monitor and channel (they're constant per fly_id)
    monitor_channel = dam_rhythm[['fly_id', 'monitor', 'channel']].drop_duplicates()
    hourly_day = hourly_day.merge(monitor_channel, on='fly_id', how='left')
    
    return hourly_day


def run_daily_cosinor(hourly_data, period=24):
    """
    Run cosinor regression for one fly-day.
    
    Model: hourly_MT ~ Mesor + A*cos(2π*ZT/period) + B*sin(2π*ZT/period)
    
    Returns:
        Series with fly_id, exp_day, Mesor, Amp, phase, Cos_p
    """
    df = hourly_data.copy()
    
    # Create cos/sin terms
    df['rad'] = 2 * np.pi * df['zt'] / period
    df['cos_term'] = np.cos(df['rad'])
    df['sin_term'] = np.sin(df['rad'])
    
    # Fit linear regression
    X = df[['cos_term', 'sin_term']].values
    y = df['hourly_MT'].values
    
    if len(y) < 3:
        # Not enough data points
        return pd.Series({
            'fly_id': df['fly_id'].iloc[0] if len(df) > 0 else None,
            'exp_day': df['exp_day'].iloc[0] if len(df) > 0 else None,
            'monitor': df['monitor'].iloc[0] if len(df) > 0 and 'monitor' in df.columns else None,
            'channel': df['channel'].iloc[0] if len(df) > 0 and 'channel' in df.columns else None,
            'Mesor': np.nan,
            'Amp': np.nan,
            'phase': np.nan,
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
        'exp_day': df['exp_day'].iloc[0],
        'monitor': df['monitor'].iloc[0] if 'monitor' in df.columns else None,
        'channel': df['channel'].iloc[0] if 'channel' in df.columns else None,
        'Mesor': intercept,
        'Amp': amplitude,
        'phase': phase_hours,
        'Cos_p': p_value
    })


def compute_rhythm_features(dam_clean, exclude_days, period):
    """Compute per-fly rhythm features (daily cosinor, then aggregate)."""
    # Prepare data
    dam_rhythm = prepare_rhythm_data(dam_clean, exclude_days)
    
    # Calculate hourly totals
    hourly_day = calculate_hourly_totals(dam_rhythm)
    
    # Run daily cosinor for each fly-day
    daily_cosinor_list = []
    
    for (fly_id, exp_day), group in hourly_day.groupby(['fly_id', 'exp_day']):
        result = run_daily_cosinor(group, period)
        daily_cosinor_list.append(result)
    
    daily_cosinor = pd.DataFrame(daily_cosinor_list)
    
    # Get metadata (preserve monitor and channel)
    metadata = dam_rhythm[['fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment']].drop_duplicates()
    
    # Merge metadata - ensure monitor/channel are present (from Series or metadata)
    if 'monitor' in daily_cosinor.columns and 'channel' in daily_cosinor.columns:
        # Monitor/channel already present from Series, just merge other metadata
        daily_features = daily_cosinor.merge(metadata[['fly_id', 'genotype', 'sex', 'treatment']], on='fly_id', how='left')
        # Fill any missing monitor/channel values from metadata
        if daily_features['monitor'].isna().any() or daily_features['channel'].isna().any():
            monitor_channel = metadata[['fly_id', 'monitor', 'channel']]
            daily_features = daily_features.drop(columns=['monitor', 'channel'], errors='ignore')
            daily_features = daily_features.merge(monitor_channel, on='fly_id', how='left')
    else:
        # Monitor/channel missing, merge full metadata
        daily_features = daily_cosinor.merge(metadata, on='fly_id', how='left')
    
    # Aggregate to per-fly means and SDs
    cosinor_features = daily_features.groupby('fly_id').agg({
        'monitor': 'first',
        'channel': 'first',
        'genotype': 'first',
        'sex': 'first',
        'treatment': 'first',
        'Mesor': ['mean', 'std'],
        'Amp': ['mean', 'std'],
        'phase': ['mean', 'std'],
        'Cos_p': lambda x: (x < 0.05).sum()  # rhythmic_days
    }).reset_index()
    
    # Flatten column names (all lowercase)
    cosinor_features.columns = [
        'fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment',
        'mesor_mean', 'mesor_sd', 'amplitude_mean', 'amplitude_sd',
        'phase_mean', 'phase_sd', 'rhythmic_days'
    ]
    
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
    if 'exp_day' in df.columns:
        df = df[~df['exp_day'].isin(exclude_days)].copy()
    
    # Use fly_id from database if available, otherwise create it (format: M{monitor}_Ch{channel:02d})
    if 'fly_id' not in df.columns or df['fly_id'].isna().all():
        df['fly_id'] = 'M' + df['monitor'].astype(str) + '_Ch' + df['channel'].astype(str).str.zfill(2)
    df['zt_num'] = pd.to_numeric(df['zt'], errors='coerce')
    
    # Rename value to movement
    df = df.rename(columns={'value': 'movement'})
    
    # Sort by fly_id, exp_day, datetime
    df = df.sort_values(['fly_id', 'exp_day', 'datetime']).reset_index(drop=True)
    
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
    df['is_day'] = df['zt_num'] < 12
    
    # Detect sleep bout starts
    df['start'] = df['sleep'] & (~df['sleep'].shift(1, fill_value=False))
    df['bout_id'] = (df['start'].cumsum() * df['sleep']).replace(0, np.nan)
    
    # Extract bouts
    # Use as_index=False to keep grouping columns as regular columns (avoids reset_index conflict)
    bouts = df[df['sleep'] & df['bout_id'].notna()].groupby(['fly_id', 'exp_day', 'bout_id'], as_index=False).agg({
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
    dark_df = df[(df['zt_num'] >= 12) & (df['zt_num'] < 24)].reset_index(drop=True)
    
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
        'exp_day': df['exp_day'].iloc[0],
        'monitor': df['monitor'].iloc[0] if 'monitor' in df.columns else None,
        'channel': df['channel'].iloc[0] if 'channel' in df.columns else None,
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
    # Prepare data
    mt_data = prepare_sleep_data(dam_clean, exclude_days)
    
    # Compute daily sleep features
    daily_sleep_list = []
    
    for (fly_id, exp_day), group in mt_data.groupby(['fly_id', 'exp_day']):
        result = compute_sleep_features_daily(group, bin_length_min, sleep_threshold_min)
        daily_sleep_list.append(result)
    
    daily_sleep_features = pd.DataFrame(daily_sleep_list)
    
    # Get metadata (preserve monitor and channel)
    metadata = mt_data[['fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment']].drop_duplicates()
    
    # Merge metadata - ensure monitor/channel are present (from Series or metadata)
    if 'monitor' in daily_sleep_features.columns and 'channel' in daily_sleep_features.columns:
        # Monitor/channel already present from Series, just merge other metadata
        daily_sleep_features = daily_sleep_features.merge(metadata[['fly_id', 'genotype', 'sex', 'treatment']], on='fly_id', how='left')
        # Fill any missing monitor/channel values from metadata
        if daily_sleep_features['monitor'].isna().any() or daily_sleep_features['channel'].isna().any():
            monitor_channel = metadata[['fly_id', 'monitor', 'channel']]
            daily_sleep_features = daily_sleep_features.drop(columns=['monitor', 'channel'], errors='ignore')
            daily_sleep_features = daily_sleep_features.merge(monitor_channel, on='fly_id', how='left')
    else:
        # Monitor/channel missing, merge full metadata
        daily_sleep_features = daily_sleep_features.merge(metadata, on='fly_id', how='left')
    
    # Aggregate to per-fly means
    sleep_ML_features = daily_sleep_features.groupby('fly_id').agg({
        'monitor': 'first',
        'channel': 'first',
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
        'exp_day': 'count'  # n_days
    }).reset_index()
    
    # Rename columns (use lowercase to match database schema)
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
        'P_wake': 'p_wake_mean',  # Use lowercase to match database schema
        'P_doze': 'p_doze_mean',  # Use lowercase to match database schema
        'sleep_latency_min': 'sleep_latency_mean',
        'WASO_min': 'waso_mean',
        'exp_day': 'n_days'
    })
    
    return sleep_ML_features


# ============================================================
#   SLIDING WINDOW (death prediction)
# ============================================================

# Step1 defaults for deriving daily labels when not in DB
_S1_REF_DAY = 4
_S1_DECLINE_THRESHOLD = 0.5
_S1_DEATH_THRESHOLD = 0.2
_S1_TRANSITION_WINDOW = 10
_S1_THRESHOLDS = {
    "A1": 12 * 60,
    "A2": 24 * 60,
    "ACTIVITY_LOW": 50,
    "INDEX_LOW": 0.02,
    "SLEEP_MAX": 1300,
    "SLEEP_BOUT": 720,
    "MISSING_MAX": 0.10
}


def _get_experiment_metadata(experiment_id):
    """Return (start_date, lights_on, lights_off) from DB."""
    if not DB_AVAILABLE:
        return None, 9, 21
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT start_date, lights_on_hour, lights_off_hour FROM experiments WHERE experiment_id = %s",
                (experiment_id,)
            )
            row = cur.fetchone()
    if not row:
        return None, 9, 21
    start_date = row[0]
    lights_on = int(row[1]) if row[1] is not None else 9
    lights_off = int(row[2]) if row[2] is not None else 21
    return start_date, lights_on, lights_off


def _get_daily_labels_from_db(experiment_id):
    """
    Load daily labels from health_reports if step 1 was run with --save-daily-health.
    Returns DataFrame with columns fly_id, report_date, status, or None if only summary rows exist.
    """
    if not DB_AVAILABLE:
        return None
    engine = create_engine(DATABASE_URL)
    query = f"SELECT fly_id, report_date, status FROM health_reports WHERE experiment_id = {int(experiment_id)}"
    df = pd.read_sql(query, engine)
    engine.dispose()
    if df.empty or len(df) == 0:
        return None
    # If we have more rows than unique flies, we have daily labels
    n_flies = df['fly_id'].nunique()
    if len(df) > n_flies:
        df['report_date'] = pd.to_datetime(df['report_date']).dt.date
        return df
    return None


def _compute_daily_labels_from_step1(dam_clean, step1, exclude_days, bin_length_min,
                                     start_date, lights_on, lights_off):
    """Derive daily fly status using step 1 helpers. Returns DataFrame with fly_id, date, STATUS."""
    dam_activity = step1.prep_data_for_health(dam_clean.copy(), exclude_days, bin_length_min)
    daily_summary = step1.calculate_daily_metrics(dam_activity, bin_length_min)
    daily_summary = step1.normalize_to_ref_day(
        daily_summary, _S1_REF_DAY, _S1_DECLINE_THRESHOLD, _S1_DEATH_THRESHOLD
    )
    transition_data = step1.startle_test(dam_activity, lights_on, lights_off, _S1_TRANSITION_WINDOW)
    thresholds = {
        "A1": _S1_THRESHOLDS["A1"],
        "A2": _S1_THRESHOLDS["A2"],
        "ACTIVITY_LOW": _S1_THRESHOLDS["ACTIVITY_LOW"],
        "INDEX_LOW": _S1_THRESHOLDS["INDEX_LOW"],
        "SLEEP_MAX": _S1_THRESHOLDS["SLEEP_MAX"],
        "SLEEP_BOUT": _S1_THRESHOLDS["SLEEP_BOUT"],
        "MISSING_MAX": _S1_THRESHOLDS["MISSING_MAX"],
    }
    fly_status = step1.classify_status(daily_summary, transition_data, thresholds)
    fly_status = step1.apply_irreversible_death(fly_status)
    # Add fly_id: merge with dam_clean (monitor, channel -> fly_id)
    mc = dam_clean[['monitor', 'channel', 'fly_id']].drop_duplicates()
    fly_status = fly_status.merge(mc, on=['monitor', 'channel'], how='left')
    fly_status['date'] = pd.to_datetime(fly_status['date']).dt.date
    return fly_status[['fly_id', 'date', 'STATUS']].rename(columns={'date': 'report_date', 'STATUS': 'status'})


def _compute_window_features_one(window_df, bin_length_min, sleep_threshold_min, lights_on_hour):
    """
    Compute features for one 24h window [9am D-1, 9am D).
    window_df: DataFrame with datetime, fly_id, value (MT counts).
    Returns dict with total_activity, activity_mean, activity_var, longest_zero_hours,
    total_sleep_min, total_bouts, mean_bout_min, max_bout_min, frag_bouts_per_hour, amplitude_24h.
    """
    step1 = import_module('1-prepare_data_and_health')
    vals = window_df['value'].astype(float)
    total_activity = float(vals.sum())
    activity_mean = float(vals.mean()) if len(vals) > 0 else np.nan
    activity_var = float(vals.var()) if len(vals) > 1 else (0.0 if len(vals) == 1 else np.nan)
    longest_zero_hours = step1.longest_zero_run(vals.values, bin_length_min) / 60.0

    # Build slice for sleep: need fly_id, exp_day, movement, zt_num (0..24 for the window)
    window_start = window_df['datetime'].min()
    w = window_df.copy()
    w['movement'] = w['value'].astype(float)
    w['zt_num'] = (w['datetime'] - window_start).dt.total_seconds() / 3600.0
    w['exp_day'] = 0
    if 'fly_id' not in w.columns and 'monitor' in w.columns and 'channel' in w.columns:
        w['fly_id'] = 'M' + w['monitor'].astype(str) + '_Ch' + w['channel'].astype(str).str.zfill(2)
    sleep_ser = compute_sleep_features_daily(w, bin_length_min, sleep_threshold_min)
    total_sleep_min = float(sleep_ser.get('total_sleep_min', np.nan))
    total_bouts = int(sleep_ser.get('total_bouts', 0))
    mean_bout_min = float(sleep_ser.get('mean_bout_min', np.nan))
    max_bout_min = float(sleep_ser.get('max_bout_min', np.nan))
    frag_bouts_per_hour = float(sleep_ser.get('fragmentation_bouts_per_hour', np.nan))

    # Cosinor on hourly totals for the window
    w['zt'] = np.floor(w['zt_num']).clip(0, 23).astype(int)
    hourly = w.groupby(['fly_id', 'exp_day', 'zt'], as_index=False)['value'].sum().rename(columns={'value': 'hourly_MT'})
    if len(hourly) >= 3:
        cos_ser = run_daily_cosinor(hourly, period=24)
        amplitude_24h = float(cos_ser.get('Amp', np.nan))
    else:
        amplitude_24h = np.nan

    return {
        'total_activity': total_activity,
        'activity_mean': activity_mean,
        'activity_var': activity_var,
        'longest_zero_hours': longest_zero_hours,
        'total_sleep_min': total_sleep_min,
        'total_bouts': total_bouts,
        'mean_bout_min': mean_bout_min,
        'max_bout_min': max_bout_min,
        'frag_bouts_per_hour': frag_bouts_per_hour,
        'amplitude_24h': amplitude_24h,
    }


def build_sliding_windows(
    dam_clean,
    experiment_id,
    step1_module,
    exclude_days,
    bin_length_min,
    sleep_threshold_min,
    exclude_qc_fail=False,
):
    """
    Build sliding-window dataset: one row per (fly_id, window_end_date).
    Features from [9am D-1, 9am D); label = status on window_end_date.
    Returns DataFrame with schema columns for features_sliding_window.
    """
    start_date, lights_on, lights_off = _get_experiment_metadata(experiment_id)
    if start_date is None:
        start_date = pd.to_datetime(dam_clean['datetime']).min().date()

    # Load death dates from flies for days_until_death
    fly_death = {}
    if DB_AVAILABLE:
        with psycopg2.connect(**DB_CONFIG) as conn:
            death_df = pd.read_sql(
                f"SELECT fly_id, death_datetime FROM flies WHERE experiment_id = {int(experiment_id)} AND death_datetime IS NOT NULL",
                conn
            )
        if not death_df.empty:
            death_df['death_date'] = pd.to_datetime(death_df['death_datetime']).dt.date
            fly_death = death_df.set_index('fly_id')['death_date'].to_dict()

    # MT only
    if 'reading' in dam_clean.columns:
        r = dam_clean[dam_clean['reading'] == 'MT'].copy()
    else:
        r = dam_clean.copy()
    if 'fly_id' not in r.columns and 'monitor' in r.columns and 'channel' in r.columns:
        r['fly_id'] = 'M' + r['monitor'].astype(str) + '_Ch' + r['channel'].astype(str).str.zfill(2)
    r['datetime'] = pd.to_datetime(r['datetime'])

    # Daily labels
    daily = _get_daily_labels_from_db(experiment_id)
    if daily is None:
        daily = _compute_daily_labels_from_step1(
            dam_clean, step1_module, exclude_days, bin_length_min,
            start_date, lights_on, lights_off
        )
    if daily is None or daily.empty:
        print("WARNING: No daily labels for sliding window; skipping.")
        return pd.DataFrame()

    # Map status to alive/dying/dead
    def map_status(s):
        if s == 'Alive':
            return 'alive'
        if s == 'Unhealthy':
            return 'dying'
        if s == 'Dead':
            return 'dead'
        return 'qc_fail'

    daily['status_mapped'] = daily['status'].map(map_status)
    if exclude_qc_fail:
        daily = daily[daily['status_mapped'] != 'qc_fail'].copy()

    # Group readings by fly_id once (avoids repeated full-table scan over 33M+ rows)
    print("  Grouping readings by fly_id...", end="", flush=True)
    by_fly = dict(list(r.groupby('fly_id')))
    print(f" {len(by_fly)} flies.")

    rows = []
    n_done = 0
    for (fly_id, report_date), grp in daily.groupby(['fly_id', 'report_date']):
        report_date = pd.Timestamp(report_date).date() if hasattr(report_date, 'year') else report_date
        end_ts = pd.Timestamp(report_date) + pd.Timedelta(hours=lights_on)
        start_ts = end_ts - pd.Timedelta(days=1)
        fly_r = by_fly.get(fly_id)
        if fly_r is None or fly_r.empty:
            continue
        window = fly_r[(fly_r['datetime'] >= start_ts) & (fly_r['datetime'] < end_ts)]
        if len(window) < 60:
            continue
        feats = _compute_window_features_one(window, bin_length_min, sleep_threshold_min, lights_on)
        _start = pd.Timestamp(start_date).date() if start_date is not None else start_date
        _end = pd.Timestamp(report_date).date() if hasattr(report_date, 'year') else report_date
        exp_day = (_end - _start).days if _start is not None else 0
        status_raw = grp['status'].iloc[0]
        status = grp['status_mapped'].iloc[0]
        if status == 'qc_fail':
            continue
        death_date = fly_death.get(fly_id)
        if death_date is not None and report_date <= death_date:
            days_until_death = (death_date - report_date).days
        else:
            days_until_death = None
        rows.append({
            'experiment_id': experiment_id,
            'fly_id': fly_id,
            'window_end_date': report_date,
            'exp_day': exp_day,
            **feats,
            'status': status,
            'status_raw': status_raw,
            'days_until_death': days_until_death,
        })
        n_done += 1
        if n_done % 500 == 0:
            print(f"  Processed {n_done} windows...", flush=True)
    if not rows:
        return pd.DataFrame()
    print(f"  Built {len(rows)} sliding-window rows.")
    out = pd.DataFrame(rows)
    return out


def _save_sliding_window_to_db(df, experiment_id):
    """Insert/upsert sliding-window rows into features_sliding_window."""
    if df.empty or not DB_AVAILABLE:
        return
    cols = [
        'experiment_id', 'fly_id', 'window_end_date', 'exp_day',
        'total_activity', 'activity_mean', 'activity_var', 'longest_zero_hours',
        'total_sleep_min', 'total_bouts', 'mean_bout_min', 'max_bout_min',
        'frag_bouts_per_hour', 'amplitude_24h', 'status', 'status_raw', 'days_until_death'
    ]
    df_db = df[cols].copy()
    df_db['status_raw'] = df_db['status_raw'].fillna('').astype(str)
    df_db['days_until_death'] = df_db['days_until_death'].apply(lambda x: None if pd.isna(x) else int(x))
    tuples = [tuple(row[c] for c in cols) for _, row in df_db.iterrows()]
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM features_sliding_window WHERE experiment_id = %s", (experiment_id,)
            )
            insert_cols = ', '.join(cols)
            execute_values(
                cur,
                f"INSERT INTO features_sliding_window ({insert_cols}) VALUES %s",
                tuples,
                template=None,
                page_size=500
            )
        conn.commit()
    print(f"Saved {len(df_db)} sliding-window rows to features_sliding_window for experiment_id={experiment_id}")


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def create_feature_table(
    exclude_days=DEFAULT_EXCLUDE_DAYS,
    sleep_threshold_min=DEFAULT_SLEEP_THRESHOLD_MIN,
    bin_length_min=DEFAULT_BIN_LENGTH_MIN,
    period=DEFAULT_PERIOD,
    experiment_id=None,
    build_sliding_window=False,
    sliding_window_output=None,
    exclude_qc_fail=False,
):
    """
    Main function to create ML feature table.
    
    Args:
        exclude_days: List of days to exclude
        sleep_threshold_min: Minimum minutes of inactivity for sleep
        bin_length_min: Length of each time bin in minutes
        period: Circadian period in hours
        experiment_id: Experiment ID to use (None = use latest)
        build_sliding_window: If True, build and save sliding-window table for death prediction
        sliding_window_output: Optional path to also export sliding-window CSV
        exclude_qc_fail: If True, exclude QC_Fail rows from sliding-window dataset
        
    Returns:
        ML_features DataFrame
    """
    # Require database
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required. Please ensure database is configured and available.")
    
    # ============================================================
    # STEP 1: Load data from database
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
    # Import database functions from step 1
    step1 = import_module('1-prepare_data_and_health')
    
    # Use provided experiment_id, or get latest if not provided
    experiment_id_param = experiment_id
    if experiment_id_param is None:
        experiment_id = step1.get_latest_experiment_id()
    else:
        experiment_id = experiment_id_param
    
    if not experiment_id:
        raise ValueError("No experiment found in database")
    
    # Load data from database
    dam_clean = step1.load_readings_from_db(experiment_id)
    if dam_clean is None or len(dam_clean) == 0:
        raise ValueError(f"No data found in database for experiment_id {experiment_id}")
    
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
    ML_features = cosinor_features.merge(
        sleep_features,
        on=['fly_id', 'monitor', 'channel', 'genotype', 'sex', 'treatment'],
        how='inner'
    )
    
    # ============================================================
    # STEP 4: Sort and save output
    # ============================================================
    # Sort by monitor and channel (direct numeric sorting - much more efficient)
    ML_features = ML_features.sort_values(['monitor', 'channel'])
    
    # ============================================================
    # STEP 5: Save features to database
    # ============================================================
    if USE_DATABASE and DB_AVAILABLE and experiment_id:
        try:
            engine = create_engine(DATABASE_URL)
            
            # Add experiment_id to features
            ML_features_db = ML_features.copy()
            ML_features_db['experiment_id'] = experiment_id
            
            # Map column names to database schema (all lowercase)
            feature_mapping = {
                'fly_id': 'fly_id',
                'experiment_id': 'experiment_id',
                'mesor_mean': 'mesor_mean',
                'mesor_sd': 'mesor_sd',
                'amplitude_mean': 'amplitude_mean',
                'amplitude_sd': 'amplitude_sd',
                'phase_mean': 'phase_mean',
                'phase_sd': 'phase_sd',
                'rhythmic_days': 'rhythmic_days',
                'total_sleep_mean': 'total_sleep_mean',
                'day_sleep_mean': 'day_sleep_mean',
                'night_sleep_mean': 'night_sleep_mean',
                'total_bouts_mean': 'total_bouts_mean',
                'day_bouts_mean': 'day_bouts_mean',
                'night_bouts_mean': 'night_bouts_mean',
                'mean_bout_mean': 'mean_bout_mean',
                'max_bout_mean': 'max_bout_mean',
                'mean_day_bout_mean': 'mean_day_bout_mean',
                'max_day_bout_mean': 'max_day_bout_mean',
                'mean_night_bout_mean': 'mean_night_bout_mean',
                'max_night_bout_mean': 'max_night_bout_mean',
                'frag_bouts_per_hour_mean': 'frag_bouts_per_hour_mean',
                'frag_bouts_per_min_sleep_mean': 'frag_bouts_per_min_sleep_mean',
                'mean_wake_bout_mean': 'mean_wake_bout_mean',
                'p_wake_mean': 'p_wake_mean',
                'p_doze_mean': 'p_doze_mean',
                'sleep_latency_mean': 'sleep_latency_mean',
                'waso_mean': 'waso_mean'
            }
            
            # Select and rename columns to match database schema
            db_columns = [col for col in feature_mapping.keys() if col in ML_features_db.columns]
            ML_features_db = ML_features_db[db_columns].rename(columns=feature_mapping)
            
            print(f"Saving {len(ML_features_db)} flies (feature rows) to database for experiment_id {experiment_id}")
            print(f"Feature columns: {len([c for c in ML_features_db.columns if c not in ['fly_id', 'experiment_id']])} features per fly")
            print(f"Columns to save: {list(ML_features_db.columns)}")
            
            if len(ML_features_db) == 0:
                print("WARNING: No features to save!")
                return ML_features
            
            # Use bulk UPSERT (INSERT ... ON CONFLICT DO UPDATE) for features
            with psycopg2.connect(**DB_CONFIG) as conn:
                with conn.cursor() as cur:
                    try:
                        # Prepare all data as tuples for bulk insert
                        # Get column names in correct order
                        column_names = list(ML_features_db.columns)
                        feature_cols = [col for col in column_names if col not in ['fly_id', 'experiment_id']]
                        
                        # Prepare tuples (all columns in order)
                        # Convert each column to list 
                        column_lists = [ML_features_db[col].values.tolist() for col in column_names]
                        # Zip columns together to create tuples
                        features_tuples = list(zip(*column_lists))
                        
                        # Build UPSERT query (INSERT ... ON CONFLICT DO UPDATE)
                        insert_cols = ', '.join(column_names)
                        placeholders = ', '.join(['%s'] * len(column_names))
                        update_set = ', '.join([f"{col} = EXCLUDED.{col}" for col in feature_cols])
                        
                        upsert_query = f"""
                            INSERT INTO features ({insert_cols})
                            VALUES %s
                            ON CONFLICT (fly_id, experiment_id)
                            DO UPDATE SET {update_set}
                        """
                        
                        # Single bulk UPSERT operation
                        execute_values(
                            cur,
                            upsert_query,
                            features_tuples,
                            template=None,
                            page_size=1000
                        )
                        
                        conn.commit()
                        
                        # Verify actual count in database (more reliable than rowcount for bulk operations)
                        cur.execute("SELECT COUNT(*) FROM features WHERE experiment_id = %s", (experiment_id,))
                        actual_count = cur.fetchone()[0]
                        print(f"Database save complete: {actual_count} flies saved (inserted or updated)")
                        
                        # Verify all flies were saved
                        if actual_count != len(ML_features_db):
                            print(f"⚠️  WARNING: Expected {len(ML_features_db)} flies, but database has {actual_count} flies")
                        else:
                            print(f"✓ Verified: All {len(ML_features_db)} flies successfully saved to database")
                        
                    except psycopg2.Error as e:
                        conn.rollback()
                        print(f"ERROR saving features: {e}")
                        raise
            
            engine.dispose()
        except psycopg2.Error as e:
            raise RuntimeError(f"Database error saving features to database: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error saving features to database: {e}")
    
    # Sliding window for death prediction (optional)
    if build_sliding_window and dam_clean is not None and experiment_id:
        print("\n📊 Building sliding-window dataset for death prediction...")
        try:
            sw_df = build_sliding_windows(
                dam_clean,
                experiment_id,
                step1,
                exclude_days,
                bin_length_min,
                sleep_threshold_min,
                exclude_qc_fail=exclude_qc_fail,
            )
            if not sw_df.empty:
                _save_sliding_window_to_db(sw_df, experiment_id)
                if sliding_window_output:
                    sw_df.to_csv(sliding_window_output, index=False)
                    print(f"Exported sliding-window CSV to {sliding_window_output}")
            else:
                print("No sliding-window rows produced.")
        except Exception as e:
            print(f"WARNING: Sliding-window build failed: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    parser.add_argument('--exclude-days', nargs='+', type=int, default=DEFAULT_EXCLUDE_DAYS,
                       help=f'Days to exclude (default: {DEFAULT_EXCLUDE_DAYS})')
    parser.add_argument('--sleep-threshold', type=int, default=DEFAULT_SLEEP_THRESHOLD_MIN,
                       help=f'Minimum minutes of inactivity for sleep (default: {DEFAULT_SLEEP_THRESHOLD_MIN})')
    parser.add_argument('--experiment-id', type=int, default=None,
                       help='Experiment ID to use (default: latest experiment)')
    parser.add_argument('--build-sliding-window', action='store_true',
                       help='Build sliding-window table for death prediction and write to DB')
    parser.add_argument('--sliding-window-output', type=str, default=None,
                       help='Optional path to also export sliding-window CSV')
    parser.add_argument('--exclude-qc-fail', action='store_true',
                       help='Exclude QC_Fail from sliding-window dataset')
    
    args = parser.parse_args()
    
    create_feature_table(
        exclude_days=args.exclude_days,
        sleep_threshold_min=args.sleep_threshold,
        experiment_id=args.experiment_id,
        build_sliding_window=args.build_sliding_window,
        sliding_window_output=args.sliding_window_output,
        exclude_qc_fail=args.exclude_qc_fail,
    )


if __name__ == "__main__":
    main()

