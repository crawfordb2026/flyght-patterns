#!/usr/bin/env python3
"""
Drosophila Circadian Rhythm Analysis (Group-Level)

Computes circadian rhythm parameters (Mesor, Amplitude, Acrophase) for each experimental group
by fitting cosinor models to group-averaged activity data.

Columns expected: Monitor, Channel, datetime, Reading, Value, Genotype, Sex, Treatment
(or lowercase: monitor, channel, datetime, reading, value, genotype, sex, treatment)

reading is optional, but if missing will always assume reading type is MT
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Optional, Dict
import warnings
import os
import sys
warnings.filterwarnings('ignore')


# ============================================================
#   SETTINGS
# ============================================================

exclude_days = [1, 7]  # Days to exclude (e.g., acclimation, washout)
period = 24  # Circadian period in hours
lights_on_ZT0 = 9  # Hour when lights turn on (ZT0)


# ============================================================
#   1. PREPARE DATA
# ============================================================

def calculate_exp_day(df: pd.DataFrame) -> pd.Series:
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


def prepare_data(dam_rhythm: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for circadian analysis.
    
    Args:
        dam_rhythm: DataFrame with columns: Monitor/monitor, Channel/channel, 
                   datetime, Reading/reading, Value/value, etc.
    
    Returns:
        DataFrame with fly_id, ZT, and value columns, filtered to MT readings
    """
    df = dam_rhythm.copy()
    
    # Normalize column names to lowercase
    col_mapping = {
        'Monitor': 'monitor', 'Channel': 'channel', 'Reading': 'reading',
        'Value': 'value', 'Genotype': 'genotype', 'Sex': 'sex', 
        'Treatment': 'treatment', 'Exp_Day': 'exp_day'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Filter to MT readings
    if 'reading' in df.columns:
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
        print(f"✓ Filtered to MT readings only")
    else:
        print(f"✓ Data is already MT-only")
    
    # Ensure datetime is datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Calculate Exp_Day if missing
    if 'exp_day' not in df.columns:
        print("✓ Calculating Exp_Day from datetime...")
        df['exp_day'] = calculate_exp_day(df)
    else:
        print("✓ Exp_Day column found")
    
    # Exclude specified days
    if 'exp_day' in df.columns:
        before = len(df)
        df = df[~df['exp_day'].isin(exclude_days)].copy()
        after = len(df)
        print(f"✓ Excluded days {exclude_days}: {before - after:,} rows removed")
    
    # Drop rows with missing metadata or ZT
    before = len(df)
    df = df.dropna(subset=['genotype', 'sex', 'treatment'])
    # Calculate ZT if not present
    if 'ZT' not in df.columns:
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['ZT'] = (df['hour'] + df['minute'] / 60 - lights_on_ZT0) % 24
    df = df.dropna(subset=['ZT'])
    after = len(df)
    if before != after:
        print(f"✓ Removed {before - after:,} rows with missing data")
    
    # Create unique fly ID
    df['fly_id'] = df['monitor'].astype(str) + '-' + df['channel'].astype(str)
    
    # Convert ZT to numeric
    df['ZT'] = pd.to_numeric(df['ZT'], errors='coerce')
    
    return df


# ============================================================
#   2. PER-FLY HOURLY TOTALS
# ============================================================

def calculate_hourly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total activity per fly per ZT hour.
    
    Args:
        df: Prepared DataFrame with fly_id, ZT, value columns
    
    Returns:
        DataFrame with total_MT per fly per ZT hour
    """
    # Group by Genotype, Sex, Treatment, ZT, fly_id and sum activity
    fly_hourly_totals = df.groupby(
        ['genotype', 'sex', 'treatment', 'ZT', 'fly_id'],
        as_index=False
    )['value'].sum().rename(columns={'value': 'total_MT'})
    
    print(f"✓ Calculated hourly totals for {len(fly_hourly_totals['fly_id'].unique())} flies")
    
    return fly_hourly_totals


# ============================================================
#   3. GROUP MEANS ACROSS FLIES
# ============================================================

def calculate_group_means(fly_hourly_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean activity per group (averaging across flies).
    
    Args:
        fly_hourly_totals: DataFrame with per-fly hourly totals
    
    Returns:
        DataFrame with group means, SD, SEM, and n_flies per ZT
    """
    hourly_group_totals = fly_hourly_totals.groupby(
        ['genotype', 'sex', 'treatment', 'ZT'],
        as_index=False
    ).agg({
        'total_MT': ['mean', 'std', 'count']
    })
    
    # Flatten column names
    hourly_group_totals.columns = ['genotype', 'sex', 'treatment', 'ZT', 
                                   'mean_total_MT', 'sd_total_MT', 'n_flies']
    
    # Calculate SEM
    hourly_group_totals['sem_total_MT'] = hourly_group_totals['sd_total_MT'] / np.sqrt(hourly_group_totals['n_flies'])
    
    # Sort by group and ZT
    hourly_group_totals = hourly_group_totals.sort_values(
        ['genotype', 'sex', 'treatment', 'ZT']
    ).reset_index(drop=True)
    
    print(f"✓ Calculated group means for {len(hourly_group_totals.groupby(['genotype', 'sex', 'treatment']))} groups")
    
    return hourly_group_totals


# ============================================================
#   4. COSINOR FIT PER GROUP
# ============================================================

def run_circasingle_group(group_data: pd.DataFrame, period: int = 24) -> Optional[Dict]:
    """
    Fit cosinor model to a group's mean activity data.
    
    Model: mean_total_MT = Mesor + A*cos(2π*ZT/period) + B*sin(2π*ZT/period)
    
    Args:
        group_data: DataFrame with ZT and mean_total_MT columns for one group
        period: Circadian period in hours (default: 24)
    
    Returns:
        Dictionary with group info and rhythmic parameters, or None if fit fails
    """
    try:
        # Create cos/sin terms
        group_data = group_data.copy()
        group_data['time_rad'] = 2 * np.pi * group_data['ZT'] / period
        group_data['cos_term'] = np.cos(group_data['time_rad'])
        group_data['sin_term'] = np.sin(group_data['time_rad'])
        
        # Prepare data for regression
        X = group_data[['cos_term', 'sin_term']].values
        y = group_data['mean_total_MT'].values
        
        # Check if we have enough data points
        if len(y) < 3:
            return None
        
        # Fit linear regression: mean_total_MT ~ cos_term + sin_term
        model = LinearRegression()
        model.fit(X, y)
        
        # Get coefficients
        intercept = model.intercept_
        cos_coef = model.coef_[0]
        sin_coef = model.coef_[1]
        
        # Calculate amplitude: sqrt(cos_coef^2 + sin_coef^2)
        amplitude = np.sqrt(cos_coef**2 + sin_coef**2)
        
        # Calculate acrophase (phase): atan2(-sin_coef, cos_coef) converted to hours
        acrophase = (period * np.arctan2(-sin_coef, cos_coef) / (2 * np.pi)) % period
        
        # Calculate p-value using F-test
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        n = len(y)
        p = 2  # number of predictors (cos, sin)
        
        if ss_tot > 0 and n > p + 1:
            f_stat = ((ss_tot - ss_res) / p) / (ss_res / (n - p - 1))
            p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
        else:
            p_value = np.nan
        
        # Extract group metadata
        genotype = group_data['genotype'].iloc[0]
        sex = group_data['sex'].iloc[0]
        treatment = group_data['treatment'].iloc[0]
        n_flies = group_data['n_flies'].iloc[0] if 'n_flies' in group_data.columns else np.nan
        
        return {
            'genotype': genotype,
            'sex': sex,
            'treatment': treatment,
            'n_flies': n_flies,
            'Mesor': intercept,
            'Amplitude': amplitude,
            'Acrophase': acrophase,
            'p_value': p_value,
            'Rhythmic': p_value < 0.05 if not np.isnan(p_value) else False
        }
    
    except Exception as e:
        print(f"  Error fitting model: {e}")
        return None


# ============================================================
#   5. APPLY MODEL TO ALL GROUPS
# ============================================================

def compute_group_circadian_features(hourly_group_totals: pd.DataFrame,
                                     period: int = 24) -> pd.DataFrame:
    """
    Apply cosinor model to all groups and extract circadian features.
    
    Args:
        hourly_group_totals: DataFrame with group means per ZT
        period: Circadian period in hours
    
    Returns:
        DataFrame with circadian features per group
    """
    # Filter out invalid groups
    valid_data = hourly_group_totals[
        (hourly_group_totals['genotype'].notna()) &
        (hourly_group_totals['sex'].notna()) &
        (hourly_group_totals['treatment'].notna()) &
        (hourly_group_totals['genotype'] != 'na') &
        (hourly_group_totals['sex'] != 'na') &
        (hourly_group_totals['treatment'] != 'na')
    ].copy()
    
    result_list = []
    failed_groups = []
    
    # Group by genotype, sex, treatment and fit model to each
    for (genotype, sex, treatment), group_data in valid_data.groupby(['genotype', 'sex', 'treatment']):
        group_name = f"{genotype}_{sex}_{treatment}"
        result = run_circasingle_group(group_data, period=period)
        
        if result is not None:
            result_list.append(result)
        else:
            failed_groups.append(group_name)
    
    if failed_groups:
        print(f"⚠ Warning: Failed to fit models for {len(failed_groups)} groups")
        print(f"  Failed groups: {', '.join(failed_groups)}")
    
    # Combine results
    if result_list:
        group_circadian_results = pd.DataFrame(result_list)
        return group_circadian_results
    else:
        print("❌ Error: No groups were successfully fitted")
        return pd.DataFrame()


# ============================================================
#   MAIN EXECUTION
# ============================================================

def main():
    """Main function to run group-level circadian analysis pipeline."""
    
    # Check for input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../../data/processed/dam_data_MT.csv'
        if not os.path.exists(input_file):
            input_file = '../../data/processed/dam_data_merged.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Usage: python compute_group_circadian.py [input_file]")
        sys.exit(1)
    
    print("=" * 60)
    print("Drosophila Group-Level Circadian Rhythm Analysis")
    print("=" * 60)
    print(f"\n📊 Loading data from: {input_file}")
    
    # Load data
    dam_rhythm = pd.read_csv(input_file)
    print(f"✓ Loaded {len(dam_rhythm):,} rows")
    
    # Prepare data
    print("\n🔧 Preparing data...")
    df_prepared = prepare_data(dam_rhythm)
    print(f"✓ Prepared {len(df_prepared):,} rows")
    
    # Calculate hourly totals
    print("\n📈 Calculating per-fly hourly totals...")
    fly_hourly_totals = calculate_hourly_totals(df_prepared)
    print(f"✓ Computed hourly totals: {len(fly_hourly_totals):,} fly-ZT combinations")
    
    # Calculate group means
    print("\n📊 Calculating group means (averaging across flies)...")
    hourly_group_totals = calculate_group_means(fly_hourly_totals)
    print(f"✓ Computed group means: {len(hourly_group_totals):,} group-ZT combinations")
    
    # Print group combinations
    print("\n📋 Experimental groups found:")
    combo_summary = hourly_group_totals[['genotype', 'sex', 'treatment']].drop_duplicates().sort_values(
        ['genotype', 'sex', 'treatment']
    )
    print(combo_summary.to_string(index=False))
    
    # Fit cosinor models
    print("\n🔬 Fitting cosinor models to each group...")
    group_circadian_results = compute_group_circadian_features(hourly_group_totals, period=period)
    
    if len(group_circadian_results) == 0:
        print("❌ No results to save. Exiting.")
        sys.exit(1)
    
    print(f"✓ Fitted models for {len(group_circadian_results)} groups")
    
    # Summary statistics
    rhythmic_count = group_circadian_results['Rhythmic'].sum()
    print(f"✓ Rhythmic groups (p < 0.05): {rhythmic_count} / {len(group_circadian_results)} ({100*rhythmic_count/len(group_circadian_results):.1f}%)")
    
    # Save results
    output_dir = '../../data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/group_circadian_features.csv'
    group_circadian_results.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved results to {output_file}")
    print(f"   Features: Mesor, Amplitude, Acrophase, p_value, Rhythmic")
    print(f"   Rows: {len(group_circadian_results)} (one per group)")
    
    # Display results
    print("\n📊 Group-Level Circadian Features:")
    print("=" * 60)
    print(group_circadian_results.to_string(index=False))
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()

