#!/usr/bin/env python3
"""
Drosophila Circadian Rhythm Features (Per-Fly Analysis)

Computes circadian rhythm parameters (Mesor, Amplitude, Phase) for each fly using cosinor regression.

Columns expected: Monitor, Channel, datetime, Reading, Value, Genotype, Sex, Treatment
(or lowercase: monitor, channel, datetime, reading, value, genotype, sex, treatment)

reading is optional, but if missing will always assume reading type is MT
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Optional
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
        DataFrame with fly_id, ZT, and total_MT columns, filtered to MT readings
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
#   3. COSINOR FIT PER FLY
# ============================================================

def run_cosinor(fly_data: pd.DataFrame, period: int = 24) -> pd.Series:
    """
    Fit cosinor model to a single fly's data.
    
    Model: total_MT = Mesor + A*cos(2π*ZT/period) + B*sin(2π*ZT/period)
    
    Args:
        fly_data: DataFrame with ZT and total_MT columns for one fly
        period: Circadian period in hours (default: 24)
    
    Returns:
        Series with fly_id, Genotype, Sex, Treatment, Mesor, Amplitude, Phase, p_value
    """
    # Create cos/sin terms
    fly_data = fly_data.copy()
    fly_data['time_rad'] = 2 * np.pi * fly_data['ZT'] / period
    fly_data['cos_term'] = np.cos(fly_data['time_rad'])
    fly_data['sin_term'] = np.sin(fly_data['time_rad'])
    
    # Prepare data for regression
    X = fly_data[['cos_term', 'sin_term']].values
    y = fly_data['total_MT'].values
    
    # Check if we have enough data points
    if len(y) < 3:
        # Not enough data points for regression
        return pd.Series({
            'fly_id': fly_data['fly_id'].iloc[0] if len(fly_data) > 0 else None,
            'genotype': fly_data['genotype'].iloc[0] if len(fly_data) > 0 else None,
            'sex': fly_data['sex'].iloc[0] if len(fly_data) > 0 else None,
            'treatment': fly_data['treatment'].iloc[0] if len(fly_data) > 0 else None,
            'Mesor': np.nan,
            'Amplitude': np.nan,
            'Phase': np.nan,
            'p_value': np.nan
        })
    
    # Fit linear regression: total_MT ~ cos_term + sin_term
    model = LinearRegression()
    model.fit(X, y)
    
    # Get coefficients
    intercept = model.intercept_
    cos_coef = model.coef_[0]
    sin_coef = model.coef_[1]
    
    # Calculate amplitude: sqrt(cos_coef^2 + sin_coef^2)
    amplitude = np.sqrt(cos_coef**2 + sin_coef**2)
    
    # Calculate phase: atan2(-sin_coef, cos_coef) converted to hours
    phase = (period * np.arctan2(-sin_coef, cos_coef) / (2 * np.pi)) % period
    
    # Calculate p-value using F-test
    # F-test for overall model significance
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
    
    # Extract metadata
    fly_id = fly_data['fly_id'].iloc[0]
    genotype = fly_data['genotype'].iloc[0]
    sex = fly_data['sex'].iloc[0]
    treatment = fly_data['treatment'].iloc[0]
    
    return pd.Series({
        'fly_id': fly_id,
        'genotype': genotype,
        'sex': sex,
        'treatment': treatment,
        'Mesor': intercept,
        'Amplitude': amplitude,
        'Phase': phase,
        'p_value': p_value
    })


# ============================================================
#   4. APPLY MODEL TO ALL FLIES
# ============================================================

def compute_circadian_features(fly_hourly_totals: pd.DataFrame, 
                               period: int = 24) -> pd.DataFrame:
    """
    Apply cosinor model to all flies and extract circadian features.
    
    Args:
        fly_hourly_totals: DataFrame with per-fly hourly totals
        period: Circadian period in hours
    
    Returns:
        DataFrame with circadian features per fly
    """
    result_list = []
    
    for fly_id, group in fly_hourly_totals.groupby('fly_id'):
        result = run_cosinor(group, period=period)
        result_list.append(result)
    
    # Combine results
    cosinor_results = pd.DataFrame(result_list)
    
    # Add Rhythmic flag (p < 0.05)
    cosinor_results['Rhythmic'] = cosinor_results['p_value'] < 0.05
    
    return cosinor_results


# ============================================================
#   MAIN EXECUTION
# ============================================================

def main():
    """Main function to run circadian feature extraction pipeline."""
    
    # Check for input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../../data/processed/dam_data_MT.csv'
        if not os.path.exists(input_file):
            input_file = '../../data/processed/dam_data_merged.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Usage: python compute_circadian_features.py [input_file]")
        sys.exit(1)
    
    print("=" * 60)
    print("Drosophila Circadian Rhythm Feature Extraction")
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
    
    # Fit cosinor models
    print("\n🔬 Fitting cosinor models to each fly...")
    cosinor_results = compute_circadian_features(fly_hourly_totals, period=period)
    print(f"✓ Fitted models for {len(cosinor_results)} flies")
    
    # Summary statistics
    rhythmic_count = cosinor_results['Rhythmic'].sum()
    print(f"✓ Rhythmic flies (p < 0.05): {rhythmic_count} / {len(cosinor_results)} ({100*rhythmic_count/len(cosinor_results):.1f}%)")
    
    # Save results
    output_dir = '../../data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f'{output_dir}/fly_circadian_features.csv'
    cosinor_results.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved results to {output_file}")
    print(f"   Features: Mesor, Amplitude, Phase, p_value, Rhythmic")
    print(f"   Rows: {len(cosinor_results)} (one per fly)")
    
    # Display summary
    print("\n📊 Summary Statistics:")
    print(cosinor_results[['Mesor', 'Amplitude', 'Phase', 'p_value']].describe())
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()

