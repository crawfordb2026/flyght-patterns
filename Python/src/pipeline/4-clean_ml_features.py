#!/usr/bin/env python3
"""
Pipeline Step 4: Clean ML Features

This script:
1. Reads ML_features.csv from Step 3
2. Removes flies with problematic feature values:
   - Zero total sleep
   - Zero sleep bouts
   - Zero/NaN P_doze
3. Removes IQR outliers (per group) for total_sleep_mean
4. Fixes NaN values (replaces with 0 or group mean)
5. Creates z-scored feature table
6. Saves cleaned and z-scored versions

Output: ML_features_clean.csv, ML_features_Z.csv

This step prepares the feature table for machine learning by removing
problematic flies and normalizing features.

Matches the functionality of MLcleaner.r in R.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
#   USER CONFIGURATION
# ============================================================

# Default file paths
DEFAULT_INPUT = 'data/processed/ML_features.csv'
DEFAULT_OUTPUT_CLEAN = 'data/processed/ML_features_clean.csv'
DEFAULT_OUTPUT_Z = 'data/processed/ML_features_Z.csv'

# IQR outlier detection settings
DEFAULT_IQR_MULTIPLIER = 1.5  # Standard IQR multiplier for outlier detection


# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def compute_iqr_bounds(df, column='total_sleep_mean', multiplier=1.5):
    """
    Compute IQR bounds for outlier detection.
    
    Args:
        df: DataFrame with the column to analyze
        column: Column name to compute IQR for
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        dict with Q1, Q3, IQR, lower_bound, upper_bound
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': Q1 - multiplier * IQR,
        'upper_bound': Q3 + multiplier * IQR
    }


# ============================================================
#   MAIN CLEANING FUNCTIONS
# ============================================================

def remove_problematic_flies(ML_features):
    """
    Remove flies with zero sleep, zero bouts, or zero/NaN P_doze.
    
    Args:
        ML_features: Input feature DataFrame
    
    Returns:
        tuple: (cleaned_df, removed_flies_df)
    """
    df = ML_features.copy()
    
    # Identify problematic flies
    df['never_slept'] = df['total_sleep_mean'] == 0
    df['zero_sleep_bouts'] = df['total_bouts_mean'] == 0
    df['zero_P_doze'] = (
        (df['P_doze_mean'] == 0) |
        df['P_doze_mean'].isna() |
        df['P_doze_mean'].isnull()
    )
    
    # Find flies to remove
    problematic_mask = df['never_slept'] | df['zero_sleep_bouts'] | df['zero_P_doze']
    removed_flies = df[problematic_mask].copy()
    
    # Select relevant columns for report
    removed_flies_report = removed_flies[[
        'fly_id', 'Genotype', 'Sex', 'Treatment',
        'total_sleep_mean', 'total_bouts_mean', 'P_doze_mean',
        'never_slept', 'zero_sleep_bouts', 'zero_P_doze'
    ]].sort_values(['Genotype', 'Sex', 'Treatment', 'fly_id'])
    
    # Remove problematic flies
    df_clean = df[~problematic_mask].copy()
    
    # Drop helper columns
    df_clean = df_clean.drop(columns=['never_slept', 'zero_sleep_bouts', 'zero_P_doze'])
    
    return df_clean, removed_flies_report


def remove_iqr_outliers(ML_features, column='total_sleep_mean', multiplier=1.5):
    """
    Remove IQR outliers per group (Genotype × Sex × Treatment).
    
    Args:
        ML_features: Input feature DataFrame
        column: Column to check for outliers
        multiplier: IQR multiplier (default: 1.5)
    
    Returns:
        tuple: (cleaned_df, iqr_bounds_df, outlier_summary_df)
    """
    df = ML_features.copy()
    
    # Compute IQR bounds per group
    iqr_results = []
    for (genotype, sex, treatment), group in df.groupby(['Genotype', 'Sex', 'Treatment']):
        bounds = compute_iqr_bounds(group, column, multiplier)
        bounds['Genotype'] = genotype
        bounds['Sex'] = sex
        bounds['Treatment'] = treatment
        iqr_results.append(bounds)
    
    iqr_bounds = pd.DataFrame(iqr_results)
    
    # Merge bounds back to main dataframe
    df = df.merge(iqr_bounds, on=['Genotype', 'Sex', 'Treatment'], how='left')
    
    # Identify outliers
    df['is_outlier_IQR_total_sleep'] = (
        (df[column] < df['lower_bound']) |
        (df[column] > df['upper_bound'])
    )
    
    # Create summary before removal
    outlier_summary = df.groupby(['Genotype', 'Sex', 'Treatment']).agg({
        'fly_id': 'count',
        'is_outlier_IQR_total_sleep': 'sum'
    }).reset_index()
    outlier_summary.columns = ['Genotype', 'Sex', 'Treatment', 'n_total', 'n_outliers']
    outlier_summary['percent_outliers'] = (outlier_summary['n_outliers'] / outlier_summary['n_total'] * 100).round(1)
    
    # Remove outliers
    df_clean = df[~df['is_outlier_IQR_total_sleep']].copy()
    
    # Drop IQR calculation columns
    qc_cols = ['Q1', 'Q3', 'IQR', 'lower_bound', 'upper_bound', 'is_outlier_IQR_total_sleep']
    df_clean = df_clean.drop(columns=[col for col in qc_cols if col in df_clean.columns])
    
    return df_clean, iqr_bounds, outlier_summary


def fix_nan_values(ML_features):
    """
    Fix NaN values by replacing with 0 or group mean.
    
    Args:
        ML_features: Input feature DataFrame
    
    Returns:
        DataFrame with NaN values fixed
    """
    df = ML_features.copy()
    
    # Columns that should be 0 when NaN (bout metrics when no sleep)
    zero_cols = [
        'day_bouts_mean', 'night_bouts_mean',
        'mean_day_bout_mean', 'max_day_bout_mean',
        'mean_night_bout_mean', 'max_night_bout_mean',
        'frag_bouts_per_min_sleep_mean'
    ]
    
    # Columns that should use group mean when NaN
    mean_cols = [
        'sleep_latency_mean', 'WASO_mean',
        'Mesor_sd', 'Amp_sd', 'Phase_sd'
    ]
    
    # Replace NaN with 0 for zero_cols
    for col in zero_cols:
        if col in df.columns:
            # fillna handles both pandas NA and numpy NaN
            df[col] = df[col].fillna(0)
    
    # Replace NaN with group mean for mean_cols
    for col in mean_cols:
        if col in df.columns:
            # Compute group means (returns Series aligned with df)
            group_means = df.groupby(['Genotype', 'Sex', 'Treatment'])[col].transform('mean')
            # fillna handles both pandas NA and numpy NaN
            df[col] = df[col].fillna(group_means)
    
    return df


def create_z_scored_features(ML_features):
    """
    Create z-scored (standardized) feature table.
    
    Args:
        ML_features: Input feature DataFrame
    
    Returns:
        DataFrame with z-scored features
    """
    df = ML_features.copy()
    
    # Metadata columns to keep
    meta_cols = ['fly_id', 'Genotype', 'Sex', 'Treatment']
    
    # Get numeric columns (exclude metadata)
    numeric_cols = [col for col in df.columns 
                   if col not in meta_cols and df[col].dtype in [np.number]]
    
    # Create z-scored version
    df_z = df[meta_cols].copy()
    
    for col in numeric_cols:
        z_col = f'{col}_Z'
        # Z-score: (x - mean) / std
        df_z[z_col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Remove n_days_Z if it exists (as in R script)
    if 'n_days_Z' in df_z.columns:
        df_z = df_z.drop(columns=['n_days_Z'])
    
    return df_z


def run_diagnostics(ML_features_clean):
    """
    Run diagnostic checks on cleaned features.
    
    Args:
        ML_features_clean: Cleaned feature DataFrame
    
    Returns:
        dict with diagnostic results
    """
    df = ML_features_clean.copy()
    
    diagnostics = {}
    
    # 1. Check for NA
    na_summary = df.isna().sum()
    na_summary = na_summary[na_summary > 0]
    diagnostics['NA_summary'] = na_summary.to_dict() if len(na_summary) > 0 else {}
    
    # 2. Check for NaN
    nan_summary = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        nan_count = df[col].apply(lambda x: np.isnan(x) if isinstance(x, (int, float)) else False).sum()
        if nan_count > 0:
            nan_summary[col] = nan_count
    diagnostics['NaN_summary'] = nan_summary
    
    # 3. List flies with any NaN
    nan_flies = df[df.select_dtypes(include=[np.number]).apply(
        lambda row: any(np.isnan(x) if isinstance(x, (int, float)) else False for x in row), axis=1
    )]
    diagnostics['flies_with_NaN'] = nan_flies[['fly_id', 'Genotype', 'Sex', 'Treatment']].copy() if len(nan_flies) > 0 else pd.DataFrame()
    
    # 4. Which columns have NaN
    nan_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].apply(lambda x: np.isnan(x) if isinstance(x, (int, float)) else False).any():
            nan_columns.append(col)
    diagnostics['NaN_columns'] = nan_columns
    
    # 5. Diagnose fragmentation metrics
    frag_cols = ['frag_bouts_per_hour_mean', 'frag_bouts_per_min_sleep_mean']
    frag_nan = df[df[frag_cols].apply(
        lambda row: any(np.isnan(x) if isinstance(x, (int, float)) else False for x in row), axis=1
    )]
    diagnostics['frag_problems'] = frag_nan[['fly_id', 'Genotype', 'Sex', 'Treatment'] + frag_cols].copy() if len(frag_nan) > 0 else pd.DataFrame()
    
    # 6. Diagnose sleep bout structure
    sleep_struct = df[
        (df['total_sleep_mean'] == 0) | (df['total_bouts_mean'] == 0)
    ][['fly_id', 'Genotype', 'Sex', 'Treatment', 'total_sleep_mean', 'total_bouts_mean']].copy()
    diagnostics['sleep_structure_problems'] = sleep_struct
    
    return diagnostics


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def clean_ml_features(
    input_file=None,
    output_clean=None,
    output_z=None,
    iqr_multiplier=DEFAULT_IQR_MULTIPLIER,
    save_diagnostics=True
):
    """
    Main function to clean ML features.
    
    Args:
        input_file: Path to ML_features.csv from Step 3
        output_clean: Path to save cleaned features
        output_z: Path to save z-scored features
        iqr_multiplier: IQR multiplier for outlier detection
        save_diagnostics: Whether to save diagnostic reports
    
    Returns:
        tuple: (cleaned_df, z_scored_df)
    """
    print("=" * 60)
    print("PIPELINE STEP 4: CLEAN ML FEATURES")
    print("=" * 60)
    
    # Set defaults
    if input_file is None:
        input_file = DEFAULT_INPUT
    if output_clean is None:
        output_clean = DEFAULT_OUTPUT_CLEAN
    if output_z is None:
        output_z = DEFAULT_OUTPUT_Z
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, input_file) if not os.path.isabs(input_file) else input_file
    output_clean = os.path.join(script_dir, output_clean) if not os.path.isabs(output_clean) else output_clean
    output_z = os.path.join(script_dir, output_z) if not os.path.isabs(output_z) else output_z
    
    # ============================================================
    # STEP 1: Load ML features
    # ============================================================
    print(f"\n[Step 4.1] Loading ML features from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run Step 3 first: python 3-create_feature_table.py")
        sys.exit(1)
    
    ML_features = pd.read_csv(input_file)
    print(f"✓ Loaded {len(ML_features)} flies")
    print(f"   Features: {len(ML_features.columns)} columns")
    
    original_count = len(ML_features)
    
    # ============================================================
    # STEP 2: Remove problematic flies
    # ============================================================
    print("\n[Step 4.2] Removing flies with zero sleep, zero bouts, or zero/NaN P_doze...")
    
    ML_features, removed_flies = remove_problematic_flies(ML_features)
    
    print(f"✓ Removed {len(removed_flies)} problematic flies")
    if len(removed_flies) > 0:
        print("\n" + "=" * 60)
        print("FLIES REMOVED FOR ZERO SLEEP / ZERO BOUTS / ZERO P_DOZE")
        print("=" * 60)
        print(removed_flies.to_string(index=False))
        print("=" * 60)
    
    # ============================================================
    # STEP 3: Remove IQR outliers
    # ============================================================
    print("\n[Step 4.3] Removing IQR outliers...")
    
    ML_features, iqr_bounds, outlier_summary = remove_iqr_outliers(
        ML_features, 
        column='total_sleep_mean',
        multiplier=iqr_multiplier
    )
    
    print(f"✓ Removed {original_count - len(ML_features)} total flies (including outliers)")
    print("\n" + "=" * 60)
    print("IQR OUTLIER SUMMARY")
    print("=" * 60)
    print(outlier_summary.to_string(index=False))
    print("=" * 60)
    
    # ============================================================
    # STEP 4: Fix NaN values
    # ============================================================
    print("\n[Step 4.4] Fixing NaN values...")
    
    ML_features = fix_nan_values(ML_features)
    print("✓ Fixed NaN values (replaced with 0 or group mean)")
    
    # ============================================================
    # STEP 5: Create z-scored features
    # ============================================================
    print("\n[Step 4.5] Creating z-scored features...")
    
    ML_features_Z = create_z_scored_features(ML_features)
    print(f"✓ Created z-scored features for {len(ML_features_Z.columns)} columns")
    
    # ============================================================
    # STEP 6: Run diagnostics
    # ============================================================
    print("\n[Step 4.6] Running diagnostic checks...")
    
    diagnostics = run_diagnostics(ML_features)
    
    # Print diagnostics
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if len(diagnostics['NA_summary']) > 0:
        print("\n1. NA values found:")
        for col, count in diagnostics['NA_summary'].items():
            print(f"   {col}: {count}")
    else:
        print("\n1. ✓ No NA values found")
    
    if len(diagnostics['NaN_summary']) > 0:
        print("\n2. NaN values found:")
        for col, count in diagnostics['NaN_summary'].items():
            print(f"   {col}: {count}")
    else:
        print("\n2. ✓ No NaN values found")
    
    if len(diagnostics['flies_with_NaN']) > 0:
        print("\n3. Flies with NaN:")
        print(diagnostics['flies_with_NaN'].to_string(index=False))
    else:
        print("\n3. ✓ No flies with NaN")
    
    if len(diagnostics['sleep_structure_problems']) > 0:
        print("\n4. Sleep structure problems (should be empty after cleaning):")
        print(diagnostics['sleep_structure_problems'].to_string(index=False))
    else:
        print("\n4. ✓ No sleep structure problems")
    
    print("=" * 60)
    
    # ============================================================
    # STEP 7: Save outputs
    # ============================================================
    print(f"\n[Step 4.7] Saving cleaned features...")
    
    os.makedirs(os.path.dirname(output_clean), exist_ok=True)
    ML_features.to_csv(output_clean, index=False)
    print(f"✓ Saved cleaned features to {output_clean}")
    print(f"   Flies: {len(ML_features)} (removed {original_count - len(ML_features)} flies)")
    
    os.makedirs(os.path.dirname(output_z), exist_ok=True)
    ML_features_Z.to_csv(output_z, index=False)
    print(f"✓ Saved z-scored features to {output_z}")
    print(f"   Flies: {len(ML_features_Z)}")
    print(f"   Features: {len(ML_features_Z.columns)} z-scored columns")
    
    # Save diagnostics if requested
    if save_diagnostics:
        diag_file = output_clean.replace('.csv', '_diagnostics.txt')
        with open(diag_file, 'w') as f:
            f.write("ML FEATURES CLEANING DIAGNOSTICS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Original flies: {original_count}\n")
            f.write(f"Cleaned flies: {len(ML_features)}\n")
            f.write(f"Removed: {original_count - len(ML_features)}\n\n")
            f.write("Removed Flies:\n")
            f.write(removed_flies.to_string(index=False))
            f.write("\n\nIQR Outlier Summary:\n")
            f.write(outlier_summary.to_string(index=False))
        print(f"✓ Saved diagnostics to {diag_file}")
    
    # ============================================================
    # STEP 8: Print summary
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Original flies: {original_count}")
    print(f"  Cleaned flies: {len(ML_features)}")
    print(f"  Removed: {original_count - len(ML_features)} ({100*(original_count - len(ML_features))/original_count:.1f}%)")
    print(f"\nOutput files:")
    print(f"  1. Cleaned features: {output_clean}")
    print(f"  2. Z-scored features: {output_z}")
    print(f"\nNext steps:")
    print(f"  - Use ML_features_clean.csv for analysis with original scales")
    print(f"  - Use ML_features_Z.csv for machine learning (normalized features)")
    
    return ML_features, ML_features_Z


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Pipeline Step 4: Clean ML features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python 4-clean_ml_features.py
  
  # Custom input/output paths
  python 4-clean_ml_features.py --input custom_features.csv --output-clean cleaned.csv
        """
    )
    
    parser.add_argument('--input', type=str, default=None,
                       help='Input ML features file from Step 3 (default: ML_features.csv)')
    parser.add_argument('--output-clean', type=str, default=None,
                       help='Output cleaned features file (default: ML_features_clean.csv)')
    parser.add_argument('--output-z', type=str, default=None,
                       help='Output z-scored features file (default: ML_features_Z.csv)')
    parser.add_argument('--iqr-multiplier', type=float, default=DEFAULT_IQR_MULTIPLIER,
                       help=f'IQR multiplier for outlier detection (default: {DEFAULT_IQR_MULTIPLIER})')
    
    args = parser.parse_args()
    
    clean_ml_features(
        input_file=args.input,
        output_clean=args.output_clean,
        output_z=args.output_z,
        iqr_multiplier=args.iqr_multiplier
    )


if __name__ == '__main__':
    main()

