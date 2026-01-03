#!/usr/bin/env python3
"""
Comparison script to verify CSV and DB pipeline outputs match.

This script loads outputs from both the old CSV pipeline and the new DB pipeline
and performs a thorough comparison to ensure they produce identical results.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path to import config and functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    import psycopg2
    from importlib import import_module
    step1 = import_module('1-prepare_data_and_health')
    DB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import database modules: {e}")
    DB_AVAILABLE = False
    USE_DATABASE = False

# Default CSV paths (from old pipeline)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_DATA = os.path.join(SCRIPT_DIR, '../../data/processed/dam_data_prepared.csv')
DEFAULT_CSV_HEALTH = os.path.join(SCRIPT_DIR, '../../data/processed/health_report.csv')


def compare_dataframes(df1, df2, name="DataFrame", tolerance=1e-6, normalize_columns=True):
    """
    Compare two DataFrames and report differences.
    
    Args:
        normalize_columns: If True, convert column names to lowercase for comparison
    
    Returns:
        dict: Comparison results with 'match' (bool) and 'details' (str)
    """
    results = {
        'match': True,
        'details': []
    }
    
    # Check if both are None
    if df1 is None and df2 is None:
        results['details'].append(f"✓ Both {name} are None")
        return results
    
    if df1 is None or df2 is None:
        results['match'] = False
        results['details'].append(f"✗ One {name} is None (df1: {df1 is None}, df2: {df2 is None})")
        return results
    
    # Normalize column names to lowercase if requested (CSV has capitalized, DB has lowercase)
    if normalize_columns:
        df1 = df1.copy()
        df2 = df2.copy()
        df1.columns = [col.lower() if isinstance(col, str) else col for col in df1.columns]
        df2.columns = [col.lower() if isinstance(col, str) else col for col in df2.columns]
        results['details'].append(f"✓ Normalized column names to lowercase for comparison")
    
    # Shape comparison
    if df1.shape != df2.shape:
        results['match'] = False
        results['details'].append(f"✗ Shape mismatch: {name} - CSV: {df1.shape}, DB: {df2.shape}")
        return results
    else:
        results['details'].append(f"✓ Shape matches: {df1.shape}")
    
    # Column comparison
    csv_cols = set(df1.columns)
    db_cols = set(df2.columns)
    
    if csv_cols != db_cols:
        results['match'] = False
        missing_in_db = csv_cols - db_cols
        missing_in_csv = db_cols - csv_cols
        if missing_in_db:
            results['details'].append(f"✗ Columns in CSV but not in DB: {missing_in_db}")
        if missing_in_csv:
            results['details'].append(f"✗ Columns in DB but not in CSV: {missing_in_csv}")
    else:
        results['details'].append(f"✓ All columns match: {len(csv_cols)} columns")
    
    # Compare common columns
    common_cols = csv_cols & db_cols
    if common_cols:
        # Sort both dataframes by all columns for consistent comparison
        sort_cols = sorted(common_cols)
        df1_sorted = df1[sort_cols].sort_values(by=sort_cols).reset_index(drop=True)
        df2_sorted = df2[sort_cols].sort_values(by=sort_cols).reset_index(drop=True)
        
        # Compare each column
        for col in sort_cols:
            if col in ['date']:
                # Date comparison (handle date objects and date vs string)
                # Convert both to string for comparison
                csv_dates = df1_sorted[col].astype(str)
                db_dates = df2_sorted[col].astype(str)
                if not csv_dates.equals(db_dates):
                    # Check if differences are just type representations (date object vs string)
                    # Normalize both to same format
                    csv_dates_norm = csv_dates.str.replace(' 00:00:00', '').str.strip()
                    db_dates_norm = db_dates.str.replace(' 00:00:00', '').str.strip()
                    
                    if not csv_dates_norm.equals(db_dates_norm):
                        results['match'] = False
                        diff_mask = csv_dates_norm != db_dates_norm
                        num_diff = diff_mask.sum()
                        results['details'].append(f"✗ Column '{col}': {num_diff} differences found")
                        if num_diff <= 5:
                            results['details'].append(f"  First differences:")
                            diff_indices = df1_sorted[diff_mask].index[:5]
                            for idx in diff_indices:
                                results['details'].append(f"    Row {idx}: CSV={csv_dates.iloc[idx]}, DB={db_dates.iloc[idx]}")
                    else:
                        results['details'].append(f"✓ Column '{col}' matches (after normalization)")
                else:
                    results['details'].append(f"✓ Column '{col}' matches")
            elif df1_sorted[col].dtype in [np.float64, np.float32] or 'float' in str(df1_sorted[col].dtype):
                # Numeric comparison with tolerance
                if not np.allclose(df1_sorted[col].fillna(0), df2_sorted[col].fillna(0), 
                                 equal_nan=True, rtol=tolerance, atol=tolerance):
                    results['match'] = False
                    diff_mask = ~np.isclose(df1_sorted[col].fillna(0), df2_sorted[col].fillna(0),
                                           equal_nan=True, rtol=tolerance, atol=tolerance)
                    num_diff = diff_mask.sum()
                    results['details'].append(f"✗ Column '{col}': {num_diff} differences found (tolerance={tolerance})")
                    if num_diff <= 5:
                        results['details'].append(f"  First differences:")
                        diff_indices = df1_sorted[diff_mask].index[:5]
                        for idx in diff_indices:
                            csv_val = df1_sorted[col].iloc[idx]
                            db_val = df2_sorted[col].iloc[idx]
                            results['details'].append(f"    Row {idx}: CSV={csv_val}, DB={db_val}")
                else:
                    results['details'].append(f"✓ Column '{col}' matches (numeric, tolerance={tolerance})")
            else:
                # Exact comparison for other types (handle NaN/None properly)
                # Convert to string for comparison to handle NaN differences
                csv_col_str = df1_sorted[col].astype(str).replace('nan', 'NaN').replace('None', 'NaN')
                db_col_str = df2_sorted[col].astype(str).replace('nan', 'NaN').replace('None', 'NaN')
                
                if not csv_col_str.equals(db_col_str):
                    # Check if differences are just NaN representations
                    diff_mask = csv_col_str != db_col_str
                    # Filter out NaN differences (both are NaN)
                    both_nan = (df1_sorted[col].isna() & df2_sorted[col].isna())
                    real_diff_mask = diff_mask & ~both_nan
                    num_diff = real_diff_mask.sum()
                    
                    if num_diff > 0:
                        results['match'] = False
                        results['details'].append(f"✗ Column '{col}': {num_diff} differences found")
                        if num_diff <= 5:
                            results['details'].append(f"  First differences:")
                            diff_indices = df1_sorted[real_diff_mask].index[:5]
                            for idx in diff_indices:
                                csv_val = df1_sorted[col].iloc[idx]
                                db_val = df2_sorted[col].iloc[idx]
                                results['details'].append(f"    Row {idx}: CSV={csv_val}, DB={db_val}")
                    else:
                        # Only NaN representation differences, which are fine
                        results['details'].append(f"✓ Column '{col}' matches (NaN differences are just representation)")
                else:
                    results['details'].append(f"✓ Column '{col}' matches")
    
    return results


def compare_main_data(csv_path, experiment_id=None):
    """Compare the main prepared data (readings)."""
    print("\n" + "="*80)
    print("COMPARING MAIN DATA (Prepared Data / Readings)")
    print("="*80)
    
    # Load CSV data
    print(f"\n[1] Loading CSV data from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"✗ ERROR: CSV file not found: {csv_path}")
        return False
    
    try:
        csv_data = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV: {csv_data.shape[0]} rows, {csv_data.shape[1]} columns")
    except Exception as e:
        print(f"✗ ERROR loading CSV: {e}")
        return False
    
    # Load DB data
    print(f"\n[2] Loading DB data (experiment_id: {experiment_id})")
    if not DB_AVAILABLE or not USE_DATABASE:
        print("✗ ERROR: Database not available")
        return False
    
    try:
        if experiment_id is None:
            experiment_id = step1.get_latest_experiment_id()
            if experiment_id is None:
                print("✗ ERROR: No experiment found in database")
                return False
            print(f"  Using latest experiment_id: {experiment_id}")
        
        db_data = step1.load_readings_from_db(experiment_id)
        if db_data is None or len(db_data) == 0:
            print("✗ ERROR: No data found in database")
            return False
        print(f"✓ Loaded DB: {db_data.shape[0]} rows, {db_data.shape[1]} columns")
    except Exception as e:
        print(f"✗ ERROR loading DB: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare (normalize columns since CSV has capitalized, DB has lowercase)
    print(f"\n[3] Comparing data...")
    results = compare_dataframes(csv_data, db_data, name="Main Data", normalize_columns=True)
    
    # Print results
    for detail in results['details']:
        print(f"  {detail}")
    
    return results['match']


def compare_health_reports(csv_path, experiment_id=None):
    """Compare health reports."""
    print("\n" + "="*80)
    print("COMPARING HEALTH REPORTS")
    print("="*80)
    
    # Load CSV health report
    print(f"\n[1] Loading CSV health report from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"✗ ERROR: CSV file not found: {csv_path}")
        return False
    
    try:
        csv_health = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV health report: {csv_health.shape[0]} rows, {csv_health.shape[1]} columns")
        print(f"  Columns: {list(csv_health.columns)}")
    except Exception as e:
        print(f"✗ ERROR loading CSV: {e}")
        return False
    
    # Load DB health report
    print(f"\n[2] Loading DB health report (experiment_id: {experiment_id})")
    if not DB_AVAILABLE or not USE_DATABASE:
        print("✗ ERROR: Database not available")
        return False
    
    try:
        if experiment_id is None:
            experiment_id = step1.get_latest_experiment_id()
            if experiment_id is None:
                print("✗ ERROR: No experiment found in database")
                return False
            print(f"  Using latest experiment_id: {experiment_id}")
        
        db_health = step1.load_health_report_from_db(experiment_id)
        if db_health is None or len(db_health) == 0:
            print("✗ ERROR: No health report found in database")
            return False
        print(f"✓ Loaded DB health report: {db_health.shape[0]} rows, {db_health.shape[1]} columns")
        print(f"  Columns: {list(db_health.columns)}")
    except Exception as e:
        print(f"✗ ERROR loading DB: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Note: DB health report may have fewer columns (only stores status, not all metrics)
    # Normalize column names for comparison (CSV has capitalized, DB has lowercase)
    print(f"\n[3] Comparing health reports...")
    print(f"  Note: DB version may have fewer columns than CSV (only stores essential fields)")
    
    csv_health_normalized = csv_health.copy()
    csv_health_normalized.columns = [col.lower() if isinstance(col, str) else col for col in csv_health_normalized.columns]
    
    db_health_normalized = db_health.copy()
    db_health_normalized.columns = [col.lower() if isinstance(col, str) else col for col in db_health_normalized.columns]
    
    # Compare common columns
    common_cols = set(csv_health_normalized.columns) & set(db_health_normalized.columns)
    if common_cols:
        print(f"  Comparing {len(common_cols)} common columns: {sorted(common_cols)}")
        csv_subset = csv_health_normalized[sorted(common_cols)]
        db_subset = db_health_normalized[sorted(common_cols)]
        
        results = compare_dataframes(csv_subset, db_subset, name="Health Report (common columns)", normalize_columns=False)
        
        for detail in results['details']:
            print(f"  {detail}")
        
        # Check for columns only in CSV
        csv_only = set(csv_health.columns) - set(db_health.columns)
        if csv_only:
            print(f"\n  Note: CSV has additional columns not in DB: {sorted(csv_only)}")
            print(f"    These are detailed metrics that aren't stored in the database schema.")
        
        return results['match']
    else:
        print("✗ ERROR: No common columns found between CSV and DB health reports")
        return False


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare CSV and DB pipeline outputs')
    parser.add_argument('--csv-data', type=str, default=DEFAULT_CSV_DATA,
                       help='Path to CSV prepared data file')
    parser.add_argument('--csv-health', type=str, default=DEFAULT_CSV_HEALTH,
                       help='Path to CSV health report file')
    parser.add_argument('--experiment-id', type=int, default=None,
                       help='Experiment ID to compare (default: latest)')
    parser.add_argument('--data-only', action='store_true',
                       help='Only compare main data, skip health report')
    parser.add_argument('--health-only', action='store_true',
                       help='Only compare health report, skip main data')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    csv_data_path = os.path.abspath(args.csv_data) if not os.path.isabs(args.csv_data) else args.csv_data
    csv_health_path = os.path.abspath(args.csv_health) if not os.path.isabs(args.csv_health) else args.csv_health
    
    print("="*80)
    print("CSV vs DB PIPELINE COMPARISON")
    print("="*80)
    print(f"\nCSV Data: {csv_data_path}")
    print(f"CSV Health: {csv_health_path}")
    print(f"Experiment ID: {args.experiment_id or 'latest'}")
    
    # Compare main data
    data_match = True
    if not args.health_only:
        data_match = compare_main_data(csv_data_path, args.experiment_id)
    
    # Compare health reports
    health_match = True
    if not args.data_only:
        health_match = compare_health_reports(csv_health_path, args.experiment_id)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if args.data_only:
        print(f"\nMain Data: {'✓ MATCH' if data_match else '✗ MISMATCH'}")
    elif args.health_only:
        print(f"\nHealth Report: {'✓ MATCH' if health_match else '✗ MISMATCH'}")
    else:
        print(f"\nMain Data: {'✓ MATCH' if data_match else '✗ MISMATCH'}")
        print(f"Health Report: {'✓ MATCH' if health_match else '✗ MISMATCH'}")
        print(f"\nOverall: {'✓ ALL MATCH' if (data_match and health_match) else '✗ MISMATCHES FOUND'}")
    
    return 0 if (data_match and health_match) else 1


if __name__ == '__main__':
    sys.exit(main())

