#!/usr/bin/env python3
"""
Debug the binning process to see why some flies are missing.
"""

import pandas as pd

def debug_binning():
    print("🔍 Debugging binning process...")
    
    # Load original data
    original = pd.read_csv('../../data/processed/dam_data_MT.csv')
    original['datetime'] = pd.to_datetime(original['datetime'])
    
    # Focus on one missing fly
    test_fly = 'M5_Ch31'
    print(f"Testing with fly: {test_fly}")
    
    fly_data = original[original['fly_id'] == test_fly].copy()
    print(f"Original rows: {len(fly_data)}")
    
    # Apply binning logic step by step
    print("\nStep 1: Floor datetime to 5-minute intervals")
    fly_data['datetime_binned'] = fly_data['datetime'].dt.floor('5min')
    
    # Show first few rows
    print("First 10 rows after flooring:")
    for i, (_, row) in enumerate(fly_data.head(10).iterrows()):
        print(f"  {row['datetime']} -> {row['datetime_binned']} (value={row['value']})")
    
    print(f"\nStep 2: Group by metadata columns")
    print("Grouping columns: ['datetime_binned', 'monitor', 'channel', 'fly_id', 'genotype', 'sex', 'treatment']")
    
    # Check if all required columns exist
    required_cols = ['datetime_binned', 'monitor', 'channel', 'fly_id', 'genotype', 'sex', 'treatment']
    missing_cols = [col for col in required_cols if col not in fly_data.columns]
    print(f"Missing columns: {missing_cols}")
    
    if 'genotype' in fly_data.columns:
        print(f"Genotype values: {fly_data['genotype'].unique()}")
    if 'sex' in fly_data.columns:
        print(f"Sex values: {fly_data['sex'].unique()}")
    if 'treatment' in fly_data.columns:
        print(f"Treatment values: {fly_data['treatment'].unique()}")
    
    # Try the grouping
    try:
        binned = fly_data.groupby(
            ['datetime_binned', 'monitor', 'channel', 'fly_id', 'genotype', 'sex', 'treatment'],
            as_index=False
        )['value'].sum()
        
        print(f"\nBinned rows: {len(binned)}")
        print("First 5 binned rows:")
        for i, (_, row) in enumerate(binned.head(5).iterrows()):
            print(f"  {row['datetime_binned']}: value={row['value']}")
            
    except Exception as e:
        print(f"Error during grouping: {e}")
        
        # Check for NaN values in grouping columns
        print("\nChecking for NaN values in grouping columns:")
        for col in required_cols:
            if col in fly_data.columns:
                nan_count = fly_data[col].isna().sum()
                print(f"  {col}: {nan_count} NaN values")
                if nan_count > 0:
                    print(f"    Sample NaN values: {fly_data[fly_data[col].isna()][col].head()}")

if __name__ == "__main__":
    debug_binning()

