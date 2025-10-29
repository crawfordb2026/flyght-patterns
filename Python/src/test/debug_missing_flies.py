#!/usr/bin/env python3
"""
Debug script to investigate missing flies after binning.
"""

import pandas as pd

def debug_missing_flies():
    # Load original and binned data
    print("🔍 Investigating missing flies...")
    
    original = pd.read_csv('../../data/processed/dam_data_MT.csv')
    binned = pd.read_csv('../../data/processed/dam_data_MT_5min.csv')
    
    original['datetime'] = pd.to_datetime(original['datetime'])
    binned['datetime'] = pd.to_datetime(binned['datetime'])
    
    original_flies = set(original['fly_id'].unique())
    binned_flies = set(binned['fly_id'].unique())
    
    missing_flies = original_flies - binned_flies
    print(f"Missing flies: {missing_flies}")
    
    # Check each missing fly
    for fly_id in missing_flies:
        print(f"\n🔍 Analyzing {fly_id}:")
        
        fly_data = original[original['fly_id'] == fly_id].copy()
        fly_data = fly_data.sort_values('datetime')
        
        print(f"  Total rows: {len(fly_data)}")
        print(f"  Date range: {fly_data['datetime'].min()} to {fly_data['datetime'].max()}")
        print(f"  Value range: {fly_data['value'].min()} to {fly_data['value'].max()}")
        print(f"  All values zero: {fly_data['value'].sum() == 0}")
        
        # Check first few rows
        print(f"  First 5 rows:")
        for i, (_, row) in enumerate(fly_data.head(5).iterrows()):
            print(f"    {row['datetime']}: value={row['value']}")
        
        # Check if all values are zero
        if fly_data['value'].sum() == 0:
            print(f"  ⚠️  All values are zero - this fly might have been filtered out!")
        else:
            print(f"  ✅ Has non-zero values - should not be missing")

if __name__ == "__main__":
    debug_missing_flies()

