#!/usr/bin/env python3
"""
Debug script to understand M6_Ch05's death detection
"""

import pandas as pd
from datetime import datetime, timedelta

def debug_fly_death(fly_id):
    # Load data
    dam_data = pd.read_csv('../../data/processed/dam_data_merged.csv')
    dam_data['datetime'] = pd.to_datetime(dam_data['datetime'])
    
    # Get MT data for this fly
    fly_data = dam_data[dam_data['fly_id'] == fly_id]
    mt_data = fly_data[fly_data['reading'] == 'MT'].copy()
    mt_data = mt_data.sort_values('datetime').reset_index(drop=True)
    
    print(f"🔍 Debugging {fly_id}")
    print(f"Total MT records: {len(mt_data)}")
    print(f"Date range: {mt_data['datetime'].min()} to {mt_data['datetime'].max()}")
    
    # Find all zero periods
    mt_data['is_zero'] = (mt_data['value'] == 0)
    mt_data['zero_group'] = (mt_data['is_zero'] != mt_data['is_zero'].shift()).cumsum()
    
    zero_periods = []
    
    for group_id in mt_data['zero_group'].unique():
        group_data = mt_data[mt_data['zero_group'] == group_id]
        
        if group_data['is_zero'].iloc[0]:  # Only process zero groups
            start_time = group_data['datetime'].min()
            end_time = group_data['datetime'].max()
            duration_minutes = (end_time - start_time).total_seconds() / 60 + 1
            
            zero_periods.append({
                'start': start_time,
                'end': end_time,
                'duration_minutes': duration_minutes,
                'duration_hours': duration_minutes / 60
            })
    
    # Sort by start time
    zero_periods.sort(key=lambda x: x['start'])
    
    print(f"\nAll zero periods (sorted by start time):")
    for i, period in enumerate(zero_periods):
        print(f"  Period {i+1}: {period['start']} to {period['end']} ({period['duration_hours']:.1f}h)")
    
    # Find periods >= 12 hours
    long_periods = [p for p in zero_periods if p['duration_minutes'] >= 720]
    
    print(f"\nPeriods >= 12 hours:")
    for i, period in enumerate(long_periods):
        print(f"  Long period {i+1}: {period['start']} to {period['end']} ({period['duration_hours']:.1f}h)")
    
    if long_periods:
        first_long = min(long_periods, key=lambda x: x['start'])
        print(f"\nFirst 12+ hour period: {first_long['start']} to {first_long['end']}")
        print(f"This matches the declared death time: {first_long['start']}")
    
    # Show data around the declared death time
    death_time = pd.to_datetime('2025-09-23 12:26:00')
    print(f"\nData around declared death time ({death_time}):")
    around_death = mt_data[
        (mt_data['datetime'] >= death_time - timedelta(hours=2)) &
        (mt_data['datetime'] <= death_time + timedelta(hours=2))
    ]
    
    for _, row in around_death.iterrows():
        status = "ZERO" if row['value'] == 0 else f"MT={row['value']}"
        print(f"  {row['datetime']}: {status}")

if __name__ == "__main__":
    debug_fly_death('M6_Ch05')
