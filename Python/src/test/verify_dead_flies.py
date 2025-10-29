#!/usr/bin/env python3
"""
Verification script to check that declared dead flies actually have
12+ consecutive hours of MT=0 inactivity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def verify_dead_fly(fly_id, dam_data, death_report):
    """
    Verify that a declared dead fly actually has 12+ consecutive hours of MT=0.
    
    Args:
        fly_id (str): Fly ID to verify
        dam_data (pd.DataFrame): Full dataset
        death_report (pd.DataFrame): Death report
        
    Returns:
        dict: Verification results
    """
    # Get death info
    death_info = death_report[death_report['fly_id'] == fly_id]
    if len(death_info) == 0:
        return {'error': 'Fly not found in death report'}
    
    time_of_death = pd.to_datetime(death_info['time_of_death'].iloc[0])
    
    # Get all MT data for this fly
    fly_data = dam_data[dam_data['fly_id'] == fly_id]
    mt_data = fly_data[fly_data['reading'] == 'MT'].copy()
    mt_data = mt_data.sort_values('datetime').reset_index(drop=True)
    
    print(f"\n🔍 Verifying {fly_id}:")
    print(f"   Declared death time: {time_of_death}")
    print(f"   Total MT records: {len(mt_data)}")
    
    # Find the longest consecutive period of MT=0
    mt_data['is_zero'] = (mt_data['value'] == 0)
    mt_data['zero_group'] = (mt_data['is_zero'] != mt_data['is_zero'].shift()).cumsum()
    
    # Analyze each group
    max_zero_duration = 0
    max_zero_period = None
    zero_periods = []
    
    for group_id in mt_data['zero_group'].unique():
        group_data = mt_data[mt_data['zero_group'] == group_id]
        
        if group_data['is_zero'].iloc[0]:  # Only process zero groups
            start_time = group_data['datetime'].min()
            end_time = group_data['datetime'].max()
            
            # Calculate duration in minutes
            duration_minutes = (end_time - start_time).total_seconds() / 60 + 1
            
            zero_periods.append({
                'start': start_time,
                'end': end_time,
                'duration_minutes': duration_minutes,
                'duration_hours': duration_minutes / 60
            })
            
            if duration_minutes > max_zero_duration:
                max_zero_duration = duration_minutes
                max_zero_period = {
                    'start': start_time,
                    'end': end_time,
                    'duration_minutes': duration_minutes,
                    'duration_hours': duration_minutes / 60
                }
    
    print(f"   Longest zero period: {max_zero_duration:.1f} minutes ({max_zero_duration/60:.1f} hours)")
    
    if max_zero_period:
        print(f"   Zero period: {max_zero_period['start']} to {max_zero_period['end']}")
        
        # Check if this period includes or is near the declared death time
        death_window_start = time_of_death - timedelta(hours=1)
        death_window_end = time_of_death + timedelta(hours=1)
        
        period_includes_death = (max_zero_period['start'] <= time_of_death <= max_zero_period['end'])
        period_near_death = (death_window_start <= max_zero_period['start'] <= death_window_end)
        
        print(f"   Period includes death time: {period_includes_death}")
        print(f"   Period near death time: {period_near_death}")
        
        # Show some sample data around death time
        print(f"\n   Sample data around death time:")
        around_death = mt_data[
            (mt_data['datetime'] >= time_of_death - timedelta(hours=2)) &
            (mt_data['datetime'] <= time_of_death + timedelta(hours=2))
        ].head(10)
        
        for _, row in around_death.iterrows():
            status = "ZERO" if row['value'] == 0 else f"MT={row['value']}"
            print(f"     {row['datetime']}: {status}")
    
    # Check if there are any periods >= 720 minutes (12 hours)
    long_zero_periods = [p for p in zero_periods if p['duration_minutes'] >= 720]
    
    verification_result = {
        'fly_id': fly_id,
        'declared_death_time': time_of_death,
        'max_zero_duration_minutes': max_zero_duration,
        'max_zero_duration_hours': max_zero_duration / 60,
        'has_12h_zero_period': len(long_zero_periods) > 0,
        'long_zero_periods': long_zero_periods,
        'total_zero_periods': len(zero_periods),
        'verification_passed': max_zero_duration >= 720
    }
    
    return verification_result


def main():
    """
    Verify all declared dead flies.
    """
    print("🔍 Dead Fly Verification")
    print("=" * 50)
    
    # Load data
    print("📂 Loading data...")
    dam_data = pd.read_csv('../../data/processed/dam_data_merged.csv')
    dam_data['datetime'] = pd.to_datetime(dam_data['datetime'])
    
    death_report = pd.read_csv('../../data/processed/dead_flies_report.csv')
    death_report['time_of_death'] = pd.to_datetime(death_report['time_of_death'])
    
    print(f"   Loaded {len(dam_data):,} rows")
    print(f"   {len(death_report)} flies declared dead")
    
    # Verify each dead fly
    print(f"\n🔍 Verifying each declared dead fly...")
    
    verification_results = []
    
    for _, death_row in death_report.iterrows():
        fly_id = death_row['fly_id']
        result = verify_dead_fly(fly_id, dam_data, death_report)
        verification_results.append(result)
        
        if result.get('verification_passed', False):
            print(f"   ✅ {fly_id}: VERIFIED (has {result['max_zero_duration_hours']:.1f}h zero period)")
        else:
            print(f"   ❌ {fly_id}: FAILED (max zero period: {result['max_zero_duration_hours']:.1f}h)")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("=== VERIFICATION SUMMARY ===")
    print("=" * 50)
    
    passed = sum(1 for r in verification_results if r.get('verification_passed', False))
    total = len(verification_results)
    
    print(f"\nVerification Results:")
    print(f"  Total flies checked: {total}")
    print(f"  Passed verification: {passed}")
    print(f"  Failed verification: {total - passed}")
    
    if passed == total:
        print(f"\n✅ ALL DEAD FLIES VERIFIED! The detection algorithm is working correctly.")
    else:
        print(f"\n⚠️  Some flies failed verification. Check the details above.")
    
    # Show detailed results
    print(f"\nDetailed Results:")
    for result in verification_results:
        fly_id = result['fly_id']
        max_hours = result['max_zero_duration_hours']
        passed = result.get('verification_passed', False)
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"  {fly_id}: {max_hours:.1f}h max zero period - {status}")


if __name__ == "__main__":
    main()
