#!/usr/bin/env python3
"""
Investigate why so many flies are being classified as "Dead".
Check the reference day and activity patterns.
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DB_CONFIG
    import psycopg2
except ImportError:
    print("Error: Could not import database config.")
    sys.exit(1)

def investigate_experiment(experiment_id):
    """Deep dive into death classifications."""
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print(f"\n{'='*70}")
    print(f"DEATH CLASSIFICATION INVESTIGATION - Experiment {experiment_id}")
    print(f"{'='*70}\n")
    
    # Get a sample of "dead" flies with their metrics
    cur.execute("""
        SELECT h.fly_id, f.monitor, f.channel, f.genotype, f.sex, f.treatment,
               h.total_activity, h.longest_zero_hours, h.rel_activity, 
               h.has_startle_response, h.missing_fraction, h.status
        FROM health_reports h
        JOIN flies f ON h.fly_id = f.fly_id AND h.experiment_id = f.experiment_id
        WHERE h.experiment_id = %s AND h.status = 'Dead'
        ORDER BY h.total_activity DESC
        LIMIT 20
    """, (experiment_id,))
    
    dead_flies_sample = cur.fetchall()
    
    print("📊 Sample of DEAD flies with HIGHEST activity:")
    print("(If truly dead, these should have very low activity)")
    print()
    print(f"{'Fly ID':<20} {'Activity':<10} {'Zero Hrs':<12} {'Rel Act':<12} {'Startle':<10}")
    print("-" * 70)
    
    for row in dead_flies_sample[:10]:
        fly_id, monitor, channel, genotype, sex, treatment, total_activity, longest_zero_hours, rel_activity, has_startle, missing_frac, status = row
        print(f"{fly_id:<20} {total_activity or 0:<10} {longest_zero_hours or 0:<12.1f} {rel_activity or 0:<12.3f} {'Yes' if has_startle else 'No':<10}")
    
    # Check reference day (day 4) activity distribution
    print("\n\n📈 Checking reference day (day 4) activity levels:")
    print("(Dead flies might have had low activity on ref day, making decline look worse)")
    print()
    
    # Get activity data for all days to find reference day activity
    # Note: exp_day is not stored in readings, we need to calculate it from datetime
    print("  (Analyzing daily activity patterns...)")
    print("  Note: This analysis uses the LAST day data from health_reports")
    print()
    
    # For now, let's look at the final day activity and compare to what rel_activity says
    # We can't easily get day 4 from readings without exp_day
    # Instead, show flies with lowest final activity but high rel_activity (alive flies)
    cur.execute("""
        SELECT h.fly_id, h.total_activity, h.rel_activity, h.status
        FROM health_reports h
        WHERE h.experiment_id = %s
        ORDER BY h.total_activity ASC
        LIMIT 20
    """, (experiment_id,))
    
    low_activity_flies = cur.fetchall()
    
    print("Flies with LOWEST final day activity:")
    print(f"{'Fly ID':<20} {'Final Activity':<15} {'Rel Activity':<15} {'Status':<12}")
    print("-" * 65)
    for fly_id, total_activity, rel_activity, status in low_activity_flies[:15]:
        print(f"{fly_id:<20} {total_activity or 0:<15} {rel_activity or 0:<15.3f} {status:<12}")
    
    # Overall statistics
    print("\n\n📊 Overall Statistics:")
    
    # Average activity levels by status
    cur.execute("""
        SELECT status, 
               AVG(total_activity) as avg_activity,
               AVG(longest_zero_hours) as avg_zero_hours,
               AVG(rel_activity) as avg_rel_activity,
               COUNT(*) as count
        FROM health_reports
        WHERE experiment_id = %s
        GROUP BY status
        ORDER BY avg_activity DESC
    """, (experiment_id,))
    
    status_stats = cur.fetchall()
    
    print(f"{'Status':<12} {'Count':<8} {'Avg Activity':<15} {'Avg Zero Hrs':<15} {'Avg Rel Act':<15}")
    print("-" * 70)
    for status, avg_act, avg_zero, avg_rel, count in status_stats:
        print(f"{status:<12} {count:<8} {avg_act or 0:<15.1f} {avg_zero or 0:<15.1f} {avg_rel or 0:<15.3f}")
    
    # Check how many dead flies are dead because of each criterion
    print("\n\n🔍 Death Classification Breakdown:")
    print("Analyzing why flies were marked as dead...")
    
    # Flies with 24+ hours zero (A2 threshold)
    cur.execute("""
        SELECT COUNT(*) 
        FROM health_reports
        WHERE experiment_id = %s AND status = 'Dead' AND longest_zero_hours >= 24
    """, (experiment_id,))
    a2_deaths = cur.fetchone()[0]
    
    # Flies with 12-24 hours zero and no startle
    cur.execute("""
        SELECT COUNT(*) 
        FROM health_reports
        WHERE experiment_id = %s AND status = 'Dead' 
        AND longest_zero_hours >= 12 AND longest_zero_hours < 24
        AND has_startle_response = FALSE
    """, (experiment_id,))
    a1_deaths = cur.fetchone()[0]
    
    # Flies with rel_activity < 0.2 (decline-based)
    cur.execute("""
        SELECT COUNT(*) 
        FROM health_reports
        WHERE experiment_id = %s AND status = 'Dead' 
        AND (rel_activity IS NULL OR rel_activity < 0.2)
    """, (experiment_id,))
    decline_deaths = cur.fetchone()[0]
    
    # Total dead
    cur.execute("""
        SELECT COUNT(*) 
        FROM health_reports
        WHERE experiment_id = %s AND status = 'Dead'
    """, (experiment_id,))
    total_deaths = cur.fetchone()[0]
    
    print(f"  24+ hours zero activity (A2):        {a2_deaths:>6} ({a2_deaths/total_deaths*100:.1f}%)")
    print(f"  12+ hours zero + no startle (A1):    {a1_deaths:>6} ({a1_deaths/total_deaths*100:.1f}%)")
    print(f"  Activity decline < 20% (decline):    {decline_deaths:>6} ({decline_deaths/total_deaths*100:.1f}%)")
    print(f"  {'─' * 50}")
    print(f"  Total dead flies:                    {total_deaths:>6}")
    
    print("\n\n💡 Recommendations:")
    if decline_deaths / total_deaths > 0.7:
        print("  ⚠️  Most deaths are from DECLINE criterion (< 20% of ref day)")
        print("     → Consider checking if reference day 4 is appropriate")
        print("     → Some flies might have been inactive on day 4 but alive")
        print("     → Try adjusting --ref-day or --death-threshold")
    
    if a2_deaths / total_deaths > 0.5:
        print("  ✓  Most deaths are from 24+ hours zero activity (strong evidence)")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Investigate death classifications')
    parser.add_argument('--experiment-id', type=int, required=True,
                       help='Experiment ID to investigate')
    args = parser.parse_args()
    
    investigate_experiment(args.experiment_id)
