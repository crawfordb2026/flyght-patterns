#!/usr/bin/env python3
"""Quick script to check health status distribution in the database."""

import sys
import os
import csv
from datetime import datetime

# Add current directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config import DB_CONFIG
    import psycopg2
except ImportError:
    print("Error: Could not import database config. Make sure config.py exists.")
    sys.exit(1)

def check_health_status(experiment_id=None):
    """Check the distribution of health statuses."""
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    output_file = None  # Track output file for final summary
    
    # If no experiment_id provided, use the latest
    if experiment_id is None:
        cur.execute("SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT 1")
        result = cur.fetchone()
        if not result:
            print("No experiments found in database")
            return
        experiment_id = result[0]
    
    print(f"\n{'='*60}")
    print(f"HEALTH STATUS REPORT - Experiment ID: {experiment_id}")
    print(f"{'='*60}\n")
    
    # Get total flies
    cur.execute("SELECT COUNT(DISTINCT fly_id) FROM flies WHERE experiment_id = %s", (experiment_id,))
    total_flies = cur.fetchone()[0]
    print(f"📊 Total flies in experiment: {total_flies}")
    
    # Get status breakdown
    cur.execute("""
        SELECT status, COUNT(*) as count
        FROM health_reports
        WHERE experiment_id = %s
        GROUP BY status
        ORDER BY count DESC
    """, (experiment_id,))
    
    print(f"\n📋 Health Status Breakdown:")
    print(f"{'Status':<15} {'Count':>10} {'Percentage':>12}")
    print(f"{'-'*40}")
    
    total_reports = 0
    status_counts = {}
    for status, count in cur.fetchall():
        status_counts[status] = count
        total_reports += count
        percentage = (count / total_flies * 100) if total_flies > 0 else 0
        print(f"{status:<15} {count:>10} {percentage:>11.1f}%")
    
    print(f"{'-'*40}")
    print(f"{'TOTAL':<15} {total_reports:>10}")
    
    # Get ALL dead flies and export to CSV
    if 'Dead' in status_counts and status_counts['Dead'] > 0:
        print(f"\n🔍 Fetching ALL dead flies ({status_counts['Dead']} total)...")
        cur.execute("""
            SELECT h.fly_id, f.monitor, f.channel, f.genotype, f.sex, f.treatment,
                   h.total_activity, h.longest_zero_hours, h.rel_activity, 
                   h.has_startle_response, h.missing_fraction
            FROM health_reports h
            JOIN flies f ON h.fly_id = f.fly_id AND h.experiment_id = f.experiment_id
            WHERE h.experiment_id = %s AND h.status = 'Dead'
            ORDER BY f.monitor, f.channel
        """, (experiment_id,))
        
        dead_flies = cur.fetchall()
        
        # Export to CSV file (use nonlocal to update the outer variable)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"dead_flies_exp{experiment_id}_{timestamp}.csv"  # This updates the function-level variable
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                'Fly ID', 'Monitor', 'Channel', 'Genotype', 'Sex', 'Treatment',
                'Total Activity', 'Longest Zero Hours', 'Rel Activity', 
                'Has Startle Response', 'Missing Fraction'
            ])
            # Write all rows
            for row in dead_flies:
                fly_id, monitor, channel, genotype, sex, treatment, total_activity, longest_zero_hours, rel_activity, has_startle, missing_frac = row
                writer.writerow([
                    fly_id, monitor, channel, genotype, sex, treatment,
                    total_activity or 0, 
                    f"{longest_zero_hours:.2f}" if longest_zero_hours is not None else "0.00",
                    f"{rel_activity:.3f}" if rel_activity is not None else "0.000",
                    "Yes" if has_startle else "No",
                    f"{missing_frac:.3f}" if missing_frac is not None else "0.000"
                ])
        
        print(f"  ✅ Exported {len(dead_flies)} dead flies to: {output_file}")
        
        # Show first 10 in console
        print(f"\n📋 Sample of dead flies (first 10, see CSV for all):")
        print(f"\n{'Fly ID':<18} {'Monitor':<13} {'Ch':<4} {'Genotype':<9} {'Sex':<8} {'Treatment':<12} {'Activity':<10} {'Zero Hrs':<10} {'Rel Act':<10} {'Startle':<8} {'Missing':<8}")
        print(f"{'-'*140}")
        for row in dead_flies[:10]:
            fly_id, monitor, channel, genotype, sex, treatment, total_activity, longest_zero_hours, rel_activity, has_startle, missing_frac = row
            startle_str = "Yes" if has_startle else "No"
            print(f"{fly_id:<18} {monitor:<13} {channel:<4} {genotype:<9} {sex:<8} {treatment:<12} "
                  f"{total_activity or 0:<10} {longest_zero_hours or 0:<10.1f} {rel_activity or 0:<10.3f} "
                  f"{startle_str:<8} {missing_frac or 0:<8.3f}")
    
    # Check if there's a pattern in dead flies
    if 'Dead' in status_counts and status_counts['Dead'] > 0:
        print(f"\n📈 Dead flies by genotype:")
        cur.execute("""
            SELECT f.genotype, COUNT(*) as count
            FROM health_reports h
            JOIN flies f ON h.fly_id = f.fly_id AND h.experiment_id = f.experiment_id
            WHERE h.experiment_id = %s AND h.status = 'Dead'
            GROUP BY f.genotype
            ORDER BY count DESC
        """, (experiment_id,))
        
        for genotype, count in cur.fetchall():
            print(f"  {genotype:<15} {count:>5} flies")
        
        print(f"\n📈 Dead flies by treatment:")
        cur.execute("""
            SELECT f.treatment, COUNT(*) as count
            FROM health_reports h
            JOIN flies f ON h.fly_id = f.fly_id AND h.experiment_id = f.experiment_id
            WHERE h.experiment_id = %s AND h.status = 'Dead'
            GROUP BY f.treatment
            ORDER BY count DESC
        """, (experiment_id,))
        
        for treatment, count in cur.fetchall():
            print(f"  {treatment:<15} {count:>5} flies")
    
    cur.close()
    conn.close()
    
    # Final summary
    if output_file:
        print(f"\n{'='*60}")
        print(f"📄 Full dead flies report saved to: {output_file}")
        print(f"{'='*60}")
    print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Check health status distribution')
    parser.add_argument('--experiment-id', type=int, default=None,
                       help='Experiment ID (default: latest)')
    args = parser.parse_args()
    
    check_health_status(args.experiment_id)
