#!/usr/bin/env python3
"""
Delete an experiment from the database.

This script deletes all data associated with an experiment_id, including:
- features_z
- features
- readings
- health_reports
- flies
- experiments

Usage:
    python delete_experiment.py --experiment-id 1
    python delete_experiment.py --experiment-id 1 --confirm
"""

import argparse
import sys
import os

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    import psycopg2
    from psycopg2.extras import execute_values
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    USE_DATABASE = False
    print("Error: Database modules not available")
    sys.exit(1)


def get_experiment_info(experiment_id):
    """Get experiment information before deletion."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT experiment_id, name, start_date, end_date, created_at
                    FROM experiments
                    WHERE experiment_id = %s
                """, (experiment_id,))
                result = cur.fetchone()
                return result
    except psycopg2.Error as e:
        print(f"Error getting experiment info: {e}")
        return None


def count_experiment_data(experiment_id):
    """Count rows in each table for this experiment."""
    counts = {}
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                # Count features_z
                try:
                    cur.execute("SELECT COUNT(*) FROM features_z WHERE experiment_id = %s", (experiment_id,))
                    counts['features_z'] = cur.fetchone()[0]
                except psycopg2.ProgrammingError:
                    counts['features_z'] = 0  # Table might not exist
                
                # Count features
                try:
                    cur.execute("SELECT COUNT(*) FROM features WHERE experiment_id = %s", (experiment_id,))
                    counts['features'] = cur.fetchone()[0]
                except psycopg2.ProgrammingError:
                    counts['features'] = 0
                
                # Count readings
                try:
                    cur.execute("SELECT COUNT(*) FROM readings WHERE experiment_id = %s", (experiment_id,))
                    counts['readings'] = cur.fetchone()[0]
                except psycopg2.ProgrammingError:
                    counts['readings'] = 0  # Table might not exist (shouldn't happen, but be safe)
                
                # Count health_reports
                try:
                    cur.execute("SELECT COUNT(*) FROM health_reports WHERE experiment_id = %s", (experiment_id,))
                    counts['health_reports'] = cur.fetchone()[0]
                except psycopg2.ProgrammingError:
                    counts['health_reports'] = 0
                
                # Count flies
                try:
                    cur.execute("SELECT COUNT(*) FROM flies WHERE experiment_id = %s", (experiment_id,))
                    counts['flies'] = cur.fetchone()[0]
                except psycopg2.ProgrammingError:
                    counts['flies'] = 0
                
                return counts
    except psycopg2.Error as e:
        print(f"Error counting experiment data: {e}")
        return None


def delete_experiment(experiment_id, confirm=False):
    """Delete an experiment and all associated data."""
    if not USE_DATABASE or not DB_AVAILABLE:
        print("Error: Database not available")
        return False
    
    # Get experiment info
    exp_info = get_experiment_info(experiment_id)
    if exp_info is None:
        print(f"Error: Experiment {experiment_id} not found")
        return False
    
    exp_id, name, start_date, end_date, created_at = exp_info
    
    # Count data to be deleted
    print(f"\nExperiment {exp_id}: {name}")
    print(f"  Start date: {start_date}")
    print(f"  End date: {end_date}")
    print(f"  Created: {created_at}")
    
    counts = count_experiment_data(experiment_id)
    if counts is None:
        print("Error: Could not count experiment data")
        return False
    
    print(f"\nData to be deleted:")
    print(f"  features_z: {counts['features_z']:,} rows")
    print(f"  features: {counts['features']:,} rows")
    print(f"  readings: {counts['readings']:,} rows")
    print(f"  health_reports: {counts['health_reports']:,} rows")
    print(f"  flies: {counts['flies']:,} rows")
    print(f"  experiments: 1 row")
    
    total_rows = sum(counts.values()) + 1
    print(f"\nTotal: {total_rows:,} rows will be deleted")
    
    # Confirm deletion
    if not confirm:
        response = input(f"\n⚠️  WARNING: This will permanently delete experiment {exp_id} and all its data.\n"
                        f"Type 'DELETE' to confirm: ")
        if response != 'DELETE':
            print("Deletion cancelled")
            return False
    
    # Delete in correct order (respecting foreign key constraints)
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                deleted_counts = {}
                
                # 1. Delete features_z (if exists)
                try:
                    cur.execute("DELETE FROM features_z WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['features_z'] = cur.rowcount
                except psycopg2.ProgrammingError:
                    deleted_counts['features_z'] = 0  # Table might not exist
                
                # 2. Delete features (if exists)
                try:
                    cur.execute("DELETE FROM features WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['features'] = cur.rowcount
                except psycopg2.ProgrammingError:
                    deleted_counts['features'] = 0  # Table might not exist
                
                # 3. Delete readings (if exists)
                try:
                    cur.execute("DELETE FROM readings WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['readings'] = cur.rowcount
                except psycopg2.ProgrammingError:
                    deleted_counts['readings'] = 0  # Table might not exist
                
                # 4. Delete health_reports (if exists)
                try:
                    cur.execute("DELETE FROM health_reports WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['health_reports'] = cur.rowcount
                except psycopg2.ProgrammingError:
                    deleted_counts['health_reports'] = 0  # Table might not exist
                
                # 5. Delete flies (if exists)
                try:
                    cur.execute("DELETE FROM flies WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['flies'] = cur.rowcount
                except psycopg2.ProgrammingError:
                    deleted_counts['flies'] = 0  # Table might not exist
                
                # 6. Delete experiment (must exist, but handle gracefully)
                try:
                    cur.execute("DELETE FROM experiments WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['experiments'] = cur.rowcount
                    if deleted_counts['experiments'] == 0:
                        print(f"\n⚠️  Warning: Experiment {experiment_id} not found in experiments table")
                        return False
                except psycopg2.ProgrammingError:
                    print(f"\n✗ Error: experiments table does not exist")
                    return False
                
                conn.commit()
                
                print(f"\n✓ Deletion complete:")
                for table, count in deleted_counts.items():
                    print(f"  {table}: {count:,} rows deleted")
                
                return True
                
    except psycopg2.Error as e:
        print(f"\n✗ Database error deleting experiment: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error deleting experiment: {e}")
        return False


def list_experiments():
    """List all experiments in the database."""
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT experiment_id, name, start_date, end_date, created_at
                    FROM experiments
                    ORDER BY created_at DESC
                """)
                results = cur.fetchall()
                
                if not results:
                    print("No experiments found in database")
                    return
                
                print("\nExperiments in database:")
                print("-" * 80)
                print(f"{'ID':<5} {'Name':<20} {'Start Date':<12} {'End Date':<12} {'Created':<20}")
                print("-" * 80)
                for exp_id, name, start_date, end_date, created_at in results:
                    end_str = str(end_date) if end_date else "None"
                    created_str = str(created_at)[:19] if created_at else "None"
                    print(f"{exp_id:<5} {name[:18]:<20} {str(start_date):<12} {end_str:<12} {created_str:<20}")
                print("-" * 80)
                
    except psycopg2.Error as e:
        print(f"Error listing experiments: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Delete an experiment from the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python delete_experiment.py --list
  
  # Delete experiment (will prompt for confirmation)
  python delete_experiment.py --experiment-id 1
  
  # Delete experiment without confirmation prompt
  python delete_experiment.py --experiment-id 1 --confirm
        """
    )
    
    parser.add_argument('--experiment-id', type=int, default=None,
                       help='Experiment ID to delete')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt (use with caution!)')
    parser.add_argument('--list', action='store_true',
                       help='List all experiments in the database')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return 0
    
    if args.experiment_id is None:
        print("Error: --experiment-id is required (or use --list to see available experiments)")
        parser.print_help()
        return 1
    
    success = delete_experiment(args.experiment_id, confirm=args.confirm)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

