#!/usr/bin/env python3
"""
Delete an experiment (or all experiments) from the database.

This script deletes all data associated with an experiment_id, including:
- features_z
- features
- readings
- health_reports
- flies
- experiments

Usage:
    python delete_experiment.py --experiment-id 1          # Delete one experiment
    python delete_experiment.py --experiment-id 1 --confirm # Delete without prompt
    python delete_experiment.py --all                      # Delete ALL experiments
    python delete_experiment.py --all --confirm --max-workers 8  # Delete all with 8 threads
    python delete_experiment.py --list                     # List all experiments
"""

import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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

# Thread-safe print lock for progress updates
print_lock = Lock()


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


def delete_experiment(experiment_id, confirm=False, show_progress=True, skip_counting=False):
    """Delete an experiment and all associated data.
    
    Args:
        experiment_id: ID of experiment to delete
        confirm: If True, skip confirmation prompt
        show_progress: If True, print progress messages
        skip_counting: If True, skip counting rows (faster)
    """
    if not USE_DATABASE or not DB_AVAILABLE:
        if show_progress:
            with print_lock:
                print("Error: Database not available")
        return False
    
    # Get experiment info
    exp_info = get_experiment_info(experiment_id)
    if exp_info is None:
        if show_progress:
            with print_lock:
                print(f"Error: Experiment {experiment_id} not found")
        return False
    
    exp_id, name, start_date, end_date, created_at = exp_info
    
    # Show info and count (if not skipping)
    if show_progress:
        with print_lock:
            print(f"\nExperiment {exp_id}: {name}")
            print(f"  Start date: {start_date}")
            print(f"  End date: {end_date}")
            print(f"  Created: {created_at}")
    
    if not skip_counting:
        counts = count_experiment_data(experiment_id)
        if counts is None:
            if show_progress:
                with print_lock:
                    print("Error: Could not count experiment data")
            return False
        
        if show_progress:
            with print_lock:
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
            if show_progress:
                with print_lock:
                    print("Deletion cancelled")
            return False
    
    # Delete in correct order (respecting foreign key constraints)
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                deleted_counts = {}
                
                # 1. Delete features_z (if exists)
                if show_progress:
                    with print_lock:
                        print(f"  Deleting features_z...", end='', flush=True)
                try:
                    cur.execute("DELETE FROM features_z WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['features_z'] = cur.rowcount
                    if show_progress:
                        with print_lock:
                            print(f" ✓ ({deleted_counts['features_z']:,} rows)")
                except psycopg2.ProgrammingError:
                    deleted_counts['features_z'] = 0  # Table might not exist
                    if show_progress:
                        with print_lock:
                            print(f" ✓ (table doesn't exist)")
                
                # 2. Delete features (if exists)
                if show_progress:
                    with print_lock:
                        print(f"  Deleting features...", end='', flush=True)
                try:
                    cur.execute("DELETE FROM features WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['features'] = cur.rowcount
                    if show_progress:
                        with print_lock:
                            print(f" ✓ ({deleted_counts['features']:,} rows)")
                except psycopg2.ProgrammingError:
                    deleted_counts['features'] = 0
                    if show_progress:
                        with print_lock:
                            print(f" ✓ (table doesn't exist)")
                
                # 3. Delete readings (usually the largest table)
                if show_progress:
                    with print_lock:
                        print(f"  Deleting readings...", end='', flush=True)
                try:
                    cur.execute("DELETE FROM readings WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['readings'] = cur.rowcount
                    if show_progress:
                        with print_lock:
                            print(f" ✓ ({deleted_counts['readings']:,} rows)")
                except psycopg2.ProgrammingError:
                    deleted_counts['readings'] = 0
                    if show_progress:
                        with print_lock:
                            print(f" ✓ (table doesn't exist)")
                
                # 4. Delete health_reports
                if show_progress:
                    with print_lock:
                        print(f"  Deleting health_reports...", end='', flush=True)
                try:
                    cur.execute("DELETE FROM health_reports WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['health_reports'] = cur.rowcount
                    if show_progress:
                        with print_lock:
                            print(f" ✓ ({deleted_counts['health_reports']:,} rows)")
                except psycopg2.ProgrammingError:
                    deleted_counts['health_reports'] = 0
                    if show_progress:
                        with print_lock:
                            print(f" ✓ (table doesn't exist)")
                
                # 5. Delete flies
                if show_progress:
                    with print_lock:
                        print(f"  Deleting flies...", end='', flush=True)
                try:
                    cur.execute("DELETE FROM flies WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['flies'] = cur.rowcount
                    if show_progress:
                        with print_lock:
                            print(f" ✓ ({deleted_counts['flies']:,} rows)")
                except psycopg2.ProgrammingError:
                    deleted_counts['flies'] = 0
                    if show_progress:
                        with print_lock:
                            print(f" ✓ (table doesn't exist)")
                
                # 6. Delete experiment (must exist, but handle gracefully)
                if show_progress:
                    with print_lock:
                        print(f"  Deleting experiment record...", end='', flush=True)
                try:
                    cur.execute("DELETE FROM experiments WHERE experiment_id = %s", (experiment_id,))
                    deleted_counts['experiments'] = cur.rowcount
                    if deleted_counts['experiments'] == 0:
                        if show_progress:
                            with print_lock:
                                print(f"\n⚠️  Warning: Experiment {experiment_id} not found in experiments table")
                        return False
                    if show_progress:
                        with print_lock:
                            print(f" ✓")
                except psycopg2.ProgrammingError:
                    if show_progress:
                        with print_lock:
                            print(f"\n✗ Error: experiments table does not exist")
                    return False
                
                conn.commit()
                
                if show_progress:
                    with print_lock:
                        print(f"  ✓ Deletion complete for experiment {exp_id}")
                
                return True
                
    except psycopg2.Error as e:
        if show_progress:
            with print_lock:
                print(f"\n✗ Database error deleting experiment {experiment_id}: {e}")
        return False
    except Exception as e:
        if show_progress:
            with print_lock:
                print(f"\n✗ Unexpected error deleting experiment {experiment_id}: {e}")
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


def delete_all_experiments(confirm=False, max_workers=4):
    """Delete ALL experiments from the database using multithreading.
    
    Args:
        confirm: If True, skip confirmation prompt
        max_workers: Number of parallel threads to use (default: 4)
    """
    if not USE_DATABASE or not DB_AVAILABLE:
        print("Error: Database not available")
        return False
    
    try:
        # Get all experiment IDs
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT experiment_id, name FROM experiments ORDER BY experiment_id")
                experiments = cur.fetchall()
        
        if not experiments:
            print("No experiments found in database")
            return True
        
        print(f"\n⚠️  WARNING: You are about to delete ALL {len(experiments)} experiments!")
        print("\nExperiments to be deleted:")
        for exp_id, name in experiments:
            print(f"  - ID {exp_id}: {name}")
        
        # Confirm deletion
        if not confirm:
            response = input(f"\n⚠️  Type 'DELETE ALL' to confirm deletion of ALL experiments: ")
            if response != 'DELETE ALL':
                print("Deletion cancelled")
                return False
        
        print(f"\nDeleting all {len(experiments)} experiments using {max_workers} parallel threads...")
        
        # Use ThreadPoolExecutor for parallel deletion
        success_count = 0
        failed_experiments = []
        completed = 0
        
        def delete_with_progress(exp_data):
            """Wrapper function for parallel deletion with progress tracking."""
            exp_id, name = exp_data
            try:
                # Skip counting for speed when deleting all
                result = delete_experiment(exp_id, confirm=True, show_progress=False, skip_counting=True)
                return (exp_id, name, result, None)
            except Exception as e:
                return (exp_id, name, False, str(e))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all deletion tasks
            future_to_exp = {
                executor.submit(delete_with_progress, (exp_id, name)): (exp_id, name)
                for exp_id, name in experiments
            }
            
            # Process completed tasks
            for future in as_completed(future_to_exp):
                exp_id, name = future_to_exp[future]
                completed += 1
                try:
                    exp_id_result, name_result, success, error = future.result()
                    if success:
                        success_count += 1
                        with print_lock:
                            print(f"[{completed}/{len(experiments)}] ✓ Deleted experiment {exp_id}: {name}")
                    else:
                        failed_experiments.append((exp_id, name, error or "Deletion failed"))
                        with print_lock:
                            print(f"[{completed}/{len(experiments)}] ✗ Failed to delete experiment {exp_id}: {name} ({error or 'Unknown error'})")
                except Exception as e:
                    failed_experiments.append((exp_id, name, str(e)))
                    with print_lock:
                        print(f"[{completed}/{len(experiments)}] ✗ Error deleting experiment {exp_id}: {name} ({e})")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"✅ Successfully deleted {success_count}/{len(experiments)} experiments")
        if failed_experiments:
            print(f"⚠️  Failed to delete {len(failed_experiments)} experiments:")
            for exp_id, name, error in failed_experiments:
                print(f"  - Experiment {exp_id}: {name} ({error})")
        print(f"{'='*60}")
        
        return success_count == len(experiments)
        
    except Exception as e:
        print(f"Error deleting all experiments: {e}")
        return False


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
  
  # Delete ALL experiments (will prompt for confirmation)
  python delete_experiment.py --all
  
  # Delete ALL experiments without confirmation prompt (with 8 parallel threads)
  python delete_experiment.py --all --confirm --max-workers 8
        """
    )
    
    parser.add_argument('--experiment-id', type=int, default=None,
                       help='Experiment ID to delete')
    parser.add_argument('--all', action='store_true',
                       help='Delete ALL experiments')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt (use with caution!)')
    parser.add_argument('--list', action='store_true',
                       help='List all experiments in the database')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel threads when deleting all experiments (default: 4)')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return 0
    
    # Check for conflicting arguments
    if args.all and args.experiment_id is not None:
        print("Error: Cannot use both --all and --experiment-id. Use one or the other.")
        parser.print_help()
        return 1
    
    if args.all:
        success = delete_all_experiments(confirm=args.confirm, max_workers=args.max_workers)
        return 0 if success else 1
    
    if args.experiment_id is None:
        print("Error: --experiment-id is required (or use --all to delete all, or --list to see available experiments)")
        parser.print_help()
        return 1
    
    success = delete_experiment(args.experiment_id, confirm=args.confirm)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

