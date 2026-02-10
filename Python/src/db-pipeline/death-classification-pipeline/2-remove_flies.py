#!/usr/bin/env python3
"""
Pipeline Step 2: Remove Flies (Optional)

This script:
1. Reads prepared data from Step 1 (database)
2. Optionally reads health report from Step 1 (database)
3. Removes flies based on various criteria:
   - Specific fly IDs
   - Specific days for certain flies
   - Metadata criteria (genotype, sex, treatment)
   - Health status from health report
4. Updates database with cleaned data (removes readings/flies from database)

This step is OPTIONAL. Run it only if you want to remove flies based on
the health report or other criteria. If you skip this step, use the
prepared data directly in Step 3.


Temporary profiling: [TMP] print markers added for bottleneck finding. Remove later by
searching for "[TMP]" or "TMP" in this file and deleting those lines.

Example command line usage: 
python 2-remove_flies.py \
  --statuses "Dead,Unhealthy" \
  --treatments "8mM His,VEH" \
  --genotypes "Fmn,Rye" \
  --sexes "Female" \
  --flies "5-ch18,6-ch5,5-ch26" \
  --per-fly-remove '{"5-ch7": [1, 2], "6-ch18": [3, 4]}'
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
import time
from pathlib import Path
from importlib import import_module

# Allow importing config from parent (db-pipeline) when run from death-classification-pipeline
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    import psycopg2
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    USE_DATABASE = False


# ============================================================
#   USER CONFIGURATION
# ============================================================


# Removal criteria (set these in the script or use command-line arguments)
REMOVE_FLIES = None  # Example: ["6-ch23", "6-ch5", "5-ch7"]
REMOVE_GENOTYPES = None  # Example: ["mutant1", "control"]
REMOVE_SEXES = None  # Example: ["Male", "Female"]
REMOVE_TREATMENTS = None  # Example: ["drug1", "vehicle"]
REMOVE_PER_FLY_DAYS = None  # Example: {"5-ch7": [1, 2], "6-ch18": [3]}
REMOVE_BY_STATUS = ["Dead"]  # Example: ["Dead", "QC_Fail"] - requires health report


# ============================================================
#   HELPER FUNCTIONS
# ============================================================

def create_fly_key(monitor, channel):
    """Create a fly key in the format "Monitor-Channel" (lowercase)."""
    channel_str = str(channel).lower()
    if channel_str.startswith('ch'):
        channel_str = channel_str[2:]
    return f"{monitor}-ch{channel_str}"


def is_ok(x):
    """Check if a value is valid (not NA, not empty, not "na")."""
    if pd.isna(x):
        return False
    x_str = str(x).lower()
    return x_str != "" and x_str != "na"


# ============================================================
#   MAIN REMOVAL FUNCTION
# ============================================================

def remove_flies(
    df,
    flies_to_remove=None,
    per_fly_remove=None,
    genotypes_to_remove=None,
    sexes_to_remove=None,
    treatments_to_remove=None,
    statuses_to_remove=None,
    health_report=None
):
    """
    Remove flies from dataset based on various criteria.
    
    Args:
        df: Input DataFrame
        flies_to_remove: List of fly keys (e.g., ["6-ch23", "6-ch5"])
        per_fly_remove: Dict mapping fly keys to lists of days to remove
        genotypes_to_remove: List of genotypes to remove
        sexes_to_remove: List of sexes to remove
        treatments_to_remove: List of treatments to remove
        statuses_to_remove: List of health statuses to remove (requires health_report)
        health_report: DataFrame with health report (for status-based removal)
        
    Returns:
        tuple: (cleaned_df, removals_summary)
    """
    print(f"[TMP] remove_flies() START", flush=True)
    df = df.copy()
    original_count = len(df)
    print(f"[TMP] remove_flies() after copy, rows={original_count}", flush=True)

    # Create fly key column
    channel_str = df['channel'].astype(str).str.lower()
    channel_str = channel_str.str.replace('^ch', '', regex=True)
    df['fly_key'] = df['monitor'].astype(str) + '-ch' + channel_str
    
    # Normalize metadata to lowercase for comparison
    if 'genotype' in df.columns:
        df['genotype_lower'] = df['genotype'].astype(str).str.lower()
    if 'sex' in df.columns:
        df['sex_lower'] = df['sex'].astype(str).str.lower()
    if 'treatment' in df.columns:
        df['treatment_lower'] = df['treatment'].astype(str).str.lower()
    
    removals_summary = {
        'flies_removed': 0,
        'rows_removed_by_fly': 0,
        'rows_removed_by_days': 0,
        'rows_removed_by_metadata': 0,
        'rows_removed_by_status': 0
    }
    
    # 1. Remove by health status (if health report provided)
    print(f"[TMP] remove_flies() before status removal block", flush=True)
    if statuses_to_remove and health_report is not None:
        before = len(df)
        # Ensure column names are lowercase (database returns lowercase)
        health_report = health_report.copy()
        health_report.columns = [col.lower() if isinstance(col, str) else col for col in health_report.columns]
        
        # Get flies with specified statuses
        flies_to_remove_by_status = health_report[
            health_report['final_status'].isin(statuses_to_remove)
        ][['monitor', 'channel']].drop_duplicates()
        
        if len(flies_to_remove_by_status) > 0:
            # Create fly keys
            channel_str = flies_to_remove_by_status['channel'].astype(str).str.lower()
            channel_str = channel_str.str.replace('^ch', '', regex=True)
            flies_to_remove_by_status['fly_key'] = flies_to_remove_by_status['monitor'].astype(str) + '-ch' + channel_str
            fly_keys_to_remove = flies_to_remove_by_status['fly_key'].str.lower().tolist()
            
            df = df[~df['fly_key'].str.lower().isin(fly_keys_to_remove)].copy()
            after = len(df)
            removals_summary['rows_removed_by_status'] = before - after
            removals_summary['flies_removed'] = len(fly_keys_to_remove)
    print(f"[TMP] remove_flies() after status removal block", flush=True)

    # 2. Remove specific flies completely
    if flies_to_remove:
        flies_to_remove_lower = [f.lower() for f in flies_to_remove]
        before = len(df)
        df = df[~df['fly_key'].str.lower().isin(flies_to_remove_lower)].copy()
        after = len(df)
        removals_summary['flies_removed'] += len(flies_to_remove)
        removals_summary['rows_removed_by_fly'] = before - after
    print(f"[TMP] remove_flies() after specific flies block", flush=True)

    # 3. Remove specific days for certain flies
    if per_fly_remove:
        if 'exp_day' not in df.columns:
            pass
        else:
            before = len(df)
            for fly_key, days in per_fly_remove.items():
                fly_key_lower = fly_key.lower()
                mask = (
                    (df['fly_key'].str.lower() == fly_key_lower) &
                    (df['exp_day'].isin(days))
                )
                df = df[~mask].copy()
            after = len(df)
            removals_summary['rows_removed_by_days'] = before - after
    print(f"[TMP] remove_flies() after per-fly days block", flush=True)

    # 4. Remove by metadata
    before = len(df)
    metadata_mask = pd.Series([True] * len(df), index=df.index)
    
    if genotypes_to_remove:
        genotypes_lower = [g.lower() for g in genotypes_to_remove]
        if 'genotype_lower' in df.columns:
            metadata_mask &= ~df['genotype_lower'].isin(genotypes_lower)
    
    if sexes_to_remove:
        sexes_lower = [s.lower() for s in sexes_to_remove]
        if 'sex_lower' in df.columns:
            metadata_mask &= ~df['sex_lower'].isin(sexes_lower)
    
    if treatments_to_remove:
        treatments_lower = [t.lower() for t in treatments_to_remove]
        if 'treatment_lower' in df.columns:
            metadata_mask &= ~df['treatment_lower'].isin(treatments_lower)
    
    df = df[metadata_mask].copy()
    after = len(df)
    removals_summary['rows_removed_by_metadata'] = before - after
    print(f"[TMP] remove_flies() after metadata block", flush=True)

    # Remove helper columns
    columns_to_drop = ['fly_key', 'genotype_lower', 'sex_lower', 'treatment_lower']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    final_count = len(df)
    total_removed = original_count - final_count
    print(f"[TMP] remove_flies() END (returning)", flush=True)

    return df, removals_summary, total_removed


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def remove_flies_from_data(
    flies_to_remove=None,
    per_fly_remove=None,
    genotypes_to_remove=None,
    sexes_to_remove=None,
    treatments_to_remove=None,
    statuses_to_remove=None,
    use_config=True,
    experiment_id=None
):
    """
    Main function to remove flies from prepared data.
    
    Args:
        flies_to_remove: List of fly keys to remove
        per_fly_remove: Dict mapping fly keys to days to remove
        genotypes_to_remove: List of genotypes to remove
        sexes_to_remove: List of sexes to remove
        treatments_to_remove: List of treatments to remove
        statuses_to_remove: List of health statuses to remove
        use_config: Whether to use hardcoded config values
        experiment_id: Experiment ID to use (None = use latest)
        
    Returns:
        Cleaned DataFrame
    """
    _t0 = time.perf_counter()
    print(f"[TMP] remove_flies_from_data() START (elapsed 0.00s)", flush=True)

    # Require database
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required. Please ensure database is configured and available.")

    # Use config values if specified
    if use_config:
        if flies_to_remove is None:
            flies_to_remove = REMOVE_FLIES
        if genotypes_to_remove is None:
            genotypes_to_remove = REMOVE_GENOTYPES
        if sexes_to_remove is None:
            sexes_to_remove = REMOVE_SEXES
        if treatments_to_remove is None:
            treatments_to_remove = REMOVE_TREATMENTS
        if per_fly_remove is None:
            per_fly_remove = REMOVE_PER_FLY_DAYS
        if statuses_to_remove is None:
            statuses_to_remove = REMOVE_BY_STATUS
            # Convert string to list if needed (for backward compatibility)
            if isinstance(statuses_to_remove, str):
                statuses_to_remove = [statuses_to_remove]
    
    # ============================================================
    # STEP 1: Load data from database
    # ============================================================
    print(f"[TMP] before STEP 1 (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
    print(f"\n📥 Loading data from database...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    # Import database functions from step 1
    print(f"[TMP] before import step1 (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
    from importlib import import_module
    step1 = import_module('1-prepare_data_and_health')
    print(f"[TMP] after import step1 (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
    
    # Use provided experiment_id, or get latest if not provided
    experiment_id_param = experiment_id
    if experiment_id_param is None:
        print(f"[TMP] before get_latest_experiment_id (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
        print(f"  Getting latest experiment ID...", end='', flush=True)
        experiment_id = step1.get_latest_experiment_id()
        print(f"[TMP] after get_latest_experiment_id (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
        print(f" ✓ (ID: {experiment_id})")
    else:
        experiment_id = experiment_id_param
        print(f"  Using experiment ID: {experiment_id}")
    
    if not experiment_id:
        raise ValueError("No experiment found in database")

    # Fast path: when only removing by health status, we need only the health report
    # (skip loading 33M+ readings and in-memory removal — saves ~3–4 minutes).
    status_only = bool(
        statuses_to_remove
        and not flies_to_remove
        and not per_fly_remove
        and not genotypes_to_remove
        and not sexes_to_remove
        and not treatments_to_remove
    )
    precomputed_removed_fly_ids = None
    df = None
    df_cleaned = None
    removals_summary = {
        'flies_removed': 0,
        'rows_removed_by_fly': 0,
        'rows_removed_by_days': 0,
        'rows_removed_by_metadata': 0,
        'rows_removed_by_status': 0
    }
    total_removed = None

    if status_only:
        print(f"[TMP] status-only fast path: loading health report only (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
        print(f"\n📊 Loading health report (status-only removal, skipping full readings load)...")
        print(f"  Loading from database...", end='', flush=True)
        try:
            health_report_fast = step1.load_health_report_from_db(experiment_id)
            print(f"[TMP] after load_health_report_from_db fast path (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
            if health_report_fast is None or len(health_report_fast) == 0:
                print(f" ⚠️")
                print("  Warning: No health report found. Falling back to full pipeline.")
                status_only = False
            else:
                health_report_fast.columns = [c.lower() if isinstance(c, str) else c for c in health_report_fast.columns]
                if 'final_status' not in health_report_fast.columns:
                    status_only = False
                else:
                    removed = health_report_fast[health_report_fast['final_status'].isin(statuses_to_remove)]
                    precomputed_removed_fly_ids = set(removed['fly_id'].unique().tolist())
                    removals_summary['flies_removed'] = len(precomputed_removed_fly_ids)
                    removals_summary['rows_removed_by_status'] = None  # not computed in fast path
                    print(f" ✓ ({len(health_report_fast)} reports, {len(precomputed_removed_fly_ids)} flies to remove by status)")
        except Exception as e:
            print(f" ❌")
            print(f"  Warning: {e}. Falling back to full pipeline.")
            status_only = False

    if not status_only:
        # Load data from database
        print(f"[TMP] before load_readings_from_db (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
        print(f"\n📥 Loading data from database...")
        print(f"  Loading readings from database (this may take several minutes)...", end='', flush=True)
        df = step1.load_readings_from_db(experiment_id)
        print(f"[TMP] after load_readings_from_db (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
        if df is None or len(df) == 0:
            raise ValueError(f"No data found in database for experiment_id {experiment_id}")
        print(f" ✓ ({len(df):,} rows loaded)")

        # STEP 2: Load health report from database (when not status-only)
        health_report = None
        if statuses_to_remove:
            print(f"[TMP] before load_health_report_from_db (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
            print(f"\n📊 Loading health report...")
            print(f"  Loading from database...", end='', flush=True)
            try:
                health_report = step1.load_health_report_from_db(experiment_id)
                print(f"[TMP] after load_health_report_from_db (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
                if health_report is None or len(health_report) == 0:
                    print(f" ⚠️")
                    print("  Warning: No health report found in database. Skipping status-based removal.")
                    statuses_to_remove = None
                else:
                    print(f" ✓ ({len(health_report)} reports loaded)")
            except psycopg2.Error as e:
                print(f" ❌")
                print(f"  Warning: Database error loading health report: {e}. Skipping status-based removal.")
                statuses_to_remove = None
            except Exception as e:
                print(f" ❌")
                print(f"  Warning: Unexpected error loading health report: {e}. Skipping status-based removal.")
                statuses_to_remove = None

        if not any([flies_to_remove, per_fly_remove, genotypes_to_remove,
                    sexes_to_remove, treatments_to_remove, statuses_to_remove]):
            print(f"\n⚠️  No removal criteria specified. No flies will be removed.")

        print(f"\n🗑️  Removing flies based on criteria...")
        if statuses_to_remove:
            print(f"  - By status: {', '.join(statuses_to_remove)}")
        if flies_to_remove:
            print(f"  - Specific flies: {len(flies_to_remove)} flies")
        if genotypes_to_remove:
            print(f"  - By genotype: {', '.join(genotypes_to_remove)}")
        if sexes_to_remove:
            print(f"  - By sex: {', '.join(sexes_to_remove)}")
        if treatments_to_remove:
            print(f"  - By treatment: {', '.join(treatments_to_remove)}")
        if per_fly_remove:
            print(f"  - Per-fly days: {len(per_fly_remove)} flies")

        print(f"[TMP] before remove_flies() call (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
        print(f"\n  Processing removals...", end='', flush=True)
        df_cleaned, removals_summary, total_removed = remove_flies(
            df,
            flies_to_remove=flies_to_remove,
            per_fly_remove=per_fly_remove,
            genotypes_to_remove=genotypes_to_remove,
            sexes_to_remove=sexes_to_remove,
            treatments_to_remove=treatments_to_remove,
            statuses_to_remove=statuses_to_remove,
            health_report=health_report
        )
        print(f"[TMP] after remove_flies() call (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
        print(f" ✓")

    # Print summary
    print(f"\n📈 Removal Summary:")
    if df is not None and df_cleaned is not None:
        print(f"  Original rows: {len(df):,}")
        print(f"  Rows after removal: {len(df_cleaned):,}")
        print(f"  Total rows removed: {total_removed:,}")
    else:
        print(f"  (status-only path: row counts not loaded)")
    print(f"  Flies removed: {removals_summary['flies_removed']}")
    if removals_summary.get('rows_removed_by_status') and removals_summary['rows_removed_by_status'] > 0:
        print(f"  - By status: {removals_summary['rows_removed_by_status']:,} rows")
    if removals_summary.get('rows_removed_by_fly', 0) > 0:
        print(f"  - By fly ID: {removals_summary['rows_removed_by_fly']:,} rows")
    if removals_summary.get('rows_removed_by_days', 0) > 0:
        print(f"  - By specific days: {removals_summary['rows_removed_by_days']:,} rows")
    if removals_summary.get('rows_removed_by_metadata', 0) > 0:
        print(f"  - By metadata: {removals_summary['rows_removed_by_metadata']:,} rows")

    # ============================================================
    # STEP 5: Save cleaned data to database (remove deleted flies)
    # ============================================================
    if USE_DATABASE and DB_AVAILABLE and experiment_id:
        print(f"[TMP] before DB update section (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
        print(f"\n💾 Updating database...")
        try:
            engine = create_engine(DATABASE_URL)
            print(f"[TMP] after create_engine (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)

            # Get list of fly_ids that were removed (precomputed in status-only path)
            print(f"  Calculating removed fly IDs...", end='', flush=True)
            if precomputed_removed_fly_ids is not None:
                removed_fly_ids = precomputed_removed_fly_ids
            else:
                original_fly_ids = set(df['fly_id'].unique())
                cleaned_fly_ids = set(df_cleaned['fly_id'].unique())
                removed_fly_ids = original_fly_ids - cleaned_fly_ids
            print(f"[TMP] after calculating removed_fly_ids (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
            print(f" ✓ ({len(removed_fly_ids)} flies to remove from database)")

            # Delete in correct order to respect foreign key constraints:
            # 1. features_z (references flies)
            # 2. features (references flies)
            # 3. health_reports (references flies)
            # 4. readings (references flies)
            # 5. flies (last, after all references are removed)
            if removed_fly_ids:
                print(f"[TMP] removed_fly_ids non-empty, entering delete loop", flush=True)
                # Ensure index exists for fast FK check when deleting from flies (helps existing DBs)
                try:
                    with psycopg2.connect(**DB_CONFIG) as conn_ix:
                        with conn_ix.cursor() as cur_ix:
                            cur_ix.execute(
                                "CREATE INDEX IF NOT EXISTS idx_readings_experiment_fly ON readings(experiment_id, fly_id)"
                            )
                        conn_ix.commit()
                except psycopg2.Error:
                    pass  # index may already exist or table may be hypertable; continue
                print(f"  Deleting from database tables (this may take several minutes)...")

                # Convert set to list for batching
                removed_fly_ids_list = list(removed_fly_ids)
                batch_size = 50  # Process 50 flies at a time to avoid query planning issues
                total_flies = len(removed_fly_ids_list)
                
                # Track totals across all batches
                total_deleted = {
                    'features_z': 0,
                    'features': 0,
                    'health_reports': 0,
                    'readings': 0,
                    'flies': 0
                }
                
                print(f"[TMP] before opening DB connection for deletes (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
                with psycopg2.connect(**DB_CONFIG) as conn:
                    with conn.cursor() as cur:
                        # Process in batches
                        num_batches = (total_flies + batch_size - 1) // batch_size
                        print(f"[TMP] num_batches={num_batches} (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)

                        for batch_idx in range(num_batches):
                            _t_batch = time.perf_counter()
                            start_idx = batch_idx * batch_size
                            end_idx = min(start_idx + batch_size, total_flies)
                            batch_fly_ids = removed_fly_ids_list[start_idx:end_idx]
                            batch_num = batch_idx + 1
                            
                            placeholders = ','.join(['%s'] * len(batch_fly_ids))
                            
                            # Show batch progress
                            if num_batches > 1:
                                print(f"    Batch {batch_num}/{num_batches} ({len(batch_fly_ids)} flies)...")
                            
                            # Delete from features_z first (if it exists)
                            if batch_idx == 0:
                                print(f"    [1/5] Deleting from features_z...", end='', flush=True)
                            try:
                                _td = time.perf_counter()
                                cur.execute(
                                    f"DELETE FROM features_z WHERE experiment_id = %s AND fly_id IN ({placeholders})",
                                    [experiment_id] + batch_fly_ids
                                )
                                if batch_idx == 0:
                                    print(f"[TMP] after DELETE features_z (elapsed {time.perf_counter() - _td:.2f}s for this query)", flush=True)
                                total_deleted['features_z'] += cur.rowcount
                                if batch_idx == num_batches - 1:
                                    print(f" ✓ ({total_deleted['features_z']} rows)")
                            except psycopg2.ProgrammingError as e:
                                if batch_idx == 0:
                                    print(f" ⚠️  (table may not exist yet)")
                            except psycopg2.Error as e:
                                if batch_idx == 0:
                                    print(f" ❌")
                                    print(f"      Warning: Database error deleting from features_z: {e}")
                            
                            # Delete from features
                            if batch_idx == 0:
                                print(f"    [2/5] Deleting from features...", end='', flush=True)
                            try:
                                _td = time.perf_counter()
                                cur.execute(
                                    f"DELETE FROM features WHERE experiment_id = %s AND fly_id IN ({placeholders})",
                                    [experiment_id] + batch_fly_ids
                                )
                                if batch_idx == 0:
                                    print(f"[TMP] after DELETE features (elapsed {time.perf_counter() - _td:.2f}s for this query)", flush=True)
                                total_deleted['features'] += cur.rowcount
                                if batch_idx == num_batches - 1:
                                    print(f" ✓ ({total_deleted['features']} rows)")
                            except psycopg2.ProgrammingError as e:
                                if batch_idx == 0:
                                    print(f" ⚠️  (table may not exist yet)")
                            except psycopg2.Error as e:
                                if batch_idx == 0:
                                    print(f" ❌")
                                    print(f"      Warning: Database error deleting from features: {e}")
                            
                            # Delete from health_reports
                            if batch_idx == 0:
                                print(f"    [3/5] Deleting from health_reports...", end='', flush=True)
                            _td = time.perf_counter()
                            cur.execute(
                                f"DELETE FROM health_reports WHERE experiment_id = %s AND fly_id IN ({placeholders})",
                                [experiment_id] + batch_fly_ids
                            )
                            if batch_idx == 0:
                                print(f"[TMP] after DELETE health_reports (elapsed {time.perf_counter() - _td:.2f}s for this query)", flush=True)
                            total_deleted['health_reports'] += cur.rowcount
                            if batch_idx == num_batches - 1:
                                print(f" ✓ ({total_deleted['health_reports']} rows)")
                            
                            # Delete from readings
                            if batch_idx == 0:
                                print(f"    [4/5] Deleting from readings (this will take the longest)...", end='', flush=True)
                            _td = time.perf_counter()
                            cur.execute(
                                f"DELETE FROM readings WHERE experiment_id = %s AND fly_id IN ({placeholders})",
                                [experiment_id] + batch_fly_ids
                            )
                            if batch_idx == 0:
                                print(f"[TMP] after DELETE readings (elapsed {time.perf_counter() - _td:.2f}s for this query)", flush=True)
                            total_deleted['readings'] += cur.rowcount
                            if batch_idx == num_batches - 1:
                                print(f" ✓ ({total_deleted['readings']:,} rows)")
                            
                            # Delete from flies (last, after all references are removed)
                            if batch_idx == 0:
                                print(f"    [5/5] Deleting from flies...", end='', flush=True)
                            _td = time.perf_counter()
                            cur.execute(
                                f"DELETE FROM flies WHERE experiment_id = %s AND fly_id IN ({placeholders})",
                                [experiment_id] + batch_fly_ids
                            )
                            if batch_idx == 0:
                                print(f"[TMP] after DELETE flies (elapsed {time.perf_counter() - _td:.2f}s for this query)", flush=True)
                            total_deleted['flies'] += cur.rowcount
                            if batch_idx == num_batches - 1:
                                print(f" ✓ ({total_deleted['flies']} rows)")

                            if batch_idx == 0:
                                print(f"[TMP] end of batch 0 (batch elapsed {time.perf_counter() - _t_batch:.2f}s)", flush=True)
                            elif batch_idx < num_batches - 1 and batch_idx % 5 == 0:
                                print(f"[TMP] end of batch {batch_idx} (batch elapsed {time.perf_counter() - _t_batch:.2f}s)", flush=True)

                        print(f"[TMP] before commit (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
                        print(f"  Committing transaction...", end='', flush=True)
                        conn.commit()
                        print(f"[TMP] after commit (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
                        print(f" ✓")
            
            engine.dispose()
            print(f"[TMP] remove_flies_from_data() before return (DB path) (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
            print(f"\n✅ Successfully updated database")
        except psycopg2.Error as e:
            raise RuntimeError(f"Database error saving cleaned data to database: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error saving cleaned data to database: {e}")
    else:
        print(f"\n⚠️  Database not available or no flies to remove")

    print(f"[TMP] remove_flies_from_data() END return (elapsed {time.perf_counter() - _t0:.2f}s)", flush=True)
    return df_cleaned


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
    print(f"[TMP] main() START", flush=True)
    parser = argparse.ArgumentParser(
        description='Pipeline Step 2: Remove flies (optional)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove specific flies
  python 2-remove_flies.py --flies "6-ch23,6-ch5"
  
  # Remove flies by health status
  python 2-remove_flies.py --statuses "Dead,QC_Fail"
  
  # Remove specific days for flies (JSON format)
  python 2-remove_flies.py --per-fly-remove '{"5-ch7": [1, 2], "6-ch18": [3]}'
  
  # Remove by metadata
  python 2-remove_flies.py --genotypes "mutant1" --sexes "Male"
        """
    )
    
    parser.add_argument('--flies', type=str,
                       help='Comma-separated list of fly keys to remove (e.g., "6-ch23,6-ch5")')
    parser.add_argument('--per-fly-remove', type=str,
                       help='JSON dict mapping fly keys to days to remove (e.g., \'{"5-ch7": [1, 2]}\')')
    parser.add_argument('--genotypes', type=str,
                       help='Comma-separated list of genotypes to remove')
    parser.add_argument('--sexes', type=str,
                       help='Comma-separated list of sexes to remove')
    parser.add_argument('--treatments', type=str,
                       help='Comma-separated list of treatments to remove')
    parser.add_argument('--statuses', type=str,
                       help='Comma-separated list of health statuses to remove (e.g., "Dead,QC_Fail")')
    parser.add_argument('--experiment-id', type=int, default=None,
                       help='Experiment ID to use (default: latest experiment)')
    
    args = parser.parse_args()
    
    # Parse arguments
    flies_to_remove = None
    if args.flies:
        flies_to_remove = [f.strip() for f in args.flies.split(',')]
    
    per_fly_remove = None
    if args.per_fly_remove:
        try:
            per_fly_remove = json.loads(args.per_fly_remove)
        except json.JSONDecodeError as e:
            sys.exit(1)
    
    genotypes_to_remove = None
    if args.genotypes:
        genotypes_to_remove = [g.strip() for g in args.genotypes.split(',')]
    
    sexes_to_remove = None
    if args.sexes:
        sexes_to_remove = [s.strip() for s in args.sexes.split(',')]
    
    treatments_to_remove = None
    if args.treatments:
        treatments_to_remove = [t.strip() for t in args.treatments.split(',')]
    
    statuses_to_remove = None
    if args.statuses:
        statuses_to_remove = [s.strip() for s in args.statuses.split(',')]
    
    # Determine if any command-line arguments were provided
    # If no args provided, use config values from the top of the script
    has_cli_args = any([
        args.flies, args.per_fly_remove, args.genotypes, args.sexes,
        args.treatments, args.statuses, args.experiment_id
    ])
    print(f"[TMP] main() before remove_flies_from_data()", flush=True)

    remove_flies_from_data(
        flies_to_remove=flies_to_remove,
        per_fly_remove=per_fly_remove,
        genotypes_to_remove=genotypes_to_remove,
        sexes_to_remove=sexes_to_remove,
        treatments_to_remove=treatments_to_remove,
        statuses_to_remove=statuses_to_remove,
        use_config=not has_cli_args,  # Use config if no CLI args provided
        experiment_id=args.experiment_id
    )
    print(f"[TMP] main() after remove_flies_from_data() (done)", flush=True)


if __name__ == '__main__':
    main()

