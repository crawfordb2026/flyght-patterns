#!/usr/bin/env python3
"""
Pipeline Step 2: Remove Flies (Optional)

This script:
1. Reads prepared data from Step 1 (dam_data_prepared.csv)
2. Optionally reads health report from Step 1 (health_report.csv)
3. Removes flies based on various criteria:
   - Specific fly IDs
   - Specific days for certain flies
   - Metadata criteria (genotype, sex, treatment)
   - Health status from health report
4. Saves cleaned data

Output: dam_data_cleaned.csv

This step is OPTIONAL. Run it only if you want to remove flies based on
the health report or other criteria. If you skip this step, use the
prepared data directly in Step 3.


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
from pathlib import Path


# ============================================================
#   USER CONFIGURATION
# ============================================================

# Default file paths
DEFAULT_INPUT = '../../data/processed/dam_data_prepared.csv'
DEFAULT_HEALTH_REPORT = '../../data/processed/health_report.csv'
DEFAULT_OUTPUT = '../../data/processed/dam_data_cleaned.csv'

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
    df = df.copy()
    original_count = len(df)
    
    # Create fly key column
    df['fly_key'] = df.apply(
        lambda row: create_fly_key(row['monitor'], row['channel']), axis=1
    )
    
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
    if statuses_to_remove and health_report is not None:
        before = len(df)
        # Get flies with specified statuses
        flies_to_remove_by_status = health_report[
            health_report['FINAL_STATUS'].isin(statuses_to_remove)
        ][['Monitor', 'Channel']].drop_duplicates()
        
        if len(flies_to_remove_by_status) > 0:
            # Create fly keys
            flies_to_remove_by_status['fly_key'] = flies_to_remove_by_status.apply(
                lambda row: create_fly_key(row['Monitor'], row['Channel']), axis=1
            )
            fly_keys_to_remove = flies_to_remove_by_status['fly_key'].str.lower().tolist()
            
            df = df[~df['fly_key'].str.lower().isin(fly_keys_to_remove)].copy()
            after = len(df)
            removals_summary['rows_removed_by_status'] = before - after
            removals_summary['flies_removed'] = len(fly_keys_to_remove)
            print(f"✓ Removed {len(fly_keys_to_remove)} flies by status {statuses_to_remove}")
            print(f"  → {removals_summary['rows_removed_by_status']:,} rows removed")
    
    # 2. Remove specific flies completely
    if flies_to_remove:
        flies_to_remove_lower = [f.lower() for f in flies_to_remove]
        before = len(df)
        df = df[~df['fly_key'].str.lower().isin(flies_to_remove_lower)].copy()
        after = len(df)
        removals_summary['flies_removed'] += len(flies_to_remove)
        removals_summary['rows_removed_by_fly'] = before - after
        print(f"✓ Removed {len(flies_to_remove)} flies: {flies_to_remove}")
        print(f"  → {removals_summary['rows_removed_by_fly']:,} rows removed")
    
    # 3. Remove specific days for certain flies
    if per_fly_remove:
        if 'Exp_Day' not in df.columns:
            print("WARNING: Exp_Day column not found. Cannot remove specific days.")
            print("  Skipping per-fly day removal.")
        else:
            before = len(df)
            for fly_key, days in per_fly_remove.items():
                fly_key_lower = fly_key.lower()
                mask = (
                    (df['fly_key'].str.lower() == fly_key_lower) &
                    (df['Exp_Day'].isin(days))
                )
                df = df[~mask].copy()
            after = len(df)
            removals_summary['rows_removed_by_days'] = before - after
            print(f"✓ Removed specific days for {len(per_fly_remove)} flies")
            print(f"  → {removals_summary['rows_removed_by_days']:,} rows removed")
            for fly_key, days in per_fly_remove.items():
                print(f"    {fly_key}: days {days}")
    
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
    
    if removals_summary['rows_removed_by_metadata'] > 0:
        print(f"✓ Removed flies by metadata")
        if genotypes_to_remove:
            print(f"  → Genotypes: {genotypes_to_remove}")
        if sexes_to_remove:
            print(f"  → Sexes: {sexes_to_remove}")
        if treatments_to_remove:
            print(f"  → Treatments: {treatments_to_remove}")
        print(f"  → {removals_summary['rows_removed_by_metadata']:,} rows removed")
    
    # Remove helper columns
    columns_to_drop = ['fly_key', 'genotype_lower', 'sex_lower', 'treatment_lower']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    final_count = len(df)
    total_removed = original_count - final_count
    
    return df, removals_summary, total_removed


# ============================================================
#   MAIN WORKFLOW
# ============================================================

def remove_flies_from_data(
    input_file=None,
    output_file=None,
    health_report_file=None,
    flies_to_remove=None,
    per_fly_remove=None,
    genotypes_to_remove=None,
    sexes_to_remove=None,
    treatments_to_remove=None,
    statuses_to_remove=None,
    use_config=True
):
    """
    Main function to remove flies from prepared data.
    
    Args:
        input_file: Path to prepared data from Step 1
        output_file: Path to save cleaned data
        health_report_file: Path to health report from Step 1 (optional)
        flies_to_remove: List of fly keys to remove
        per_fly_remove: Dict mapping fly keys to days to remove
        genotypes_to_remove: List of genotypes to remove
        sexes_to_remove: List of sexes to remove
        treatments_to_remove: List of treatments to remove
        statuses_to_remove: List of health statuses to remove
        use_config: Whether to use hardcoded config values
        
    Returns:
        Cleaned DataFrame
    """
    print("=" * 60)
    print("PIPELINE STEP 2: REMOVE FLIES (OPTIONAL)")
    print("=" * 60)
    
    # Set defaults
    if input_file is None:
        input_file = DEFAULT_INPUT
    if output_file is None:
        output_file = DEFAULT_OUTPUT
    if health_report_file is None:
        health_report_file = DEFAULT_HEALTH_REPORT
    
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
    
    # Convert relative paths to absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, input_file) if not os.path.isabs(input_file) else input_file
    output_file = os.path.join(script_dir, output_file) if not os.path.isabs(output_file) else output_file
    health_report_file = os.path.join(script_dir, health_report_file) if not os.path.isabs(health_report_file) else health_report_file
    
    # ============================================================
    # STEP 1: Load data
    # ============================================================
    print(f"\n[Step 2.1] Loading prepared data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run Step 1 first: python 1-prepare_data_and_health.py")
        sys.exit(1)
    
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} rows")
    print(f"   Unique flies: {df.groupby(['monitor', 'channel']).ngroups}")
    
    # ============================================================
    # STEP 2: Load health report (if needed)
    # ============================================================
    health_report = None
    if statuses_to_remove:
        print(f"\n[Step 2.2] Loading health report from {health_report_file}...")
        
        if not os.path.exists(health_report_file):
            print(f"WARNING: Health report not found: {health_report_file}")
            print("  Status-based removal will be skipped.")
            print("  Please run Step 1 first: python 1-prepare_data_and_health.py")
            statuses_to_remove = None
        else:
            health_report = pd.read_csv(health_report_file)
            print(f"✓ Loaded health report for {len(health_report)} flies")
    
    # ============================================================
    # STEP 3: Check if any removal criteria specified
    # ============================================================
    if not any([flies_to_remove, per_fly_remove, genotypes_to_remove, 
                sexes_to_remove, treatments_to_remove, statuses_to_remove]):
        print("\nWARNING: No removal criteria specified.")
        print("  Nothing will be removed. The output will be identical to input.")
        print("  Use --help to see available options.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # ============================================================
    # STEP 4: Remove flies
    # ============================================================
    print("\n[Step 2.3] Removing flies...")
    
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
    
    # ============================================================
    # STEP 5: Print summary
    # ============================================================
    print("\n" + "=" * 60)
    print("REMOVAL SUMMARY")
    print("=" * 60)
    print(f"Flies removed completely: {removals_summary['flies_removed']}")
    print(f"Rows removed by fly removal: {removals_summary['rows_removed_by_fly']:,}")
    print(f"Rows removed by day removal: {removals_summary['rows_removed_by_days']:,}")
    print(f"Rows removed by metadata: {removals_summary['rows_removed_by_metadata']:,}")
    print(f"Rows removed by status: {removals_summary['rows_removed_by_status']:,}")
    print(f"Total rows removed: {total_removed:,}")
    print()
    
    remaining_flies = df_cleaned.groupby(['monitor', 'channel']).ngroups
    print(f"Remaining flies: {remaining_flies}")
    print(f"Remaining rows: {len(df_cleaned):,}")
    
    # ============================================================
    # STEP 5: Save output
    # ============================================================
    print(f"\n[Step 2.4] Saving cleaned data to {output_file}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_cleaned.to_csv(output_file, index=False)
    print(f"✓ Saved cleaned data successfully")
    
    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review the cleaned data: {output_file}")
    print(f"  2. Run: python 3-create_feature_table.py")
    
    return df_cleaned


# ============================================================
#   COMMAND-LINE INTERFACE
# ============================================================

def main():
    """Main function with command-line argument parsing."""
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
    
    parser.add_argument('--input', type=str, default=None,
                       help='Input file from Step 1 (default: dam_data_prepared.csv)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output cleaned data file (default: dam_data_cleaned.csv)')
    parser.add_argument('--health-report', type=str, default=None,
                       help='Health report file from Step 1 (default: health_report.csv)')
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
            print(f"ERROR: Invalid JSON for --per-fly-remove: {e}")
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
        args.treatments, args.statuses, args.input, args.output, args.health_report
    ])
    
    remove_flies_from_data(
        input_file=args.input,
        output_file=args.output,
        health_report_file=args.health_report,
        flies_to_remove=flies_to_remove,
        per_fly_remove=per_fly_remove,
        genotypes_to_remove=genotypes_to_remove,
        sexes_to_remove=sexes_to_remove,
        treatments_to_remove=treatments_to_remove,
        statuses_to_remove=statuses_to_remove,
        use_config=not has_cli_args  # Use config if no CLI args provided
    )


if __name__ == '__main__':
    main()

