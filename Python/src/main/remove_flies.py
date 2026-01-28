#!/usr/bin/env python3
"""
Remove Specific Flies or Days from DAM Data

This script removes flies from the dataset based on:
1. Specific fly IDs (Monitor-Channel combinations)
2. Specific days for certain flies
3. Metadata criteria (genotype, sex, treatment)

Usage:
    python remove_flies.py [input_file] [output_file] [options]
    
Examples:
    # Remove specific flies
    python remove_flies.py --flies "6-ch23,6-ch5"
    
    # Remove specific days for flies
    python remove_flies.py --per-fly-remove '{"5-ch7": [1, 2], "6-ch18": [3]}'
    
    # Remove by metadata
    python remove_flies.py --genotypes "mutant1,mutant2" --sexes "Male"
    
    # All options
    python remove_flies.py input.csv output.csv --flies "6-ch23" --genotypes "bad_genotype"
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path


def create_fly_key(monitor, channel):
    """
    Create a fly key in the format "Monitor-Channel" (lowercase).
    
    Args:
        monitor: Monitor number
        channel: Channel number or string
        
    Returns:
        String key like "5-ch7"
    """
    # Convert channel to string and remove "ch" prefix if present
    channel_str = str(channel).lower()
    if channel_str.startswith('ch'):
        channel_str = channel_str[2:]
    
    return f"{monitor}-ch{channel_str}"


def is_ok(x):
    """
    Check if a value is valid (not NA, not empty, not "na").
    """
    if pd.isna(x):
        return False
    x_str = str(x).lower()
    return x_str != "" and x_str != "na"


def remove_flies(df, flies_to_remove=None, per_fly_remove=None, 
                 genotypes_to_remove=None, sexes_to_remove=None, 
                 treatments_to_remove=None):
    """
    Remove flies from dataset based on various criteria.
    
    Args:
        df: Input DataFrame
        flies_to_remove: List of fly keys (e.g., ["6-ch23", "6-ch5"])
        per_fly_remove: Dict mapping fly keys to lists of days to remove
                       (e.g., {"5-ch7": [1, 2], "6-ch18": [3]})
        genotypes_to_remove: List of genotypes to remove
        sexes_to_remove: List of sexes to remove
        treatments_to_remove: List of treatments to remove
        
    Returns:
        Cleaned DataFrame and summary statistics
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
        'rows_removed_by_metadata': 0
    }
    
    # 1. Remove specific flies completely
    if flies_to_remove:
        flies_to_remove_lower = [f.lower() for f in flies_to_remove]
        before = len(df)
        df = df[~df['fly_key'].str.lower().isin(flies_to_remove_lower)].copy()
        after = len(df)
        removals_summary['flies_removed'] = len(flies_to_remove)
        removals_summary['rows_removed_by_fly'] = before - after
        print(f"✓ Removed {removals_summary['flies_removed']} flies: {flies_to_remove}")
        print(f"  → {removals_summary['rows_removed_by_fly']:,} rows removed")
    
    # 2. Remove specific days for certain flies
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
    
    # 3. Remove by metadata
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


def print_summary(df, removals_summary, total_removed):
    """
    Print summary of remaining flies.
    
    Args:
        df: Cleaned DataFrame
        removals_summary: Dictionary with removal statistics
        total_removed: Total number of rows removed
    """
    print("\n" + "=" * 60)
    print("REMOVAL SUMMARY")
    print("=" * 60)
    print(f"Flies removed completely: {removals_summary['flies_removed']}")
    print(f"Rows removed by fly removal: {removals_summary['rows_removed_by_fly']:,}")
    print(f"Rows removed by day removal: {removals_summary['rows_removed_by_days']:,}")
    print(f"Rows removed by metadata: {removals_summary['rows_removed_by_metadata']:,}")
    print(f"Total rows removed: {total_removed:,}")
    print()
    
    # Get remaining flies
    if 'monitor' in df.columns and 'channel' in df.columns:
        remaining_flies = df[['monitor', 'channel']].drop_duplicates().sort_values(['monitor', 'channel'])
        print(f"Remaining flies: {len(remaining_flies)}")
        print()
        print("Remaining flies (Monitor-Channel):")
        for _, row in remaining_flies.head(20).iterrows():
            fly_key = create_fly_key(row['monitor'], row['channel'])
            print(f"  {fly_key}")
        if len(remaining_flies) > 20:
            print(f"  ... and {len(remaining_flies) - 20} more")
    print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Remove specific flies or days from DAM data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove specific flies
  python remove_flies.py --flies "6-ch23,6-ch5"
  
  # Remove specific days for flies (JSON format)
  python remove_flies.py --per-fly-remove '{"5-ch7": [1, 2], "6-ch18": [3]}'
  
  # Remove by metadata
  python remove_flies.py --genotypes "mutant1" --sexes "Male"
  
  # All options
  python remove_flies.py input.csv output.csv --flies "6-ch23" --genotypes "bad_genotype"
        """
    )
    
    parser.add_argument('input_file', nargs='?', 
                       default='../../data/processed/dam_data_with_flies.csv',
                       help='Input CSV file (default: ../../data/processed/dam_data_with_flies.csv)')
    parser.add_argument('output_file', nargs='?',
                       default='../../data/processed/dam_data_cleaned.csv',
                       help='Output CSV file (default: ../../data/processed/dam_data_cleaned.csv)')
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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Remove Flies from DAM Data")
    print("=" * 60)
    print(f"\n📊 Input file: {args.input_file}")
    print(f"📄 Output file: {args.output_file}")
    print()
    
    # Read data
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)
    
    print("Reading data...")
    df = pd.read_csv(args.input_file)
    print(f"✓ Loaded {len(df):,} rows")
    
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
    
    # Check if any removal criteria specified
    if not any([flies_to_remove, per_fly_remove, genotypes_to_remove, 
                sexes_to_remove, treatments_to_remove]):
        print("\nWARNING: No removal criteria specified. Nothing will be removed.")
        print("Use --help to see available options.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Perform removals
    print("\nRemoving flies...")
    df_cleaned, removals_summary, total_removed = remove_flies(
        df,
        flies_to_remove=flies_to_remove,
        per_fly_remove=per_fly_remove,
        genotypes_to_remove=genotypes_to_remove,
        sexes_to_remove=sexes_to_remove,
        treatments_to_remove=treatments_to_remove
    )
    
    # Print summary
    print_summary(df_cleaned, removals_summary, total_removed)
    
    # Save output
    print(f"\n💾 Saving cleaned data to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_cleaned.to_csv(args.output_file, index=False)
    print(f"✓ Saved {len(df_cleaned):,} rows successfully")
    print()


if __name__ == '__main__':
    main()

