#!/usr/bin/env python3
"""
Sex Difference Analysis Pipeline
=================================
Analyzes sex differences within clusters and genotypes

This script performs:
1. Counts of Genotype × Sex per Cluster
2. Sex distribution tests (Fisher test) within each genotype
3. Genotype-specific sex comparison heatmaps

Usage:
    python sexdiff_analysis.py [--experiment-id ID] [--umap-clusters umap_clusters.csv] [--genotype Rye] [--output-dir analysis_results/sexdiff]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import fisher_exact
from importlib import import_module
import warnings
warnings.filterwarnings('ignore')

try:
    from config import DB_CONFIG, DATABASE_URL, USE_DATABASE
    from sqlalchemy import create_engine
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    USE_DATABASE = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Top 10 features (matching UMAP script)
# Note: Using lowercase _z suffix to match database convention
TOP_FEATURES = [
    "night_sleep_mean_z",
    "total_sleep_mean_z",
    "frag_bouts_per_min_sleep_mean_z",
    "max_bout_mean_z",
    "p_wake_mean_z",
    "mean_bout_mean_z",
    "max_night_bout_mean_z",
    "mean_night_bout_mean_z",
    "sleep_latency_mean_z",
    "mean_wake_bout_mean_z"
]

# Default genotype for sex comparison (can be overridden via command line)
DEFAULT_GENOTYPE = "Rye"


def load_umap_clusters(umap_clusters_file):
    """Load UMAP clusters data."""
    print(f"[Loading] Reading {umap_clusters_file}...")
    umap_df = pd.read_csv(umap_clusters_file)
    # Normalize column names to lowercase for consistency with database
    umap_df.columns = [col.lower() if isinstance(col, str) else col for col in umap_df.columns]
    print(f"✓ Loaded {len(umap_df)} flies with cluster assignments")
    print(f"  Clusters: {sorted([c for c in umap_df['cluster'].unique() if c != -1])}")
    print(f"  Noise points: {len(umap_df[umap_df['cluster'] == -1])}")
    return umap_df


def load_feature_data_from_db(experiment_id=None):
    """Load ML features Z-scored data from database."""
    if not USE_DATABASE or not DB_AVAILABLE:
        raise RuntimeError("Database is required. Please ensure database is configured and available.")
    
    # Import database functions from step 1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(script_dir))
    step1 = import_module('1-prepare_data_and_health')
    
    # Use provided experiment_id, or get latest if not provided
    if experiment_id is None:
        experiment_id = step1.get_latest_experiment_id()
        if experiment_id is None:
            raise ValueError("No experiment found in database. Please specify --experiment-id")
        print(f"[Loading] Using latest experiment_id: {experiment_id}")
    else:
        print(f"[Loading] Loading experiment_id: {experiment_id}")
    
    # Load z-scored features from database
    engine = create_engine(DATABASE_URL)
    query = f"""
        SELECT 
            fz.*,
            fl.genotype,
            fl.sex,
            fl.treatment,
            fl.monitor,
            fl.channel
        FROM features_z fz
        JOIN flies fl ON fz.fly_id = fl.fly_id AND fz.experiment_id = fl.experiment_id
        WHERE fz.experiment_id = {experiment_id}
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    if df is None or len(df) == 0:
        raise ValueError(f"No z-scored features found in database for experiment_id {experiment_id}")
    
    # Normalize column names to lowercase
    df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
    
    # Remove feature_id if present
    if 'feature_id' in df.columns:
        df = df.drop(columns=['feature_id'])
    
    print(f"✓ Loaded {len(df)} flies")
    return df


def subset_vehicle(df):
    """Filter to vehicle treatment only."""
    print("\n[Subset] Filtering to vehicle (VEH) treatment...")
    df_veh = df[df['treatment'].str.upper() == 'VEH'].copy()
    print(f"✓ {len(df_veh)} flies in vehicle group")
    return df_veh


def cluster_genotype_sex_counts(umap_df, output_dir):
    """Analyze counts of Genotype × Sex per Cluster."""
    print("\n" + "="*60)
    print("STEP 1: CLUSTER × GENOTYPE × SEX COUNTS")
    print("="*60)
    
    # Basic counts
    cluster_geno_sex_counts = umap_df.groupby(['cluster', 'Genotype', 'Sex']).size().reset_index(name='count')
    cluster_geno_sex_counts = cluster_geno_sex_counts.sort_values(['cluster', 'Genotype', 'Sex'])
    
    print("\n[Counts] Cluster × Genotype × Sex:")
    print(cluster_geno_sex_counts.to_string(index=False))
    
    # Percentages within each cluster
    cluster_geno_sex_percent = cluster_geno_sex_counts.groupby('cluster').apply(
        lambda x: x.assign(percent=round(100 * x['count'] / x['count'].sum(), 1))
    ).reset_index(drop=True)
    
    print("\n[Percentages] Within each cluster:")
    print(cluster_geno_sex_percent.to_string(index=False))
    
    # Wide table format
    cluster_geno_sex_wide = cluster_geno_sex_counts.copy()
    cluster_geno_sex_wide['Genotype_Sex'] = cluster_geno_sex_wide['genotype'] + '_' + cluster_geno_sex_wide['sex']
    cluster_geno_sex_wide = cluster_geno_sex_wide.pivot_table(
        index='cluster',
        columns='Genotype_Sex',
        values='count',
        fill_value=0
    ).reset_index()
    
    print("\n[Wide Table] Counts by cluster:")
    print(cluster_geno_sex_wide.to_string(index=False))
    
    # Visualization
    print("\n[Plot] Creating cluster × genotype × sex bar chart...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out noise cluster for cleaner visualization
    plot_data = cluster_geno_sex_counts[cluster_geno_sex_counts['cluster'] != -1].copy()
    
    if len(plot_data) > 0:
        # Create grouped bar chart
        clusters = sorted(plot_data['cluster'].unique())
        genotypes = sorted(plot_data['genotype'].unique())
        sexes = sorted(plot_data['sex'].unique())
        
        x = np.arange(len(clusters))
        width = 0.8 / (len(genotypes) * len(sexes))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(genotypes)))
        color_map = dict(zip(genotypes, colors))
        
        offset = -0.4 + width/2
        for genotype in genotypes:
            for sex in sexes:
                subset = plot_data[(plot_data['genotype'] == genotype) & (plot_data['sex'] == sex)]
                if len(subset) > 0:
                    counts = [subset[subset['cluster'] == c]['count'].sum() if len(subset[subset['cluster'] == c]) > 0 else 0 
                             for c in clusters]
                    ax.bar(x + offset, counts, width, label=f'{genotype} {sex}', 
                          color=color_map[genotype], alpha=0.8)
                    offset += width
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Number of Flies', fontsize=12)
        ax.set_title('Counts per Cluster by Genotype and Sex', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Cluster {c}' for c in clusters])
        ax.legend(title='Genotype × Sex', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
    
    path = os.path.join(output_dir, 'cluster_genotype_sex_counts.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()
    
    # Save tables
    cluster_geno_sex_counts.to_csv(
        os.path.join(output_dir, 'cluster_genotype_sex_counts.csv'),
        index=False
    )
    cluster_geno_sex_percent.to_csv(
        os.path.join(output_dir, 'cluster_genotype_sex_percentages.csv'),
        index=False
    )
    cluster_geno_sex_wide.to_csv(
        os.path.join(output_dir, 'cluster_genotype_sex_wide.csv'),
        index=False
    )
    
    return cluster_geno_sex_counts, cluster_geno_sex_percent


def sex_distribution_tests(umap_df, output_dir):
    """Test sex distribution within each genotype across clusters using Fisher's exact test."""
    print("\n" + "="*60)
    print("STEP 2: SEX DISTRIBUTION TESTS")
    print("="*60)
    
    # Create contingency table per genotype
    sex_table_by_geno = umap_df.groupby(['Genotype', 'Sex', 'cluster']).size().reset_index(name='n')
    
    print("\n[Testing] Fisher's exact test for sex distribution within each genotype...")
    
    genotypes = sorted(umap_df['genotype'].unique())
    sex_cluster_tests = []
    
    for genotype in genotypes:
        geno_data = sex_table_by_geno[sex_table_by_geno['genotype'] == genotype]
        
        if len(geno_data) == 0:
            continue
        
        # Create contingency table: Sex × Cluster
        contingency = geno_data.pivot_table(
            index='Sex',
            columns='cluster',
            values='n',
            fill_value=0
        )
        
        # Check if we have at least 2 sexes and 2 clusters (excluding noise)
        valid_clusters = [c for c in contingency.columns if c != -1]
        if len(contingency.index) < 2 or len(valid_clusters) < 2:
            sex_cluster_tests.append({
                'Genotype': genotype,
                'p_value': np.nan,
                'note': 'Insufficient data (need ≥2 sexes and ≥2 clusters)'
            })
            continue
        
        # Filter to valid clusters only
        contingency_clean = contingency[valid_clusters]
        
        # Fisher's exact test
        # Note: fisher_exact works on 2x2 tables, so we need to handle larger tables
        # For larger tables, we'll use a chi-square approximation or test each pair
        if contingency_clean.shape == (2, 2):
            # Perfect 2x2 case
            oddsratio, p_value = fisher_exact(contingency_clean.values)
        else:
            # For larger tables, use chi-square test (Fisher's exact for >2x2 is computationally expensive)
            from scipy.stats import chi2_contingency
            chi2, p_value, dof, expected = chi2_contingency(contingency_clean.values)
            # Note: This is an approximation; true Fisher's exact for >2x2 requires different methods
        
        sex_cluster_tests.append({
            'Genotype': genotype,
            'p_value': p_value
        })
    
    sex_cluster_tests_df = pd.DataFrame(sex_cluster_tests)
    print(sex_cluster_tests_df.to_string(index=False))
    
    # Save results
    sex_cluster_tests_df.to_csv(
        os.path.join(output_dir, 'sex_distribution_tests.csv'),
        index=False
    )
    
    return sex_cluster_tests_df


def genotype_sex_comparison(df_veh, genotype, output_dir):
    """Compare sexes within a specific genotype using heatmap."""
    print("\n" + "="*60)
    print(f"STEP 3: SEX COMPARISON FOR {genotype.upper()} GENOTYPE")
    print("="*60)
    
    # Filter to specified genotype
    geno_df = df_veh[df_veh['genotype'].str.upper() == genotype.upper()].copy()
    
    if len(geno_df) == 0:
        print(f"\n⚠ Warning: No flies found for genotype '{genotype}'")
        print(f"   Available genotypes: {sorted(df_veh['genotype'].unique())}")
        return None
    
    sexes = sorted(geno_df['sex'].unique())
    if len(sexes) < 2:
        print(f"\n⚠ Warning: Only one sex found for {genotype}: {sexes}")
        print("   Cannot compare sexes. Skipping heatmap.")
        return None
    
    print(f"\n[Analysis] Comparing sexes for {genotype}:")
    print(f"  Sexes: {sexes}")
    print(f"  Flies: {len(geno_df)} total")
    for sex in sexes:
        count = len(geno_df[geno_df['sex'] == sex])
        print(f"    {sex}: {count}")
    
    # Compute medians per sex per feature
    available_features = [f for f in TOP_FEATURES if f in geno_df.columns]
    
    if len(available_features) == 0:
        print(f"\n⚠ Warning: No matching features found in data")
        return None
    
    print(f"\n[Summary] Computing medians per sex for {len(available_features)} features...")
    
    geno_summary = geno_df.melt(
        id_vars=['sex'],
        value_vars=available_features,
        var_name='feature',
        value_name='value'
    ).groupby(['Sex', 'feature'])['value'].median().reset_index()
    
    # Create matrix for heatmap
    geno_mat = geno_summary.pivot_table(
        index='Sex',
        columns='feature',
        values='value'
    )
    
    print("\n[Plot] Creating sex comparison heatmap...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use row scaling to compare sexes within each feature
    sns.heatmap(
        geno_mat,
        cmap='RdBu_r',
        center=0,
        robust=True,
        square=False,
        linewidths=0.5,
        cbar_kws={'label': 'Median Z-Score (row-scaled)'},
        ax=ax,
        annot=False
    )
    
    ax.set_title(f'Behavioral Signatures: {genotype} Males vs Females (Median Z Scores)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Sex', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    path = os.path.join(output_dir, f'{genotype}_sex_comparison_heatmap.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()
    
    # Save summary
    geno_summary.to_csv(
        os.path.join(output_dir, f'{genotype}_sex_summary.csv'),
        index=False
    )
    
    # Also save the matrix
    geno_mat.to_csv(
        os.path.join(output_dir, f'{genotype}_sex_matrix.csv')
    )
    
    return geno_summary, geno_mat


def main():
    parser = argparse.ArgumentParser(
        description='Sex Difference Analysis: Cluster and genotype sex comparisons',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sexdiff_analysis.py
  python sexdiff_analysis.py --experiment-id 1 --genotype Rye
  python sexdiff_analysis.py --experiment-id 1 --umap-clusters analysis_results/umap/umap_clusters.csv --genotype Fmn
  python sexdiff_analysis.py --experiment-id 1 --genotype All  # Run for all genotypes with both sexes
        """
    )
    
    parser.add_argument(
        '--experiment-id',
        type=int,
        default=None,
        help='Experiment ID to use (default: latest experiment)'
    )
    
    parser.add_argument(
        '--umap-clusters',
        type=str,
        default=None,
        help='UMAP clusters CSV file (default: auto-detect in analysis_results/umap/)'
    )
    
    parser.add_argument(
        '--genotype',
        type=str,
        default=None,
        help=f'Genotype for sex comparison (default: {DEFAULT_GENOTYPE}, or "All" for all genotypes)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots and tables (default: analysis_results/sexdiff)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect UMAP clusters file if not provided
    if args.umap_clusters is None:
        # Look for umap_clusters.csv in analysis_results/umap within db-pipeline/analysis/ folder
        script_dir = Path(__file__).parent
        default_umap = script_dir / 'analysis_results' / 'umap' / 'umap_clusters.csv'
        if default_umap.exists():
            args.umap_clusters = str(default_umap)
        else:
            print("Error: Could not auto-detect umap_clusters.csv")
            print(f"  Expected at: {default_umap}")
            print("\nPlease specify --umap-clusters path")
            sys.exit(1)
    
    # Set genotype (use default if not specified)
    if args.genotype is None:
        args.genotype = DEFAULT_GENOTYPE
    
    # Set output directory
    if args.output_dir is None:
        # Use analysis_results/sexdiff within db-pipeline/analysis/ folder
        script_dir = Path(__file__).parent
        args.output_dir = str(script_dir / 'analysis_results' / 'sexdiff')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("SEX DIFFERENCE ANALYSIS PIPELINE")
    print("Cluster and Genotype Sex Comparisons")
    print("="*60)
    
    # Load data
    umap_df = load_umap_clusters(args.umap_clusters)
    df = load_feature_data_from_db(experiment_id=args.experiment_id)
    df_veh = subset_vehicle(df)
    
    if len(df_veh) == 0:
        print("\n❌ Error: No vehicle flies found in dataset!")
        print("   Make sure Treatment column contains 'VEH' values")
        sys.exit(1)
    
    # Run analyses
    cluster_geno_sex_counts, cluster_geno_sex_percent = cluster_genotype_sex_counts(umap_df, args.output_dir)
    sex_cluster_tests_df = sex_distribution_tests(umap_df, args.output_dir)
    
    # Genotype-specific sex comparison
    if args.genotype.upper() == 'ALL':
        print("\n[Genotype Comparison] Running sex comparison for all genotypes with both sexes...")
        genotypes_with_both_sexes = []
        for genotype in sorted(df_veh['genotype'].unique()):
            geno_data = df_veh[df_veh['genotype'] == genotype]
            sexes = sorted(geno_data['sex'].unique())
            if len(sexes) >= 2:
                genotypes_with_both_sexes.append(genotype)
        
        print(f"  Found {len(genotypes_with_both_sexes)} genotypes with both sexes: {genotypes_with_both_sexes}")
        
        for genotype in genotypes_with_both_sexes:
            genotype_sex_comparison(df_veh, genotype, args.output_dir)
    else:
        genotype_sex_comparison(df_veh, args.genotype, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n✓ All outputs saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - cluster_genotype_sex_counts.csv")
    print("  - cluster_genotype_sex_percentages.csv")
    print("  - cluster_genotype_sex_wide.csv")
    print("  - cluster_genotype_sex_counts.png")
    print("  - sex_distribution_tests.csv")
    if args.genotype.upper() != 'ALL':
        print(f"  - {args.genotype}_sex_comparison_heatmap.png")
        print(f"  - {args.genotype}_sex_summary.csv")
        print(f"  - {args.genotype}_sex_matrix.csv")
    else:
        print("  - [Genotype]_sex_comparison_heatmap.png (for each genotype)")
        print("  - [Genotype]_sex_summary.csv (for each genotype)")
        print("  - [Genotype]_sex_matrix.csv (for each genotype)")
    print()


if __name__ == '__main__':
    main()

