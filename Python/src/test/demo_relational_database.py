#!/usr/bin/env python3
"""
Demonstration of the Relational Database Design

This script shows how to use the two-table relational structure for analysis.
The key benefit is that fly metadata (genotype, sex, treatment) is stored only
once per fly, not millions of times in the time-series data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_database():
    """Load both tables from the relational database."""
    print("📊 Loading relational database...")
    
    # Load the two separate tables
    time_series = pd.read_csv('../../data/processed/time_series_data.csv')
    metadata = pd.read_csv('../../data/processed/fly_metadata.csv')
    
    print(f"✅ Loaded {len(time_series):,} time-series measurements")
    print(f"✅ Loaded {len(metadata)} fly metadata records")
    
    return time_series, metadata

def demonstrate_join():
    """Demonstrate how the tables connect via JOIN operation."""
    print("\n🔗 DEMONSTRATING RELATIONAL JOIN")
    print("=" * 50)
    
    time_series, metadata = load_database()
    
    # Join the tables on (monitor, channel) foreign key
    full_data = time_series.merge(metadata, on=['monitor', 'channel'])
    
    print(f"Joined data: {len(full_data):,} rows")
    print(f"Columns: {list(full_data.columns)}")
    
    print("\nFirst 5 rows of joined data:")
    print(full_data.head())
    
    return full_data

def analyze_by_genotype(full_data):
    """Analyze sleep patterns by genotype."""
    print("\n🧬 ANALYSIS BY GENOTYPE")
    print("=" * 50)
    
    # Calculate average movement per genotype
    genotype_stats = full_data.groupby('genotype').agg({
        'mt': ['mean', 'std', 'count'],
        'ct': ['mean', 'std'],
        'pn': ['mean', 'std']
    }).round(2)
    
    print("Movement statistics by genotype:")
    print(genotype_stats)
    
    # Show genotype distribution
    print(f"\nGenotype distribution:")
    genotype_counts = full_data['genotype'].value_counts()
    for genotype, count in genotype_counts.items():
        print(f"  {genotype}: {count:,} measurements")

def analyze_by_treatment(full_data):
    """Analyze sleep patterns by treatment."""
    print("\n💊 ANALYSIS BY TREATMENT")
    print("=" * 50)
    
    # Calculate average movement per treatment
    treatment_stats = full_data.groupby('treatment').agg({
        'mt': ['mean', 'std', 'count'],
        'ct': ['mean', 'std'],
        'pn': ['mean', 'std']
    }).round(2)
    
    print("Movement statistics by treatment:")
    print(treatment_stats)
    
    # Show treatment distribution
    print(f"\nTreatment distribution:")
    treatment_counts = full_data['treatment'].value_counts()
    for treatment, count in treatment_counts.items():
        print(f"  {treatment}: {count:,} measurements")

def create_visualization(full_data):
    """Create a simple visualization of the data."""
    print("\n📈 CREATING VISUALIZATION")
    print("=" * 50)
    
    # Sample data for visualization (every 100th row to keep it manageable)
    sample_data = full_data.iloc[::100].copy()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Movement by genotype
    sample_data.boxplot(column='mt', by='genotype', ax=axes[0,0])
    axes[0,0].set_title('Movement Count (MT) by Genotype')
    axes[0,0].set_xlabel('Genotype')
    axes[0,0].set_ylabel('Movement Count')
    
    # Plot 2: Movement by treatment
    sample_data.boxplot(column='mt', by='treatment', ax=axes[0,1])
    axes[0,1].set_title('Movement Count (MT) by Treatment')
    axes[0,1].set_xlabel('Treatment')
    axes[0,1].set_ylabel('Movement Count')
    
    # Plot 3: Time series of one fly
    fly_data = sample_data[sample_data['fly_id'] == 'M5_Ch01'].copy()
    if len(fly_data) > 0:
        axes[1,0].plot(fly_data['datetime'], fly_data['mt'], 'b-', alpha=0.7)
        axes[1,0].set_title(f'Movement over time - {fly_data.iloc[0]["fly_id"]} ({fly_data.iloc[0]["genotype"]})')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Movement Count')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Heatmap of movement by genotype and treatment
    pivot_data = sample_data.groupby(['genotype', 'treatment'])['mt'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.1f', ax=axes[1,1], cmap='YlOrRd')
    axes[1,1].set_title('Average Movement by Genotype and Treatment')
    
    plt.tight_layout()
    plt.savefig('../../results/figures/relational_database_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to results/figures/relational_database_analysis.png")

def main():
    """Main demonstration function."""
    print("🚀 RELATIONAL DATABASE DEMONSTRATION")
    print("=" * 60)
    print("This shows how the two-table design enables efficient analysis")
    print("without storing redundant information millions of times.")
    
    # Demonstrate the join
    full_data = demonstrate_join()
    
    # Show analysis capabilities
    analyze_by_genotype(full_data)
    analyze_by_treatment(full_data)
    
    # Create visualization
    create_visualization(full_data)
    
    print("\n✅ DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key benefits of the relational design:")
    print("✅ No redundancy - genotype stored once, not 530K times")
    print("✅ Small file sizes - time_series stays lean (16.5 MB)")
    print("✅ Easy updates - change fly info in one place")
    print("✅ Flexible - add new metadata columns without touching time_series")
    print("✅ Professional database design")

if __name__ == "__main__":
    main()
