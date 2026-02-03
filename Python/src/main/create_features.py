#!/usr/bin/env python3
"""
Drosophila Sleep Metrics (Individual-Fly Analysis)

Works with 15-beam DAM data structured like dam_sleep

Columns expected: Monitor, Channel, datetime, Reading, Value, Genotype, Sex, Treatment
(or lowercase: monitor, channel, datetime, reading, value, genotype, sex, treatment)

reading is optional, but if missing will always assume reading type is MT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================
#   1. PREPARE DATA
# ============================================================

def prepare_data(dam_sleep: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for sleep analysis.
    
    Args:
        dam_sleep: DataFrame with columns: Monitor/monitor, Channel/channel, 
                   datetime, Reading/reading, Value/value, etc.
    
    Returns:
        DataFrame with fly_id and value columns, filtered to MT readings
    """
    df = dam_sleep.copy()
    
    # Normalize column names to lowercase
    col_mapping = {
        'Monitor': 'monitor', 'Channel': 'channel', 'Reading': 'reading',
        'Value': 'value', 'Genotype': 'genotype', 'Sex': 'sex', 
        'Treatment': 'treatment'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # 1.1 Create unique fly ID
    df['fly_id'] = df['monitor'].astype(str) + '-' + df['channel'].astype(str)
    
    # 1.2 Keep only movement data (MT) for sleep analysis
    if 'reading' in df.columns:
        # Filter to MT only
        df = df[df['reading'] == 'MT'].copy()
        df = df.drop('reading', axis=1)
        print(f"✓ Filtered to MT readings only")
    else:
        print(f"✓ Data is already MT-only")

    return df


# ============================================================
#   2. DEFINE FUNCTION
# ============================================================

def compute_sleep_metrics_full(data: pd.DataFrame,
                               bin_length_min: int = 1,
                               sleep_threshold_min: int = 5,
                               lights_on_ZT0: int = 9,
                               lights_off_ZT12: int = 21) -> pd.DataFrame:
    """
    Compute comprehensive sleep metrics for each fly per day.
    
    Args:
        data: DataFrame with fly_id, datetime, value columns
        bin_length_min: Length of each time bin in minutes (default: 1)
        sleep_threshold_min: Minimum minutes of inactivity to count as sleep (default: 5)
        lights_on_ZT0: Hour when lights turn on (default: 9)
        lights_off_ZT12: Hour when lights turn off (default: 21)
    
    Returns:
        DataFrame with sleep metrics per fly per day
    """
    df = data.copy()
    
    # Calculate threshold in bins
    thr_bins = int(np.ceil(sleep_threshold_min / bin_length_min))
    
    # Ensure datetime is datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Add day, ZT, and phase
    df['day'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['ZT'] = (df['hour'] + df['minute'] / 60 - lights_on_ZT0) % 24
    df['phase'] = df['ZT'].apply(lambda x: 'Light' if x < 12 else 'Dark')
    
    # Sort by fly_id, day, datetime
    df = df.sort_values(['fly_id', 'day', 'datetime']).reset_index(drop=True)
    
    # Group by fly_id and day to compute sleep
    result_list = []
    
    for (fly_id, day), group in df.groupby(['fly_id', 'day']):
        group = group.copy().reset_index(drop=True)
        
        # Detect inactivity and sleep
        group['inactive'] = (group['value'] == 0)
        
        # Create run IDs for consecutive inactive/active periods
        group['run_id'] = (group['inactive'] != group['inactive'].shift()).cumsum()
        
        # Calculate run lengths
        run_lengths = group.groupby('run_id')['inactive'].transform('count')
        group['run_len'] = run_lengths
        
        # Sleep = inactive AND run length >= threshold
        group['sleep'] = group['inactive'] & (group['run_len'] >= thr_bins)
        
        # ----- Bout-level information -----
        # Detect sleep bout starts
        group['sleep_start'] = group['sleep'] & (~group['sleep'].shift(1, fill_value=False))
        
        # Create bout IDs
        bout_starts = group['sleep_start'].cumsum()
        group['bout_id'] = bout_starts * group['sleep'].astype(int)
        
        # Filter to sleep periods and compute bout summaries
        sleep_periods = group[group['sleep']].copy()
        
        if len(sleep_periods) == 0:
            # No sleep detected for this fly/day
            bout_summary = pd.DataFrame({
                'fly_id': [fly_id],
                'day': [day],
                'total_sleep_min': [0],
                'light_sleep_min': [0],
                'dark_sleep_min': [0],
                'n_bouts_total': [0],
                'mean_bout_len_min': [np.nan],
                'max_bout_len_min': [0],
                'n_bouts_light': [0],
                'mean_bout_len_light_min': [np.nan],
                'max_bout_len_light_min': [0],
                'n_bouts_dark': [0],
                'mean_bout_len_dark_min': [np.nan],
                'max_bout_len_dark_min': [0],
                'sleep_latency_min': [np.nan]
            })
        else:
            # Compute bout-level metrics
            bouts = sleep_periods.groupby('bout_id').agg({
                'bout_id': 'count',  # bout_len_bins
                'phase': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # bout_phase
                'ZT': 'first'  # bout_start_ZT
            }).rename(columns={'bout_id': 'bout_len_bins', 'phase': 'bout_phase', 'ZT': 'bout_start_ZT'})
            
            # Compute daily bout summary
            bout_summary = pd.DataFrame({
                'fly_id': [fly_id],
                'day': [day],
                'total_sleep_min': [bouts['bout_len_bins'].sum() * bin_length_min],
                'light_sleep_min': [bouts[bouts['bout_phase'] == 'Light']['bout_len_bins'].sum() * bin_length_min],
                'dark_sleep_min': [bouts[bouts['bout_phase'] == 'Dark']['bout_len_bins'].sum() * bin_length_min],
                'n_bouts_total': [len(bouts)],
                'mean_bout_len_min': [bouts['bout_len_bins'].mean() * bin_length_min],
                'max_bout_len_min': [bouts['bout_len_bins'].max() * bin_length_min],
                'n_bouts_light': [(bouts['bout_phase'] == 'Light').sum()],
                'mean_bout_len_light_min': [bouts[bouts['bout_phase'] == 'Light']['bout_len_bins'].mean() * bin_length_min if (bouts['bout_phase'] == 'Light').any() else np.nan],
                'max_bout_len_light_min': [bouts[bouts['bout_phase'] == 'Light']['bout_len_bins'].max() * bin_length_min if (bouts['bout_phase'] == 'Light').any() else 0],
                'n_bouts_dark': [(bouts['bout_phase'] == 'Dark').sum()],
                'mean_bout_len_dark_min': [bouts[bouts['bout_phase'] == 'Dark']['bout_len_bins'].mean() * bin_length_min if (bouts['bout_phase'] == 'Dark').any() else np.nan],
                'max_bout_len_dark_min': [bouts[bouts['bout_phase'] == 'Dark']['bout_len_bins'].max() * bin_length_min if (bouts['bout_phase'] == 'Dark').any() else 0],
                'sleep_latency_min': [bouts[bouts['bout_phase'] == 'Dark']['bout_start_ZT'].min() * 60 if (bouts['bout_phase'] == 'Dark').any() else np.nan]
            })
        
        # ----- Transition probabilities -----
        sleep_array = group['sleep'].values
        n = len(sleep_array)
        
        if n > 1:
            # N_S_to_W: sleep to wake transitions
            N_S_to_W = np.sum((sleep_array[1:] == False) & (sleep_array[:-1] == True))
            N_S = np.sum(sleep_array)
            
            # N_W_to_S: wake to sleep transitions
            N_W_to_S = np.sum((sleep_array[1:] == True) & (sleep_array[:-1] == False))
            N_W = np.sum(~sleep_array)
            
            P_wake = N_S_to_W / N_S if N_S > 0 else np.nan
            P_doze = N_W_to_S / N_W if N_W > 0 else np.nan
        else:
            P_wake = np.nan
            P_doze = np.nan
        
        # Add transition probabilities to bout summary
        bout_summary['P_wake'] = P_wake
        bout_summary['P_doze'] = P_doze
        
        result_list.append(bout_summary)
    
    # Combine all results
    final = pd.concat(result_list, ignore_index=True)
    
    # Add derived metrics
    final['dark_phase_min'] = 12 * 60
    final['sleep_efficiency'] = (final['dark_sleep_min'] / final['dark_phase_min']) * 100
    final['WASO_min'] = final['total_sleep_min'] - final['dark_sleep_min']
    final.loc[final['total_sleep_min'].isna(), 'WASO_min'] = np.nan
    
    return final


# ============================================================
#   3. RUN ANALYSIS
# ============================================================

def run_sleep_analysis(dam_sleep: pd.DataFrame,
                      bin_length_min: int = 1,
                      sleep_threshold_min: int = 5,
                      lights_on_ZT0: int = 9) -> pd.DataFrame:
    """
    Run complete sleep analysis pipeline.
    
    Args:
        dam_sleep: Input DataFrame with sleep data
        bin_length_min: Length of each time bin in minutes
        sleep_threshold_min: Minimum minutes of inactivity to count as sleep
        lights_on_ZT0: Hour when lights turn on
    
    Returns:
        DataFrame with sleep metrics per fly per day, including metadata
    """
    # Prepare data
    dam_mt = prepare_data(dam_sleep)
    
    # Compute sleep metrics
    fly_sleep_summary = compute_sleep_metrics_full(
        dam_mt,
        bin_length_min=bin_length_min,
        sleep_threshold_min=sleep_threshold_min,
        lights_on_ZT0=lights_on_ZT0
    )
    
    # Add metadata back
    metadata = dam_mt[['fly_id', 'genotype', 'sex', 'treatment']].drop_duplicates()
    fly_sleep_summary = fly_sleep_summary.merge(metadata, on='fly_id', how='left')
    
    return fly_sleep_summary


# ============================================================
#   4. OPTIONAL: COLLAPSE ACROSS DAYS
# ============================================================

def collapse_across_days(fly_sleep_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Average metrics across days for each fly.
    
    Args:
        fly_sleep_summary: Per-day sleep metrics
    
    Returns:
        Per-fly averaged metrics
    """
    numeric_cols = fly_sleep_summary.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['monitor', 'channel']]
    
    fly_sleep_mean = fly_sleep_summary.groupby(
        ['fly_id', 'genotype', 'sex', 'treatment']
    )[numeric_cols].mean().reset_index()
    
    return fly_sleep_mean


# ============================================================
#   5. GROUP-LEVEL ANALYSIS & VISUALIZATION
# ============================================================

def compute_group_summaries(fly_sleep_mean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean ± SEM by experimental group.
    
    Args:
        fly_sleep_mean: Per-fly averaged metrics
    
    Returns:
        Group-level summaries with mean and SEM
    """
    # Clean data
    fly_sleep_mean_clean = fly_sleep_mean.copy()
    
    # Replace infinite and NaN values
    numeric_cols = fly_sleep_mean_clean.select_dtypes(include=[np.number]).columns
    fly_sleep_mean_clean[numeric_cols] = fly_sleep_mean_clean[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    
    # Filter out missing metadata
    fly_sleep_mean_clean = fly_sleep_mean_clean[
        fly_sleep_mean_clean['genotype'].notna() &
        fly_sleep_mean_clean['sex'].notna() &
        fly_sleep_mean_clean['treatment'].notna() &
        (fly_sleep_mean_clean['genotype'] != 'na')
    ].copy()
    
    # Compute SEM function
    def sem(x):
        x_clean = x.dropna()
        if len(x_clean) == 0:
            return np.nan
        return x_clean.std() / np.sqrt(len(x_clean))
    
    # Group summaries
    group_sleep_summary = fly_sleep_mean_clean.groupby(
        ['genotype', 'sex', 'treatment']
    ).agg({
        **{col: ['mean', sem] for col in numeric_cols},
        'fly_id': 'count'  # n_flies
    }).reset_index()
    
    # Flatten column names
    group_sleep_summary.columns = [
        '_'.join(col).strip('_') if col[1] else col[0]
        for col in group_sleep_summary.columns.values
    ]
    group_sleep_summary = group_sleep_summary.rename(columns={'fly_id_count': 'n_flies'})
    
    return group_sleep_summary


def plot_metrics(group_sleep_summary: pd.DataFrame,
                metrics: list,
                metric_labels: dict,
                title_prefix: str = "",
                output_dir: Optional[str] = None):
    """
    Create bar plots with error bars for sleep metrics.
    
    Args:
        group_sleep_summary: Group-level summaries
        metrics: List of metric names to plot
        metric_labels: Dictionary mapping metric names to display labels
        title_prefix: Prefix for plot titles
        output_dir: Directory to save plots (if None, displays plots)
    """
    for metric in metrics:
        mean_col = f'{metric}_mean'
        sem_col = f'{metric}_sem'
        
        if mean_col not in group_sleep_summary.columns or sem_col not in group_sleep_summary.columns:
            print(f"Warning: {metric} not found in group summary, skipping...")
            continue
        
        subdata = group_sleep_summary[['genotype', 'sex', 'treatment', mean_col, sem_col]].copy()
        subdata = subdata.rename(columns={mean_col: 'Mean', sem_col: 'SEM'})
        subdata = subdata.dropna(subset=['Mean', 'SEM'])
        
        if len(subdata) == 0:
            continue
        
        label = metric_labels.get(metric, metric)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        x_pos = np.arange(len(subdata['genotype'].unique()))
        width = 0.25
        
        for i, sex in enumerate(subdata['sex'].unique()):
            sex_data = subdata[subdata['sex'] == sex]
            for j, treatment in enumerate(sex_data['treatment'].unique()):
                treat_data = sex_data[sex_data['treatment'] == treatment]
                genotypes = treat_data['genotype'].unique()
                
                means = [treat_data[treat_data['genotype'] == g]['Mean'].values[0] 
                        if len(treat_data[treat_data['genotype'] == g]) > 0 else 0
                        for g in subdata['genotype'].unique()]
                sems = [treat_data[treat_data['genotype'] == g]['SEM'].values[0] 
                       if len(treat_data[treat_data['genotype'] == g]) > 0 else 0
                       for g in subdata['genotype'].unique()]
                
                offset = (i * len(subdata['treatment'].unique()) + j) * width - width
                ax.bar(x_pos + offset, means, width, yerr=sems, 
                      label=f'{sex} - {treatment}', capsize=5)
        
        ax.set_xlabel('Genotype', fontsize=12)
        ax.set_ylabel(f'{label} (Mean ± SEM)', fontsize=12)
        ax.set_title(f'{title_prefix}{label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(subdata['genotype'].unique(), rotation=45, ha='right')
        ax.legend(title='Sex - Treatment', loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f'{output_dir}/{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# ============================================================
#   MAIN EXECUTION
# ============================================================

def main():
    """Main function to run sleep analysis pipeline."""
    import sys
    import os
    
    # Define metric labels
    metric_labels = {
        'total_sleep_min': 'Total Sleep (min)',
        'light_sleep_min': 'Daytime Sleep (min)',
        'dark_sleep_min': 'Nighttime Sleep (min)',
        'sleep_efficiency': 'Sleep Efficiency (%)',
        'n_bouts_total': 'Number of Sleep Bouts',
        'mean_bout_len_min': 'Average Bout Length (min)',
        'max_bout_len_min': 'Longest Sleep Bout (min)',
        'WASO_min': 'Wake After Sleep Onset (min)',
        'P_wake': 'Probability of Waking (Pwake)',
        'P_doze': 'Probability of Falling Asleep (Pdoze)',
        'sleep_latency_min': 'Sleep Latency (min)'
    }
    
    core_metrics = ['total_sleep_min', 'light_sleep_min', 
                   'dark_sleep_min', 'sleep_efficiency']
    
    next_metrics = ['n_bouts_total', 'mean_bout_len_min', 'max_bout_len_min',
                   'WASO_min', 'P_wake', 'P_doze', 'sleep_latency_min']
    
    # Check for input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../../data/processed/dam_data_MT.csv'
        if not os.path.exists(input_file):
            input_file = '../../data/processed/dam_data_merged.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Usage: python compute_sleep_features.py [input_file]")
        sys.exit(1)
    
    print("=" * 60)
    print("Drosophila Sleep Metrics Analysis")
    print("=" * 60)
    print(f"\n📊 Loading data from: {input_file}")
    
    # Load data
    dam_sleep = pd.read_csv(input_file)
    print(f"✓ Loaded {len(dam_sleep):,} rows")
    
    # Run analysis
    print("\n🔬 Computing sleep metrics...")
    fly_sleep_summary = run_sleep_analysis(dam_sleep)
    print(f"✓ Computed metrics for {len(fly_sleep_summary):,} fly-day combinations")
    
    # Collapse across days
    print("\n📈 Averaging across days...")
    fly_sleep_mean = collapse_across_days(fly_sleep_summary)
    print(f"✓ Averaged metrics for {len(fly_sleep_mean)} flies")
    
    # Compute group summaries
    print("\n📊 Computing group-level summaries...")
    group_sleep_summary = compute_group_summaries(fly_sleep_mean)
    print(f"✓ Computed summaries for {len(group_sleep_summary)} groups")
    
    # Save results
    output_dir = '../../data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    fly_sleep_summary.to_csv(f'{output_dir}/fly_sleep_summary.csv', index=False)
    fly_sleep_mean.to_csv(f'{output_dir}/fly_sleep_mean.csv', index=False)
    group_sleep_summary.to_csv(f'{output_dir}/group_sleep_summary.csv', index=False)
    
    print(f"\n💾 Saved results to {output_dir}/")
    print("   - fly_sleep_summary.csv (per-day metrics)")
    print("   - fly_sleep_mean.csv (per-fly averages)")
    print("   - group_sleep_summary.csv (group-level summaries)")
    
    # Create plots
    print("\n📊 Generating plots...")
    plot_dir = f'{output_dir}/sleep_plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_metrics(group_sleep_summary, core_metrics, metric_labels, 
                "Core Sleep Metric: ", plot_dir)
    plot_metrics(group_sleep_summary, next_metrics, metric_labels,
                "Sleep Architecture Metric: ", plot_dir)
    
    print(f"✓ Saved plots to {plot_dir}/")
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()