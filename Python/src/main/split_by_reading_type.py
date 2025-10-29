#!/usr/bin/env python3
"""
Split DAM Data by Reading Type

This script splits the filtered DAM data into 3 separate files by reading type:
- dam_data_MT.csv: Movement readings (for sleep/activity analysis)
- dam_data_CT.csv: Cumulative total readings (for activity tracking)
- dam_data_Pn.csv: Pause readings (for inactivity analysis)

Each file contains only one reading type, making analysis more focused and files smaller.
"""

import pandas as pd
import os
from datetime import datetime


def split_by_reading_type(input_file, output_dir):
    """
    Split the filtered DAM data into 3 files by reading type.
    
    Args:
        input_file (str): Path to dam_data_filtered.csv
        output_dir (str): Directory to save the split files
        
    Returns:
        dict: Summary information about the split files
    """
    print("📂 Loading filtered data...")
    
    # Load the filtered data
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"   Loaded {len(df):,} rows")
    print(f"   Reading types: {sorted(df['reading'].unique())}")
    
    # Get reading type counts
    reading_counts = df['reading'].value_counts().sort_index()
    print(f"\n   Rows per reading type:")
    for reading, count in reading_counts.items():
        percentage = (count / len(df)) * 100
        print(f"     {reading}: {count:,} rows ({percentage:.1f}%)")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split by reading type
    split_files = {}
    
    for reading_type in ['MT', 'CT', 'Pn']:
        print(f"\n📊 Creating {reading_type} file...")
        
        # Filter data for this reading type
        reading_data = df[df['reading'] == reading_type].copy()
        
        # Drop the 'reading' column since it's redundant
        reading_data = reading_data.drop('reading', axis=1)
        
        # Define output filename
        output_file = os.path.join(output_dir, f'dam_data_{reading_type}.csv')
        
        # Save to CSV
        reading_data.to_csv(output_file, index=False)
        
        # Calculate file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        # Store summary info
        split_files[reading_type] = {
            'filename': output_file,
            'rows': len(reading_data),
            'columns': list(reading_data.columns),
            'date_range': (reading_data['datetime'].min(), reading_data['datetime'].max()),
            'unique_flies': reading_data['fly_id'].nunique(),
            'file_size_mb': round(file_size, 1)
        }
        
        print(f"   ✓ Saved {len(reading_data):,} rows to {os.path.basename(output_file)}")
    
    return split_files


def create_split_summary(split_files, input_file, input_rows):
    """
    Create a summary file documenting the split operation.
    
    Args:
        split_files (dict): Summary info from split_by_reading_type
        input_file (str): Original input file path
        input_rows (int): Number of rows in input file
    """
    summary_file = os.path.join(os.path.dirname(input_file), 'SPLIT_SUMMARY.txt')
    
    with open(summary_file, 'w') as f:
        f.write("DAM Data Split Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Split performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {os.path.basename(input_file)}\n")
        f.write(f"Input rows: {input_rows:,}\n\n")
        
        f.write("Output files:\n")
        f.write("-" * 30 + "\n")
        
        total_output_rows = 0
        for reading_type in ['MT', 'CT', 'Pn']:
            info = split_files[reading_type]
            f.write(f"\n{reading_type} file: {os.path.basename(info['filename'])}\n")
            f.write(f"  - Rows: {info['rows']:,}\n")
            f.write(f"  - Columns: {', '.join(info['columns'])}\n")
            f.write(f"  - Date range: {info['date_range'][0].strftime('%Y-%m-%d')} to {info['date_range'][1].strftime('%Y-%m-%d')}\n")
            f.write(f"  - Unique flies: {info['unique_flies']}\n")
            f.write(f"  - File size: {info['file_size_mb']} MB\n")
            
            total_output_rows += info['rows']
        
        f.write(f"\nVerification: {'✓' if total_output_rows == input_rows else '✗'} ")
        f.write(f"Sum of output rows ({total_output_rows:,}) = Input rows ({input_rows:,})\n\n")
        
        f.write("Usage Guidelines:\n")
        f.write("-" * 30 + "\n")
        f.write("• MT file: Use for sleep/activity analysis (movement is key indicator)\n")
        f.write("• CT file: Use for cumulative activity tracking over time\n")
        f.write("• Pn file: Use for pause/inactivity behavior analysis\n\n")
        
        f.write("Note: All files contain the same metadata (fly_id, genotype, sex, treatment)\n")
        f.write("      but different measurement values. Each file is self-contained.\n\n")
        
        f.write("To recombine if needed:\n")
        f.write("-" * 30 + "\n")
        f.write("```python\n")
        f.write("import pandas as pd\n\n")
        f.write("mt = pd.read_csv('dam_data_MT.csv')\n")
        f.write("ct = pd.read_csv('dam_data_CT.csv')\n")
        f.write("pn = pd.read_csv('dam_data_Pn.csv')\n\n")
        f.write("mt['reading'] = 'MT'\n")
        f.write("ct['reading'] = 'CT'\n")
        f.write("pn['reading'] = 'Pn'\n\n")
        f.write("combined = pd.concat([mt, ct, pn])\n")
        f.write("```\n")
    
    print(f"📝 Created summary file: {os.path.basename(summary_file)}")


def main():
    """
    Main function to split the filtered DAM data by reading type.
    """
    print("📊 Splitting DAM Data by Reading Type")
    print("=" * 50)
    
    # Define file paths
    input_file = '../../data/processed/dam_data_marked.csv'
    output_dir = '../../data/processed'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        print("   Please run mark_dead_flies.py first to generate the marked data.")
        return
    
    # Load input file info
    print(f"\n📂 Input file: {os.path.basename(input_file)}")
    df = pd.read_csv(input_file)
    input_rows = len(df)
    
    print(f"   Total rows: {input_rows:,}")
    print(f"   Unique reading types: {', '.join(sorted(df['reading'].unique()))}")
    
    # Get reading type distribution
    reading_counts = df['reading'].value_counts().sort_index()
    print(f"\n   Rows per reading type:")
    for reading, count in reading_counts.items():
        percentage = (count / input_rows) * 100
        print(f"     {reading}: {count:,} rows ({percentage:.1f}%)")
    
    # Split the data
    print(f"\n🔧 Creating split files...")
    split_files = split_by_reading_type(input_file, output_dir)
    
    # Create summary file
    print(f"\n📝 Creating summary documentation...")
    create_split_summary(split_files, input_file, input_rows)
    
    # Print final summary
    print(f"\n" + "=" * 50)
    print("=== SPLITTING DAM DATA BY READING TYPE ===")
    print("=" * 50)
    
    print(f"\nInput file: {os.path.basename(input_file)}")
    print(f"Total rows: {input_rows:,}")
    print(f"Unique reading types: {', '.join(sorted(df['reading'].unique()))}")
    
    print(f"\nRows per reading type:")
    for reading, count in reading_counts.items():
        percentage = (count / input_rows) * 100
        print(f"- {reading}: {count:,} rows ({percentage:.1f}%)")
    
    print(f"\nCreating split files...")
    
    total_output_rows = 0
    for reading_type in ['MT', 'CT', 'Pn']:
        info = split_files[reading_type]
        print(f"\n✓ {os.path.basename(info['filename'])}")
        print(f"  - Rows: {info['rows']:,}")
        print(f"  - Columns: {', '.join(info['columns'])}")
        print(f"  - Date range: {info['date_range'][0].strftime('%Y-%m-%d')} to {info['date_range'][1].strftime('%Y-%m-%d')}")
        print(f"  - Unique flies: {info['unique_flies']}")
        print(f"  - File size: {info['file_size_mb']} MB")
        
        total_output_rows += info['rows']
    
    print(f"\nVerification: {'✓' if total_output_rows == input_rows else '✗'} ", end="")
    print(f"Sum of output rows ({total_output_rows:,}) = Input rows ({input_rows:,})")
    
    print(f"\nFiles saved to: {output_dir}/")
    print(f"\n✅ Data splitting complete!")
    
    print(f"\n💡 Usage for future analysis:")
    print(f"   • Sleep analysis: Use dam_data_MT.csv (movement is key)")
    print(f"   • Activity tracking: Use dam_data_CT.csv (cumulative totals)")
    print(f"   • Pause behavior: Use dam_data_Pn.csv (inactivity patterns)")


if __name__ == "__main__":
    main()
