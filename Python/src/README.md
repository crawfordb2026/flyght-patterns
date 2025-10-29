# Script Organization

This directory contains the organized scripts for the Drosophila Activity Monitor (DAM) data processing pipeline.

## Directory Structure

### `main/` - Core Pipeline Scripts
These are the main scripts that form the core data processing pipeline:

- **`create_database.py`** - Creates the initial merged dataset from raw monitor files and details.txt
- **`filter_empty_channels.py`** - Removes data from channels with missing metadata
- **`mark_dead_flies.py`** - Identifies and marks flies that meet death criteria (24+ hours of inactivity)
- **`split_by_reading_type.py`** - Splits the dataset into separate files by reading type (MT, CT, Pn)
- **`bin_hourly.py`** - Bins raw DAM data into hourly intervals
- **`BIN_HOURLY_README.md`** - Documentation for the hourly binning script

### `test/` - Debugging and Verification Scripts
These scripts are used for testing, debugging, and verification:

- **`detect_dead_flies.py`** - Original dead fly detection script (superseded by mark_dead_flies.py)
- **`bin_data.py`** - Original 5-minute binning script (superseded by bin_hourly.py)
- **`debug_*.py`** - Various debugging scripts for troubleshooting issues
- **`verify_*.py`** - Scripts for verifying data integrity and processing results
- **`demo_relational_database.py`** - Demo script for the original relational database approach

## Usage

### Running the Main Pipeline
From the `Python/src/main/` directory:

```bash
# Step 1: Create initial database
python3 create_database.py

# Step 2: Filter empty channels
python3 filter_empty_channels.py

# Step 3: Mark dead flies
python3 mark_dead_flies.py

# Step 4: Split by reading type
python3 split_by_reading_type.py

# Step 5: Bin to hourly intervals
python3 bin_hourly.py ../../data/processed/dam_data_MT.csv
python3 bin_hourly.py ../../data/processed/dam_data_CT.csv
python3 bin_hourly.py ../../data/processed/dam_data_Pn.csv
```

### Running Test Scripts
From the `Python/src/test/` directory:

```bash
# Verify data integrity
python3 verify_split_files.py
python3 verify_dead_flies.py

# Debug specific issues
python3 debug_missing_flies.py
python3 debug_m6_ch05.py
```

## File Paths
All scripts have been updated to use relative paths from their new locations:
- Main scripts: `../../data/processed/` for data files
- Test scripts: `../../data/processed/` for data files

## Data Flow
1. Raw files (`Monitor5.txt`, `Monitor6.txt`, `details.txt`) → `dam_data_merged.csv`
2. `dam_data_merged.csv` → `dam_data_with_flies.csv` (filtered)
3. `dam_data_with_flies.csv` → `dam_data_marked.csv` (dead flies marked)
4. `dam_data_marked.csv` → `dam_data_MT.csv`, `dam_data_CT.csv`, `dam_data_Pn.csv` (split)
5. Split files → `dam_data_MT_hourly.csv`, etc. (hourly binned)
