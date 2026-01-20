# Fly Sleep Analysis Pipeline

This folder contains a complete, ordered pipeline for processing raw Drosophila Activity Monitor (DAM) data into ML-ready feature tables.

## Pipeline Overview

```
Raw Files → Step 1 → Step 2 (optional) → Step 3 → Step 4
           Prepare  Remove              Features  Clean
           Data +   Flies                         Features
           Health
           Report
```

## Step-by-Step Guide

### Step 1: Prepare Data and Generate Health Report
**Script:** `1-prepare_data_and_health.py`

**What it does:**
- Loads raw DAM files (Monitor5.txt, Monitor6.txt) and metadata (details.txt)
- Merges data into long format
- Calculates time variables: Date, Time, ZT (Zeitgeber Time), Phase (Light/Dark)
- Optionally filters by date range
- Calculates Exp_Day (experimental day) using global experiment start
- Generates health report (using in-memory data, no file I/O):
  - Filters to MT data only
  - Calculates daily health metrics per fly
  - Classifies fly status (Alive, Unhealthy, Dead, QC_Fail) using decision tree
  - Applies irreversible death rule
  - Generates per-fly summary health report

**Output:** `dam_data_prepared.csv`, `health_report.csv`

**Usage:**
```bash
cd Python/src/pipeline
python 1-prepare_data_and_health.py
```

**Options:**
- `--dam-files`: List of Monitor*.txt files (default: Monitor5.txt, Monitor6.txt)
- `--meta-path`: Metadata file path (default: details.txt)
- `--lights-on`: Hour when lights turn on (default: 9)
- `--apply-date-filter`: Enable date filtering
- `--exp-start`: Experiment start date (YYYY-MM-DD, default: auto-detect)
- `--exp-end`: Experiment end date (YYYY-MM-DD, default: auto-detect)
- `--ref-day`: Reference day for normalization (default: 4)
- `--exclude-days`: Days to exclude from health analysis (default: 1 7)
- `--output-data`: Output file path for prepared data (default: dam_data_prepared.csv)
- `--output-health`: Output file path for health report (default: health_report.csv)

**Important:** Review the health report to decide which flies to remove in Step 2.

**Health Report Death Criteria:**
The health report classifies flies as "Dead" if they meet any of these criteria:
- 24 consecutive hours with zero movement (MT counts = 0)
- 12 consecutive hours with zero movement AND no startle response at light transitions
- Activity drops below 20% of reference day (day 4) activity

---

### Step 2: Remove Flies (Optional)
**Script:** `2-remove_flies.py`

**What it does:**
- Reads prepared data from Step 1
- Optionally reads health report from Step 1
- Removes flies based on various criteria:
  - Specific fly IDs
  - Specific days for certain flies
  - Metadata criteria (genotype, sex, treatment)
  - Health status from health report

**Output:** `dam_data_cleaned.csv`

**Usage:**
```bash
python 2-remove_flies.py
```

**Options:**
- `--input`: Input file from Step 1 (default: dam_data_prepared.csv)
- `--output`: Output cleaned data file (default: dam_data_cleaned.csv)
- `--health-report`: Health report file from Step 1 (default: health_report.csv)
- `--flies`: Comma-separated fly keys (e.g., "6-ch23,6-ch5")
- `--statuses`: Comma-separated health statuses (e.g., "Dead,QC_Fail")
- `--genotypes`: Comma-separated genotypes to remove
- `--sexes`: Comma-separated sexes to remove
- `--treatments`: Comma-separated treatments to remove
- `--per-fly-remove`: JSON dict mapping fly keys to days (e.g., '{"5-ch7": [1, 2]}')

**Note:** This step is OPTIONAL. If you skip it, Step 3 will use the prepared data directly.

---

### Step 3: Create Feature Table
**Script:** `3-create_feature_table.py`

**What it does:**
- Reads cleaned data (from Step 1 or Step 2)
- Extracts RHYTHM features (circadian):
  - Calculates hourly totals per fly per day per ZT
  - Runs daily cosinor regression (per fly per day)
  - Aggregates to per-fly means and SDs
- Extracts SLEEP features:
  - Calculates daily sleep metrics per fly per day
  - Aggregates to per-fly means
- Merges rhythm + sleep features into final ML_features table

**Output:** `ML_features.csv`

**Usage:**
```bash
python 3-create_feature_table.py
```

**Options:**
- `--input`: Input cleaned/prepared data file (default: dam_data_cleaned.csv)
- `--output`: Output ML features file (default: ML_features.csv)
- `--exclude-days`: Days to exclude (default: 1 7)
- `--sleep-threshold`: Minimum minutes of inactivity for sleep (default: 5)

**Note:** Step 3 automatically falls back to `dam_data_prepared.csv` if `dam_data_cleaned.csv` doesn't exist. The output is sorted by monitor and channel for easy reading.

**Note:** Step 3 automatically falls back to `dam_data_prepared.csv` if `dam_data_cleaned.csv` doesn't exist. The output is sorted by monitor and channel for easy reading.

---

### Step 4: Clean ML Features
**Script:** `4-clean_ml_features.py`

**What it does:**
- Reads ML_features.csv from Step 3
- Removes flies with problematic feature values:
  - Zero total sleep
  - Zero sleep bouts
  - Zero/NaN P_doze (probability of falling asleep)
- Removes IQR outliers per group (Genotype × Sex × Treatment)
- Fixes NaN values (replaces with 0 or group mean)
- Creates z-scored (standardized) feature table
- Saves both cleaned and z-scored versions

**Output:** `ML_features_clean.csv`, `ML_features_Z.csv`

**Usage:**
```bash
python 4-clean_ml_features.py
```

**Options:**
- `--input`: Input ML features file from Step 3 (default: ML_features.csv)
- `--output-clean`: Output cleaned features file (default: ML_features_clean.csv)
- `--output-z`: Output z-scored features file (default: ML_features_Z.csv)
- `--iqr-multiplier`: IQR multiplier for outlier detection (default: 1.5)

**This is the final step!** The output is ready for ML analysis.

---

## Quick Start

Run all steps in order:

```bash
cd Python/src/pipeline

# Step 1: Prepare data and generate health report
python 1-prepare_data_and_health.py

# Step 2: Remove flies (optional - review health report first!)
python 2-remove_flies.py --statuses "Dead,QC_Fail"

# Step 3: Create feature table
python 3-create_feature_table.py

# Step 4: Clean features (optional - recommended for ML)
python 4-clean_ml_features.py
```

---

## Configuration

Each script has a **USER CONFIGURATION** section at the top where you can set default values. For example, in `2-remove_flies.py`:

```python
# Removal criteria (set these in the script or use command-line arguments)
REMOVE_FLIES = None  # Example: ["6-ch23", "6-ch5"]
REMOVE_BY_STATUS = ["Dead"]  # Remove flies with these statuses
```

**Note:** If you run Step 2 without any command-line arguments, it will use the config values from the script. Command-line arguments override config values.

---

## Output Files

All output files are saved to `Python/src/pipeline/data/processed/`:

1. `dam_data_prepared.csv` - Prepared data with time variables
2. `health_report.csv` - Per-fly health status summary
3. `dam_data_cleaned.csv` - Cleaned data (if Step 2 is run)
4. `ML_features.csv` - ML-ready feature table (from Step 3)
5. `ML_features_clean.csv` - Cleaned feature table (if Step 4 is run)
6. `ML_features_Z.csv` - Z-scored (normalized) feature table (if Step 4 is run)

---

## File Structure

```
pipeline/
├── README.md                        # This file
├── 1-prepare_data_and_health.py    # Step 1: Data preparation + health report
├── 2-remove_flies.py               # Step 2: Fly removal (optional)
├── 3-create_feature_table.py       # Step 3: Feature extraction
└── 4-clean_ml_features.py          # Step 4: Feature cleaning (optional)
```

---

## Notes

- **Step 2 is optional**: You can skip fly removal and go directly from Step 1 to Step 3
- **Step 4 is optional**: Recommended for ML analysis, but you can use ML_features.csv directly
- **Health report review**: Always review the health report before deciding what to remove
- **Efficient workflow**: Step 1 generates both prepared data and health report in one pass (no intermediate file I/O)
- **Path handling**: All scripts handle relative paths automatically
- **Error handling**: Scripts check for required input files and provide helpful error messages

---

## Troubleshooting

**"Input file not found"**
- Make sure you've run the previous steps in order
- Check that files are in the expected locations

**"No removal criteria specified"**
- Step 2 will warn if no removal criteria are set
- You can skip Step 2 if you don't want to remove any flies

**Import errors**
- Make sure you're running from the `pipeline/` directory
- The scripts import functions from `main/` directory

---

## Testing

A comprehensive test suite is available for Step 2 (`2-remove_flies.py`) in `Python/src/test-scripts/test_remove_flies.py`. The tests cover:
- All individual removal criteria
- Multiple criteria combinations
- Edge cases and error handling

Run tests with:
```bash
cd Python/src/test-scripts
pytest test_remove_flies.py -v
```

---

## Comparison with R Scripts

This pipeline replicates the functionality of:
- `pipeline.r` → Step 1 (data prep + health report)
- `feature-table.r` → Step 3 (feature extraction)

The Python pipeline is more modular, allowing you to review intermediate results and make decisions at each step. Step 1 combines data preparation and health report generation for efficiency.
