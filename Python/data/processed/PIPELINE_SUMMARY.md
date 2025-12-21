# Fly-ML Data Processing Pipeline Summary

## Overview

This pipeline processes raw Drosophila Activity Monitor (DAM) data from monitor files and metadata into clean, analyzed datasets ready for research. The pipeline uses comprehensive health analysis to identify problematic flies, which can then be removed using flexible criteria.

---

## Complete Pipeline Workflow

```
RAW DATA
├── Monitor5.txt, Monitor6.txt (raw measurements)
└── details.txt (fly metadata)
    ↓
[Step 1: create_database.py]
    ↓
dam_data_merged.csv (all flies, all readings: MT/CT/Pn)
    ↓
[Step 2: filter_empty_channels.py]
    ↓
dam_data_with_flies.csv (empty channels removed)
    ↓
[Step 3: generate_health_report.py]
    ↓
health_report.csv (fly health classifications)
    ↓
[Step 4: remove_flies.py] ← Optional, based on health report
    ↓
dam_data_cleaned.csv (flies removed as needed)
    ↓
[Step 5: split_by_reading_type.py] ← Optional
    ↓
dam_data_MT.csv, dam_data_CT.csv, dam_data_Pn.csv
```

---

## Step-by-Step Guide

### **Step 1: Create Merged Database**

**Script:** `src/main/create_database.py`

**Command:**
```bash
cd Python/src/main
python3 create_database.py
```

**What it does:**
- Reads `Monitor5.txt` and `Monitor6.txt` (raw time-series data)
- Reads `details.txt` (fly metadata: genotype, sex, treatment)
- Merges into long-format database
- Creates unique `fly_id` for each fly (format: `M{monitor}_Ch{channel:02d}`)

**Input:**
- `../../Monitor5.txt`
- `../../Monitor6.txt`
- `../../details.txt`

**Output:**
- `../../data/processed/dam_data_merged.csv`
  - Columns: `datetime, monitor, channel, reading, value, fly_id, genotype, sex, treatment`
  - ~1.6 million rows (all flies, all readings)

---

### **Step 2: Filter Empty Channels**

**Script:** `src/main/filter_empty_channels.py`

**Command:**
```bash
python3 filter_empty_channels.py
```

**What it does:**
- Removes channels without flies (typically channels 31-32)
- Keeps only channels with valid metadata

**Input:**
- `../../data/processed/dam_data_merged.csv`

**Output:**
- `../../data/processed/dam_data_with_flies.csv`
  - ~1.5 million rows (empty channels removed)

---

### **Step 3: Generate Health Report**

**Script:** `src/main/generate_health_report.py`

**Command:**
```bash
python3 generate_health_report.py
```

**What it does:**
- Analyzes fly health using MT (movement) data
- Calculates daily metrics: activity, sleep, inactivity periods
- Tests startle response at light transitions
- Normalizes activity to reference day (day 4)
- Classifies each fly as: **Alive**, **Unhealthy**, **Dead**, or **QC_Fail**
- Applies irreversible death rule (once dead, always dead)

**Input:**
- `../../data/processed/dam_data_MT.csv` (preferred, MT-only)
- OR `../../data/processed/dam_data_with_flies.csv` (all readings)
- OR `../../data/processed/dam_data_merged.csv`

**Output:**
- `../../data/processed/health_report.csv`
  - Columns: `Monitor, Channel, Genotype, Sex, Treatment, DAYS_ANALYZED, DAYS_ALIVE, DAYS_UNHEALTHY, DAYS_DEAD, DAYS_QC_FAIL, FIRST_UNHEALTHY_DAY, FIRST_DEAD_DAY, LAST_ALIVE_DAY, FINAL_STATUS`
  - One row per fly with comprehensive health summary

**Health Classification Criteria:**
- **Dead**: 24+ hours inactivity, OR 12+ hours + no startle, OR activity < 20% of baseline
- **Unhealthy**: Low activity + no startle, OR activity < 50% of baseline, OR excessive sleep
- **QC_Fail**: >10% missing data
- **Alive**: None of the above

---

### **Step 4: Remove Flies (Optional)**

**Script:** `src/main/remove_flies.py`

**Command:**
```bash
# Remove specific flies by ID
python3 remove_flies.py --flies "6-ch23,6-ch5"

# Remove flies by metadata
python3 remove_flies.py --genotypes "bad_genotype" --sexes "Male"

# Remove specific days for certain flies
python3 remove_flies.py --per-fly-remove '{"5-ch7": [1, 2], "6-ch18": [3]}'

# Combine multiple criteria
python3 remove_flies.py --flies "6-ch23" --genotypes "mutant1"
```

**What it does:**
- Removes flies based on:
  1. Specific fly IDs (Monitor-Channel combinations)
  2. Specific days for certain flies
  3. Metadata criteria (genotype, sex, treatment)
- Case-insensitive matching
- Preserves all other data

**Input:**
- `../../data/processed/dam_data_with_flies.csv` (default)
- OR any CSV file with monitor, channel, genotype, sex, treatment columns

**Output:**
- `../../data/processed/dam_data_cleaned.csv`
  - Same structure as input, with specified flies/days removed

**Typical Workflow:**
1. Review `health_report.csv`
2. Identify flies to remove (e.g., all with `FINAL_STATUS == "Dead"` or `"QC_Fail"`)
3. Use `remove_flies.py` to remove them

---

### **Step 5: Split by Reading Type (Optional)**

**Script:** `src/main/split_by_reading_type.py`

**Command:**
```bash
python3 split_by_reading_type.py
```

**What it does:**
- Splits merged data into 3 separate files by reading type
- Removes redundant `reading` column from each file
- Creates summary documentation

**Input:**
- `../../data/processed/dam_data_with_flies.csv` (preferred)
- OR `../../data/processed/dam_data_cleaned.csv` (if flies were removed)
- OR `../../data/processed/dam_data_merged.csv`

**Output:**
- `../../data/processed/dam_data_MT.csv` (~500K rows, movement data)
- `../../data/processed/dam_data_CT.csv` (~500K rows, cumulative totals)
- `../../data/processed/dam_data_Pn.csv` (~500K rows, pause data)
- `../../data/processed/SPLIT_SUMMARY.txt` (documentation)

**Benefits:**
- Smaller files (~25 MB each vs ~45 MB combined)
- Faster loading (only load what you need)
- Clearer analysis (each file has one measurement type)

---

## Quick Start Commands

**Complete pipeline:**
```bash
cd Python/src/main

# Step 1: Create database
python3 create_database.py

# Step 2: Filter empty channels
python3 filter_empty_channels.py

# Step 3: Generate health report
python3 generate_health_report.py

# Step 4: Review health_report.csv, then remove problematic flies
python3 remove_flies.py --flies "6-ch23,6-ch5"  # Example

# Step 5: Split by reading type (optional)
python3 split_by_reading_type.py
```

**Minimal pipeline (no removal):**
```bash
python3 create_database.py
python3 filter_empty_channels.py
python3 generate_health_report.py
python3 split_by_reading_type.py  # Works with dam_data_with_flies.csv
```

---

## Key Features

### **No Dead Fly Marking Step**
- The old `mark_dead_flies.py` step has been removed
- Health report provides comprehensive analysis instead
- Use `remove_flies.py` to remove flies based on health report findings

### **Flexible Removal**
- Remove by specific fly IDs
- Remove by metadata (genotype, sex, treatment)
- Remove specific days for certain flies
- Combine multiple criteria

### **Comprehensive Health Analysis**
- Multiple health metrics (activity, sleep, startle response)
- Reference day normalization
- Daily status tracking
- Irreversible death rule

### **Modular Design**
- Each step can run independently
- Optional steps clearly marked
- Works with multiple input file formats

---

## File Structure

```
data/processed/
├── dam_data_merged.csv          # Step 1 output
├── dam_data_with_flies.csv      # Step 2 output
├── health_report.csv             # Step 3 output (analysis)
├── dam_data_cleaned.csv          # Step 4 output (optional)
├── dam_data_MT.csv               # Step 5 output (optional)
├── dam_data_CT.csv               # Step 5 output (optional)
├── dam_data_Pn.csv               # Step 5 output (optional)
└── SPLIT_SUMMARY.txt             # Step 5 documentation
```

---

## Analysis Workflow

1. **Run pipeline** → Get `health_report.csv`
2. **Review health report** → Identify problematic flies
3. **Remove flies** → Use `remove_flies.py` with appropriate criteria
4. **Analyze clean data** → Use `dam_data_cleaned.csv` or split files

---

## Health Report Interpretation

**FINAL_STATUS values:**
- **Alive**: Fly survived entire experiment with normal activity
- **Unhealthy**: Fly showed signs of decline but survived
- **Dead**: Fly died during experiment (behavioral death)
- **QC_Fail**: Data quality issues (>10% missing)

**Key columns:**
- `DAYS_ALIVE`, `DAYS_UNHEALTHY`, `DAYS_DEAD`: Counts per category
- `FIRST_UNHEALTHY_DAY`, `FIRST_DEAD_DAY`: When problems started
- `LAST_ALIVE_DAY`: Last day fly was classified as alive

---

## Notes

- **No LIKELY_DEAD column**: The pipeline no longer uses this flag. Use health report + removal instead.
- **Health report is analysis-only**: It doesn't modify the data, just provides classifications.
- **Removal is manual**: Review health report and decide which flies to remove based on your research needs.
- **All steps are optional after Step 2**: You can stop at any point depending on your needs.

---

## Questions?

The pipeline is designed to be:
- **Flexible**: Choose which steps to run
- **Transparent**: Health report shows exactly why flies are classified
- **Reversible**: Original data preserved, removal creates new file
- **Comprehensive**: Health analysis uses multiple behavioral indicators

