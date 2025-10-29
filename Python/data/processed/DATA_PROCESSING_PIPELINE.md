# Data Processing Pipeline

## Complete Step-by-Step Process: From Raw Files to Filtered Data

This document explains how we go from the original monitor files and metadata to a clean dataset with no dead flies.

---

## **Overview of Input Files**

### **Raw Data Files:**
1. **`Monitor5.txt`** - Time-series measurements for Monitor 5 (channels 1-32)
2. **`Monitor6.txt`** - Time-series measurements for Monitor 6 (channels 1-32)
3. **`details.txt`** - Fly metadata (genotype, sex, treatment for each channel)

### **What Each File Contains:**

**Monitor5.txt & Monitor6.txt:**
- Tab-separated values with columns:
  - ID, date, time, port, movement_type, [32 channel values]
- Three movement types: MT, CT, Pn
- Measurements every 1 minute
- 32 channels per monitor (channels 1-32)
- Total: ~24,858 rows × 3 movement types = ~74,574 data points per monitor

**details.txt:**
- Tab-separated metadata with columns:
  - Monitor, Channel, Genotype, Sex, Treatment
- One row per fly
- Contains information for 64 flies (32 per monitor)

---

## **STEP 1: Create Merged Database**

**Script:** `src/create_database.py`

**Command:**
```bash
cd Python/src
python3 create_database.py
```

**What it does:**
1. Reads `details.txt` to extract fly metadata
2. Reads `Monitor5.txt` and `Monitor6.txt` to extract time-series data
3. Creates **LONG format** data where:
   - Each timestamp has **3 rows per channel** (MT, CT, Pn)
   - `reading` column indicates the measurement type
   - `value` column contains the measurement
   - Metadata is included in every row

**Output:**
- **`data/processed/dam_data_merged.csv`**
  - ~1.6 million rows
  - Columns: `datetime, monitor, channel, reading, value, fly_id, genotype, sex, treatment`
  - Contains ALL flies, ALL readings, ALL metadata

**Example row:**
```
2025-09-19 11:46:00, 5, 1, MT, 15, M5_Ch01, SSS, Female, 2mM His
2025-09-19 11:46:00, 5, 1, CT, 26, M5_Ch01, SSS, Female, 2mM His
2025-09-19 11:46:00, 5, 1, Pn, 15, M5_Ch01, SSS, Female, 2mM His
```

**Key Features:**
- Converts wide format (32 columns) to long format (1 value column)
- Merges metadata with time-series data
- Creates unique `fly_id` for each fly: `M{monitor}_Ch{channel:02d}`
- Removes empty channels (channels with NA genotype)

---

## **STEP 2: Detect and Filter Dead Flies**

**Script:** `src/detect_dead_flies.py`

**Command:**
```bash
python3 detect_dead_flies.py
```

**What it does:**
1. **Detects dead flies** using the definition:
   - Dead = 12+ consecutive hours (720 minutes) of MT=0
   - Uses only MT (movement) readings for detection
   - More reliable than CT or Pn readings

2. **Creates death report:**
   - Saves `data/processed/dead_flies_report.csv`
   - Contains: fly_id, genotype, sex, treatment, time_of_death, hours_survived

3. **Filters out ALL data from dead flies:**
   - Removes completely from dataset (both before and after death)
   - Keeps only data from flies that survived the entire experiment

**Output:**
- **`data/processed/dead_flies_report.csv`**
  - Death information for flies that died
  - Example: 3 flies died during experiment

- **`data/processed/dam_data_filtered.csv`**
  - Clean data with NO dead flies
  - ~1.5 million rows (down from 1.6 million)
  - Only contains flies that survived entire experiment
  - 61 flies total (64 - 3 dead = 61 alive)

**Example of detected dead fly:**
```
fly_id: M6_Ch05
genotype: Rye
sex: Female
treatment: 8mM His
time_of_death: 2025-09-23 12:26:00
hours_survived: 96.7
```

**Verification:**
- All declared dead flies have legitimate 12+ hour periods of MT=0
- No false positives
- All data from dead flies removed completely

---

## **STEP 3: Split by Reading Type** (OPTIONAL)

**Script:** `src/split_by_reading_type.py`

**Command:**
```bash
python3 split_by_reading_type.py
```

**What it does:**
1. Reads `dam_data_filtered.csv`
2. Splits into 3 separate files by reading type:
   - **`dam_data_MT.csv`** - Movement readings only
   - **`dam_data_CT.csv`** - Cumulative total readings only
   - **`dam_data_Pn.csv`** - Pause readings only

3. Removes redundant `reading` column from each file
4. Creates `SPLIT_SUMMARY.txt` with documentation

**Output:**
- **`data/processed/dam_data_MT.csv`** (~505K rows)
- **`data/processed/dam_data_CT.csv`** (~505K rows)
- **`data/processed/dam_data_Pn.csv`** (~505K rows)

**Why split?**
- Smaller files: ~25 MB each vs ~45 MB combined
- Faster loading: only load what you need
- Clearer analysis: each file has one measurement type
- Less confusion: no need to filter by reading type

**Usage:**
- Sleep analysis: Use `dam_data_MT.csv` (movement is key)
- Activity tracking: Use `dam_data_CT.csv` (cumulative totals)
- Pause behavior: Use `dam_data_Pn.csv` (inactivity patterns)

---

## **STEP 4: Bin to Time Intervals** (OPTIONAL)

**Script:** `src/bin_data.py`

**Command:**
```bash
# 5-minute bins (most common)
python3 bin_data.py ../data/processed/dam_data_MT.csv 5

# 30-minute bins (for longer-term analysis)
python3 bin_data.py ../data/processed/dam_data_CT.csv 30

# 60-minute bins (hourly data)
python3 bin_data.py ../data/processed/dam_data_Pn.csv 60
```

**What it does:**
1. Sums values within each time interval
2. Reduces data size significantly
3. Preserves all flies and metadata
4. Detects reading type from filename automatically

**Output:**
- **`data/processed/dam_data_MT_5min.csv`** (if binned from MT file)
- **`data/processed/dam_data_CT_30min.csv`** (if binned from CT file)
- **`data/processed/dam_data_Pn_60min.csv`** (if binned from Pn file)

**Example binning:**
- **Before (1-minute):**
  - 11:46:00, value=15
  - 11:47:00, value=15
  - 11:48:00, value=15
  - 11:49:00, value=15
  - Total: 60

- **After (5-minute bin):**
  - 11:45:00, value=60

**Benefits:**
- 80% reduction in data size
- Faster analysis
- Summary statistics by interval
- Still contains all metadata

**Verification:**
- All flies present
- No data loss (totals match)
- Correct binning

---

## **Complete Workflow Summary**

```
INPUT FILES
├── Monitor5.txt          (~74K rows, raw measurements)
├── Monitor6.txt          (~74K rows, raw measurements)
└── details.txt           (64 rows, fly metadata)

    ↓ (Step 1: create_database.py)

INTERMEDIATE OUTPUT
└── dam_data_merged.csv  (1.6M rows, all flies, all readings)

    ↓ (Step 2: detect_dead_flies.py)

MAIN OUTPUT
├── dead_flies_report.csv        (3 rows, death information)
└── dam_data_filtered.csv        (1.5M rows, alive flies only)
                                  61 flies, no dead flies

    ↓ (Step 3: split_by_reading_type.py) [OPTIONAL]

SPLIT FILES
├── dam_data_MT.csv      (505K rows, movement only)
├── dam_data_CT.csv      (505K rows, cumulative only)
└── dam_data_Pn.csv      (505K rows, pause only)

    ↓ (Step 4: bin_data.py) [OPTIONAL]

BINNED FILES
└── dam_data_MT_5min.csv (101K rows, 5-minute intervals)
```

---

## **Quick Start Commands**

```bash
# Step 1: Create merged database
cd Python/src
python3 create_database.py

# Step 2: Filter dead flies
python3 detect_dead_flies.py

# Step 3: Split by reading type (optional)
python3 split_by_reading_type.py

# Step 4: Bin to time intervals (optional)
python3 bin_data.py ../data/processed/dam_data_MT.csv 5
python3 bin_data.py ../data/processed/dam_data_CT.csv 5
python3 bin_data.py ../data/processed/dam_data_Pn.csv 5
```

---

## **Final Data Structure**

After all steps, you have clean, ready-to-analyze data:

**Main file:**
- **`dam_data_filtered.csv`** - Complete dataset, no dead flies, all reading types

**Split files (optional):**
- **`dam_data_MT.csv`** - Movement only
- **`dam_data_CT.csv`** - Cumulative only
- **`dam_data_Pn.csv`** - Pause only

**Binned files (optional):**
- **`dam_data_MT_5min.csv`** - Movement, 5-minute bins
- **`dam_data_CT_5min.csv`** - Cumulative, 5-minute bins
- **`dam_data_Pn_5min.csv`** - Pause, 5-minute bins

**Metadata columns in all files:**
- `datetime` - Timestamp
- `monitor` - Monitor number (5 or 6)
- `channel` - Channel number (1-32)
- `value` - Measurement value
- `fly_id` - Unique fly identifier (M5_Ch01, etc.)
- `genotype` - Fly genotype (SSS, Rye, Fmn, Iso)
- `sex` - Fly sex (Female)
- `treatment` - Treatment condition (2mM His, 8mM His, VEH)

---

## **Key Points**

✅ **Step 1** (Required): Creates the merged database  
✅ **Step 2** (Required): Removes dead flies  
⭕ **Step 3** (Optional): Splits by reading type for convenience  
⭕ **Step 4** (Optional): Bins to time intervals for faster analysis  

**Minimum workflow:** Steps 1 and 2 (creates `dam_data_filtered.csv`)  
**Recommended workflow:** Steps 1, 2, and 3 (creates split files)  
**Full workflow:** Steps 1, 2, 3, and 4 (includes binned files)

---

## **Verification Scripts**

Created for testing and debugging:
- `verify_dead_flies.py` - Verifies dead fly detection is accurate
- `verify_split_files.py` - Verifies correct reading types in split files
- `debug_missing_flies.py` - Debugs missing flies during binning
- `debug_binning.py` - Debugs binning process

---

## **Questions?**

The data processing pipeline is designed to:
1. Preserve all biological information
2. Remove unreliable data (dead flies)
3. Provide flexible analysis options
4. Maintain data integrity throughout
5. Enable reproducible science

If you have questions about any step, refer to the individual script docstrings or ask!

