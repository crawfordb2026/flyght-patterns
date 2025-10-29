# Updated DAM Data Processing Pipeline

## Overview

This document describes the **updated pipeline** for processing Drosophila DAM data. The key change is that the pipeline now **marks** dead flies instead of **removing** them.

---

## Pipeline Flow

```
INPUT FILES
├── Monitor5.txt
├── Monitor6.txt
└── details.txt

    ↓ [Step 1: create_database.py]

dam_data_merged.csv
  - All flies, all readings
  - NEW: LIKELY_DEAD column (default=False)

    ↓ [Step 2: filter_empty_channels.py]

dam_data_with_flies.csv
  - Removed empty channels (channels 31-32)
  - Only flies with metadata
  - LIKELY_DEAD column preserved

    ↓ [Step 3: mark_dead_flies.py]

dam_data_marked.csv
  - SAME number of rows as input
  - LIKELY_DEAD=True for dead flies (24h+ inactive)
  - LIKELY_DEAD=False for alive flies
  - All data preserved

    ↓ [Step 4: split_by_reading_type.py] [OPTIONAL]

SPLIT FILES
├── dam_data_MT.csv      (LIKELY_DEAD column preserved)
├── dam_data_CT.csv      (LIKELY_DEAD column preserved)
└── dam_data_Pn.csv      (LIKELY_DEAD column preserved)
```

---

## Key Changes from Previous Pipeline

### 1. **No Data Removal**
- **Old**: `detect_dead_flies.py` removed all rows from dead flies
- **New**: `mark_dead_flies.py` marks rows but keeps all data
- **Benefit**: You can analyze dead flies separately or filter in analysis

### 2. **24-Hour Threshold**
- **Old**: 12 consecutive hours of inactivity = dead
- **New**: 24 consecutive hours of inactivity = dead
- **Benefit**: More conservative threshold reduces false positives

### 3. **LIKELY_DEAD Flag**
- **Old**: Dead flies completely removed from dataset
- **New**: `LIKELY_DEAD=True/False` flag marks dead flies
- **Benefit**: Flexible analysis - you can include or exclude dead flies

### 4. **New Empty Channel Filter**
- **New**: `filter_empty_channels.py` removes channels 31-32 (no flies)
- **Benefit**: Cleaner data without empty monitoring positions

---

## Step-by-Step Process

### **Step 1: Create Database** (MODIFIED)

**Script:** `src/create_database.py`

**Changes:**
- Added `LIKELY_DEAD` column (default=False) to output

**Command:**
```bash
python3 src/create_database.py
```

**Input:**
- `Monitor5.txt`
- `Monitor6.txt`
- `details.txt`

**Output:**
- `data/processed/dam_data_merged.csv`
- Columns: `datetime, monitor, channel, reading, value, fly_id, genotype, sex, treatment, LIKELY_DEAD`
- ~1.6 million rows

---

### **Step 2: Filter Empty Channels** (NEW)

**Script:** `src/filter_empty_channels.py`

**Purpose:** Remove channels that don't have actual flies (channels 31-32)

**Command:**
```bash
python3 src/filter_empty_channels.py
```

**Input:**
- `data/processed/dam_data_merged.csv`

**Output:**
- `data/processed/dam_data_with_flies.csv`
- Removed ~200,000 rows from empty channels
- Only channels with actual flies
- LIKELY_DEAD column preserved

**Summary:**
```
Removed X rows from Y empty channels.
Remaining: Z rows from W channels.
```

---

### **Step 3: Mark Dead Flies** (MODIFIED)

**Script:** `src/mark_dead_flies.py` (renamed from `detect_dead_flies.py`)

**Changes:**
- 24-hour threshold (instead of 12 hours)
- NO row removal
- Sets `LIKELY_DEAD=True` for dead flies
- Only marks AFTER the 24-hour inactivity period starts

**Command:**
```bash
python3 src/mark_dead_flies.py
```

**Input:**
- `data/processed/dam_data_with_flies.csv`

**Output:**
- `data/processed/dam_data_marked.csv`
- SAME number of rows as input
- `LIKELY_DEAD=True` for dead flies
- `LIKELY_DEAD=False` for alive flies
- All data preserved

**Logic:**
1. For each fly, detect if it has 24+ consecutive hours of MT=0
2. If yes, mark that fly's `LIKELY_DEAD=True` from the death timestamp forward
3. Keep earlier time points as `LIKELY_DEAD=False`

**Summary:**
```
Total flies: 64
Marked as likely dead: X
Flies remaining alive: Y
```

---

### **Step 4: Split by Reading Type** (UPDATED)

**Script:** `src/split_by_reading_type.py`

**Changes:**
- Now uses `dam_data_marked.csv` as input (instead of `dam_data_filtered.csv`)
- Preserves `LIKELY_DEAD` column in all output files

**Command:**
```bash
python3 src/split_by_reading_type.py
```

**Input:**
- `data/processed/dam_data_marked.csv`

**Output:**
- `data/processed/dam_data_MT.csv`
- `data/processed/dam_data_CT.csv`
- `data/processed/dam_data_Pn.csv`
- All include `LIKELY_DEAD` column

---

## Migration Guide

### **From Old to New Pipeline**

**Old commands:**
```bash
python3 create_database.py
python3 detect_dead_flies.py
python3 split_by_reading_type.py
```

**New commands:**
```bash
python3 create_database.py          # Creates dam_data_merged.csv
python3 filter_empty_channels.py   # Creates dam_data_with_flies.csv
python3 mark_dead_flies.py         # Creates dam_data_marked.csv
python3 split_by_reading_type.py   # Splits dam_data_marked.csv
```

---

## Analysis Options

### **Option 1: Include All Flies**
```python
data = pd.read_csv('dam_data_marked.csv')
# Analysis includes both alive and dead flies
```

### **Option 2: Exclude Dead Flies**
```python
data = pd.read_csv('dam_data_marked.csv')
alive_data = data[data['LIKELY_DEAD'] == False]
# Analysis only on flies that survived
```

### **Option 3: Analyze Dead Flies Separately**
```python
data = pd.read_csv('dam_data_marked.csv')
dead_data = data[data['LIKELY_DEAD'] == True]
alive_data = data[data['LIKELY_DEAD'] == False]
# Compare dead vs alive flies
```

---

## Key Benefits

✅ **Flexible Analysis**: Choose to include or exclude dead flies  
✅ **No Data Loss**: All rows preserved  
✅ **More Conservative**: 24-hour threshold reduces false positives  
✅ **Empty Channels Removed**: Cleaner dataset  
✅ **Backward Compatible**: Can filter to match old pipeline  

---

## Summary of Files Created

```
data/processed/
├── dam_data_merged.csv           (Step 1 output)
├── dam_data_with_flies.csv       (Step 2 output)
├── dam_data_marked.csv            (Step 3 output - USE THIS!)
├── dam_data_MT.csv                (Step 4 output - optional)
├── dam_data_CT.csv                (Step 4 output - optional)
└── dam_data_Pn.csv                (Step 4 output - optional)
```

**Main file for analysis:** `dam_data_marked.csv`

---

## Questions?

The updated pipeline gives you more flexibility:
- Keep all data for complete analysis
- Filter by `LIKELY_DEAD` flag as needed
- Analyze dead flies separately
- Replicate old pipeline results by filtering

All scripts include detailed documentation and error checking.

