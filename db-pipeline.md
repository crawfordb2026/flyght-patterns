Visual overview of how the system works:

## System overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR COMPUTER                                 │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  FILESYSTEM (Raw Data)                                   │   │
│  │  ├── Monitor1.txt  (huge file, ~800K rows)              │   │
│  │  ├── Monitor2.txt                                        │   │
│  │  ├── ...                                                  │   │
│  │  ├── Monitor30.txt                                       │   │
│  │  └── metadata.txt  (small file, ~960 rows)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  PYTHON PIPELINE (Pandas Processing)                      │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │ Step 1: prepare_data_and_health.py                 │ │ │
│  │  │ • Parse Monitor*.txt files                          │ │ │
│  │  │ • Convert to long format (MT, CT, Pn)              │ │ │
│  │  │ • Join with metadata                                │ │ │
│  │  │ • Generate health reports                           │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  │                    │                                       │ │
│  │                    ▼                                       │ │
│  │            [Insert to TimescaleDB]                       │ │
│  └────────────────────┬──────────────────────────────────────┘ │
│                       │                                        │
│                       ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  TIMESCALEDB DATABASE                                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │  readings    │  │ health_reports│ │  features    │    │ │
│  │  │  (24M rows)  │  │  (few rows)   │ │  (960 rows)  │    │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │ │
│  └────────────────────┬──────────────────────────────────────┘ │
│                       │                                        │
│                       ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  PYTHON ANALYSIS (Pandas + scikit-learn)                 │ │
│  │  • Load small features table                             │ │
│  │  • PCA, UMAP, DBSCAN                                     │ │
│  │  • Statistical tests                                     │ │
│  │  • Visualizations                                        │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data flow

### Step 1: Raw files → database (`1-prepare_data_and_health.py`)

```
┌─────────────┐
│ Monitor1.txt│  ───┐
│ (800K rows) │     │
└─────────────┘     │
                    │
┌─────────────┐     │
│ Monitor2.txt│  ───┤
│ (800K rows) │     │
└─────────────┘     │
                    │
      ...           │  [Pandas reads files sequentially]
                    │
┌─────────────┐     │
│MonitorN.txt │  ───┘
│ (800K rows) │
└─────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  PANDAS PROCESSING                      │
│  ┌──────────────────────────────────┐ │
│  │ Parse each file                   │ │
│  │ • Read tab-separated values       │ │
│  │ • Extract datetime, channels      │ │
│  │ • Convert to long format          │ │
│  │   (MT, CT, Pn as separate rows)   │ │
│  └──────────────────────────────────┘ │
│              │                         │
│              ▼                         │
│  ┌──────────────────────────────────┐ │
│  │ Join with metadata               │ │
│  │ • Add fly_id, genotype, etc.    │ │
│  └──────────────────────────────────┘ │
└──────────────┬─────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  TIMESCALEDB: readings table            │
│  ┌────────────────────────────────────┐ │
│  │ datetime  │ fly_id │ reading │ val │ │
│  │───────────┼────────┼─────────┼─────│ │
│  │ 2025-09-19│ M1_Ch01│   MT    │  15 │ │
│  │ 2025-09-19│ M1_Ch01│   CT    │  26 │ │
│  │ 2025-09-19│ M1_Ch01│   Pn    │  15 │ │
│  │ 2025-09-19│ M1_Ch02│   MT    │  12 │ │
│  │    ...    │  ...   │  ...    │ ... │ │
│  │           │        │         │     │ │
│  │       24 MILLION ROWS TOTAL        │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Step 2: Remove flies (optional) (`2-remove_flies.py`)

```
┌─────────────────────────────────────────┐
│  TIMESCALEDB: readings table            │
│  (24M rows, all flies, all timepoints)  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  PANDAS: Filter flies                   │
│  ┌──────────────────────────────────┐ │
│  │ Remove based on:                  │ │
│  │ • Health status (Dead, Unhealthy) │ │
│  │ • Specific fly IDs                │ │
│  │ • Metadata (genotype, sex, etc.)  │ │
│  │ • Per-fly day removal             │ │
│  └──────────────────────────────────┘ │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  TIMESCALEDB: readings table (filtered) │
│  (fewer rows, dead/unhealthy removed)   │
└─────────────────────────────────────────┘

Note: Health reports are generated in Step 1, not Step 2.
Step 2 is optional and uses health reports to filter flies.
```

### Step 3: Feature extraction (`3-create_feature_table.py`)

```
┌─────────────────────────────────────────┐
│  TIMESCALEDB: readings table            │
│  (filtered: dead flies removed)          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  PANDAS: Feature Extraction             │
│                                         │
│  ┌──────────────────────────────────┐ │
│  │ SLEEP FEATURES                   │ │
│  │ Group by: fly_id, day             │ │
│  │ Calculate:                        │ │
│  │ • Total sleep (minutes)            │ │
│  │ • Number of sleep bouts           │ │
│  │ • Mean bout length                │ │
│  │ • Sleep latency                   │ │
│  │ • WASO (wake after sleep onset)   │ │
│  │ • Fragmentation                   │ │
│  └──────────────────────────────────┘ │
│              │                          │
│  ┌──────────────────────────────────┐ │
│  │ CIRCADIAN FEATURES                │ │
│  │ Group by: fly_id, day             │ │
│  │ Calculate hourly totals            │ │
│  │ Fit cosinor regression:            │ │
│  │ • Mesor (baseline)                │ │
│  │ • Amplitude (rhythm strength)     │ │
│  │ • Phase (peak timing)              │ │
│  └──────────────────────────────────┘ │
│              │                          │
│              ▼                          │
│  ┌──────────────────────────────────┐ │
│  │ Aggregate to per-fly means       │ │
│  │ (average across all days)         │ │
│  └──────────────────────────────────┘ │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  TIMESCALEDB: features table            │
│  ┌────────────────────────────────────┐ │
│  │ fly_id │ total_sleep │ bouts │ ... │ │
│  │────────┼─────────────┼───────┼─────│ │
│  │M1_Ch01 │   450.2     │  12   │ ... │ │
│  │M1_Ch02 │   380.5     │  15   │ ... │ │
│  │  ...   │    ...      │  ...  │ ... │ │
│  │        │             │       │     │ │
│  │    960 rows (one per fly)          │ │
│  │    25+ feature columns             │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Step 4: Clean ML features (`4-clean_ml_features.py`)

```
┌─────────────────────────────────────────┐
│  TIMESCALEDB: features table            │
│  (960 rows, 25+ columns)                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  PANDAS: Clean & Normalize              │
│  ┌──────────────────────────────────┐ │
│  │ Remove problematic flies:          │ │
│  │ • Zero total sleep                 │ │
│  │ • Zero sleep bouts                 │ │
│  │ • Zero/NaN P_doze                  │ │
│  └──────────────────────────────────┘ │
│              │                          │
│  ┌──────────────────────────────────┐ │
│  │ Remove IQR outliers               │ │
│  │ • Per group (genotype/treatment)   │ │
│  │ • Based on total_sleep_mean        │ │
│  └──────────────────────────────────┘ │
│              │                          │
│  ┌──────────────────────────────────┐ │
│  │ Fix NaN values                    │ │
│  │ • Replace with 0 or group mean    │ │
│  └──────────────────────────────────┘ │
│              │                          │
│  ┌──────────────────────────────────┐ │
│  │ Z-score normalization              │ │
│  │ • Per feature, across all flies   │ │
│  └──────────────────────────────────┘ │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  TIMESCALEDB: features_z table          │
│  (z-scored, ready for ML)               │
│  ┌────────────────────────────────────┐ │
│  │ fly_id │ total_sleep_z │ bouts_z │...│ │
│  │────────┼───────────────┼──────────┼───│ │
│  │M1_Ch01 │    0.234      │  -0.156 │...│ │
│  │M1_Ch02 │   -0.567      │   0.891 │...│ │
│  │  ...   │     ...       │   ...   │...│ │
│  │        │               │         │   │ │
│  │    960 rows × 25+ features         │ │
│  └────────────────────────────────────┘ │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  PANDAS + SCIKIT-LEARN ANALYSIS         │
│  • Load features_z from database         │
│  • PCA, UMAP, DBSCAN                     │
│  • Statistical tests                     │
│  • Visualizations                        │
└─────────────────────────────────────────┘
```

## Database structure

The database schema is defined in `schema.sql` and can be set up using `setup_database.py`.

```
TIMESCALEDB DATABASE
│
├── experiments (small table)
│   └── experiment_id, name, start_date, end_date, 
│       lights_on_hour, lights_off_hour, created_at
│
├── flies (small table)
│   └── fly_id (PK), experiment_id (FK), monitor, channel,
│       genotype, sex, treatment
│
├── readings (HUGE TABLE - millions of rows, TimescaleDB hypertable)
│   │
│   ├── measurement_id, experiment_id (FK), fly_id (FK),
│   │   datetime, reading_type (MT/CT/Pn), value, monitor
│   │
│   ├── Partitioned by datetime (1 day chunks)
│   └── Indexes on (fly_id, datetime) and experiment_id
│
├── health_reports (medium table)
│   └── health_report_id, experiment_id (FK), fly_id (FK),
│       report_date, status (Alive/Unhealthy/Dead/QC_Fail),
│       total_activity, longest_zero_hours, rel_activity,
│       has_startle_response, missing_fraction
│
├── features (small table)
│   └── feature_id, experiment_id (FK), fly_id (FK),
│       mesor_mean, mesor_sd, amplitude_mean, amplitude_sd,
│       phase_mean, phase_sd, rhythmic_days,
│       total_sleep_mean, day_sleep_mean, night_sleep_mean,
│       total_bouts_mean, day_bouts_mean, night_bouts_mean,
│       mean_bout_mean, max_bout_mean, mean_day_bout_mean,
│       max_day_bout_mean, mean_night_bout_mean,
│       max_night_bout_mean, frag_bouts_per_hour_mean,
│       frag_bouts_per_min_sleep_mean, mean_wake_bout_mean,
│       p_wake_mean, p_doze_mean, sleep_latency_mean, waso_mean
│
└── features_z (small table)
    └── feature_id, experiment_id (FK), fly_id (FK),
        All features from 'features' table with '_z' suffix
        (z-scored versions for ML analysis)
```

## Pipeline file structure

```
Python/src/db-pipeline/
│
├── config.py                    # Database configuration
│   └── DB_CONFIG, DATABASE_URL, USE_DATABASE
│
├── setup_database.py            # Database setup script
│   └── Creates database and runs schema.sql
│
├── schema.sql                   # Database schema definition
│   └── All table definitions and indexes
│
├── 1-prepare_data_and_health.py # Step 1: Load data & health reports
│   └── Parses Monitor*.txt files, creates readings & health_reports
│
├── 2-remove_flies.py            # Step 2: Optional fly removal
│   └── Filters flies based on health status, metadata, etc.
│
├── 3-create_feature_table.py    # Step 3: Feature extraction
│   └── Extracts sleep and circadian features, creates features table
│
├── 4-clean_ml_features.py       # Step 4: Clean & normalize
│   └── Removes outliers, fixes NaN, creates z-scored features_z table
│
├── delete_experiment.py        # Utility: Delete experiment from database
│   └── Removes all data for a given experiment_id
│
└── analysis/                   # Analysis scripts
    ├── pca_analysis.py         # PCA analysis on features_z
    ├── umap_dbscan_analysis.py # UMAP and DBSCAN clustering
    └── sexdiff_analysis.py     # Sex difference analysis
```

## Usage

### 0. Install TimescaleDB (Required)
TimescaleDB is a PostgreSQL extension required for time-series data optimization. Install it before setting up the database:

**macOS (Homebrew):**
```bash
brew tap timescale/tap
brew install timescaledb
brew services restart postgresql@16  # Adjust version if needed
```

**Other platforms:** See [TimescaleDB installation guide](https://docs.timescale.com/install/latest/)

### 1. Setup database
```bash
cd Python/src/db-pipeline
python3 setup_database.py
```

### 2. Configure database connection
Set environment variables or modify `config.py`:
- `DB_HOST` (default: localhost)
- `DB_PORT` (default: 5432)
- `DB_NAME` (default: fly_ml_db)
- `DB_USER` (default: postgres)
- `DB_PASSWORD` (default: postgres)
- `USE_DATABASE` (default: true)

### 3. Run pipeline steps
```bash
# Step 1: Load raw data and generate health reports
python3 1-prepare_data_and_health.py

# Step 2: (Optional) Remove flies
python3 2-remove_flies.py --statuses "Dead,Unhealthy"
# Optional: --experiment-id specifies which experiment to use (default uses latest experiment)  
python3 2-remove_flies.py --experiment-id 1 --statuses "Dead"

# Step 3: Extract features
python3 3-create_feature_table.py

# Step 4: Clean and normalize features
python3 4-clean_ml_features.py
```

### 4. Utility scripts

**Delete an experiment:**
```bash
# List all experiments
python3 delete_experiment.py --list

# Delete an experiment (will prompt for confirmation)
python3 delete_experiment.py --experiment-id 1

# Delete without confirmation prompt
python3 delete_experiment.py --experiment-id 1 --confirm
```

## Memory usage

```
┌─────────────────────────────────────────────────────────────┐
│  PANDAS ZONE (Data Processing)                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Processing Monitor Files                               │ │
│  │ • Files processed sequentially                         │ │
│  │ • ~2-4 GB RAM per monitor file                         │ │
│  │ • Releases memory after each file                      │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Feature Extraction                                     │ │
│  │ • Processes data in chunks from database              │ │
│  │ • ~4-6 GB RAM peak                                     │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                              │
│  Total: ~8-12 GB RAM                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PANDAS ZONE (ML Analysis)                                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Features Table                                         │ │
│  │ • 960 rows × 25 columns                               │ │
│  │ • ~200 KB in memory                                   │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ PCA/UMAP Results                                       │ │
│  │ • 960 rows × 2-3 columns                              │ │
│  │ • ~50 KB in memory                                    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                              │
│  Total: ~500 MB RAM                                         │
└─────────────────────────────────────────────────────────────┘
```

## Complete flow

```
START
 │
 ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: prepare_data_and_health.py                        │
│  • Read Monitor*.txt files (sequential)                    │
│  • Parse with pandas                                        │
│  • Convert to long format (MT, CT, Pn)                      │
│  • Join with metadata (details.txt)                         │
│  • Generate health reports                                  │
│  • Insert into TimescaleDB:                                │
│    - experiments, flies, readings, health_reports           │
│  ⏱️ Time: ~20-30 minutes                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: remove_flies.py (OPTIONAL)                        │
│  • Query readings from database                            │
│  • Filter based on health status, metadata, etc.          │
│  • Updates readings table (removes filtered rows)          │
│  ⏱️ Time: ~1-2 minutes                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: create_feature_table.py                           │
│  • Query readings from database                            │
│  • Calculate sleep features (pandas)                       │
│  • Calculate circadian features (cosinor regression)       │
│  • Aggregate to per-fly means                             │
│  • Insert into TimescaleDB.features                        │
│  ⏱️ Time: ~10-15 minutes                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: clean_ml_features.py                              │
│  • Query features from database                            │
│  • Remove problematic flies (zero sleep, etc.)             │
│  • Remove IQR outliers                                      │
│  • Fix NaN values                                           │
│  • Z-score normalization                                    │
│  • Insert into TimescaleDB.features_z                      │
│  ⏱️ Time: ~1-2 minutes                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  MACHINE LEARNING (separate scripts)                        │
│  • Load features_z from database (pandas)                 │
│  • PCA analysis                                             │
│  • UMAP analysis                                            │
│  • DBSCAN clustering                                         │
│  • Statistical tests                                        │
│  • Generate visualizations                                  │
│  ⏱️ Time: ~1-5 minutes                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
                   END
```

## Why this design

1. **TimescaleDB** stores everything and handles time-series queries efficiently
   - Automatic partitioning by time (hypertables)
   - Fast queries on time-ranges and fly_ids
   - Handles 24M+ rows efficiently

2. **pandas** for data processing
   - Familiar API and extensive functionality
   - Good integration with database (SQLAlchemy)
   - Sufficient performance for this dataset size

3. **Database-first approach**
   - All data stored in one place (no CSV files)
   - Easy to query and filter
   - Supports multiple experiments
   - Can add new features without reprocessing raw data

4. **Modular pipeline**
   - Each step is independent
   - Can re-run individual steps
   - Easy to debug and modify

5. **Configuration via environment variables**
   - Easy to switch between databases
   - Can disable database mode for testing
   - Flexible deployment

## Key features

- **Health report generation**: Automatic detection of dead/unhealthy flies
- **Feature extraction**: Sleep and circadian rhythm features
- **Data cleaning**: Automatic outlier removal and normalization
- **Z-scoring**: Features normalized for ML analysis
- **Multiple experiments**: Support for multiple experiments in one database
- **Experiment management**: Delete experiments and all associated data

## Notes

- The pipeline uses **pandas** for all data processing
- Files are processed **sequentially** (not in parallel)
- Health reports are generated in **Step 1**, not Step 2
- Step 2 is **optional** and used to filter flies based on health status
- All data is stored in **TimescaleDB** (no intermediate CSV files)
- The `USE_DATABASE` flag allows running without database (for testing)
- TimescaleDB extension is optional - schema works with regular PostgreSQL