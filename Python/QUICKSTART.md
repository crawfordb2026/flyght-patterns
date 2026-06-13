# Quick Start Guide

This guide walks you through the flyght-patterns pipeline from raw data to ML-ready features and analysis. No database required if you use the CSV pipeline.

---

## Which pipeline should I use?

| | **CSV Pipeline** (`csv_pipeline`) | **SQL Pipeline** (`sql_db_pipeline`) |
|---|---|---|
| **Setup** | Just Python + files | Requires PostgreSQL database |
| **Best for** | Getting started, small datasets | Large datasets, team sharing |
| **Analysis scripts** | Yes (with `--csv-input`) | Yes (default) |

**If you are new here, start with the CSV pipeline.**

---

## Before you start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your data files

```
Python/
├── Monitors_raw/       ← Put your Monitor*.txt files here
└── metadata.txt         ← Fly metadata (see format below)
```

**`metadata.txt` format** (tab or space separated):

```
Monitor  Channel  Genotype   Sex     Treatment
51       ch01     WT         Male    VEH
51       ch02     sleep_mut  Female  VEH
```

---

## CSV Pipeline (csv_pipeline)

Run all scripts from the `Python/src/csv_pipeline/` directory.

### Step 0 — Filter by date (optional)

Only needed if your raw monitor files span multiple experiments.

```bash
cd Python/src/csv_pipeline
python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 7 --offset 0
```

- `--input`: Monitor name (without `.txt`)
- `--load`: Date your flies were loaded (MM/DD/YY)
- `--days`: How many days to include
- `--offset`: Days to skip before starting (0 = start on load date)
- **Output:** `Python/Monitors_date_filtered/Monitor51_06_20_25.txt`

### Step 1 — Prepare data and health report

```bash
python 1-prepare_data_and_health.py
```

- **Input:** Files in `Monitors_date_filtered/` + `Python/metadata.txt`
- **Output:**
  - `data/processed/dam_data_prepared.csv` — all readings (MT, CT, Pn) per fly per minute
  - `data/processed/health_report.csv` — daily health status (Alive/Unhealthy/Dead/QC_Fail) per fly

### Step 2 — Remove flies (optional)

Remove dead, unhealthy, or specific flies before feature extraction.

```bash
python 2-remove_flies.py --statuses Dead QC_Fail
```

- **Output:** `data/processed/dam_data_cleaned.csv`
- Skip this step to keep all flies.

### Step 3 — Create feature table

```bash
python 3-create_feature_table.py
```

- **Input:** `dam_data_cleaned.csv` (or `dam_data_prepared.csv` if Step 2 was skipped)
- **Output:** `data/processed/ML_features.csv` — one row per fly with ~40 circadian + sleep features

### Step 4 — Clean and normalize

```bash
python 4-clean_ml_features.py
```

- **Input:** `ML_features.csv`
- **Output:**
  - `data/processed/ML_features_clean.csv` — cleaned, original scale
  - `data/processed/ML_features_Z.csv` — z-scored, ready for ML

---

## SQL Pipeline (sql_db_pipeline)

Requires PostgreSQL. See `sql_db_pipeline.md` for database setup instructions.

Run all scripts from the `Python/src/sql_db_pipeline/` directory.

### Setup database (first time only)

```bash
cd Python/src/sql_db_pipeline
python setup_database.py
```

Set these environment variables (or edit `config.py`):

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fly_ml_db
DB_USER=your_username
DB_PASSWORD=your_password
```

### Steps 0–4

The scripts are identical in usage to the CSV pipeline, but store data in PostgreSQL instead of CSV files.

```bash
python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 7 --offset 0
python 1-prepare_data_and_health.py
python 2-remove_flies.py --statuses Dead QC_Fail   # optional
python 3-create_feature_table.py
python 4-clean_ml_features.py
```

---

## Running Analysis Scripts

All analysis scripts are in `Python/src/sql_db_pipeline/analysis/`.

### With the CSV pipeline (no database)

Pass `--csv-input` pointing to your `ML_features_Z.csv`:

```bash
cd Python/src/sql_db_pipeline/analysis

# 1. PCA
python pca_analysis.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv

# 2. UMAP clustering (requires pca_scores.csv from step above)
python umap_dbscan_analysis.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv

# 3. Random Forest
python random-forest.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv

# 4. Sex differences (requires umap_clusters.csv from step above)
python sexdiff_analysis.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv

# 5. Cluster characterization (requires umap_clusters.csv)
python cluster_characterization.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv

# 6. RF vs PCA loadings comparison (reads from analysis_results/ automatically)
python rf_vs_pca_loadings.py
```

### With the SQL pipeline

```bash
cd Python/src/sql_db_pipeline/analysis

python pca_analysis.py
python umap_dbscan_analysis.py
python random-forest.py
python sexdiff_analysis.py
python cluster_characterization.py
python rf_vs_pca_loadings.py
```

> **Note:** `death_prediction_xgboost.py` requires the SQL pipeline (it uses rolling time-series features not available in the CSV pipeline).

### Analysis outputs

All outputs go to `analysis_results/` inside the `analysis/` folder:

```
analysis_results/
├── pca/               ← PCA scores, loadings, genotype heatmaps
├── umap/              ← UMAP coordinates, cluster assignments
├── random_forest/     ← Feature importances, confusion matrix
├── sexdiff/           ← Sex difference results per cluster
├── cluster_characterization/  ← Per-cluster feature profiles
└── RFvsPCAloadings/   ← RF vs PCA feature comparison
```

---

## Recommended order (CSV pipeline, full run)

```bash
cd Python/src/csv_pipeline

python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 7
python 1-prepare_data_and_health.py
python 2-remove_flies.py --statuses Dead
python 3-create_feature_table.py
python 4-clean_ml_features.py

cd ../sql_db_pipeline/analysis
CSV=../../csv_pipeline/data/processed/ML_features_Z.csv

python pca_analysis.py --csv-input $CSV
python umap_dbscan_analysis.py --csv-input $CSV
python random-forest.py --csv-input $CSV
python sexdiff_analysis.py --csv-input $CSV
python cluster_characterization.py --csv-input $CSV
python rf_vs_pca_loadings.py
```

---

## Common issues

**"Input file not found"** — Make sure you ran the previous step, or check the `data/processed/` directory.

**"No vehicle flies found"** — Analysis scripts default to VEH (vehicle) treatment. Make sure your `metadata.txt` has a `Treatment` column with `VEH` values, or your treatment name contains "VEH".

**Missing `pca_scores.csv` when running UMAP** — Run `pca_analysis.py` first; UMAP reads its output.

**Missing `umap_clusters.csv` when running sexdiff or cluster_characterization** — Run `umap_dbscan_analysis.py` first.
