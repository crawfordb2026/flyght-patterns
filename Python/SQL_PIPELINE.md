# SQL Pipeline — Step-by-Step Guide

This guide walks you through the full SQL/database pipeline: from raw monitor files to finished analysis plots, with all data stored in a PostgreSQL database.

**What this pipeline does:** The same thing as the CSV pipeline, but it stores all data in a proper database rather than CSV files. This makes it better suited for large datasets, multiple experiments, or sharing data across a team. It also enables the death prediction analysis, which requires the rolling time-series data stored in the database.

**Not sure if you should use this?** See `QUICKSTART.md` in this folder to compare your options.

---

## Before You Begin

### What you need

- **Python 3.8 or higher** — [Download here](https://www.python.org/downloads/)
  - On Windows: when installing, check the box that says "Add Python to PATH"
  - To check your version: open a terminal and run `python --version`
- **PostgreSQL 13 or higher** — a free, open-source database system. [Download here](https://www.postgresql.org/download/)
  - What is PostgreSQL? It's a database program that runs on your computer and stores data in a structured, queryable way. Think of it like a very powerful spreadsheet engine running in the background.
- **A terminal** — On Windows this is Command Prompt or PowerShell. On Mac it's Terminal.
- **Your monitor files** and **metadata file** (details below)

---

## Part 1 — Install and Set Up PostgreSQL

### Install PostgreSQL

**macOS (using Homebrew):**
```bash
brew install postgresql@16
brew services start postgresql@16
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Windows:**
Download the installer from [postgresql.org/download/windows](https://www.postgresql.org/download/windows/). Run it, follow the prompts, and note down the password you set for the `postgres` user — you'll need it later.

### Install Python dependencies

Open a terminal, navigate to the flyght-patterns folder, and run:

```bash
pip install -r requirements.txt
```

---

## Part 2 — Set Up the Database

### 1. Configure your database connection

Open `Python/src/sql_db_pipeline/config.py` in any text editor. You'll see settings like this:

```python
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "fly_ml_db"
DB_USER = "postgres"
DB_PASSWORD = "yourpassword"
```

Change `DB_PASSWORD` to the password you set when installing PostgreSQL. The other defaults should work for a local installation.

Alternatively, you can set environment variables instead of editing the file:

```bash
# Mac/Linux (add to your terminal session):
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=fly_ml_db
export DB_USER=postgres
export DB_PASSWORD=yourpassword

# Windows (PowerShell):
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_NAME = "fly_ml_db"
$env:DB_USER = "postgres"
$env:DB_PASSWORD = "yourpassword"
```

### 2. Create the database

Navigate to the sql_db_pipeline folder and run the setup script:

```bash
cd Python/src/sql_db_pipeline
python setup_database.py
```

**What you should see:** A message confirming that the database `fly_ml_db` was created and the schema was applied. If you see a connection error, double-check that PostgreSQL is running and your password in `config.py` is correct.

---

## Part 3 — Set Up Your Data Files

### 1. Place your monitor files

Copy your raw `Monitor*.txt` files into this folder:

```
Python/Monitors_raw/
```

It should look like this:
```
Python/
└── Monitors_raw/
    ├── Monitor51.txt
    ├── Monitor52.txt
    └── Monitor53.txt
```

### 2. Create your metadata file

Create a plain text file at `Python/metadata.txt`. This tells the pipeline which fly is in which monitor channel, along with each fly's genotype, sex, and treatment.

**Format** (columns separated by tabs or spaces):

```
Monitor        Channel  Genotype   Sex     Treatment
51_1_5_26       ch01     WT         Male    VEH
51_1_5_26       ch02     WT         Female  VEH
51_1_5_26       ch03     sleep_mut  Male    VEH
52_1_5_26       ch01     WT         Male    VEH
```

**Important notes:**
- `Monitor` is just the monitor number (ex. 51) and the load date (1_5_26)
- `Channel` is the channel identifier exactly as it appears in your monitor file (e.g., `ch01`, `ch02`)
- `Treatment` should be `VEH` for vehicle/control flies (the analysis scripts look for this by default)
- Every fly you want to analyze needs a row here

---

## Part 4 — Run the Pipeline

All scripts live in `Python/src/sql_db_pipeline/`. Navigate there first:

```bash
cd Python/src/sql_db_pipeline
```

Then run the steps below in order.

---

### Step 0 — Filter by Date *(optional)*

**When to use this:** Only needed if your monitor files span a longer time period than your experiment, or if you have multiple experiments in the same files. If your files already contain only the data you want, skip this step.

```bash
python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 7 --offset 0
```

**What each option means:**
- `--input Monitor51` — the monitor file to filter (without `.txt`)
- `--load "06/20/25"` — the date your flies were loaded into the monitor (MM/DD/YY format)
- `--days 7` — how many days of data to include
- `--offset 0` — how many days after the load date to start (0 = start on the load date itself)

Run this once for each monitor file that needs filtering.

**What you should see:** A new file created at `Python/Monitors_date_filtered/Monitor51_06_20_25.txt`

---

### Step 1 — Prepare Data and Health Report *(required)*

This is the main data loading step. It reads your monitor files, pairs each channel with its metadata, generates health reports for each fly, and loads everything into the database.

```bash
python 1-prepare_data_and_health.py
```

> **Note:** This script reads from `Monitors_date_filtered/` if that folder has files, otherwise falls back to `Monitors_raw/`.

**What you should see:** Progress messages as each monitor file is processed, then a summary. This step loads around 24 million rows — expect 5–15 minutes.

**What gets stored in the database:**
- `experiments` table — metadata about this run
- `flies` table — one row per fly
- `readings` table — all minute-by-minute activity data (~24M rows)
- `health_reports` table — daily health status for each fly (Alive/Unhealthy/Dead/QC_Fail)

Each time you run Step 1, it creates a new "experiment" in the database. You can have multiple experiments coexist. See [Managing Multiple Experiments](#managing-multiple-experiments) below.

---

### Step 2 — Remove Flies *(optional)*

**When to use this:** Use this step if you want to exclude dead flies, QC failures, or any other flies before feature extraction.

Remove flies by health status (most common):
```bash
python 2-remove_flies.py --statuses "Dead,Unhealthy"
```

Remove specific fly IDs:
```bash
python 2-remove_flies.py --flies "51-ch03,52-ch07"
```

Remove by genotype or sex:
```bash
python 2-remove_flies.py --genotypes "sleep_mut" --sexes "Female"
```

By default, this operates on the most recently created experiment. To target a specific experiment:
```bash
python 2-remove_flies.py --experiment-id 2 --statuses "Dead"
```

**What you should see:** A count of how many flies were removed and how many remain.

---

### Step 3 — Create Feature Table *(required)*

This step extracts all behavioral and circadian features for each fly.

```bash
python 3-create_feature_table.py
```

**What it computes per fly:**
- **Circadian features:** mesor, amplitude, phase (from cosinor regression), plus periodogram period and power
- **Sleep totals:** total sleep, day sleep, night sleep (in minutes)
- **Sleep structure:** number of bouts, mean/max bout length, fragmentation, latency, WASO
- **Activity transitions:** P_wake (probability of waking), P_doze (probability of falling asleep)
- **Rhythm regularity:** interdaily stability, activity onset and offset times

This also populates the `features_sliding_window` table used by the death prediction analysis.

**What you should see:** Progress per fly or group, then a summary. Expect 10–20 minutes.

**What gets stored:** `features` table in the database — one row per fly, 25+ columns.

---

### Step 4 — Clean and Normalize Features *(required)*

This step removes outlier flies, fills missing values, and z-scores all features.

```bash
python 4-clean_ml_features.py
```

**What it does:**
- Removes flies with zero sleep (likely dead or sensor errors)
- Removes statistical outliers using IQR filtering (within each genotype group)
- Fills any remaining missing values
- Z-scores all features (each feature has mean 0, SD 1 across all flies)

**What you should see:** A count of flies before and after cleaning.

**What gets stored:** `features_z` table in the database — z-scored version of the features table.

---

## Part 5 — Run Analysis Scripts

Navigate to the analysis folder:

```bash
cd analysis
```

All scripts run without any extra flags when using the SQL pipeline.

---

### Analysis 1 — PCA *(run this first)*

Principal Component Analysis — finds the major axes of variation in your fly population and shows which features drive genotype differences.

```bash
python pca_analysis.py
```

To target a specific experiment (default is the latest):
```bash
python pca_analysis.py --experiment-id 2
```

**Output** (saved to `analysis_results/pca/`):
- `pca_pc1_pc2.png` — scatter plot of flies in behavioral space, colored by genotype
- `genotype_signature_heatmap.png` — which features are high or low for each genotype
- `effect_size_heatmap.png` — which features differ most between genotypes
- `posthoc_results/` — pairwise statistical comparisons per feature

---

### Analysis 2 — UMAP + Clustering *(requires PCA first)*

Reduces the data to 2D using UMAP, then finds natural groupings (clusters) of flies with similar behavior.

```bash
python umap_dbscan_analysis.py
```

**Output** (saved to `analysis_results/umap/`):
- `umap_genotype.png` — UMAP map colored by genotype
- `umap_sex.png` — same map colored by sex
- `hdbscan_cluster_genotype_pies.png` — what genotypes make up each cluster
- `cluster_signatures_heatmap.png` — what behaviors define each cluster
- `cluster_genotype_enrichment.csv` — which genotypes are over/under-represented per cluster

---

### Analysis 3 — Random Forest

Trains a classifier to predict genotype from behavioral features. Shows which features are most informative.

```bash
python random-forest.py
```

**Output** (saved to `analysis_results/random_forest/`):
- `feature_importance.png` — ranked list of most discriminating features
- `confusion_matrix.png` — how well the model classifies each genotype
- `classification_results.txt` — accuracy, F1, AUC per genotype

---

### Analysis 4 — Sex Differences *(requires UMAP first)*

Compares male and female behavioral profiles within each genotype and cluster.

```bash
python sexdiff_analysis.py
```

---

### Analysis 5 — Cluster Characterization *(requires UMAP first)*

Profiles each behavioral cluster to show what makes it distinct from the others.

```bash
python cluster_characterization.py
```

---

### Analysis 6 — RF vs PCA Comparison *(run after RF and PCA)*

Checks whether the features that separate genotypes in PCA are the same features the Random Forest finds important.

```bash
python rf_vs_pca_loadings.py
```

---

### Analysis 7 — Death Prediction *(SQL pipeline only)*

Uses XGBoost on rolling 5-day windows of behavior to predict which flies are approaching death. This is the only analysis that requires the SQL pipeline.

```bash
python death_prediction_xgboost.py
```

**Output** (saved to `analysis_results/death_prediction/`):
- `shap_summary.png` — which features most predict imminent death
- `feature_importance.csv` — feature rankings
- `confusion_matrix.png` — alive/dying/dead classification accuracy

---

## Recommended Full Run (Copy-Paste Ready)

```bash
# From repo root — run pipeline
cd Python/src/sql_db_pipeline
python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 7
python 1-prepare_data_and_health.py
python 2-remove_flies.py --statuses "Dead,QC_Fail"
python 3-create_feature_table.py
python 4-clean_ml_features.py

# Run analysis
cd analysis
python pca_analysis.py
python umap_dbscan_analysis.py
python random-forest.py
python sexdiff_analysis.py
python cluster_characterization.py
python rf_vs_pca_loadings.py
python death_prediction_xgboost.py
```

---

## Where Are My Results?

All analysis outputs go to:
```
Python/src/sql_db_pipeline/analysis/analysis_results/
├── pca/
├── umap/
├── random_forest/
├── sexdiff/
├── cluster_characterization/
├── RFvsPCAloadings/
└── death_prediction/
```

Plots are saved as `.png` files. They are not displayed on screen — open the files in your file browser to view them.

---

## Managing Multiple Experiments

Each time you run Step 1, a new experiment is created in the database. The other steps default to the most recently created experiment.

```bash
# List all experiments in the database
python delete_experiment.py --list

# Run a specific step on experiment 2 (instead of the latest)
python 3-create_feature_table.py --experiment-id 2

# Delete an experiment and all its data
python delete_experiment.py --experiment-id 1
```

---

## Common Problems

**"could not connect to server" or database connection error**
PostgreSQL might not be running. On Mac: `brew services start postgresql@16`. On Linux: `sudo systemctl start postgresql`. On Windows: open Services and check if PostgreSQL is running, or restart it from the Start menu.

**"password authentication failed"**
The password in `config.py` doesn't match what you set when installing PostgreSQL. Update `DB_PASSWORD` in `config.py`.

**"database fly_ml_db does not exist"**
Run `python setup_database.py` from the `sql_db_pipeline/` folder first.

**"No vehicle flies found" or "No VEH flies"**
The analysis scripts filter to `VEH` treatment by default. Make sure your `metadata.txt` has a `Treatment` column and that vehicle/control flies are labeled `VEH`.

**Step 1 is very slow**
That's expected — it's loading 24 million rows into the database. 5–15 minutes is normal.

**"Missing pca_scores.csv" when running UMAP**
Run `pca_analysis.py` first — UMAP reads from its output.

**"Missing umap_clusters.csv" when running sexdiff or cluster_characterization**
Run `umap_dbscan_analysis.py` first.

**Plots aren't showing up**
The scripts save plots to files — they don't open automatically. Check `analysis_results/` in your file browser.

**Want to start over with fresh data**
Delete the experiment and re-run from Step 1:
```bash
python delete_experiment.py --experiment-id 1
python 1-prepare_data_and_health.py
```
