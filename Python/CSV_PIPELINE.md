# CSV Pipeline — Step-by-Step Guide

This guide walks you through the full CSV pipeline: from raw monitor files to finished analysis plots. No database required.

**What this pipeline does:** It takes raw activity recordings from Drosophila Activity Monitors (DAM), extracts sleep and circadian rhythm features for each fly, and runs a suite of statistical and machine learning analyses. Everything is saved as CSV files on your computer.

---

## Before You Begin

### What you need

- **Python 3.8 or higher** — [Download here](https://www.python.org/downloads/)
  - On Windows: when installing, check the box that says "Add Python to PATH"
  - To check your version: open a terminal and run `python --version`
- **A terminal** — On Windows this is Command Prompt or PowerShell. On Mac it's Terminal.
- **Your monitor files** — Raw `.txt` files from your DAM system (named like `Monitor51.txt`)
- **A metadata file** — A simple text file listing which fly is in which channel (you'll create this, format below)

### Install dependencies

Open a terminal, navigate to the flyght-patterns folder, and run:

```bash
pip install -r requirements.txt
```

This installs all required Python packages. It may take a few minutes the first time.

---

## Setting Up Your Data Files

You need two things in place before running any scripts.

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
Monitor  Channel  Genotype   Sex     Treatment
51       ch01     WT         Male    VEH
51       ch02     WT         Female  VEH
51       ch03     sleep_mut  Male    VEH
52       ch01     WT         Male    VEH
```

**Important notes:**
- `Monitor` is just the monitor number (not the filename — use `51`, not `Monitor51`)
- `Channel` is the channel identifier exactly as it appears in your monitor file (e.g., `ch01`, `ch02`)
- `Treatment` should be `VEH` for vehicle/control flies (the analysis scripts look for this by default)
- Every fly you want to analyze needs a row here

---

## Running the Pipeline

All scripts live in `Python/src/csv_pipeline/`. Open a terminal and navigate there first:

```bash
cd Python/src/csv_pipeline
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

This is the main data loading step. It reads your monitor files, pairs each channel with its metadata, and generates a health report classifying each fly as Alive, Unhealthy, Dead, or QC_Fail.

```bash
python 1-prepare_data_and_health.py
```

> **Note:** This script reads from `Monitors_date_filtered/` if that folder has files, otherwise falls back to `Monitors_raw/`. Make sure the right folder has your data.

**What you should see:** Progress messages as each monitor file is processed, then a summary of flies loaded. This step takes a few minutes depending on dataset size.

**Output files created:**
- `data/processed/dam_data_prepared.csv` — all activity readings for every fly, every minute
- `data/processed/health_report.csv` — daily health status for each fly

---

### Step 2 — Remove Flies *(optional)*

**When to use this:** Use this step if you want to exclude dead flies, QC failures, or any other flies before feature extraction. You can also skip it and keep all flies.

Remove flies by health status (most common):
```bash
python 2-remove_flies.py --statuses Dead QC_Fail
```

Remove by specific fly IDs:
```bash
python 2-remove_flies.py --flies "51-ch03,52-ch07"
```

Remove by genotype or sex:
```bash
python 2-remove_flies.py --genotypes "sleep_mut" --sexes "Female"
```

**What you should see:** A count of how many flies were removed and how many remain.

**Output file created:** `data/processed/dam_data_cleaned.csv`

---

### Step 3 — Create Feature Table *(required)*

This step extracts all behavioral and circadian features for each fly. It reads from `dam_data_cleaned.csv` if Step 2 was run, otherwise from `dam_data_prepared.csv`.

```bash
python 3-create_feature_table.py
```

**What it computes per fly:**
- **Circadian features:** mesor, amplitude, phase (from cosinor regression), plus periodogram period and power
- **Sleep totals:** total sleep, day sleep, night sleep (in minutes)
- **Sleep structure:** number of bouts, mean/max bout length, fragmentation, latency, WASO
- **Activity transitions:** P_wake (probability of waking), P_doze (probability of falling asleep)
- **Rhythm regularity:** interdaily stability, activity onset and offset times

**What you should see:** Progress messages per fly or group, then a summary. This is the longest step — expect 10–20 minutes for a full dataset.

**Output file created:** `data/processed/ML_features.csv` — one row per fly, one column per feature

---

### Step 4 — Clean and Normalize Features *(required)*

This step removes outlier flies, fills missing values, and z-scores all features so they're ready for machine learning.

```bash
python 4-clean_ml_features.py
```

**What it does:**
- Removes flies with zero sleep (likely dead or sensor errors)
- Removes statistical outliers using IQR filtering (within each genotype group)
- Fills any remaining missing values
- Z-scores all features (each feature has mean 0, SD 1 across all flies)

**What you should see:** A count of flies before and after cleaning, and confirmation that the output files were written.

**Output files created:**
- `data/processed/ML_features_clean.csv` — cleaned features at original scale
- `data/processed/ML_features_Z.csv` — z-scored features, ready for analysis

---

## Running Analysis Scripts

The analysis scripts live in a different folder: `Python/src/sql_db_pipeline/analysis/`

Navigate there, then run the scripts with the `--csv-input` flag pointing to your z-scored features file.

```bash
cd ../sql_db_pipeline/analysis
```

Use this shorthand to avoid retyping the path:

```bash
# On Mac/Linux:
CSV=../../csv_pipeline/data/processed/ML_features_Z.csv

# On Windows (PowerShell):
$CSV = "../../csv_pipeline/data/processed/ML_features_Z.csv"
```

---

### Analysis 1 — PCA *(run this first)*

Principal Component Analysis — finds the major axes of variation in your fly population and shows which features drive genotype differences.

```bash
# Mac/Linux:
python pca_analysis.py --csv-input $CSV

# Windows:
python pca_analysis.py --csv-input $CSV
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
python umap_dbscan_analysis.py --csv-input $CSV
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
python random-forest.py --csv-input $CSV
```

**Output** (saved to `analysis_results/random_forest/`):
- `feature_importance.png` — ranked list of most discriminating features
- `confusion_matrix.png` — how well the model classifies each genotype
- `classification_results.txt` — accuracy, F1, AUC per genotype

---

### Analysis 4 — Sex Differences *(requires UMAP first)*

Compares male and female behavioral profiles within each genotype and cluster.

```bash
python sexdiff_analysis.py --csv-input $CSV
```

**Output** (saved to `analysis_results/sexdiff/`): Feature comparison plots and statistics by sex.

---

### Analysis 5 — Cluster Characterization *(requires UMAP first)*

Profiles each behavioral cluster to show what makes it distinct from the others.

```bash
python cluster_characterization.py --csv-input $CSV
```

**Output** (saved to `analysis_results/cluster_characterization/`):
- Per-cluster feature profiles
- Plots showing the top distinguishing features per cluster

---

### Analysis 6 — RF vs PCA Comparison *(run after RF and PCA)*

Checks whether the features that separate genotypes in PCA are the same features the Random Forest finds important.

```bash
python rf_vs_pca_loadings.py
```

> **Note:** This script doesn't need `--csv-input`. It reads the output files from the previous analyses automatically.

---

## Recommended Full Run (Copy-Paste Ready)

```bash
# From repo root — run pipeline
cd Python/src/csv_pipeline
python 0-filter_dates.py --input Monitor51 --load "06/20/25" --days 7
python 1-prepare_data_and_health.py
python 2-remove_flies.py --statuses Dead QC_Fail
python 3-create_feature_table.py
python 4-clean_ml_features.py

# Run analysis
cd ../sql_db_pipeline/analysis
python pca_analysis.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv
python umap_dbscan_analysis.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv
python random-forest.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv
python sexdiff_analysis.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv
python cluster_characterization.py --csv-input ../../csv_pipeline/data/processed/ML_features_Z.csv
python rf_vs_pca_loadings.py
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
└── RFvsPCAloadings/
```

Plots are saved as `.png` files. They are not displayed on screen — open the files in your file browser to view them.

---

## Common Problems

**"Input file not found"**
Make sure you ran the previous step. Each script depends on the output of the one before it. Check the `data/processed/` folder to see what files exist.

**"No vehicle flies found" or "No VEH flies"**
The analysis scripts filter to `VEH` treatment by default. Make sure your `metadata.txt` has a `Treatment` column and that vehicle/control flies are labeled `VEH`.

**Step 3 is very slow**
That's normal for large datasets. It's doing a lot of math per fly. Let it run.

**"Missing pca_scores.csv" when running UMAP**
Run `pca_analysis.py` first — UMAP reads from its output.

**"Missing umap_clusters.csv" when running sexdiff or cluster_characterization**
Run `umap_dbscan_analysis.py` first.

**Plots aren't showing up**
The scripts save plots to files — they don't open automatically. Check `analysis_results/` in your file browser.

**Something else went wrong**
Read the full error message — Python error messages almost always tell you what the problem is in plain English. If a file isn't found, check the path. If a package is missing, run `pip install <package-name>`.
