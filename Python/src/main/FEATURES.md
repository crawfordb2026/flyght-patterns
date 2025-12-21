# Feature Extraction Scripts

This document briefly explains the three main feature extraction scripts for Drosophila sleep and circadian rhythm analysis.

## 1. `create_features.py` - Sleep Features

**Purpose:** Extracts sleep-related features for each fly per day.

**What it does:**
- Detects sleep bouts (periods of inactivity ≥ 5 minutes)
- Calculates sleep metrics: total sleep, light/dark sleep, bout counts, sleep efficiency, latency, transition probabilities
- Produces per-fly, per-day features that can be averaged across days

**Output:**
- `fly_sleep_summary.csv` - Per-day sleep metrics for each fly
- `fly_sleep_mean.csv` - Per-fly averaged sleep metrics
- `group_sleep_summary.csv` - Group-level summaries (mean ± SEM)
- `sleep_plots/` - Visualization plots for all metrics

**Use when:** You need sleep behavior features (bouts, efficiency, latency, etc.) for downstream analysis or machine learning.

---

## 2. `cosinor.py` - Per-Fly Circadian Features

**Purpose:** Extracts circadian rhythm parameters for each individual fly.

**What it does:**
- Fits cosinor regression models to each fly's hourly activity data
- Extracts rhythm parameters: Mesor (baseline), Amplitude (rhythm strength), Phase (peak timing)
- Calculates significance of rhythmicity per fly

**Output:**
- `fly_circadian_features.csv` - One row per fly with Mesor, Amplitude, Phase, p_value, Rhythmic flag

**Use when:** You need per-fly circadian features to combine with sleep features or for individual-level analysis.

---

## 3. `circasingle.py` - Group-Level Circadian Analysis

**Purpose:** Characterizes circadian rhythms at the group level (averaging across flies).

**What it does:**
- Averages activity across flies within each experimental group
- Fits cosinor models to group-averaged activity curves
- Produces group-level rhythm parameters for statistical comparisons

**Output:**
- `group_circadian_features.csv` - One row per group (Genotype × Sex × Treatment) with group-level Mesor, Amplitude, Acrophase, p_value

**Use when:** You want to directly compare circadian rhythms between experimental groups or need group-level rhythm characterization.

---

## Summary

| Script | Level | Features | Best For |
|--------|-------|----------|----------|
| `create_features.py` | Per-fly, per-day | Sleep metrics (17 features) | Sleep behavior analysis |
| `cosinor.py` | Per-fly | Circadian parameters (4 features) | Individual rhythm features |
| `circasingle.py` | Group-level | Circadian parameters (4 features) | Group comparisons |

**Note:** All scripts accept `dam_data_MT.csv` or `dam_data_merged.csv` as input and handle both uppercase and lowercase column names.

