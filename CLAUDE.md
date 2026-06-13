# Project: flyght-patterns (pipeline analysis)

## Goals
- Prioritize small, safe, readable changes.
- Fix root causes; avoid overengineering.
- Keep behavior consistent unless explicitly asked to change it.

## Keeping CLAUDE.md Updated
Update this file whenever:
- New scripts are added to either pipeline
- Naming conventions or column names change
- New CLI flags or established patterns are introduced across scripts
- The repo structure or documentation layout changes
- A non-obvious design decision is made that future sessions should know about

The goal is that any new conversation can read CLAUDE.md and immediately understand the project's architecture and conventions without exploring the codebase from scratch.

## Environment
- OS: Windows (PowerShell)
- Prefer absolute paths in commands and file references.
- Python scripts are under:
  - `Python/src/sql_db_pipeline` (SQL/database pipeline)
  - `Python/src/csv_pipeline` (CSV-based pipeline)
  - `Python/src/sql_db_pipeline/analysis` (analysis scripts, shared by both pipelines)
- Main outputs are under:
  - `Python/src/sql_db_pipeline/analysis/analysis_results/`

## Documentation Structure
User-facing guides live in `Python/`:
- `Python/QUICKSTART/QUICKSTART.md` — pipeline selection guide (CSV vs SQL), short
- `Python/QUICKSTART/CSV_PIPELINE.md` — full step-by-step guide for csv_pipeline
- `Python/QUICKSTART/SQL_PIPELINE.md` — full step-by-step guide for sql_db_pipeline

Root-level docs:
- `README.md` — what the project is and what it does; not a how-to guide
- `sql_db_pipeline.md` — technical reference for SQL pipeline internals
- `HOW_TO_USE.md` — local reference only (gitignored)

Do not add how-to instructions to README.md. Keep it as a scientific/technical overview.

## Dual-Pipeline Architecture
Both pipelines (csv_pipeline and sql_db_pipeline) run the same 5 steps (0-4) and produce equivalent outputs. Key invariants:
- Both pipelines output **identical lowercase column names** (`genotype`, `sex`, `treatment`, `fly_id`, etc.)
- Feature column names: `mesor_mean`, `amplitude_mean`, `phase_mean`, `p_wake_mean`, `p_doze_mean`, `waso_mean`, `amplitude_sd`, etc. — all lowercase, all the way through
- csv_pipeline output: `data/processed/ML_features_Z.csv`
- sql_db_pipeline output: `features_z` table in PostgreSQL

## Analysis Script CSV Mode Pattern
All 6 main analysis scripts support `--csv-input <path>` to load from `ML_features_Z.csv` instead of the database:
- `pca_analysis.py`
- `umap_dbscan_analysis.py`
- `random-forest.py`
- `sexdiff_analysis.py`
- `cluster_characterization.py`
- `rf_vs_pca_loadings.py` (reads from saved analysis outputs — no `--csv-input` needed)

**Do NOT add `--csv-input` to these scripts — by design:**
- `death_prediction_xgboost.py` — requires `features_sliding_window` table (rolling time-series), no CSV equivalent
- `rf_vs_pca_loadings.py` — reads from `analysis_results/` directly, not from feature input

The CSV loading pattern in each script:
```python
def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]  # normalize
    return df
```
Then at the top of `main()`: `if args.csv_input: df = load_data_from_csv(...) else: df = load_data_from_db(...)`

## Data/Analysis Assumptions
- UMAP input must come from `analysis_results/pca/pca_scores.csv` (PC columns). Do not re-fit PCA inside UMAP.
- When `umap_dbscan_analysis.py` loads from CSV, set `experiment_id_int = None` — the PCA scores filter already handles None gracefully.
- For downstream behavioral tests, align rows by `fly_id` and ensure matching sets.
- Avoid index/column ambiguity in pandas merges (`fly_id` should be unambiguous).
- Random Forest `feature_importance.csv` should contain all features, sorted.
- Comparison scripts should use existing outputs (no duplicate output files unless asked).

## Code Style / Editing Rules
- Keep code simple and explicit.
- Do not introduce unnecessary abstractions.
- Do not rename files/functions unless requested.
- Preserve existing output filenames unless requested.
- Add comments only when needed for non-obvious logic.
- Do not touch unrelated files.

## Robustness Expectations
- Add clear checks for missing required input files with helpful error messages.
- Handle column name normalization defensively (lowercase where needed).
- Keep row-order consistency when alignment matters.
- Prefer deterministic behavior (`random_state` where applicable).

## Plotting/Runtime
- Use non-interactive matplotlib backend for CLI runs on Windows:
  - `matplotlib.use('Agg')`
- Avoid GUI/Tkinter dependencies in scripts run from terminal.

## Dependencies
- Keep `requirements.txt` updated when adding imports.
- Pin to reasonable minimum versions.

## Validation Before Finishing
- Run lints on edited files if available.
- Run the directly affected script(s) when feasible.
- Verify expected outputs are written to `analysis_results/...`.
- Report what changed and why in concise bullets.

## Communication Style
- Be concise and practical.
- If uncertain, ask one focused clarifying question.
- Offer next command(s) to verify.
