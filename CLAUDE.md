# Project: flyght-patterns (pipeline analysis)

## Goals
- Prioritize small, safe, readable changes.
- Fix root causes; avoid overengineering.
- Keep behavior consistent unless explicitly asked to change it.

## Environment
- OS: Windows (PowerShell)
- Prefer absolute paths in commands and file references.
- Python scripts are under:
  - `Python/src/sql_db_pipeline` (SQL/database pipeline)
  - `Python/src/csv_pipeline` (CSV-based pipeline)
  - `Python/src/sql_db_pipeline/analysis`
- Main outputs are under:
  - `Python/src/sql_db_pipeline/analysis/analysis_results/`

## Code Style / Editing Rules
- Keep code simple and explicit.
- Do not introduce unnecessary abstractions.
- Do not rename files/functions unless requested.
- Preserve existing output filenames unless requested.
- Add comments only when needed for non-obvious logic.
- Do not touch unrelated files.

## Data/Analysis Assumptions
- UMAP input should come from `analysis_results/pca/pca_scores.csv` (PC columns).
- Do not re-fit PCA inside UMAP script.
- For downstream behavioral tests, align rows by `fly_id` and ensure matching sets.
- Avoid index/column ambiguity in pandas merges (`fly_id` should be unambiguous).
- Random Forest `feature_importance.csv` should contain all features, sorted.
- Comparison scripts should use existing outputs (no duplicate output files unless asked).

## Robustness Expectations
- Add clear checks for missing required input files with helpful error messages.
- Handle column name normalization defensively (e.g., lowercase where needed).
- Keep row-order consistency when alignment matters.
- Prefer deterministic behavior (`random_state` where applicable).

## Plotting/Runtime
- Use non-interactive matplotlib backend for CLI runs on Windows:
  - `matplotlib.use('Agg')`
- Avoid GUI/Tkinter dependencies in scripts run from terminal.

## Dependencies
- Keep `requirements.txt` updated when adding imports.
- If adding package requirements, pin to reasonable minimum versions.

## Validation Before Finishing
- Run lints on edited files if available.
- Run the directly affected script(s) when feasible.
- Verify expected outputs are written to `analysis_results/...`.
- Report what changed and why in concise bullets.

## Communication Style
- Be concise and practical.
- If uncertain, ask one focused clarifying question.
- Offer next command(s) to verify.
