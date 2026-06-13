# Quick Start

Welcome to flyght-patterns. This one-page guide helps you pick the right pipeline and points you to the full instructions.

---

## Which pipeline should you use?

Read the descriptions below and pick whichever fits:

```
+---------------------------+-----------------------------------+----------------------------------+
|                           | CSV Pipeline                      | SQL Pipeline                     |
+---------------------------+-----------------------------------+----------------------------------+
| Requirements              | Python only                       | Python + PostgreSQL              |
| Setup time                | ~15 minutes                       | ~1 hour                          |
| Best for                  | Smaller datasets                  | Larger Datasets                  |
| Data stored in            | CSV files on your computer        | A database on your computer      |
| Analysis scripts          | Yes, with --csv-input flag        | Yes, default mode                |
| Ease of Use               | Slighly easier, no SQL/DB setup   | Takes time to setup SQL database |
+---------------------------+-----------------------------------+----------------------------------+
```

> **If you are not sure, use the CSV pipeline.** It is simpler, requires no database setup, and produces the same results for most use cases. You can always switch to the SQL pipeline later.

---

## What you need (both pipelines)

- **Python 3.8 or higher** -- download at https://www.python.org/downloads/
  - On Windows: during install, check "Add Python to PATH"
  - Verify it works: open a terminal and type `python --version`
- **Your monitor files** -- `.txt` files from your DAM system (e.g. `Monitor51.txt`)
- **A metadata file** -- you will create this before running Step 1 (format shown in each guide)

Install Python dependencies (run once from the repo root):

```
pip install -r requirements.txt
```

---

## Go to your pipeline guide

- **CSV Pipeline** (no database needed):
  Open `Python/QUICKSTART/CSV_PIPELINE.md`

- **SQL Pipeline** (stores data in PostgreSQL):
  Open `Python/QUICKSTART/SQL_PIPELINE.md`

---

## Not sure what this project does?

See `README.md` at the repo root for an overview of the science and what the pipeline produces.
