# Useful SQL Queries for the Database

## List All Experiments


# Connect to your database (it will prompt for password)
psql -U postgres -d fly_ml_db

Basic query:
```sql
SELECT experiment_id, name, start_date, end_date, created_at FROM experiments ORDER BY created_at DESC;
```

With more details:
```sql
SELECT 
    experiment_id,
    name,
    start_date,
    end_date,
    lights_on_hour,
    lights_off_hour,
    created_at
FROM experiments
ORDER BY created_at DESC;
```

## Count Data per Experiment

```sql
SELECT 
    e.experiment_id,
    e.name,
    COUNT(DISTINCT f.fly_id) as num_flies,
    COUNT(DISTINCT r.measurement_id) as num_readings,
    COUNT(DISTINCT h.health_report_id) as num_health_reports
FROM experiments e
LEFT JOIN flies f ON e.experiment_id = f.experiment_id
LEFT JOIN readings r ON e.experiment_id = r.experiment_id
LEFT JOIN health_reports h ON e.experiment_id = h.experiment_id
GROUP BY e.experiment_id, e.name
ORDER BY e.experiment_id;
```

## List Flies in an Experiment

```sql
SELECT 
    fly_id,
    monitor,
    channel,
    genotype,
    sex,
    treatment
FROM flies
WHERE experiment_id = 1  -- Replace 1 with your experiment_id
ORDER BY monitor, channel;
```

## Count Readings per Experiment

```sql
SELECT 
    experiment_id,
    COUNT(*) as total_readings,
    MIN(datetime) as first_reading,
    MAX(datetime) as last_reading
FROM readings
GROUP BY experiment_id
ORDER BY experiment_id;
```

## List Health Reports

```sql
SELECT 
    experiment_id,
    fly_id,
    report_date,
    status,
    total_activity,
    longest_zero_hours
FROM health_reports
WHERE experiment_id = 1  -- Replace 1 with your experiment_id
ORDER BY fly_id, report_date;
```

## Check if TimescaleDB is Installed

```sql
SELECT * FROM pg_available_extensions WHERE name = 'timescaledb';
```

## Check if readings is a Hypertable

```sql
SELECT * FROM timescaledb_information.hypertables 
WHERE hypertable_name = 'readings';
```

## Running SQL Queries

### Using psql (Command Line)

1. **Connect to the database:**
   ```powershell
   # Add PostgreSQL to PATH
   $env:Path += ";C:\Program Files\PostgreSQL\18\bin"
   
   # Connect (will prompt for password)
   psql -U postgres -d fly_ml_db
   ```

2. **Run queries:**
   ```sql
   SELECT * FROM experiments;
   ```

3. **Exit psql:**
   ```sql
   \q
   ```

### Using psql with a Single Query

```powershell
$env:Path += ";C:\Program Files\PostgreSQL\18\bin"
psql -U postgres -d fly_ml_db -c "SELECT * FROM experiments;"
```

### Using Python

```python
import psycopg2
from config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("SELECT * FROM experiments ORDER BY created_at DESC")
results = cur.fetchall()
for row in results:
    print(row)
cur.close()
conn.close()
```
