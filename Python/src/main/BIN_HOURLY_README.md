# Hourly Binning Script

## Overview

`bin_hourly.py` bins DAM data from 1-minute intervals to 1-hour intervals, aggregating activity counts and maintaining the LIKELY_DEAD flag.

## Features

✅ **Sums activity values** within each hour  
✅ **Preserves LIKELY_DEAD** flag (uses OR logic: True if any reading in that hour has LIKELY_DEAD=True)  
✅ **Fills missing hours** with activity_count=0  
✅ **Complete hourly timeline** - every fly has data for every hour  
✅ **Handles LIKELY_DEAD forward-filling** for missing hours  

## Usage

### Single File
```bash
cd Python/src
python3 bin_hourly.py ../../data/processed/dam_data_MT.csv
# Output: dam_data_MT_hourly.csv
```

### Multiple Files
```bash
python3 bin_hourly.py \
  ../../data/processed/dam_data_MT.csv \
  ../../data/processed/dam_data_CT.csv \
  ../../data/processed/dam_data_Pn.csv
# Output: three _hourly.csv files
```

## Input Requirements

- CSV file with columns: `datetime, monitor, channel, value, fly_id, genotype, sex, treatment`
- Optional: `LIKELY_DEAD` column (will be handled automatically if present)
- Data should have `datetime` in pandas-readable format

## Output

- Same filename with `_hourly` added before `.csv`
- Same columns as input
- One row per fly per hour
- Continuous hourly timeline (no gaps)

## Example

**Input (1-minute data):**
```
datetime,channel,fly_id,genotype,sex,treatment,value,LIKELY_DEAD
2025-09-19 11:46:00,1,M5_Ch01,SSS,Female,2mM His,15,False
2025-09-19 11:47:00,1,M5_Ch01,SSS,Female,2mM His,15,False
2025-09-19 11:48:00,1,M5_Ch01,SSS,Female,2mM His,15,False
2025-09-19 11:49:00,1,M5_Ch01,SSS,Female,2mM His,15,False
2025-09-19 11:50:00,1,M5_Ch01,SSS,Female,2mM His,15,False
2025-09-19 12:51:00,1,M5_Ch01,SSS,Female,2mM His,12,False
```

**Output (1-hour data):**
```
datetime,channel,fly_id,genotype,sex,treatment,value,LIKELY_DEAD
2025-09-19 11:00:00,1,M5_Ch01,SSS,Female,2mM His,75,False
2025-09-19 12:00:00,1,M5_Ch01,SSS,Female,2mM His,12,False
```

Note: If there were 60 rows in the 11:XX hour, they would sum to one row with value=sum of all 60 values.

## Aggregation Logic

- **value**: SUM all values within the hour
- **LIKELY_DEAD**: True if ANY reading in that hour has LIKELY_DEAD=True
- **Missing hours**: Filled with value=0, LIKELY_DEAD forward-filled from previous hour

## Data Reduction

Typical reduction: ~98% (from 505,000 rows to ~10,000 rows for 61 flies over 5 days)

## Integration with Pipeline

```
dam_data_MT.csv (1-min data)
    ↓
bin_hourly.py
    ↓
dam_data_MT_hourly.csv (1-hour data)
```

Use hourly data for:
- Sleep analysis plots
- Long-term trends
- Statistical modeling
- Quick visualization

## Notes

- Replaces the old `bin_data.py` (5-minute intervals)
- Works with all reading types (MT, CT, Pn)
- Maintains all metadata columns
- Preserves dead fly tracking through LIKELY_DEAD flag

