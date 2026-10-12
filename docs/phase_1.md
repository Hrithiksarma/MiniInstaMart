# Phase 1: Simulator and raw data

Requirements touched: D2, B1.

## Decisions
- Simulated orders with known effect sizes in `config.yaml`, so the ETA model can later be checked against ground truth.
- Delivery time = prep + rider wait + travel + handover. Rain at peak hours has an extra interaction effect.
- Defects (missing weather 4%, impossible coordinates 0.3%, negative times 0.2%, duplicates 1%) hit disjoint rows; leakage column `delivered_at` is added to every row.
- Hidden components are saved in a separate truth file and are never features.
- Raw parquet files are written read-only.

## What failed
- The first defect-rate test counted NaNs including duplicated rows; fixed to count on de-duplicated rows.
- The first drift dataset spread 100k orders over 90 days, so pressure was lower and drift looked faster than normal. Fixed by matching order density (18 days) and starting on a Monday.

## Findings (simulated data)
- 500k clean orders: mean delivery 24.9 min, median 23.7, max 143.5.
- Rain +4.6 min and peak +3.4 min on average; the rain-and-peak rider wait is larger than the sum of the two separate effects.
- Public dataset download is deferred (see data_sources.md).
