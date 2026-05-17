# Dataset Extraction

## Overview

This project uses the **Glassdoor Comments** dataset, a CSV file containing
employee reviews scraped from Glassdoor. The data is stored locally in the
repository and loaded via `src/ingestion.py`.

## Dataset Location

```text
data/raw/glassdoor_comments.csv
```

## Required Columns

The ingestion module expects exactly these three text columns:

| Column     | Description                                      |
|------------|--------------------------------------------------|
| `headline` | Short title or summary of the employee review    |
| `pros`     | Positive aspects mentioned by the employee       |
| `cons`     | Negative aspects mentioned by the employee       |

> Any other columns present in the CSV are ignored during ingestion.

## Encoding Handling

The loader tries multiple encodings automatically in this order:

```python
encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]
```

This ensures compatibility with CSV files exported from different systems
and operating systems without manual intervention.

## Ingestion Pipeline

Data loading is handled by `src/ingestion.py` via `run_data_ingestion()`:

```python
from ingestion import run_data_ingestion
df = run_data_ingestion("data/raw/glassdoor_comments.csv")
```

### Steps executed

| Step | Function | Description |
|---|---|---|
| 1 | `load_glassdoor_csv()` | Load CSV with encoding fallback |
| 2 | `validate_dataframe_input()` | Check it's a non-empty DataFrame |
| 3 | `validate_required_text_columns()` | Verify headline, pros, cons exist |
| 4 | `select_required_text_columns()` | Keep only the 3 required columns |
| 5 | `cast_text_columns()` | Fill NaN with "" and cast to str |
| 6 | `remove_invalid_records()` | Drop duplicates and all-empty rows |
| 7 | `build_review_text()` | Concatenate columns into review_text |

## Output — `review_text`

After ingestion, a new column `review_text` is created by concatenating
the three text columns:

```python
df["review_text"] = (
    df["headline"].str.strip()
    + " "
    + df["pros"].str.strip()
    + " "
    + df["cons"].str.strip()
).str.strip()
```

This single field becomes the input for all downstream preprocessing
and sentiment analysis steps.

## Data Quality Rules

| Rule | Implementation |
|---|---|
| No empty DataFrame | `validate_dataframe_input()` raises ValueError |
| No missing columns | `validate_required_text_columns()` raises ValueError |
| No duplicate rows | `drop_duplicates()` |
| No all-empty rows | Filter rows where all text fields are blank |
| No empty review_text | Filter rows where concatenated text has length 0 |

## Configuration

The input path is defined in `params.yaml`:

```yaml
data:
  input_path: "data/raw/glassdoor_comments.csv"
  sentiment_results_path: "data/processed/glassdoor_sentiment_results.csv"
```

## Processed Output

After the full pipeline runs, results are saved to:

```text
data/processed/glassdoor_sentiment_results.csv
```

This file contains the original columns plus all preprocessing
and sentiment analysis results.
