# Project Architecture

## Overview

This project follows a flat, sequential module architecture where each
script in `src/` handles a single responsibility. The pipeline is
orchestrated by `src/mlflow_pipeline.py`, which is the main entry point
and integrates all stages under a single MLflow run.

---

## Directory Structure

```text
challenge-basico/
├── data/
│   ├── raw/
│   │   └── glassdoor_comments.csv        ← Source dataset
│   └── processed/
│       └── glassdoor_sentiment_results.csv ← Full results with sentiment labels
│
├── docs/
│   ├── architecture.md
│   ├── dataset_extraction.md
│   ├── model_construction.md
│   ├── mlops.md
│   ├── results.md
│   └── technical_run_guide.md
│
├── outputs/
│   ├── figures/
│   │   └── sentiment_distribution.png    ← Bar chart: VADER vs pysentimiento
│   └── reports/
│       └── model_comparison_report.txt   ← Agreement rate + distributions
│
├── src/
│   ├── ingestion.py                      ← CSV loading and validation
│   ├── preprocessing.py                  ← Text normalization and tokenization
│   ├── sentiment_vader.py                ← VADER sentiment scoring
│   ├── sentiment_pysentimiento.py        ← pysentimiento transformer scoring
│   ├── run_sentiment_analysis.py         ← Pipeline without MLflow
│   ├── evaluation.py                     ← Metrics, plots and report
│   └── mlflow_pipeline.py               ← Full pipeline with MLflow tracking
│
├── params.yaml                           ← Single source of truth for all config
├── requirements.txt
└── README.md
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `ingestion.py` | Load CSV, validate columns, build `review_text` |
| `preprocessing.py` | Normalize text, tokenize, remove stopwords, lemmatize |
| `sentiment_vader.py` | Apply VADER lexical model → `vader_sentiment` |
| `sentiment_pysentimiento.py` | Apply pysentimiento transformer → `pysentimiento_sentiment` |
| `evaluation.py` | Agreement rate, distributions, crosstab, plots, report |
| `run_sentiment_analysis.py` | Lightweight pipeline without MLflow |
| `mlflow_pipeline.py` | Full pipeline with MLflow tracking and artifact logging |

---

## Text Processing Layers

One of the key design decisions is having **two separate normalization paths**
for different purposes:review_text
│
├── normalize_for_sentiment()     ← Minimal cleaning
│   Preserves: negations, intensifiers, case, punctuation
│   Used by: VADER, pysentimiento
│   Output: sentiment_text
│
└── normalize_for_features()      ← Strong cleaning
Removes: URLs, HTML, punctuation, numbers
Lowercase only
Used by: TF-IDF, n-grams, classical NLP
Output: clean_text → tokens → tokens_no_stopwords → lemmas → processed_text

---

## Full Data Flow
data/raw/glassdoor_comments.csv
│
▼
run_data_ingestion()
├── load_glassdoor_csv()          ← encoding fallback
├── validate_dataframe_input()
├── validate_required_text_columns()
├── cast_text_columns()           ← NaN → "" → str
├── remove_invalid_records()      ← dedup + all-empty rows
└── build_review_text()           ← headline + pros + cons
│
▼
preprocess_dataframe()
├── normalize_for_sentiment()     → sentiment_text
├── normalize_for_features()      → clean_text
├── tokenize_text()               → tokens
├── remove_stopwords_for_features() → tokens_no_stopwords
└── lemmatize_tokens()            → lemmas → processed_text
│
├──────────────────────────────┐
▼                              ▼
apply_vader_sentiment()    apply_pysentimiento_sentiment()
│                              │
│  VADER lexical scoring       │  RoBERTa transformer
│  compound score → label      │  multilingual model
│  positive / neutral /        │  positive / neutral /
│  negative                    │  negative
│                              │
└──────────────┬───────────────┘
▼
calculate_model_agreement()
build_sentiment_distribution()
build_model_comparison_table()
│
├── save_evaluation_report()
│   outputs/reports/model_comparison_report.txt
│
└── save_sentiment_distribution_plot()
outputs/figures/sentiment_distribution.png

---

## MLflow Integration

`mlflow_pipeline.py` wraps the entire flow inside a single MLflow run:
mlflow.set_tracking_uri()
mlflow.set_experiment("glassdoor_sentiment_analysis")
│
└── mlflow.start_run(run_name="vader_vs_pysentimiento_pipeline")
│
├── mlflow.log_param()    ← input_path, text_columns,
│                            vader_thresholds, model names
│
├── [full pipeline executes here]
│
├── mlflow.log_metric()   ← rows_after_ingestion,
│                            rows_after_preprocessing,
│                            agreement_rate,
│                            vader_/pysentimiento_ shares
│
└── mlflow.log_artifact() ← report .txt, figure .png,
sentiment results .csv

---

## Configuration — `params.yaml`

All paths and parameters are centralized in `params.yaml`:

```yaml
data:
  input_path: "data/raw/glassdoor_comments.csv"
  sentiment_results_path: "data/processed/glassdoor_sentiment_results.csv"

outputs:
  report_path: "outputs/reports/model_comparison_report.txt"
  figure_path: "outputs/figures/sentiment_distribution.png"

mlflow:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "glassdoor_sentiment_analysis"
  run_name: "vader_vs_pysentimiento_pipeline"

pipeline:
  text_columns: ["headline", "pros", "cons"]
  sentiment_input_column: "sentiment_text"
  vader_threshold_positive: 0.05
  vader_threshold_negative: -0.05
```

---

## Stopword Strategy

The preprocessing module uses a custom three-tier stopword system:

| Set | Purpose |
|---|---|
| `BASIC_STOPWORDS` | Standard English function words |
| `NEGATION_WORDS` | Words preserved for sentiment (no, not, never...) |
| `INTENSIFIER_WORDS` | Words preserved for sentiment (very, extremely...) |
| `STOPWORDS_FOR_FEATURES` | `BASIC_STOPWORDS - NEGATION_WORDS - INTENSIFIER_WORDS` |

> Negations and intensifiers are **never removed** from `sentiment_text`
> because they are critical for VADER and pysentimiento accuracy.

---

## Design Principles

| Principle | Implementation |
|---|---|
| Single source of truth | `params.yaml` centralizes all paths and parameters |
| Dual normalization | Separate cleaning for sentiment vs feature extraction |
| Preserved negations | NEGATION_WORDS excluded from stopword removal |
| Encoding resilience | Multiple encoding fallbacks in `load_glassdoor_csv()` |
| Modular scripts | Each file handles one responsibility |
| Two execution modes | `run_sentiment_analysis.py` (lightweight) vs `mlflow_pipeline.py` (full) |
