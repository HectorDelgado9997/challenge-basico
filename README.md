# Analysis of Comments on Glassdoor

## Project Overview

This project develops a Natural Language Processing pipeline for sentiment analysis applied to Glassdoor company reviews.

The analysis focuses exclusively on three textual fields:

- `headline`
- `pros`
- `cons`

These fields are combined into a single corpus column named `review_text`, which is then processed and used for sentiment classification.

The solution compares two sentiment analysis approaches:

- **VADER** as a lexical baseline model.
- **pysentimiento** as the main transformer-based sentiment classifier.

The project also includes local experiment tracking using **MLflow**.

---

## Repository Structure

```text
challenge-basico/
├── data/
│   ├── raw/
│   │   └── glassdoor_comments.csv
│   └── processed/
├── docs/
│   ├── model_construction.md
│   ├── mlops.md
│   └── results.md
├── notebooks/
├── outputs/
│   ├── figures/
│   │   └── sentiment_distribution.png
│   └── reports/
│       └── model_comparison_report.txt
├── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── sentiment_vader.py
│   ├── sentiment_pysentimiento.py
│   ├── run_sentiment_analysis.py
│   ├── evaluation.py
│   └── mlflow_pipeline.py
├── .gitignore
├── params.yaml
├── README.md
└── requirements.txt


## Tech used

Python
Git
GitHub
VS Code

Project Status:
Completed.

Author:
Héctor Manuel Delgado Zambrano


