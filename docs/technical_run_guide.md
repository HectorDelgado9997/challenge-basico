# Technical Run Guide

## Prerequisites

| Tool       | Version recommended |
|------------|---------------------|
| Python     | 3.9+                |
| Git        | Any recent version  |
| Git Bash   | Windows users       |

---

## 1. Clone the Repository

```bash
git clone https://github.com/HectorDelgado9997/challenge-basico.git
cd challenge-basico
```

---

## 2. Create and Activate Virtual Environment

```bash
# Create the environment
python -m venv .venv

# Activate — Windows Git Bash
source .venv/Scripts/activate

# Activate — Linux / Mac
source .venv/bin/activate
```

> You should see `(.venv)` at the start of your terminal prompt.

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

| Package          | Purpose                                        |
|------------------|------------------------------------------------|
| pandas           | Data loading and manipulation                  |
| numpy            | Numerical operations                           |
| scikit-learn     | Text utilities and preprocessing support       |
| matplotlib       | Sentiment distribution plots                   |
| vaderSentiment   | Lexical rule-based sentiment analysis          |
| pysentimiento    | Transformer-based sentiment analysis           |
| transformers     | HuggingFace backbone for pysentimiento         |
| torch            | Deep learning backend for transformers         |
| mlflow           | Experiment tracking                            |
| PyYAML           | params.yaml configuration loading             |
| langdetect       | Language detection utilities                   |
| pytest           | Unit testing                                   |

> ⚠️ `torch` is a large package (~2GB). Installation may take several
> minutes depending on your internet connection.

---

## 4. Verify the Dataset

Make sure the source file exists at:

```text
data/raw/glassdoor_comments.csv
```

The file must contain at least these columns:

```text
headline, pros, cons
```

---

## 5. Review Configuration

All parameters are defined in `params.yaml` at the project root.
You can modify paths, thresholds and MLflow settings there:

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

## 6. Execution Options

### Option A — Full pipeline with MLflow tracking (recommended)

Open **two terminals**:

**Terminal 1 — Start MLflow server:**
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

**Terminal 2 — Run the pipeline:**
```bash
cd src
python mlflow_pipeline.py
```

Then open `http://127.0.0.1:5000` to explore the tracked experiment.

---

### Option B — Lightweight pipeline without MLflow

```bash
cd src
python run_sentiment_analysis.py
```

This runs the full ingestion → preprocessing → VADER → pysentimiento
→ evaluation flow and saves all outputs, but does not log to MLflow.

---

### Option C — Run each module independently

```bash
cd src

# Step 1 — Load and validate the dataset
python ingestion.py

# Step 2 — Preprocess text
python preprocessing.py

# Step 3 — Evaluate results (requires processed CSV)
python evaluation.py
```

---

## 7. Check the Outputs

After a successful run the following files are generated:

```text
data/processed/
└── glassdoor_sentiment_results.csv     ← Full results with sentiment labels

outputs/
├── figures/
│   └── sentiment_distribution.png     ← Bar chart VADER vs pysentimiento
└── reports/
    └── model_comparison_report.txt     ← Agreement rate + distributions
```

---

## 8. Understanding the Output Columns

`glassdoor_sentiment_results.csv` contains:

| Column                    | Description                                      |
|---------------------------|--------------------------------------------------|
| `headline`                | Original review headline                         |
| `pros`                    | Original pros text                               |
| `cons`                    | Original cons text                               |
| `review_text`             | Concatenated headline + pros + cons              |
| `sentiment_text`          | Minimally cleaned text for sentiment models      |
| `clean_text`              | Strongly cleaned text for feature extraction     |
| `tokens`                  | Tokenized clean_text                             |
| `tokens_no_stopwords`     | Tokens after stopword removal                    |
| `lemmas`                  | Lemmatized tokens                                |
| `processed_text`          | Final joined lemmas string                       |
| `vader_sentiment`         | VADER label: positive / neutral / negative       |
| `pysentimiento_sentiment` | pysentimiento label: positive / neutral / negative|

---

## 9. VADER Thresholds

VADER assigns a compound score between -1.0 and 1.0. The thresholds
for classification are defined in `params.yaml`:

| Compound score       | Label      |
|----------------------|------------|
| score >= 0.05        | positive   |
| score <= -0.05       | negative   |
| -0.05 < score < 0.05 | neutral    |

---

## 10. Run the Tests

```bash
pytest -v
```

---

## Common Errors

| Error | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Virtual env not activated | Run `source .venv/Scripts/activate` |
| `FileNotFoundError` on CSV | Dataset missing | Add CSV to `data/raw/` |
| `ValueError: Missing required text columns` | CSV missing headline/pros/cons | Check column names in the CSV |
| `UnicodeDecodeError` | Encoding not in fallback list | Check file encoding manually |
| `mlflow.exceptions` | MLflow server not running | Run `mlflow ui --host 127.0.0.1 --port 5000` first |
| `torch` install timeout | Large package | Run `pip install torch` separately first |
| `ModuleNotFoundError: No module named ingestion` | Running from wrong directory | Run scripts from inside `src/` |
