# Model Construction

## Objective

The objective of this project is to build a Natural Language Processing solution for sentiment analysis applied to Glassdoor reviews.

Input Data

The raw input file is located at:

data/raw/glassdoor_comments.csv

The model pipeline focuses only on textual review information from the following columns:

- `headline`
- `pros`
- `cons`
These columns are selected because they contain the main textual information required for sentiment analysis.
These fields are combined into a single text corpus named `review_text`.

---

Data Ingestion

The ingestion stage performs the following operations:

Loads the CSV file.
Validates that the input is a pandas DataFrame.
Validates that headline, pros, and cons exist.
Selects only the required textual columns.
Converts these columns to string.
Replaces missing values with empty strings.
Removes duplicated records.
Removes records without useful text.
Creates review_text by combining headline, pros, and cons.