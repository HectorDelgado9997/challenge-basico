from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "company_name",
    "responsibility",
    "date_review",
    "job_title",
    "employment_details",
    "location",
    "overall_rating",
    "work_life_balance",
    "culture_values",
    "diversity_inclusion",
    "career_opp",
    "comp_benefits",
    "senior_mgmt",
    "recommend",
    "ceo_approv",
    "outlook",
    "headline",
    "pros",
    "cons",
    "source",
]


def load_glassdoor_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]

    last_error = None

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=encoding)

            if df.empty:
                raise ValueError("The CSV file was loaded, but the DataFrame is empty.")

            print(f"CSV loaded using encoding: {encoding}")
            return df

        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Unable to decode CSV with tested encodings. Last error: {last_error}",
    )


def validate_expected_columns(df: pd.DataFrame) -> None:
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def main() -> None:
    csv_path = "data/raw/glassdoor_comments.csv"

    df = load_glassdoor_csv(csv_path)
    validate_expected_columns(df)

    print("CSV loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()