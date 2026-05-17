from pathlib import Path

import pandas as pd

from logger import get_logger


logger = get_logger(__name__)

REQUIRED_TEXT_COLUMNS = ["headline", "pros", "cons"]


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

            logger.info("CSV loaded using encoding: %s", encoding)
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


def validate_dataframe_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")


def validate_required_text_columns(df: pd.DataFrame) -> None:
    missing_columns = set(REQUIRED_TEXT_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required text columns: {sorted(missing_columns)}")


def select_required_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[REQUIRED_TEXT_COLUMNS].copy()


def cast_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in REQUIRED_TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str)

    return df


def remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    initial_rows = len(df)

    df = df.drop_duplicates()

    df = df[
        df[REQUIRED_TEXT_COLUMNS]
        .apply(lambda row: any(value.strip() for value in row), axis=1)
    ].copy()

    removed_rows = initial_rows - len(df)
    logger.info("Invalid or duplicated records removed: %d", removed_rows)

    return df


def build_review_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["review_text"] = (
        df["headline"].str.strip()
        + " "
        + df["pros"].str.strip()
        + " "
        + df["cons"].str.strip()
    ).str.strip()

    df = df[df["review_text"].str.len() > 0].copy()

    return df


def run_data_ingestion(csv_path: str) -> pd.DataFrame:
    df = load_glassdoor_csv(csv_path)

    validate_dataframe_input(df)
    validate_required_text_columns(df)

    df = select_required_text_columns(df)
    df = cast_text_columns(df)
    df = remove_invalid_records(df)
    df = build_review_text(df)

    logger.info("Data ingestion completed. Rows: %d", df.shape[0])

    return df


def main() -> None:
    csv_path = "data/raw/glassdoor_comments.csv"

    df = run_data_ingestion(csv_path)

    logger.info("Columns: %s", df.columns.tolist())
    logger.info("Sample review_text: %s", df["review_text"].iloc[0])


if __name__ == "__main__":
    main()
