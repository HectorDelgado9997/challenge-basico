from pathlib import Path

import pandas as pd

from ingestion import run_data_ingestion
from preprocessing import preprocess_dataframe
from sentiment_pysentimiento import apply_pysentimiento_sentiment
from sentiment_vader import apply_vader_sentiment


def calculate_model_agreement(df: pd.DataFrame) -> float:
    if "vader_sentiment" not in df.columns:
        raise ValueError("Missing required column: vader_sentiment")

    if "pysentimiento_sentiment" not in df.columns:
        raise ValueError("Missing required column: pysentimiento_sentiment")

    return float((df["vader_sentiment"] == df["pysentimiento_sentiment"]).mean())


def run_sentiment_analysis_pipeline(
    input_path: str = "data/raw/glassdoor_comments.csv",
    output_path: str = "data/processed/glassdoor_sentiment_results.csv",
) -> pd.DataFrame:
    df = run_data_ingestion(input_path)
    df = preprocess_dataframe(df)

    df = apply_vader_sentiment(df)
    df = apply_pysentimiento_sentiment(df)

    agreement_rate = calculate_model_agreement(df)

    print("Sentiment analysis pipeline completed successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"VADER vs pysentimiento agreement rate: {agreement_rate:.4f}")

    print("\nVADER sentiment distribution:")
    print(df["vader_sentiment"].value_counts())

    print("\npysentimiento sentiment distribution:")
    print(df["pysentimiento_sentiment"].value_counts())

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return df


def main() -> None:
    run_sentiment_analysis_pipeline()


if __name__ == "__main__":
    main()