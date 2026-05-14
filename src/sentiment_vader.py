import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def validate_vader_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if "sentiment_text" not in df.columns:
        raise ValueError("Missing required column: sentiment_text")


def map_vader_compound_to_label(compound_score: float) -> str:
    if compound_score >= 0.05:
        return "positive"

    if compound_score <= -0.05:
        return "negative"

    return "neutral"


def apply_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    validate_vader_input(df)

    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()

    vader_scores = df["sentiment_text"].apply(analyzer.polarity_scores)

    df["vader_negative_score"] = vader_scores.apply(lambda scores: scores["neg"])
    df["vader_neutral_score"] = vader_scores.apply(lambda scores: scores["neu"])
    df["vader_positive_score"] = vader_scores.apply(lambda scores: scores["pos"])
    df["vader_compound_score"] = vader_scores.apply(lambda scores: scores["compound"])

    df["vader_sentiment"] = df["vader_compound_score"].apply(
        map_vader_compound_to_label
    )

    return df


def main() -> None:
    from ingestion import run_data_ingestion
    from preprocessing import preprocess_dataframe

    csv_path = "data/raw/glassdoor_comments.csv"

    df = run_data_ingestion(csv_path)
    df = preprocess_dataframe(df)
    df = apply_vader_sentiment(df)

    print("VADER sentiment analysis completed successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nVADER sentiment distribution:")
    print(df["vader_sentiment"].value_counts())

    print("\nSample:")
    print(
        df[
            [
                "sentiment_text",
                "vader_negative_score",
                "vader_neutral_score",
                "vader_positive_score",
                "vader_compound_score",
                "vader_sentiment",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()