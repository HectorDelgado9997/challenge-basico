import pandas as pd
from pysentimiento import create_analyzer


PYSENTIMIENTO_LABEL_MAP = {
    "POS": "positive",
    "NEG": "negative",
    "NEU": "neutral",
}


def validate_pysentimiento_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if "sentiment_text" not in df.columns:
        raise ValueError("Missing required column: sentiment_text")


def apply_pysentimiento_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    validate_pysentimiento_input(df)

    analyzer = create_analyzer(task="sentiment", lang="en")
    df = df.copy()

    predicted_labels = []
    positive_probabilities = []
    neutral_probabilities = []
    negative_probabilities = []
    max_probabilities = []

    for text in df["sentiment_text"]:
        prediction = analyzer.predict(text)

        predicted_labels.append(
            PYSENTIMIENTO_LABEL_MAP.get(prediction.output, prediction.output)
        )

        positive_probabilities.append(prediction.probas.get("POS", 0.0))
        neutral_probabilities.append(prediction.probas.get("NEU", 0.0))
        negative_probabilities.append(prediction.probas.get("NEG", 0.0))
        max_probabilities.append(max(prediction.probas.values()))

    df["pysentimiento_positive_probability"] = positive_probabilities
    df["pysentimiento_neutral_probability"] = neutral_probabilities
    df["pysentimiento_negative_probability"] = negative_probabilities
    df["pysentimiento_max_probability"] = max_probabilities
    df["pysentimiento_sentiment"] = predicted_labels

    return df


def main() -> None:
    from ingestion import run_data_ingestion
    from preprocessing import preprocess_dataframe

    csv_path = "data/raw/glassdoor_comments.csv"

    df = run_data_ingestion(csv_path)
    df = preprocess_dataframe(df)
    df = apply_pysentimiento_sentiment(df)

    print("pysentimiento sentiment analysis completed successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\npysentimiento sentiment distribution:")
    print(df["pysentimiento_sentiment"].value_counts())

    print("\nSample:")
    print(
        df[
            [
                "sentiment_text",
                "pysentimiento_positive_probability",
                "pysentimiento_neutral_probability",
                "pysentimiento_negative_probability",
                "pysentimiento_max_probability",
                "pysentimiento_sentiment",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()