import re
from typing import List

import pandas as pd


NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "nothing",
    "neither",
    "nor",
    "without",
    "cannot",
    "can't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
}


INTENSIFIER_WORDS = {
    "very",
    "too",
    "so",
    "really",
    "extremely",
    "highly",
    "quite",
    "rather",
    "absolutely",
    "completely",
    "totally",
}


BASIC_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "while",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "you",
    "your",
    "they",
    "their",
    "we",
    "our",
    "i",
    "me",
    "my",
    "he",
    "she",
    "his",
    "her",
    "them",
}


STOPWORDS_FOR_FEATURES = BASIC_STOPWORDS - NEGATION_WORDS - INTENSIFIER_WORDS


def validate_preprocessing_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if "review_text" not in df.columns:
        raise ValueError("Missing required column: review_text")


def normalize_for_sentiment(text: str) -> str:
    """
    Minimal cleaning for sentiment models.

    This version preserves negations, intensifiers and sentence-level information
    as much as possible. It is intended for VADER and pysentimiento.
    """
    if not isinstance(text, str):
        text = ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_for_features(text: str) -> str:
    """
    Stronger cleaning for classical NLP features such as TF-IDF and n-grams.
    """
    if not isinstance(text, str):
        text = ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    return text.split()


def remove_stopwords_for_features(tokens: List[str]) -> List[str]:
    return [
        token
        for token in tokens
        if token not in STOPWORDS_FOR_FEATURES and len(token) > 2
    ]


def simple_lemmatize_token(token: str) -> str:
    """
    Lightweight lemmatization approximation.

    This avoids external dependencies at this stage. A later improvement can
    replace this with spaCy lemmatization.
    """
    if token in NEGATION_WORDS:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("ing") and len(token) > 5:
        return token[:-3]

    if token.endswith("ed") and len(token) > 4:
        return token[:-2]

    if token.endswith("s") and len(token) > 3:
        return token[:-1]

    return token


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    return [simple_lemmatize_token(token) for token in tokens]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validate_preprocessing_input(df)

    df = df.copy()

    df["sentiment_text"] = df["review_text"].apply(normalize_for_sentiment)

    df["clean_text"] = df["review_text"].apply(normalize_for_features)
    df["tokens"] = df["clean_text"].apply(tokenize_text)
    df["tokens_no_stopwords"] = df["tokens"].apply(remove_stopwords_for_features)
    df["lemmas"] = df["tokens_no_stopwords"].apply(lemmatize_tokens)
    df["processed_text"] = df["lemmas"].apply(lambda tokens: " ".join(tokens))

    df = df[df["sentiment_text"].str.len() > 0].copy()
    df = df[df["processed_text"].str.len() > 0].copy()

    return df


def main() -> None:
    from ingestion import run_data_ingestion

    csv_path = "data/raw/glassdoor_comments.csv"

    df = run_data_ingestion(csv_path)
    df = preprocess_dataframe(df)

    print("Text preprocessing completed successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nSample:")
    print(
        df[
            [
                "review_text",
                "sentiment_text",
                "processed_text",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()