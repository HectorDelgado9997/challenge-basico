import pandas as pd
import pytest

import sys
sys.path.insert(0, "src")

from evaluation import (
    calculate_agreement_rate,
    build_sentiment_distribution,
    build_model_comparison_table,
    validate_evaluation_input,
)
from exceptions import EvaluationError


def make_df(vader, pysentimiento):
    return pd.DataFrame({
        "vader_sentiment": vader,
        "pysentimiento_sentiment": pysentimiento,
    })


# ─── validate_evaluation_input ───

def test_validate_evaluation_input_valid():
    df = make_df(["positive"], ["positive"])
    validate_evaluation_input(df)


def test_validate_evaluation_input_not_dataframe():
    with pytest.raises(Exception):
        validate_evaluation_input({"a": 1})


def test_validate_evaluation_input_missing_columns():
    df = pd.DataFrame({"vader_sentiment": ["positive"]})
    with pytest.raises(Exception):
        validate_evaluation_input(df)


# ─── calculate_agreement_rate ───

def test_agreement_rate_full_agreement():
    df = make_df(
        ["positive", "negative", "neutral"],
        ["positive", "negative", "neutral"]
    )
    assert calculate_agreement_rate(df) == 1.0


def test_agreement_rate_no_agreement():
    df = make_df(
        ["positive", "positive"],
        ["negative", "neutral"]
    )
    assert calculate_agreement_rate(df) == 0.0


def test_agreement_rate_partial():
    df = make_df(
        ["positive", "positive", "negative", "negative"],
        ["positive", "negative", "negative", "positive"]
    )
    assert calculate_agreement_rate(df) == 0.5


# ─── build_sentiment_distribution ───

def test_build_sentiment_distribution_shape():
    df = make_df(
        ["positive", "negative", "positive"],
        ["positive", "neutral", "negative"]
    )
    result = build_sentiment_distribution(df)
    assert "vader" in result.columns
    assert "pysentimiento" in result.columns


# ─── build_model_comparison_table ───

def test_build_model_comparison_table_returns_dataframe():
    df = make_df(
        ["positive", "negative", "neutral"],
        ["positive", "positive", "negative"]
    )
    result = build_model_comparison_table(df)
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "vader"
    assert result.columns.name == "pysentimiento"
