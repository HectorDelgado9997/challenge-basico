import pandas as pd
import pytest

import sys
sys.path.insert(0, "src")

from ingestion import (
    build_review_text,
    cast_text_columns,
    remove_invalid_records,
    validate_dataframe_input,
    validate_required_text_columns,
)
from exceptions import DataIngestionError


# ─── validate_dataframe_input ───

def test_validate_dataframe_input_valid():
    df = pd.DataFrame({"a": [1, 2]})
    validate_dataframe_input(df)


def test_validate_dataframe_input_not_dataframe():
    with pytest.raises(DataIngestionError):
        validate_dataframe_input([1, 2, 3])


def test_validate_dataframe_input_empty():
    with pytest.raises(DataIngestionError):
        validate_dataframe_input(pd.DataFrame())


# ─── validate_required_text_columns ───

def test_validate_required_text_columns_valid():
    df = pd.DataFrame({"headline": [], "pros": [], "cons": []})
    validate_required_text_columns(df)


def test_validate_required_text_columns_missing():
    df = pd.DataFrame({"headline": [], "pros": []})
    with pytest.raises(DataIngestionError):
        validate_required_text_columns(df)


# ─── cast_text_columns ───

def test_cast_text_columns_fills_nan():
    df = pd.DataFrame({
        "headline": [None, "Good company"],
        "pros": ["Great team", None],
        "cons": [None, None],
    })
    result = cast_text_columns(df)
    assert result["headline"].iloc[0] == ""
    assert result["pros"].iloc[1] == ""
    assert result["cons"].iloc[0] == ""


def test_cast_text_columns_casts_to_str():
    df = pd.DataFrame({
        "headline": [123],
        "pros": [456],
        "cons": [789],
    })
    result = cast_text_columns(df)
    assert result["headline"].dtype == object


# ─── remove_invalid_records ───

def test_remove_invalid_records_removes_duplicates():
    df = pd.DataFrame({
        "headline": ["Good", "Good"],
        "pros": ["Nice", "Nice"],
        "cons": ["None", "None"],
    })
    result = remove_invalid_records(df)
    assert len(result) == 1


def test_remove_invalid_records_removes_all_empty():
    df = pd.DataFrame({
        "headline": ["", "Good company"],
        "pros": ["", "Great benefits"],
        "cons": ["", "Long hours"],
    })
    result = remove_invalid_records(df)
    assert len(result) == 1
    assert result["headline"].iloc[0] == "Good company"


# ─── build_review_text ───

def test_build_review_text_concatenates_columns():
    df = pd.DataFrame({
        "headline": ["Great place"],
        "pros": ["Good team"],
        "cons": ["Low salary"],
    })
    result = build_review_text(df)
    assert result["review_text"].iloc[0] == "Great place Good team Low salary"


def test_build_review_text_strips_whitespace():
    df = pd.DataFrame({
        "headline": ["  Great  "],
        "pros": ["  Good  "],
        "cons": ["  Bad  "],
    })
    result = build_review_text(df)
    assert result["review_text"].iloc[0] == "Great Good Bad"


def test_build_review_text_raises_if_all_empty():
    df = pd.DataFrame({
        "headline": [""],
        "pros": [""],
        "cons": [""],
    })
    with pytest.raises(DataIngestionError):
        build_review_text(df)
