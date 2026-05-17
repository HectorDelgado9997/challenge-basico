import pytest

import sys
sys.path.insert(0, "src")

from preprocessing import (
    normalize_for_sentiment,
    normalize_for_features,
    tokenize_text,
    remove_stopwords_for_features,
    simple_lemmatize_token,
    lemmatize_tokens,
)


# ─── normalize_for_sentiment ───

def test_normalize_for_sentiment_lowercase():
    assert normalize_for_sentiment("GREAT Company") == "great company"


def test_normalize_for_sentiment_removes_urls():
    result = normalize_for_sentiment("Visit https://example.com for more")
    assert "https" not in result
    assert "example.com" not in result


def test_normalize_for_sentiment_preserves_negation():
    result = normalize_for_sentiment("Not a good place")
    assert "not" in result


def test_normalize_for_sentiment_preserves_intensifier():
    result = normalize_for_sentiment("Very good benefits")
    assert "very" in result


def test_normalize_for_sentiment_handles_non_string():
    result = normalize_for_sentiment(None)
    assert result == ""


# ─── normalize_for_features ───

def test_normalize_for_features_lowercase():
    assert "GREAT" not in normalize_for_features("GREAT Company")


def test_normalize_for_features_removes_punctuation():
    result = normalize_for_features("Great! Amazing. Wonderful?")
    assert "!" not in result
    assert "." not in result


def test_normalize_for_features_removes_urls():
    result = normalize_for_features("Visit https://example.com today")
    assert "https" not in result


# ─── tokenize_text ───

def test_tokenize_text_splits_on_spaces():
    result = tokenize_text("good team culture")
    assert result == ["good", "team", "culture"]


def test_tokenize_text_empty_string():
    result = tokenize_text("")
    assert result == []


# ─── remove_stopwords_for_features ───

def test_remove_stopwords_removes_basic_stopwords():
    tokens = ["the", "company", "is", "great"]
    result = remove_stopwords_for_features(tokens)
    assert "the" not in result
    assert "is" not in result
    assert "company" in result
    assert "great" in result


def test_remove_stopwords_preserves_negations():
    tokens = ["not", "good", "at", "all"]
    result = remove_stopwords_for_features(tokens)
    assert "not" in result


def test_remove_stopwords_preserves_intensifiers():
    tokens = ["very", "good", "the", "team"]
    result = remove_stopwords_for_features(tokens)
    assert "very" in result


# ─── simple_lemmatize_token ───

def test_lemmatize_removes_ing():
    assert simple_lemmatize_token("working") == "work"


def test_lemmatize_removes_ed():
    assert simple_lemmatize_token("worked") == "work"


def test_lemmatize_removes_s():
    assert simple_lemmatize_token("teams") == "team"


def test_lemmatize_preserves_negation():
    assert simple_lemmatize_token("not") == "not"


def test_lemmatize_ies_to_y():
    assert simple_lemmatize_token("companies") == "company"


# ─── lemmatize_tokens ───

def test_lemmatize_tokens_applies_to_all():
    tokens = ["working", "teams", "not"]
    result = lemmatize_tokens(tokens)
    assert "work" in result
    assert "team" in result
    assert "not" in result
