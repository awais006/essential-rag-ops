import pytest

from essential_rag.scorers import (
    is_not_empty,
    keyword_present,
    not_hallucinating,
    response_length_ok,
)


@pytest.mark.parametrize("text, expected", [
    ("hello world", True),
    ("", False),
    ("   ", False),          # whitespace-only is empty
])
def test_is_not_empty(text, expected):
    assert is_not_empty(outputs=text) == expected


@pytest.mark.parametrize("text, expected", [
    ("MLflow is a platform for the ML lifecycle.", True),   # a real answer
    ("I don't know based on the context.", False),          # a refusal phrase
    ("That is not in the context provided.", False),
])
def test_not_hallucinating(text, expected):
    assert not_hallucinating(outputs=text) == expected


def test_keyword_present():
    assert keyword_present("MLflow does machine learning", {"keyword": "machine learning"}) is True
    assert keyword_present("unrelated", {"keyword": "machine learning"}) is False
    assert keyword_present("anything", {}) is False


@pytest.mark.parametrize("word_count, expected", [
    (4, False),     # below the 5-word floor
    (5, True),      # lower boundary
    (100, True),    # upper boundary
    (101, False),   # above the 100-word ceiling
])
def test_response_length_ok(word_count, expected):
    text = " ".join(["word"] * word_count)
    assert response_length_ok(outputs=text) == expected