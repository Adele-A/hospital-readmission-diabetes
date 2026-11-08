"""Tests for the ICD-9 clinical grouping (Strack et al. scheme)."""

import pytest

from src.features.icd9_groups import map_icd9


@pytest.mark.parametrize(
    "code,expected",
    [
        ("250", "diabetes"),
        ("250.01", "diabetes"),
        ("250.83", "diabetes"),
        ("414", "circulatory"),
        ("410.0", "circulatory"),
        ("785", "circulatory"),   # specific symptom code
        ("486", "respiratory"),
        ("786", "respiratory"),
        ("530", "digestive"),
        ("787", "digestive"),
        ("590", "genitourinary"),
        ("788", "genitourinary"),
        ("715", "musculoskeletal"),
        ("200", "neoplasms"),
        ("140", "neoplasms"),
        ("850", "injury"),
        ("999", "injury"),
        ("V45", "other"),
        ("E909", "other"),
        ("780", "other"),         # symptom code not carved out
    ],
)
def test_known_codes(code, expected):
    assert map_icd9(code) == expected


@pytest.mark.parametrize("missing", ["?", "", "   ", None, "nan", float("nan")])
def test_missing_codes(missing):
    assert map_icd9(missing) == "Missing"


def test_range_boundaries_inclusive():
    assert map_icd9("390") == "circulatory"
    assert map_icd9("459") == "circulatory"
    assert map_icd9("389") == "other"  # just below circulatory
    assert map_icd9("460") == "respiratory"
