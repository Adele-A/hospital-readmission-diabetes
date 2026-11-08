"""Group ICD-9 diagnosis codes into clinical categories (Strack et al. 2014).

The scheme matches the one used in the original Diabetes 130-US paper so results
are comparable:

    circulatory      390-459, 785
    respiratory      460-519, 786
    digestive        520-579, 787
    diabetes         250.xx
    injury           800-999
    musculoskeletal  710-739
    genitourinary    580-629, 788
    neoplasms        140-239
    other            everything else, including V- and E-codes

Missing codes ("?", empty, None) map to ``"Missing"``. This module is versioned
and tested (``tests/test_icd9_groups.py``) because the grouping is a modelling
decision, not an incidental detail.
"""

from __future__ import annotations

MISSING = "Missing"
OTHER = "other"

# (inclusive low, inclusive high, label) ranges on the integer part of the code.
_RANGES: list[tuple[int, int, str]] = [
    (390, 459, "circulatory"),
    (460, 519, "respiratory"),
    (520, 579, "digestive"),
    (800, 999, "injury"),
    (710, 739, "musculoskeletal"),
    (580, 629, "genitourinary"),
    (140, 239, "neoplasms"),
]

# Specific symptom codes carved out to their system.
_SPECIFIC: dict[int, str] = {
    785: "circulatory",
    786: "respiratory",
    787: "digestive",
    788: "genitourinary",
}


def map_icd9(code: str | float | None) -> str:
    """Map a single ICD-9 code (string like ``"250.01"``, ``"414"``, ``"V45"``)
    to its clinical group. Returns ``"Missing"`` for missing values."""
    if code is None:
        return MISSING
    text = str(code).strip()
    if text == "" or text == "?" or text.lower() == "nan":
        return MISSING
    # V- and E-codes are not numeric diagnoses in this scheme -> other.
    if text[0] in ("V", "v", "E", "e"):
        return OTHER
    try:
        whole = int(float(text))
    except ValueError:
        return OTHER
    if whole == 250:  # 250.xx -> diabetes
        return "diabetes"
    if whole in _SPECIFIC:
        return _SPECIFIC[whole]
    for low, high, label in _RANGES:
        if low <= whole <= high:
            return label
    return OTHER
