"""Feature schema for modelling — which columns are features and of what type.

Stateless, row-wise feature construction lives here (ICD-9 grouping of the three
diagnosis codes, mapping the ``"?"`` marker to an explicit ``Missing`` category,
numeric casting). These transforms cannot leak because they do not learn anything
from the data distribution.

Everything that *does* learn from the data (imputation statistics, which rare
levels to group, scaling) is defined in ``src/models/pipeline.py`` and is fit on
the training folds only.

Never-features (identifiers / outcome / derived-from-outcome): ``encounter_id``,
``patient_nbr``, ``readmitted``, ``y``, ``is_first_encounter``; plus ``weight``
(dropped, ~97% missing) and the raw ``diag_1..3`` (replaced by their groups).
"""

from __future__ import annotations

import pandas as pd

from src.features.icd9_groups import map_icd9

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Columns that must never be used as features.
DROP_ALWAYS = {
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "y",
    "is_first_encounter",
    "weight",
    "diag_1",
    "diag_2",
    "diag_3",
}

MISSING_TOKEN = "Missing"


def add_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add diag_1/2/3_group columns from the raw ICD-9 codes (stateless)."""
    for i in (1, 2, 3):
        df[f"diag_{i}_group"] = df[f"diag_{i}"].map(map_icd9)
    return df


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (X, numeric_features, categorical_features) for the given cohort df.

    - adds the three diagnosis-group columns,
    - casts numeric features to float (missing -> NaN, imputed in the pipeline),
    - maps the ``"?"`` marker to an explicit ``Missing`` category in object columns.
    """
    df = df.copy()
    df = add_diagnosis_groups(df)

    for c in NUMERIC_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    categorical = sorted(
        c
        for c in df.columns
        if c not in DROP_ALWAYS and c not in NUMERIC_FEATURES
    )
    # Stateless "?" -> Missing on categorical columns only.
    for c in categorical:
        if df[c].dtype == object:
            df[c] = df[c].replace("?", MISSING_TOKEN)

    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    x = df[numeric + categorical].copy()
    return x, numeric, categorical
