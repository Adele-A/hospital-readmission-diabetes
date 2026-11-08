"""Leakage-safe preprocessing for modelling.

All *learned* transforms live inside this ``ColumnTransformer`` so they are fit on
the training folds only (never on the held-out test set):

  - numeric: median imputation (+ optional standardisation for linear models),
  - categorical: constant ``Missing`` imputation, then one-hot encoding with
    **rare-level grouping** (``min_frequency``) and safe handling of categories
    unseen at fit time (``handle_unknown="infrequent_if_exist"``).

The one-hot encoder learns its categories and which levels are "infrequent" from
the training data passed to ``fit`` — so wrapping it in a scikit-learn pipeline
and fitting on the train split is what prevents leakage.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale: bool = True,
    min_frequency: float = 0.01,
) -> ColumnTransformer:
    """Build the fit-on-train-only preprocessing ColumnTransformer.

    Parameters
    ----------
    scale:
        Standardise numeric features (useful for logistic regression; harmless but
        unnecessary for gradient boosting, which can set ``scale=False``).
    min_frequency:
        Categorical levels rarer than this fraction of the training rows are
        grouped into a single ``infrequent`` level — learned on train only.
    """
    num_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    numeric_tf = Pipeline(num_steps)

    categorical_tf = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_features),
            ("cat", categorical_tf, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
