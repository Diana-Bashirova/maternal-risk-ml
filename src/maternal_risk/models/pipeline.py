"""Создание ML-pipeline."""

from __future__ import annotations

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from maternal_risk.config import settings


def create_pipeline() -> Pipeline:
    imputer = IterativeImputer(
        max_iter=10, random_state=settings.data.random_state, add_indicator=False
    )
    return Pipeline(
        [
            ("imputer", imputer),
            (
                "classifier",
                XGBClassifier(
                    tree_method="hist",
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=settings.data.random_state,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )
