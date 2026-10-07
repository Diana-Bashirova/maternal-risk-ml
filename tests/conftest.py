"""Общие фикстуры для тестов."""

import pandas as pd
import pytest


@pytest.fixture
def sample_patients_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [25, 30, 45, 50, 17],
            "SystolicBP": [120, 130, 140, 150, 110],
            "DiastolicBP": [80, 85, 90, 95, 70],
            "BS": [90, 100, 110, 120, 85],
            "BodyTemp": [37.0, 37.1, 37.2, 37.5, 36.8],
            "HeartRate": [75, 80, 85, 90, 70],
            "RiskLevel": ["low risk", "mid risk", "high risk", "high risk", "low risk"],
        }
    )
