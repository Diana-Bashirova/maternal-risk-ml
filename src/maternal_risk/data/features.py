"""Инжиниринг признаков и кодирование целевой переменной."""

from __future__ import annotations

import pandas as pd

RISK_MAPPING = {"low risk": 0, "mid risk": 1, "high risk": 2}
CLASS_NAMES = {v: k for k, v in RISK_MAPPING.items()}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PulsePressure"] = df["SystolicBP"] - df["DiastolicBP"]
    for col in ["SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    df = df.copy()
    df["RiskLevel"] = df["RiskLevel"].map(RISK_MAPPING)
    return df, CLASS_NAMES
