"""Бизнес-правила: возраст > 42 → направление к специалисту."""

from __future__ import annotations

import pandas as pd

from maternal_risk.config import settings


def split_by_age(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет на 'молодых' (для ML) и 'возрастных' (для бизнес-правила)."""
    threshold = settings.data.age_threshold
    young = df[df["Age"] <= threshold].copy()
    older = df[df["Age"] > threshold].copy()
    return young, older


def apply_age_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Для пациенток > 42 лет автоматически присваивает маршрут 'referral'."""
    df = df.copy()
    df["route"] = "ml"
    df.loc[df["Age"] > settings.data.age_threshold, "route"] = "referral"
    return df
