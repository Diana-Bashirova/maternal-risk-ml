"""Загрузка и очистка данных."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from maternal_risk.config import settings


def load_and_clean_data(file_path: str | Path) -> pd.DataFrame:
    """Загружает данные и фильтрует некорректные записи."""
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"Файл {file_path} не найден. Генерирую синтетические данные.")
        df = _generate_synthetic_data(n_samples=1200, seed=settings.data.random_state)
        df.to_excel(file_path, index=False)

    df = pd.read_excel(file_path)
    n_before = len(df)
    df = df[df["Age"] >= settings.data.age_min].copy().reset_index(drop=True)

    logger.info(
        f"Загружено: {n_before} | После фильтрации (<{settings.data.age_min} лет): {len(df)}"
    )
    return df


def _generate_synthetic_data(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Age": rng.integers(16, 50, n_samples),
            "SystolicBP": rng.normal(120, 15, n_samples).astype(int),  # type: ignore[attr-defined]
            "DiastolicBP": rng.normal(80, 10, n_samples).astype(int),  # type: ignore[attr-defined]
            "BS": rng.normal(100, 25, n_samples),
            "BodyTemp": rng.normal(37.0, 0.5, n_samples),
            "HeartRate": rng.normal(80, 12, n_samples).astype(int),  # type: ignore[attr-defined]
            "RiskLevel": rng.choice(
                ["low risk", "mid risk", "high risk"], n_samples, p=[0.65, 0.25, 0.10]
            ),
        }
    )
    df["SystolicBP"] = df["SystolicBP"].clip(90, 180)
    df["DiastolicBP"] = df["DiastolicBP"].clip(50, 110)
    df["BS"] = df["BS"].clip(70, 250)
    df["BodyTemp"] = df["BodyTemp"].clip(36.0, 39.0)
    df["HeartRate"] = df["HeartRate"].clip(60, 130)
    return df
