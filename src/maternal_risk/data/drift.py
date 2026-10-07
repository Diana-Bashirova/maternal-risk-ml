"""Проверка дрейфа данных KS-тестом."""

from __future__ import annotations

import pandas as pd
from loguru import logger
from scipy import stats


def check_data_drift(X_train: pd.DataFrame, X_new: pd.DataFrame, threshold: float = 0.05) -> bool:
    numeric_cols = X_train.select_dtypes(include=["number"]).columns
    drift_detected = False
    logger.info(f"Проверка дрейфа данных (KS-тест, порог p={threshold}):")

    for col in numeric_cols:
        train_vals = X_train[col].dropna().values
        new_vals = X_new[col].dropna().values
        if len(train_vals) < 2 or len(new_vals) < 2:
            continue
        _, p_value = stats.ks_2samp(train_vals, new_vals)
        status = "[!] ДРЕЙФ" if p_value < threshold else "[OK]"
        logger.info(f"  {status} | {col:<20} | p-value={p_value:.4f}")
        if p_value < threshold:
            drift_detected = True

    logger.info("Обнаружен дрейф данных!" if drift_detected else "Дрейф не обнаружен.")
    return drift_detected
