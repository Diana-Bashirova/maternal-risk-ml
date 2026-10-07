"""Inference: загрузка текущей модели и предсказания."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger


class SecurityError(Exception):
    pass


def verify_model_integrity(model_path: Path, metadata_path: Path) -> bool:
    """Проверка контрольной суммы модели перед загрузкой."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = metadata["sha256"]
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return bool(actual == expected)


def load_current_model() -> tuple[Any, Any]:
    models_root = Path(__file__).parents[3] / "models"
    current_link = models_root / "current"

    if current_link.is_symlink():
        version = current_link.resolve().name
    elif current_link.exists():
        version = current_link.read_text().strip()
    else:
        raise FileNotFoundError("Нет текущей версии модели (models/current).")

    current_dir = models_root / version
    model_path = current_dir / "model.joblib"
    metadata_path = current_dir / "metadata.json"

    if not verify_model_integrity(model_path, metadata_path):
        raise SecurityError("Model integrity check failed!")

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def predict(df: pd.DataFrame) -> pd.DataFrame:
    model, metadata = load_current_model()
    threshold = metadata["threshold"]

    proba = model.predict_proba(df)
    max_probs = proba.max(axis=1)
    confident = max_probs >= threshold

    result = df.copy()
    result["predicted_class"] = -1
    result.loc[confident, "predicted_class"] = proba[confident].argmax(axis=1)
    result["max_proba"] = max_probs
    result["route"] = np.where(confident, "auto", "manual_review")

    logger.info(
        f"[PREDICT] Всего: {len(df)}, Авто: {confident.sum()}, "
        f"Ручная проверка: {(~confident).sum()}"
    )
    return result
