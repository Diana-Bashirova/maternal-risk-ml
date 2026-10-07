"""Обучение, подбор порога, сохранение модели с версионированием."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, make_scorer, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from maternal_risk.business.rules import split_by_age
from maternal_risk.config import settings
from maternal_risk.data.drift import check_data_drift
from maternal_risk.data.features import encode_target, engineer_features
from maternal_risk.data.loading import load_and_clean_data
from maternal_risk.models.pipeline import create_pipeline


def train() -> None:
    # 1. Данные
    df = load_and_clean_data(settings.data.path)
    df = engineer_features(df)
    df, class_names = encode_target(df)
    young_df, older_df = split_by_age(df)

    train_df, test_df = train_test_split(
        young_df,
        test_size=settings.data.test_size,
        random_state=settings.data.random_state,
        stratify=young_df["RiskLevel"],
    )

    X_train_raw = train_df.drop(columns=["RiskLevel"])
    y_train = train_df["RiskLevel"]
    X_test_raw = test_df.drop(columns=["RiskLevel"])
    _y_test = test_df["RiskLevel"]

    check_data_drift(X_train_raw, X_test_raw)

    # 2. Pipeline + веса классов
    pipeline = create_pipeline()
    sample_weights = np.ones(len(y_train), dtype=np.float64)
    sample_weights[y_train == 1] = settings.weights["mid"]
    sample_weights[y_train == 2] = settings.weights["high"]

    recall_scorer = make_scorer(recall_score, labels=[2], average="macro", zero_division=0)
    cv = StratifiedKFold(
        n_splits=settings.model.cv_splits,
        shuffle=True,
        random_state=settings.data.random_state,
    )

    search = RandomizedSearchCV(
        pipeline,
        {
            "classifier__n_estimators": settings.model.n_estimators,
            "classifier__max_depth": settings.model.max_depth,
            "classifier__learning_rate": settings.model.learning_rate,
            "classifier__subsample": [0.8, 1.0],
        },
        n_iter=15,
        cv=cv,
        scoring=recall_scorer,
        n_jobs=1,
        random_state=settings.data.random_state,
        verbose=1,
    )
    logger.info("[TUNE] Оптимизация гиперпараметров...")
    search.fit(X_train_raw, y_train, classifier__sample_weight=sample_weights)
    best_model = search.best_estimator_
    logger.info(f"[BEST] Параметры: {search.best_params_}")

    # 3. Подбор порога через Out-of-Fold
    oof_proba = cross_val_predict(
        best_model, X_train_raw, y_train, cv=cv, method="predict_proba", n_jobs=1
    )
    max_probs = oof_proba.max(axis=1)
    lo, hi, step = settings.threshold["search_range"]
    best_threshold, best_score = 0.5, -1.0

    for thr in np.arange(lo, hi, step):  # type: ignore[attr-defined]
        mask = max_probs >= thr
        if mask.sum() == 0:
            continue
        y_pred = oof_proba[mask].argmax(axis=1)
        acc = accuracy_score(y_train[mask], y_pred)
        rec = recall_score(y_train[mask], y_pred, labels=[2], average="macro", zero_division=0)
        auto_ratio = mask.sum() / len(y_train)
        score = 0.3 * acc + 0.5 * rec + 0.2 * auto_ratio
        if score > best_score:
            best_score, best_threshold = score, float(thr)

    logger.info(f"[THRESHOLD] Оптимальный порог: {best_threshold:.2f}")

    # 4. Финальное обучение
    best_model.fit(X_train_raw, y_train, classifier__sample_weight=sample_weights)

    # 5. Сохранение с версией
    save_model(best_model, best_threshold, class_names)


def save_model(model: Any, threshold: float, class_names: dict[int, str]) -> None:
    """Версионирует модель: models/v{YYYYMMDD_HHMMSS}/."""
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    models_dir = Path(__file__).parents[3] / "models" / version
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Контрольная сумма для защиты от подмены
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "threshold": threshold,
        "class_names": {str(k): v for k, v in class_names.items()},
        "sha256": sha256,
        "config": settings.model_dump(),
    }
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    # Обновляем symlink 'current'
    current_link = Path(__file__).parents[3] / "models" / "current"
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    try:
        current_link.symlink_to(version)
    except OSError:
        # Windows: fallback на текстовый файл
        current_link.write_text(version)

    logger.info(f"[SAVE] Модель сохранена: models/{version}")
    logger.info(f"[SAVE] SHA-256: {sha256}")


if __name__ == "__main__":
    train()
