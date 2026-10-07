"""Расчёт экономического эффекта."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from maternal_risk.config import settings


def compute_business_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,  # type: ignore[type-arg]
    confident_mask: np.ndarray,  # type: ignore[type-arg]
) -> dict[str, int]:
    if confident_mask.sum() == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "saved": 0}

    y_t, y_p = y_true[confident_mask], y_pred[confident_mask]
    tp = int(((y_p == 2) & (y_t == 2)).sum())
    fp = int(((y_p == 2) & (y_t != 2)).sum())
    fn = int(((y_p != 2) & (y_t == 2)).sum())

    cost_fn = settings.costs["fn_high_risk"]
    cost_fp = settings.costs["fp_high_risk"]
    saved = int((y_t == 2).sum()) * cost_fn - (fn * cost_fn + fp * cost_fp)

    logger.info(f"[BUSINESS] TP: {tp}, FP: {fp}, FN: {fn}")
    logger.info(f"[BUSINESS] Экономия: {saved:,} руб.")
    return {"tp": tp, "fp": fp, "fn": fn, "saved": saved}
