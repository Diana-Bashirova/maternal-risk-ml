"""Централизованная конфигурация с валидацией через Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    path: Path
    age_threshold: int = Field(gt=0, lt=120)
    age_min: int = Field(ge=0, lt=100)
    test_size: float = Field(gt=0.0, lt=1.0)
    random_state: int


class ModelConfig(BaseModel):
    n_estimators: list[int]
    max_depth: list[int]
    learning_rate: list[float]
    cv_splits: int = Field(gt=1)


class Settings(BaseModel):
    data: DataConfig
    model: ModelConfig
    costs: dict[str, int]
    threshold: dict[str, Any]
    weights: dict[str, float]

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        with open(path, encoding="utf-8") as f:
            return cls(**yaml.safe_load(f))


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "config.yaml"
settings = Settings.from_yaml(CONFIG_PATH)
