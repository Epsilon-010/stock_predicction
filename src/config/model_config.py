"""Typed loader for `model_config.yaml`.

Validates the YAML against Pydantic schemas at load time, so a typo in the
config file fails fast instead of half-way through a 30-minute training run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config.settings import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "model_config.yaml"


class _Base(BaseModel):
    """Forbid unknown keys — catches typos in YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Base):
    market: Literal["jp", "us"] = "jp"
    universe: str = "tse_all"
    min_history_days: int = Field(default=250, ge=1)
    start_date: str = "2005-01-01"
    end_date: str | None = None


class TargetConfig(_Base):
    type: Literal["direction", "return", "volatility"] = "direction"
    horizon_days: int = Field(default=5, ge=1, le=60)
    threshold: float = 0.0


class _FeatureGroup(_Base):
    enabled: bool = True


class PriceFeatures(_FeatureGroup):
    windows: list[int] = Field(default_factory=lambda: [5, 20, 60])


class VolumeFeatures(_FeatureGroup):
    windows: list[int] = Field(default_factory=lambda: [5, 20])


class TechnicalFeatures(_FeatureGroup):
    indicators: list[str] = Field(default_factory=list)


class CalendarFeatures(_FeatureGroup):
    pass


class CrossSectionalFeatures(_FeatureGroup):
    market_features: list[str] = Field(default_factory=list)


class LagFeatures(_FeatureGroup):
    lags: list[int] = Field(default_factory=lambda: [1, 5])


class FeaturesConfig(_Base):
    price: PriceFeatures = Field(default_factory=PriceFeatures)
    volume: VolumeFeatures = Field(default_factory=VolumeFeatures)
    technical: TechnicalFeatures = Field(default_factory=TechnicalFeatures)
    calendar: CalendarFeatures = Field(default_factory=CalendarFeatures)
    cross_sectional: CrossSectionalFeatures = Field(default_factory=CrossSectionalFeatures)
    lag: LagFeatures = Field(default_factory=LagFeatures)


class OutlierClipping(_Base):
    enabled: bool = True
    method: Literal["winsorize", "iqr"] = "winsorize"
    quantiles: tuple[float, float] = (0.005, 0.995)


class Scaling(_Base):
    method: Literal["standard", "robust", "none"] = "standard"
    fit_on: Literal["train_only", "all"] = "train_only"


class PreprocessingConfig(_Base):
    drop_na_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    imputation: Literal["forward_fill", "zero", "median"] = "forward_fill"
    outlier_clipping: OutlierClipping = Field(default_factory=OutlierClipping)
    scaling: Scaling = Field(default_factory=Scaling)


class SplitConfig(_Base):
    strategy: Literal["walk_forward", "expanding_window"] = "walk_forward"
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str | None = None
    embargo_days: int = Field(default=5, ge=0)


class ModelSpec(_Base):
    enabled: bool = True
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    early_stopping_rounds: int | None = None


class BacktestConfig(_Base):
    enabled: bool = True
    initial_capital: float = 1_000_000
    transaction_cost_bps: float = 5
    max_position_size_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "weekly"


class EvaluationConfig(_Base):
    primary_metric: str = "roc_auc"
    metrics: list[str] = Field(default_factory=lambda: ["roc_auc", "accuracy"])
    classification_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


class MLflowYamlConfig(_Base):
    log_models: bool = True
    log_artifacts: bool = True
    log_data_profile: bool = False
    register_best_model: bool = True
    model_registry_name: str = "stock_direction_classifier"


class ModelConfig(_Base):
    """Root validated ML configuration."""

    dataset: DatasetConfig
    target: TargetConfig
    features: FeaturesConfig
    preprocessing: PreprocessingConfig
    split: SplitConfig
    models: dict[str, ModelSpec]
    evaluation: EvaluationConfig
    mlflow: MLflowYamlConfig

    @field_validator("models")
    @classmethod
    def _at_least_one_enabled(cls, models: dict[str, ModelSpec]) -> dict[str, ModelSpec]:
        if not any(m.enabled for m in models.values()):
            raise ValueError("At least one model must have `enabled: true`")
        return models

    def enabled_models(self) -> dict[str, ModelSpec]:
        """Return only the model specs marked as enabled."""
        return {name: spec for name, spec in self.models.items() if spec.enabled}


@lru_cache(maxsize=4)
def load_model_config(path: Path | str | None = None) -> ModelConfig:
    """Load and validate `model_config.yaml`. Cached by path."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ModelConfig.model_validate(raw)
