"""Feature engineering pipeline (silver → processed).

Modules:
  * `technical`        — RSI, MACD, Bollinger, ATR, OBV, Stochastic, volatility
  * `calendar`         — day-of-week / month / quarter / month-end flags
  * `lag`              — lagged features (returns, RSI, MACD, …)
  * `cross_sectional`  — per-day rank within universe, market-relative returns
  * `labels`           — forward returns + direction labels (y)
  * `build_features`   — top-level pipeline
"""

from src.features.build_features import build_features, build_ticker_features
from src.features.calendar import add_calendar_features
from src.features.cross_sectional import add_cross_sectional_ranks, add_market_relative_returns
from src.features.labels import add_direction_label, add_forward_returns, add_labels
from src.features.lag import add_default_lags, add_lagged_columns
from src.features.technical import (
    add_all_technical,
    add_atr,
    add_bollinger_bands,
    add_macd,
    add_obv,
    add_returns,
    add_rsi,
    add_stochastic,
    add_volatility,
    add_volume_zscore,
)

__all__ = [  # noqa: RUF022 (grouped by section, not alphabetical)
    # ── top-level ──
    "build_features",
    "build_ticker_features",
    # ── technical ──
    "add_all_technical",
    "add_atr",
    "add_bollinger_bands",
    "add_macd",
    "add_obv",
    "add_returns",
    "add_rsi",
    "add_stochastic",
    "add_volatility",
    "add_volume_zscore",
    # ── calendar ──
    "add_calendar_features",
    # ── lag ──
    "add_default_lags",
    "add_lagged_columns",
    # ── cross-sectional ──
    "add_cross_sectional_ranks",
    "add_market_relative_returns",
    # ── labels ──
    "add_direction_label",
    "add_forward_returns",
    "add_labels",
]
